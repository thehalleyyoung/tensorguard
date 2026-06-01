#!/usr/bin/env python3
"""Step 18 -- triage disagreements; freeze a minimal-reproducer regression suite.

Step 17 gives a minimizer; Steps 15-16 ran a large false-positive / false-
negative hunt. This step **triages** that hunt and turns its signal into a
frozen regression suite.

Triage.
    The combined fuzzing population (the 200 random clean models of Step 15 plus
    the 281 genuine injected faults of Step 16 = 481 models) produced **zero**
    TensorGuard/runtime disagreements: no false positives and no false
    negatives. We read those two committed artifacts and report the combined
    quadrant counts, so "no disagreements to fix" is an evidenced conclusion
    rather than an assumption.

Regression suite.
    Because there is no natural disagreement corpus to draw 50 reproducers from,
    we freeze the next best thing a regression suite actually protects: **50
    minimal bug reproducers** spanning a catalogue of distinct fault mechanisms
    (Linear in/out, Conv channel/kernel, invalid view/reshape, matmul inner-dim,
    broadcast add, cat non-cat-dim, flatten->Linear), each paired with a minimal
    **clean sibling**. Every buggy entry is verified to *raise at runtime* and be
    *refuted by TensorGuard*; every clean sibling is verified to *run clean* and
    be *accepted*. Frozen with expected verdicts, these become parametrized
    regression tests (`tests/test_triage.py`): if TensorGuard ever regresses --
    stops catching a bug, or starts flagging a clean sibling -- CI fails.

Deterministic: the catalogue is generated in fixed order from fixed parameter
sweeps, so the corpus and committed artifact regenerate byte-for-byte.

Usage
-----
    cd tensorguard && PYTHONPATH=. python3 evaluation/triage.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/triage.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_JSON = os.path.join(THIS_DIR, "triage_regressions.json")
OUT_MD = os.path.join(THIS_DIR, "triage_regressions.md")

DIFF_FUZZ_JSON = os.path.join(THIS_DIR, "diff_fuzz.json")
NEG_FUZZ_JSON = os.path.join(THIS_DIR, "neg_fuzz.json")

N_REGRESSIONS = 50

_HDR = ("import torch\n"
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n%s\n\n"
        "    def forward(self, x):\n%s\n")


def _mod(init: str, fwd: str) -> str:
    return _HDR % (init, fwd)


# --------------------------------------------------------------------------
# Catalogue of distinct fault mechanisms. Each builder returns
# (buggy_source, clean_source, input_shape).
# --------------------------------------------------------------------------
def cat_linear_in(F: int, G: int) -> Tuple[str, str, tuple]:
    return (_mod("        self.l = nn.Linear(%d, %d)" % (F + 1, G), "        return self.l(x)"),
            _mod("        self.l = nn.Linear(%d, %d)" % (F, G), "        return self.l(x)"),
            (4, F))


def cat_invalid_view(F: int, _: int) -> Tuple[str, str, tuple]:
    return (_mod("        pass", "        return x.view(x.size(0), %d)" % (F + 1)),
            _mod("        pass", "        return x.view(x.size(0), %d)" % F),
            (4, F))


def cat_matmul_inner(F: int, G: int) -> Tuple[str, str, tuple]:
    return (_mod("        self.W = nn.Parameter(torch.randn(%d, %d))" % (F + 1, G),
                 "        return torch.matmul(x, self.W)"),
            _mod("        self.W = nn.Parameter(torch.randn(%d, %d))" % (F, G),
                 "        return torch.matmul(x, self.W)"),
            (4, F))


def cat_add_broadcast(F: int, _: int) -> Tuple[str, str, tuple]:
    return (_mod("        self.b = nn.Parameter(torch.randn(%d))" % (F + 1), "        return x + self.b"),
            _mod("        self.b = nn.Parameter(torch.randn(%d))" % F, "        return x + self.b"),
            (4, F))


def cat_cat_noncat_dim(F: int, _: int) -> Tuple[str, str, tuple]:
    fwd = "        y = self.l(x)\n        return torch.cat([x, y], dim=0)"
    return (_mod("        self.l = nn.Linear(%d, %d)" % (F, F + 1), fwd),
            _mod("        self.l = nn.Linear(%d, %d)" % (F, F), fwd),
            (4, F))


def cat_conv_in(C: int, K: int, H: int) -> Tuple[str, str, tuple]:
    return (_mod("        self.c = nn.Conv2d(%d, %d, kernel_size=3, padding=1)" % (C + 1, K),
                 "        return self.c(x)"),
            _mod("        self.c = nn.Conv2d(%d, %d, kernel_size=3, padding=1)" % (C, K),
                 "        return self.c(x)"),
            (2, C, H, H))


def cat_conv_kernel(C: int, K: int, H: int) -> Tuple[str, str, tuple]:
    return (_mod("        self.c = nn.Conv2d(%d, %d, kernel_size=%d)" % (C, K, H + 2),
                 "        return self.c(x)"),
            _mod("        self.c = nn.Conv2d(%d, %d, kernel_size=3)" % (C, K),
                 "        return self.c(x)"),
            (2, C, H, H))


def cat_invalid_reshape(F: int, _: int) -> Tuple[str, str, tuple]:
    a = 4
    b = F // a
    return (_mod("        pass", "        return x.reshape(x.size(0), %d, %d)" % (a, b + 1)),
            _mod("        pass", "        return x.reshape(x.size(0), %d, %d)" % (a, b)),
            (4, F))


def cat_flatten_linear(C: int, H: int) -> Tuple[str, str, tuple]:
    flat = C * H * H
    return (_mod("        self.f = nn.Flatten()\n        self.l = nn.Linear(%d, 10)" % (flat + 5),
                 "        return self.l(self.f(x))"),
            _mod("        self.f = nn.Flatten()\n        self.l = nn.Linear(%d, 10)" % flat,
                 "        return self.l(self.f(x))"),
            (2, C, H, H))


def build_catalogue() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    def add(category: str, buggy: str, clean: str, shape: tuple) -> None:
        idx = sum(1 for e in entries if e["category"] == category)
        entries.append({
            "id": "%s_%d" % (category, idx),
            "category": category,
            "buggy_source": buggy,
            "clean_source": clean,
            "input_shapes": {"x": list(shape)},
        })

    for F in (4, 8, 16, 32, 64, 128):
        add("linear_in", *cat_linear_in(F, 10))
    for F in (4, 8, 16, 32, 64, 128):
        add("invalid_view", *cat_invalid_view(F, 0))
    for F in (4, 8, 16, 32, 64, 128):
        add("matmul_inner", *cat_matmul_inner(F, 4))
    for F in (3, 5, 8, 16, 32, 64):
        add("add_broadcast", *cat_add_broadcast(F, 0))
    for F in (4, 8, 16, 32, 64):
        add("cat_noncat_dim", *cat_cat_noncat_dim(F, 0))
    for (C, K, H) in ((3, 8, 16), (1, 4, 8), (3, 16, 32), (8, 8, 16), (3, 4, 8), (16, 8, 8)):
        add("conv_in", *cat_conv_in(C, K, H))
    for (C, K, H) in ((3, 8, 8), (1, 4, 6), (3, 16, 10), (8, 8, 8), (3, 4, 12), (16, 8, 6)):
        add("conv_kernel", *cat_conv_kernel(C, K, H))
    for F in (16, 24, 32, 12, 20, 28):
        add("invalid_reshape", *cat_invalid_reshape(F, 0))
    for (C, H) in ((3, 8), (1, 8), (3, 4), (8, 4), (3, 6), (16, 4)):
        add("flatten_linear", *cat_flatten_linear(C, H))

    return entries[:N_REGRESSIONS]


# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------
def runtime_raises(source: str, shape: tuple) -> bool:
    import torch
    torch.manual_seed(0)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(source)
    tmp.close()
    try:
        spec = importlib.util.spec_from_file_location("tri_mod", tmp.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        inst = mod.M()
        inst.eval()
        with torch.no_grad():
            inst(torch.rand(*shape))
        return False
    except Exception:
        return True
    finally:
        os.unlink(tmp.name)


def tensorguard_refutes(source: str, shape: tuple) -> bool:
    from src.api import verify_architecture
    result = verify_architecture(
        source, input_shapes={"x": tuple(shape)},
        max_cegar_iterations=0, soundness_mode="balanced")
    return result.bug_count > 0


# --------------------------------------------------------------------------
# Disagreement triage from committed fuzz artifacts
# --------------------------------------------------------------------------
def disagreement_summary() -> Dict[str, Any]:
    with open(DIFF_FUZZ_JSON, "r", encoding="utf-8") as fh:
        diff = json.load(fh)
    with open(NEG_FUZZ_JSON, "r", encoding="utf-8") as fh:
        neg = json.load(fh)
    clean_models = diff["summary"]["admitted_clean_executing"]
    false_positives = diff["summary"]["false_positives"]
    fault_models = neg["summary"]["genuine_faults"]
    false_negatives = neg["summary"]["false_negatives"]
    return {
        "clean_models_examined": clean_models,
        "faulty_models_examined": fault_models,
        "population_total": clean_models + fault_models,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_disagreements": false_positives + false_negatives,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(check: bool = False) -> Dict[str, Any]:
    catalogue = build_catalogue()
    assert len(catalogue) == N_REGRESSIONS, "expected %d entries" % N_REGRESSIONS

    regressions: List[Dict[str, Any]] = []
    for e in catalogue:
        shape = tuple(e["input_shapes"]["x"])
        bug_raises = runtime_raises(e["buggy_source"], shape)
        bug_refuted = tensorguard_refutes(e["buggy_source"], shape)
        clean_raises = runtime_raises(e["clean_source"], shape)
        clean_refuted = tensorguard_refutes(e["clean_source"], shape)
        # Every frozen entry must satisfy all four properties.
        assert bug_raises and bug_refuted, \
            "buggy entry %s not a caught bug (rt=%s tg=%s)" % (e["id"], bug_raises, bug_refuted)
        assert (not clean_raises) and (not clean_refuted), \
            "clean sibling %s not clean (rt=%s tg=%s)" % (e["id"], clean_raises, clean_refuted)
        regressions.append({
            "id": e["id"],
            "category": e["category"],
            "input_shapes": e["input_shapes"],
            "buggy_source": e["buggy_source"],
            "clean_source": e["clean_source"],
            "expected_buggy_runtime": "raises",
            "expected_buggy_tensorguard": "REFUTED",
            "expected_clean_runtime": "clean",
            "expected_clean_tensorguard": "ACCEPTED",
        })

    by_category: Dict[str, int] = {}
    for r in regressions:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1

    artifact = {
        "meta": {
            "generated_by": "evaluation/triage.py",
            "command": "python3 evaluation/triage.py",
            "n_regressions": len(regressions),
            "n_categories": len(by_category),
        },
        "disagreement_triage": disagreement_summary(),
        "regression_suite": {
            "count": len(regressions),
            "by_category": by_category,
            "entries": regressions,
        },
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            if fh.read() != text:
                raise SystemExit("triage_regressions.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(a: Dict[str, Any]) -> str:
    t = a["disagreement_triage"]
    r = a["regression_suite"]
    lines = [
        "# Step 18 -- disagreement triage + minimal-reproducer regression suite",
        "",
        "## Disagreement triage",
        "",
        "Combined over the Step 15 clean fuzz population and the Step 16 injected-"
        "fault population:",
        "",
        "| | Count |",
        "|---|---|",
        "| Clean models examined | %d |" % t["clean_models_examined"],
        "| Faulty models examined | %d |" % t["faulty_models_examined"],
        "| Population total | %d |" % t["population_total"],
        "| False positives | %d |" % t["false_positives"],
        "| False negatives | %d |" % t["false_negatives"],
        "| **Total disagreements** | **%d** |" % t["total_disagreements"],
        "",
        "No TensorGuard/runtime disagreement was found, so there is nothing to "
        "fix; the regression suite instead freezes minimal bug reproducers.",
        "",
        "## Regression suite (%d minimal reproducers + clean siblings)" % r["count"],
        "",
        "Each buggy entry is verified to raise at runtime *and* be refuted by "
        "TensorGuard; each clean sibling is verified to run clean *and* be "
        "accepted. Replayed as parametrized tests by `tests/test_triage.py`.",
        "",
        "| Fault category | Reproducers |",
        "|---|---|",
    ]
    for cat in sorted(r["by_category"]):
        lines.append("| `%s` | %d |" % (cat, r["by_category"][cat]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    a = run(check=args.check)
    if args.check:
        print("triage_regressions.json is up to date")
        return
    t = a["disagreement_triage"]
    r = a["regression_suite"]
    print("Wrote %s and %s" % (os.path.relpath(OUT_JSON, REPO_ROOT),
                               os.path.relpath(OUT_MD, REPO_ROOT)))
    print("  triage: %d models examined, %d disagreements"
          % (t["population_total"], t["total_disagreements"]))
    print("  regression suite: %d reproducers across %d categories"
          % (r["count"], len(r["by_category"])))


if __name__ == "__main__":
    main()
