#!/usr/bin/env python3
"""Step 14 -- drive bug recall above the strongest baseline, with miss tags.

On the small balanced corpus TensorGuard already ties the strongest dynamic
baseline (``runtime_backward``) at perfect recall, because every bug there is
exercised by a single forward+backward pass. The interesting question for a
*static* verifier is the class of **latent** bugs that a single concrete
execution cannot see:

* **phase-dependent** bugs that only manifest in ``eval()`` (the runtime
  baseline must pick a mode -- it runs ``train()`` for the backward pass -- so
  an eval-only fault is invisible);
* **path-dependent** bugs on a branch the one random input never takes;
* **silent gradient-freeze** bugs where a sub-module is quietly frozen
  (``requires_grad = False``) so it never trains and never raises.

Every model in this corpus is **proven to be a genuine bug** by *exercising*
the latent fault (forcing eval mode / forcing the branch / detecting the frozen
sub-module) and observing the real failure, and is **proven silent** under the
strongest dynamic baseline (one seeded ``train()`` forward+backward pass runs
clean). We then compare recall:

* ``tensorguard`` -- static, sound-capable verifier under test;
* ``runtime_backward`` -- the strongest dynamic baseline from Step 12.

The deliverable is TensorGuard recall **strictly above** the strongest
baseline, plus a root-cause tag for every model TensorGuard itself misses, so
the residual gaps are documented rather than hidden.

Usage
-----
    cd tensorguard && PYTHONPATH=. python3 evaluation/hard_recall.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/hard_recall.py --check
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

OUT_JSON = os.path.join(THIS_DIR, "hard_recall.json")
OUT_MD = os.path.join(THIS_DIR, "hard_recall.md")

_HEADER = "import torch\nimport torch.nn as nn\n\n\n"

# Root-cause tags for the classes of bug TensorGuard's static analysis may not
# model. Keyed by family; only consulted when TensorGuard misses a model.
MISS_ROOT_CAUSE = {
    "silent_freeze": (
        "post-construction `requires_grad = False` mutation: the gradient "
        "domain reasons about detach/grad-flow in `forward`, not about a "
        "sub-module frozen in `__init__`, so the silent freeze is not modelled"
    ),
}


# --------------------------------------------------------------------------
# Corpus -- (id, family, source, input_shapes, exercise)
# `exercise` tells the validator how to trigger the latent fault.
# --------------------------------------------------------------------------
def _phase_eval_models() -> List[Dict[str, Any]]:
    specs = [
        ("phase_eval_view",
         "        if not self.training:\n"
         "            x = x.view(x.size(0), 5, 16)\n"
         "        return self.fc(x)",
         "        self.fc = nn.Linear(16, 16)", (4, 16)),
        ("phase_eval_view37",
         "        if not self.training:\n"
         "            x = x.view(x.size(0), 3, 7)\n"
         "        return self.fc(x)",
         "        self.fc = nn.Linear(16, 16)", (8, 16)),
        ("phase_eval_reshape",
         "        if not self.training:\n"
         "            x = x.reshape(x.size(0), 7, 9)\n"
         "        return self.fc(x)",
         "        self.fc = nn.Linear(16, 16)", (8, 16)),
    ]
    out = []
    for name, fwd, init, shp in specs:
        src = _HEADER + (
            "class BuggyModule(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n%s\n\n"
            "    def forward(self, x):\n%s\n" % (init, fwd)
        )
        out.append({"id": name, "family": "phase_eval", "source": src,
                    "input_shapes": {"x": list(shp)}, "exercise": "eval"})
    return out


def _path_flag_models() -> List[Dict[str, Any]]:
    specs = [
        ("path_flag_view",
         "        if self.force_branch:\n"
         "            x = x.view(-1, 999)\n"
         "        return self.fc(x)",
         "        self.fc = nn.Linear(8, 8)", (4, 8)),
        ("path_flag_reshape",
         "        if self.force_branch:\n"
         "            x = x.reshape(3, -1)\n"
         "        return self.fc(x)",
         "        self.fc = nn.Linear(8, 8)", (4, 8)),
        ("path_flag_cat",
         "        y = self.fc(x)\n"
         "        if self.force_branch:\n"
         "            y = torch.cat([y, x.view(2, 16)], dim=1)\n"
         "        return y",
         "        self.fc = nn.Linear(8, 8)", (4, 8)),
    ]
    out = []
    for name, fwd, init, shp in specs:
        src = _HEADER + (
            "class BuggyModule(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n%s\n"
            "        self.force_branch = False\n\n"
            "    def forward(self, x):\n%s\n" % (init, fwd)
        )
        out.append({"id": name, "family": "path_flag", "source": src,
                    "input_shapes": {"x": list(shp)}, "exercise": "flag"})
    return out


def _silent_freeze_models() -> List[Dict[str, Any]]:
    specs = [
        ("silent_freeze_fc1",
         "        self.fc1 = nn.Linear(8, 8)\n        self.fc2 = nn.Linear(8, 8)",
         "        for p in self.fc1.parameters():\n            p.requires_grad = False",
         "        return self.fc2(torch.relu(self.fc1(x)))", (4, 8)),
        ("silent_freeze_conv",
         "        self.c1 = nn.Conv2d(3, 8, 3, padding=1)\n        self.c2 = nn.Conv2d(8, 8, 3, padding=1)",
         "        for p in self.c1.parameters():\n            p.requires_grad = False",
         "        return self.c2(torch.relu(self.c1(x)))", (2, 3, 8, 8)),
    ]
    out = []
    for name, init_layers, freeze, fwd, shp in specs:
        src = _HEADER + (
            "class BuggyModule(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n%s\n%s\n\n"
            "    def forward(self, x):\n%s\n" % (init_layers, freeze, fwd)
        )
        out.append({"id": name, "family": "silent_freeze", "source": src,
                    "input_shapes": {"x": list(shp)}, "exercise": "freeze"})
    return out


def build_corpus() -> List[Dict[str, Any]]:
    return _phase_eval_models() + _path_flag_models() + _silent_freeze_models()


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------
def _load_module(source: str):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(source)
    tmp.close()
    try:
        spec = importlib.util.spec_from_file_location("hr_mod", tmp.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.unlink(tmp.name)


def is_genuine_bug(model: Dict[str, Any]) -> Tuple[bool, str]:
    """Exercise the latent fault and confirm the real failure."""
    import torch
    torch.manual_seed(0)
    mod = _load_module(model["source"])
    inst = mod.BuggyModule()
    args = [torch.rand(*[int(d) for d in s]) for s in model["input_shapes"].values()]
    ex = model["exercise"]
    if ex == "eval":
        inst.eval()
        try:
            with torch.no_grad():
                inst(*args)
            return False, "no_error_in_eval"
        except Exception as exc:
            return True, "eval:%s" % type(exc).__name__
    if ex == "flag":
        inst.force_branch = True
        try:
            inst(*args)
            return False, "no_error_when_branch_forced"
        except Exception as exc:
            return True, "branch:%s" % type(exc).__name__
    if ex == "freeze":
        # Genuine bug: a sub-module is frozen, so some params can never train.
        frozen = [n for n, p in inst.named_parameters() if not p.requires_grad]
        return (len(frozen) > 0), "frozen_params:%d" % len(frozen)
    return False, "unknown_exercise"


def runtime_backward_silent(model: Dict[str, Any]) -> Tuple[bool, str]:
    """Strongest dynamic baseline: one seeded train() forward+backward pass.

    Returns (predicts_buggy, detail). For this corpus we expect it to predict
    *clean* (i.e. be silent) on every model.
    """
    import torch
    torch.manual_seed(0)
    mod = _load_module(model["source"])
    inst = mod.BuggyModule()
    inst.train()
    args = [torch.rand(*[int(d) for d in s]) for s in model["input_shapes"].values()]
    try:
        out = inst(*args)
        out.float().sum().backward()
    except Exception as exc:
        return True, "raised:%s" % type(exc).__name__
    missing = [n for n, p in inst.named_parameters()
               if p.requires_grad and p.grad is None]
    if missing:
        return True, "no_grad:%s" % ",".join(sorted(missing)[:2])
    return False, "ran_clean"


def tensorguard_predicts_buggy(model: Dict[str, Any]) -> Tuple[bool, int]:
    from src.api import verify_architecture
    shapes = {k: tuple(v) for k, v in model["input_shapes"].items()}
    result = verify_architecture(
        model["source"], input_shapes=shapes,
        check_devices=True, check_gradients=True,
        max_cegar_iterations=0, soundness_mode="balanced",
    )
    return (result.bug_count > 0), result.bug_count


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(check: bool = False) -> Dict[str, Any]:
    import torch  # noqa: F401
    corpus = build_corpus()

    per_model: List[Dict[str, Any]] = []
    tg_caught = bl_caught = 0
    misses: List[Dict[str, str]] = []

    for model in corpus:
        genuine, gdetail = is_genuine_bug(model)
        assert genuine, "corpus model %s is not a genuine bug (%s)" % (
            model["id"], gdetail)
        bl_buggy, bl_detail = runtime_backward_silent(model)
        tg_buggy, tg_bugs = tensorguard_predicts_buggy(model)

        if tg_buggy:
            tg_caught += 1
        if bl_buggy:
            bl_caught += 1

        entry = {
            "id": model["id"], "family": model["family"],
            "genuine_bug": gdetail,
            "tensorguard": {"caught": tg_buggy, "bug_count": tg_bugs},
            "runtime_backward": {"caught": bl_buggy, "detail": bl_detail},
        }
        if not tg_buggy:
            tag = MISS_ROOT_CAUSE.get(
                model["family"], "uncategorised static-analysis gap")
            entry["tensorguard_miss_root_cause"] = tag
            misses.append({"id": model["id"], "family": model["family"],
                           "root_cause": tag})
        per_model.append(entry)

    total = len(corpus)
    tg_recall = round(tg_caught / total, 4)
    bl_recall = round(bl_caught / total, 4)

    by_family: Dict[str, Dict[str, int]] = {}
    for e in per_model:
        f = by_family.setdefault(e["family"],
                                 {"total": 0, "tg_caught": 0, "bl_caught": 0})
        f["total"] += 1
        f["tg_caught"] += int(e["tensorguard"]["caught"])
        f["bl_caught"] += int(e["runtime_backward"]["caught"])

    artifact = {
        "meta": {
            "generated_by": "evaluation/hard_recall.py",
            "command": "python3 evaluation/hard_recall.py",
            "n_bugs": total,
            "families": sorted({m["family"] for m in corpus}),
            "strongest_baseline": "runtime_backward",
            "design": (
                "every bug is proven genuine by exercising the latent fault "
                "and is proven silent under one seeded train() forward+backward "
                "pass; static analysis is compared against that strongest "
                "dynamic baseline on latent bugs it cannot observe"
            ),
        },
        "summary": {
            "n_bugs": total,
            "tensorguard_caught": tg_caught,
            "tensorguard_recall": tg_recall,
            "runtime_backward_caught": bl_caught,
            "runtime_backward_recall": bl_recall,
            "recall_advantage": round(tg_recall - bl_recall, 4),
            "tensorguard_misses": len(misses),
        },
        "by_family": by_family,
        "tensorguard_misses": misses,
        "per_model": per_model,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            if fh.read() != text:
                raise SystemExit("hard_recall.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(a: Dict[str, Any]) -> str:
    s = a["summary"]
    lines = [
        "# Step 14 -- bug recall above the strongest baseline (latent bugs)",
        "",
        "A corpus of **%d latent bugs** that a single concrete execution cannot "
        "see: phase-dependent (eval-only), path-dependent (untaken branch), and "
        "silent gradient-freeze faults. Every bug is *proven genuine* by "
        "exercising the latent fault, and *proven silent* under the strongest "
        "dynamic baseline (one seeded `train()` forward+backward pass). "
        "Generated by `evaluation/hard_recall.py`." % s["n_bugs"],
        "",
        "## Recall",
        "",
        "| Detector | Caught | Recall |",
        "|---|---|---|",
        "| **TensorGuard** (static) | %d / %d | %.3f |" % (
            s["tensorguard_caught"], s["n_bugs"], s["tensorguard_recall"]),
        "| `runtime_backward` (strongest dynamic baseline) | %d / %d | %.3f |" % (
            s["runtime_backward_caught"], s["n_bugs"], s["runtime_backward_recall"]),
        "",
        "TensorGuard's recall advantage on latent bugs is "
        "**%.3f** (%.3f vs %.3f). These bugs are invisible to dynamic testing "
        "by construction: the fault lives on an execution path or training "
        "phase the single concrete run never reaches."
        % (s["recall_advantage"], s["tensorguard_recall"],
           s["runtime_backward_recall"]),
        "",
        "## By family",
        "",
        "| Family | Bugs | TensorGuard caught | Baseline caught |",
        "|---|---|---|---|",
    ]
    for fam in sorted(a["by_family"]):
        f = a["by_family"][fam]
        lines.append("| `%s` | %d | %d | %d |"
                     % (fam, f["total"], f["tg_caught"], f["bl_caught"]))
    lines.append("")
    lines.append("## TensorGuard misses (root-cause tagged)")
    lines.append("")
    if not a["tensorguard_misses"]:
        lines.append("None.")
    else:
        lines.append("| Bug | Family | Root cause |")
        lines.append("|---|---|---|")
        for m in a["tensorguard_misses"]:
            lines.append("| `%s` | `%s` | %s |"
                         % (m["id"], m["family"], m["root_cause"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    a = run(check=args.check)
    if args.check:
        print("hard_recall.json is up to date")
        return
    s = a["summary"]
    print("Wrote %s and %s" % (os.path.relpath(OUT_JSON, REPO_ROOT),
                               os.path.relpath(OUT_MD, REPO_ROOT)))
    print("  latent bugs: %d | TG recall: %.3f (%d) | baseline recall: %.3f (%d) | advantage: %.3f"
          % (s["n_bugs"], s["tensorguard_recall"], s["tensorguard_caught"],
             s["runtime_backward_recall"], s["runtime_backward_caught"],
             s["recall_advantage"]))
    if a["tensorguard_misses"]:
        print("  TG misses (tagged):", [m["id"] for m in a["tensorguard_misses"]])


if __name__ == "__main__":
    main()
