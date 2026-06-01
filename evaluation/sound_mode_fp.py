#!/usr/bin/env python3
"""Step 13 -- drive the sound-mode false-positive rate to 0% on clean code.

A single false positive destroys trust in a tool that is meant to ship inside
PyTorch, so this harness hunts aggressively for one. It generates a large,
diverse corpus of *clean* PyTorch ``nn.Module``s -- every model is constructed
so its dimensions match and is then **validated to execute without error in
eager PyTorch** before it is admitted -- and runs TensorGuard in the strict
``sound`` soundness mode on each. The single hard requirement is:

    no clean, executing model may be Refuted (bug_count > 0).

Abstention is allowed (sound mode is conservative); a false *alarm* is not.
We additionally report **coverage** (the fraction of clean models that
TensorGuard actually verifies SAFE rather than abstaining), because a
"0% false positives" claim is only meaningful if the verifier is not trivially
abstaining on everything.

The generated models are produced from seeded templates, so the corpus -- and
therefore the verdicts and the committed artifact -- are deterministic.

Usage
-----
    cd tensorguard && PYTHONPATH=. python3 evaluation/sound_mode_fp.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/sound_mode_fp.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import tempfile
from typing import Any, Dict, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from real_benchmarks import load  # noqa: E402

OUT_JSON = os.path.join(THIS_DIR, "sound_mode_fp.json")
OUT_MD = os.path.join(THIS_DIR, "sound_mode_fp.md")

SEED = 1234
N_PER_FAMILY = 12  # generated models attempted per family


# --------------------------------------------------------------------------
# Clean-model generators. Each returns (source, input_shapes) for a model that
# is dimensionally consistent by construction.
# --------------------------------------------------------------------------
_ACTS = ["torch.relu", "torch.tanh", "torch.sigmoid", "torch.nn.functional.gelu"]


def _mlp(rng: random.Random) -> Tuple[str, Dict[str, tuple]]:
    depth = rng.randint(2, 5)
    batch = rng.choice([1, 8, 16, 32])
    dims = [rng.choice([16, 32, 64, 128, 256])]
    for _ in range(depth):
        dims.append(rng.choice([10, 16, 32, 64, 128]))
    layers = "\n".join(
        "        self.fc%d = nn.Linear(%d, %d)" % (i, dims[i], dims[i + 1])
        for i in range(depth)
    )
    body = "x"
    for i in range(depth):
        act = rng.choice(_ACTS)
        body = "%s(self.fc%d(%s))" % (act, i, body) if i < depth - 1 else "self.fc%d(%s)" % (i, body)
    src = _MODEL_TMPL.format(init=layers, forward="        return %s" % body)
    return src, {"x": (batch, dims[0])}


def _residual_mlp(rng: random.Random) -> Tuple[str, Dict[str, tuple]]:
    width = rng.choice([32, 64, 128])
    blocks = rng.randint(1, 4)
    batch = rng.choice([4, 8, 16])
    lines = []
    for i in range(blocks):
        lines.append("        self.a%d = nn.Linear(%d, %d)" % (i, width, width))
        lines.append("        self.b%d = nn.Linear(%d, %d)" % (i, width, width))
    fwd = ["        h = x"]
    for i in range(blocks):
        act = rng.choice(_ACTS)
        fwd.append("        h = h + self.b%d(%s(self.a%d(h)))" % (i, act, i))
    fwd.append("        return h")
    src = _MODEL_TMPL.format(init="\n".join(lines), forward="\n".join(fwd))
    return src, {"x": (batch, width)}


def _cnn(rng: random.Random) -> Tuple[str, Dict[str, tuple]]:
    batch = rng.choice([1, 2, 4, 8])
    cin = rng.choice([1, 3])
    side = rng.choice([16, 28, 32])
    chans = [cin]
    convs = rng.randint(1, 3)
    for _ in range(convs):
        chans.append(rng.choice([8, 16, 32]))
    lines = []
    for i in range(convs):
        lines.append(
            "        self.c%d = nn.Conv2d(%d, %d, kernel_size=3, padding=1)"
            % (i, chans[i], chans[i + 1])
        )
        if rng.random() < 0.5:
            lines.append("        self.bn%d = nn.BatchNorm2d(%d)" % (i, chans[i + 1]))
    have_bn = ["bn%d" % i in "\n".join(lines) for i in range(convs)]
    fwd = ["        h = x"]
    for i in range(convs):
        act = rng.choice(_ACTS)
        if have_bn[i]:
            fwd.append("        h = %s(self.bn%d(self.c%d(h)))" % (act, i, i))
        else:
            fwd.append("        h = %s(self.c%d(h))" % (act, i))
    fwd.append("        return h")
    src = _MODEL_TMPL.format(init="\n".join(lines), forward="\n".join(fwd))
    return src, {"x": (batch, cin, side, side)}


def _layernorm_mlp(rng: random.Random) -> Tuple[str, Dict[str, tuple]]:
    width = rng.choice([32, 64, 128])
    seq = rng.choice([4, 8, 16])
    batch = rng.choice([2, 4])
    out = rng.choice([10, 16, 32])
    lines = [
        "        self.ln = nn.LayerNorm(%d)" % width,
        "        self.fc1 = nn.Linear(%d, %d)" % (width, width),
        "        self.fc2 = nn.Linear(%d, %d)" % (width, out),
    ]
    act = rng.choice(_ACTS)
    fwd = [
        "        h = self.ln(x)",
        "        h = %s(self.fc1(h))" % act,
        "        return self.fc2(h)",
    ]
    src = _MODEL_TMPL.format(init="\n".join(lines), forward="\n".join(fwd))
    return src, {"x": (batch, seq, width)}


def _attention(rng: random.Random) -> Tuple[str, Dict[str, tuple]]:
    dim = rng.choice([32, 64, 128])
    seq = rng.choice([4, 8, 16])
    batch = rng.choice([1, 2, 4])
    lines = [
        "        self.q = nn.Linear(%d, %d)" % (dim, dim),
        "        self.k = nn.Linear(%d, %d)" % (dim, dim),
        "        self.v = nn.Linear(%d, %d)" % (dim, dim),
        "        self.o = nn.Linear(%d, %d)" % (dim, dim),
    ]
    fwd = [
        "        q = self.q(x)",
        "        k = self.k(x)",
        "        v = self.v(x)",
        "        scores = torch.matmul(q, k.transpose(-2, -1))",
        "        attn = torch.softmax(scores, dim=-1)",
        "        ctx = torch.matmul(attn, v)",
        "        return self.o(ctx)",
    ]
    src = _MODEL_TMPL.format(init="\n".join(lines), forward="\n".join(fwd))
    return src, {"x": (batch, seq, dim)}


def _groupnorm_conv(rng: random.Random) -> Tuple[str, Dict[str, tuple]]:
    batch = rng.choice([1, 2, 4])
    chans = rng.choice([8, 16, 32])
    groups = rng.choice([1, 2, 4])
    side = rng.choice([8, 16, 32])
    lines = [
        "        self.c1 = nn.Conv2d(%d, %d, kernel_size=3, padding=1)" % (chans, chans),
        "        self.gn = nn.GroupNorm(%d, %d)" % (groups, chans),
    ]
    act = rng.choice(_ACTS)
    fwd = [
        "        h = self.c1(x)",
        "        h = self.gn(h)",
        "        return %s(h)" % act,
    ]
    src = _MODEL_TMPL.format(init="\n".join(lines), forward="\n".join(fwd))
    return src, {"x": (batch, chans, side, side)}


GENERATORS = {
    "mlp": _mlp,
    "residual_mlp": _residual_mlp,
    "cnn": _cnn,
    "layernorm_mlp": _layernorm_mlp,
    "attention": _attention,
    "groupnorm_conv": _groupnorm_conv,
}

_MODEL_TMPL = '''import torch
import torch.nn as nn


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
{init}

    def forward(self, x):
{forward}
'''


# --------------------------------------------------------------------------
# Eager validation + TensorGuard sound-mode verdict
# --------------------------------------------------------------------------
def _executes_clean(source: str, input_shapes: Dict[str, tuple]) -> bool:
    import torch
    torch.manual_seed(0)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        tmp.write(source)
        tmp.close()
        spec = importlib.util.spec_from_file_location("gen_clean", tmp.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.CleanModule()
        model.eval()
        args = [torch.rand(*[int(d) for d in s]) for s in input_shapes.values()]
        with torch.no_grad():
            model(*args)
        return True
    except Exception:
        return False
    finally:
        os.unlink(tmp.name)


def _sound_verdict(source: str, input_shapes: Dict[str, tuple]) -> Tuple[str, int]:
    from src.api import verify_architecture
    shapes = {k: tuple(v) for k, v in input_shapes.items()}
    result = verify_architecture(
        source, input_shapes=shapes, max_cegar_iterations=0,
        soundness_mode="sound",
    )
    if result.bug_count > 0:
        return "REFUTED", result.bug_count
    if getattr(result, "abstained", False):
        return "ABSTAIN", 0
    return "SAFE", 0


def generate_corpus() -> List[Dict[str, Any]]:
    """Deterministically generate validated clean models (seeded)."""
    rng = random.Random(SEED)
    corpus: List[Dict[str, Any]] = []
    for family, gen in GENERATORS.items():
        kept = 0
        attempts = 0
        # Try until N_PER_FAMILY validated-clean models are admitted.
        while kept < N_PER_FAMILY and attempts < N_PER_FAMILY * 4:
            attempts += 1
            source, shapes = gen(rng)
            if not _executes_clean(source, shapes):
                continue
            corpus.append({
                "id": "%s_%02d" % (family, kept),
                "family": family,
                "source": source,
                "input_shapes": {k: list(v) for k, v in shapes.items()},
            })
            kept += 1
    return corpus


def run(check: bool = False) -> Dict[str, Any]:
    import torch  # noqa: F401  (ensures torch present for generation/verification)

    generated = generate_corpus()

    rows: List[Dict[str, Any]] = []
    # Real, hand-written clean half of the frozen ground-truth corpus first.
    for item in load.load_items():
        if item["label"] != "clean":
            continue
        src = load.read_source(item)
        verdict, bug_count = _sound_verdict(src, item["input_shapes"])
        rows.append({
            "id": item["id"], "family": "real_benchmarks_clean",
            "source": "frozen", "verdict": verdict, "bug_count": bug_count,
        })
    # Generated clean models.
    for m in generated:
        verdict, bug_count = _sound_verdict(m["source"], m["input_shapes"])
        rows.append({
            "id": m["id"], "family": m["family"], "source": "generated",
            "verdict": verdict, "bug_count": bug_count,
        })

    total = len(rows)
    false_positives = [r for r in rows if r["verdict"] == "REFUTED"]
    n_safe = sum(1 for r in rows if r["verdict"] == "SAFE")
    n_abstain = sum(1 for r in rows if r["verdict"] == "ABSTAIN")

    by_family: Dict[str, Dict[str, int]] = {}
    for r in rows:
        fam = by_family.setdefault(
            r["family"], {"total": 0, "SAFE": 0, "ABSTAIN": 0, "REFUTED": 0})
        fam["total"] += 1
        fam[r["verdict"]] += 1

    artifact = {
        "meta": {
            "generated_by": "evaluation/sound_mode_fp.py",
            "command": "python3 evaluation/sound_mode_fp.py",
            "soundness_mode": "sound",
            "seed": SEED,
            "n_models": total,
            "n_generated": len(generated),
            "n_real_clean": total - len(generated),
            "families": sorted(GENERATORS.keys()),
            "invariant": (
                "every model is validated to execute without error in eager "
                "PyTorch before admission; the hard requirement is zero "
                "Refuted (false-positive) verdicts in sound mode"
            ),
        },
        "summary": {
            "total": total,
            "false_positives": len(false_positives),
            "false_positive_rate": round(len(false_positives) / total, 6) if total else None,
            "verified_safe": n_safe,
            "abstained": n_abstain,
            "coverage_safe": round(n_safe / total, 4) if total else None,
        },
        "by_family": by_family,
        "false_positive_ids": [r["id"] for r in false_positives],
        # Drop the large source strings from the deterministic artifact; keep
        # only the per-model verdicts (id/family/source-kind/verdict/bug_count).
        "per_model": [
            {"id": r["id"], "family": r["family"], "source": r["source"],
             "verdict": r["verdict"], "bug_count": r["bug_count"]}
            for r in rows
        ],
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            if fh.read() != text:
                raise SystemExit("sound_mode_fp.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(a: Dict[str, Any]) -> str:
    m, s = a["meta"], a["summary"]
    lines = [
        "# Step 13 -- sound-mode false-positive rate on clean code",
        "",
        "TensorGuard run in the strict `sound` soundness mode over **%d clean, "
        "executing** PyTorch models (%d real hand-written + %d generated across "
        "%d templates). Every model is validated to run without error in eager "
        "PyTorch before admission. Generated by `evaluation/sound_mode_fp.py` "
        "(seed %d)." % (s["total"], m["n_real_clean"], m["n_generated"],
                        len(m["families"]), m["seed"]),
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| Clean models tested | %d |" % s["total"],
        "| **False positives (Refuted)** | **%d** |" % s["false_positives"],
        "| False-positive rate | %s |" % (
            "0%" if s["false_positives"] == 0 else "%.4f" % s["false_positive_rate"]),
        "| Verified SAFE (non-trivial coverage) | %d |" % s["verified_safe"],
        "| Abstained (allowed in sound mode) | %d |" % s["abstained"],
        "| SAFE coverage | %.1f%% |" % (100 * s["coverage_safe"]),
        "",
        "A sound-mode verifier is allowed to abstain, but it must **never** "
        "raise a false alarm on code that runs cleanly. The false-positive "
        "count is %d. SAFE coverage of %.1f%% confirms the zero-FP result is "
        "not achieved by trivially abstaining on everything."
        % (s["false_positives"], 100 * s["coverage_safe"]),
        "",
        "## By model family",
        "",
        "| Family | Total | SAFE | Abstain | Refuted (FP) |",
        "|---|---|---|---|---|",
    ]
    for fam in sorted(a["by_family"]):
        f = a["by_family"][fam]
        lines.append("| `%s` | %d | %d | %d | %d |"
                     % (fam, f["total"], f["SAFE"], f["ABSTAIN"], f["REFUTED"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    a = run(check=args.check)
    if args.check:
        print("sound_mode_fp.json is up to date")
        return
    s = a["summary"]
    print("Wrote %s and %s" % (os.path.relpath(OUT_JSON, REPO_ROOT),
                               os.path.relpath(OUT_MD, REPO_ROOT)))
    print("  clean models: %d | false positives: %d | SAFE: %d | abstain: %d | coverage: %.1f%%"
          % (s["total"], s["false_positives"], s["verified_safe"],
             s["abstained"], 100 * s["coverage_safe"]))
    if s["false_positives"]:
        print("  FALSE POSITIVES:", a["false_positive_ids"])


if __name__ == "__main__":
    main()
