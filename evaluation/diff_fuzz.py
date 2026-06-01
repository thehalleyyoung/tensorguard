#!/usr/bin/env python3
"""Step 15 -- differential fuzzing for false positives.

A tool that ships inside PyTorch must never cry wolf. This harness *generates
random valid* ``nn.Module``s, runs each one once in eager PyTorch, and asserts
that TensorGuard -- in the strict ``sound`` mode -- never reports a bug on a
model that executed cleanly. Concretely this is a **differential** check:

    runtime says "ran clean"   vs   TensorGuard says "bug"   ==> false positive.

Unlike the template-based clean corpus of Step 13, the models here are built by
a *random architecture fuzzer*: it grows a random chain of shape-transforming
layers (Linear / Conv2d / BatchNorm / LayerNorm / pooling / flatten /
activations), threading the running tensor shape through each op so the network
is dimensionally valid *by construction*, then it is additionally **validated
to execute without error** in eager PyTorch before it is admitted. The random
topology -- depth, widths, channel counts, op choices, rank-2 vs rank-4 entry,
and the flatten transition -- explores a far larger space than fixed templates.

The single hard requirement is **zero false positives**: no cleanly-executing
model may be Refuted. Abstention is permitted (sound mode is conservative); we
also report SAFE coverage so the zero-FP claim is not vacuous.

The fuzzer is fully seeded, so the corpus, the verdicts, and the committed
artifact are deterministic and regenerate byte-for-byte.

Usage
-----
    cd tensorguard && PYTHONPATH=. python3 evaluation/diff_fuzz.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/diff_fuzz.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_JSON = os.path.join(THIS_DIR, "diff_fuzz.json")
OUT_MD = os.path.join(THIS_DIR, "diff_fuzz.md")

N_MODELS = 200  # random architectures attempted (seeds 0 .. N_MODELS-1)

_ACTS = ["torch.relu", "torch.tanh", "torch.sigmoid",
         "torch.nn.functional.gelu", "torch.nn.functional.silu"]

_MODEL_TMPL = '''import torch
import torch.nn as nn


class FuzzModule(nn.Module):
    def __init__(self):
        super().__init__()
%s

    def forward(self, x):
%s
'''


# --------------------------------------------------------------------------
# Random architecture fuzzer. Threads the running shape through each op so the
# emitted module is dimensionally valid by construction.
# --------------------------------------------------------------------------
class _Builder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.init: List[str] = []
        self.fwd: List[str] = []
        self.idx = 0

    def _name(self, stem: str) -> str:
        self.idx += 1
        return "%s%d" % (stem, self.idx)

    # -- rank-2 ops: shape == [B, F] ------------------------------------
    def _linear(self, shape: List[int]) -> List[int]:
        f_in = shape[-1]
        f_out = self.rng.choice([8, 16, 32, 64, 128])
        n = self._name("fc")
        self.init.append("        self.%s = nn.Linear(%d, %d)" % (n, f_in, f_out))
        self.fwd.append("        h = self.%s(h)" % n)
        return shape[:-1] + [f_out]

    def _layernorm(self, shape: List[int]) -> List[int]:
        n = self._name("ln")
        self.init.append("        self.%s = nn.LayerNorm(%d)" % (n, shape[-1]))
        self.fwd.append("        h = self.%s(h)" % n)
        return shape

    # -- rank-4 ops: shape == [B, C, H, W] ------------------------------
    def _conv(self, shape: List[int]) -> List[int]:
        c_in = shape[1]
        c_out = self.rng.choice([4, 8, 16, 32])
        n = self._name("conv")
        self.init.append(
            "        self.%s = nn.Conv2d(%d, %d, kernel_size=3, padding=1)"
            % (n, c_in, c_out))
        self.fwd.append("        h = self.%s(h)" % n)
        return [shape[0], c_out, shape[2], shape[3]]

    def _batchnorm(self, shape: List[int]) -> List[int]:
        n = self._name("bn")
        self.init.append("        self.%s = nn.BatchNorm2d(%d)" % (n, shape[1]))
        self.fwd.append("        h = self.%s(h)" % n)
        return shape

    def _maxpool(self, shape: List[int]) -> List[int]:
        n = self._name("pool")
        self.init.append("        self.%s = nn.MaxPool2d(kernel_size=2)" % n)
        self.fwd.append("        h = self.%s(h)" % n)
        return [shape[0], shape[1], shape[2] // 2, shape[3] // 2]

    # -- shared ops -----------------------------------------------------
    def _act(self, shape: List[int]) -> List[int]:
        act = self.rng.choice(_ACTS)
        self.fwd.append("        h = %s(h)" % act)
        return shape

    def _flatten(self, shape: List[int]) -> List[int]:
        n = self._name("flat")
        self.init.append("        self.%s = nn.Flatten()" % n)
        self.fwd.append("        h = self.%s(h)" % n)
        flat = 1
        for d in shape[1:]:
            flat *= d
        return [shape[0], flat]

    def build(self) -> Tuple[str, Dict[str, tuple]]:
        rng = self.rng
        rank4 = rng.random() < 0.5
        if rank4:
            batch = rng.choice([1, 2, 4])
            cin = rng.choice([1, 3])
            side = rng.choice([8, 16, 32])  # power-of-two friendly for pooling
            shape = [batch, cin, side, side]
            input_shape = (batch, cin, side, side)
            n_ops = rng.randint(2, 5)
            for _ in range(n_ops):
                choices = ["conv", "batchnorm", "act"]
                if shape[2] >= 2 and shape[3] >= 2 and shape[2] % 2 == 0 \
                        and shape[3] % 2 == 0:
                    choices.append("maxpool")
                op = rng.choice(choices)
                shape = getattr(self, "_" + op)(shape)
            # Optionally flatten + linear head.
            if rng.random() < 0.7:
                shape = self._flatten(shape)
                for _ in range(rng.randint(1, 3)):
                    if rng.random() < 0.5:
                        shape = self._act(shape)
                    else:
                        shape = self._linear(shape)
        else:
            batch = rng.choice([1, 4, 8, 16])
            feat = rng.choice([8, 16, 32, 64, 128])
            shape = [batch, feat]
            input_shape = (batch, feat)
            n_ops = rng.randint(2, 6)
            for _ in range(n_ops):
                op = rng.choice(["linear", "act", "layernorm"])
                shape = getattr(self, "_" + op)(shape)

        self.fwd.insert(0, "        h = x")
        self.fwd.append("        return h")
        init = "\n".join(self.init) if self.init else "        pass"
        src = _MODEL_TMPL % (init, "\n".join(self.fwd))
        return src, {"x": input_shape}


def build_model(seed: int) -> Tuple[str, Dict[str, tuple]]:
    return _Builder(random.Random(seed)).build()


# --------------------------------------------------------------------------
# Differential check: runtime execution vs TensorGuard sound-mode verdict
# --------------------------------------------------------------------------
def _load(source: str):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(source)
    tmp.close()
    try:
        spec = importlib.util.spec_from_file_location("fuzz_mod", tmp.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.unlink(tmp.name)


def runtime_runs_clean(source: str, input_shapes: Dict[str, tuple]) -> bool:
    import torch
    torch.manual_seed(0)
    try:
        mod = _load(source)
        model = mod.FuzzModule()
        model.eval()
        args = [torch.rand(*[int(d) for d in s]) for s in input_shapes.values()]
        with torch.no_grad():
            model(*args)
        return True
    except Exception:
        return False


def tensorguard_verdict(source: str, input_shapes: Dict[str, tuple]) -> Tuple[str, int]:
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


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(check: bool = False, n_models: int = N_MODELS,
        write: bool = True) -> Dict[str, Any]:
    admitted = 0
    skipped_nonexec = 0
    safe = abstain = false_pos = 0
    false_positives: List[Dict[str, Any]] = []
    rank_counts = {"rank2": 0, "rank4": 0}

    for seed in range(n_models):
        source, shapes = build_model(seed)
        if not runtime_runs_clean(source, shapes):
            skipped_nonexec += 1
            continue
        admitted += 1
        rank = "rank4" if len(next(iter(shapes.values()))) == 4 else "rank2"
        rank_counts[rank] += 1
        verdict, bug_count = tensorguard_verdict(source, shapes)
        if verdict == "REFUTED":
            false_pos += 1
            false_positives.append(
                {"seed": seed, "bug_count": bug_count,
                 "input_shapes": {k: list(v) for k, v in shapes.items()},
                 "source": source})
        elif verdict == "SAFE":
            safe += 1
        else:
            abstain += 1

    coverage = round(safe / admitted, 4) if admitted else 0.0
    artifact = {
        "meta": {
            "generated_by": "evaluation/diff_fuzz.py",
            "command": "python3 evaluation/diff_fuzz.py",
            "n_seeds": n_models,
            "soundness_mode": "sound",
            "design": (
                "randomly grow a valid nn.Module by threading the running "
                "tensor shape through each op, validate it executes in eager "
                "PyTorch, then assert TensorGuard never Refutes a cleanly "
                "executing model (differential false-positive hunt)"
            ),
        },
        "summary": {
            "seeds_attempted": n_models,
            "admitted_clean_executing": admitted,
            "skipped_nonexecuting": skipped_nonexec,
            "false_positives": false_pos,
            "verified_safe": safe,
            "abstained": abstain,
            "safe_coverage": coverage,
            "rank_breakdown": rank_counts,
        },
        "false_positive_models": false_positives,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            if fh.read() != text:
                raise SystemExit("diff_fuzz.json is stale; regenerate it")
        return artifact

    if not write:
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(a: Dict[str, Any]) -> str:
    s = a["summary"]
    lines = [
        "# Step 15 -- differential fuzzing for false positives",
        "",
        "A random-architecture fuzzer grows valid `nn.Module`s by threading the "
        "running tensor shape through each op (Linear / Conv2d / BatchNorm / "
        "LayerNorm / pooling / flatten / activations), validates each one "
        "**executes without error** in eager PyTorch, then asserts TensorGuard "
        "in strict `sound` mode never Refutes a cleanly-executing model. "
        "Generated by `evaluation/diff_fuzz.py` (fully seeded).",
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| Random seeds attempted | %d |" % s["seeds_attempted"],
        "| Admitted (executed clean) | %d |" % s["admitted_clean_executing"],
        "| Skipped (did not execute) | %d |" % s["skipped_nonexecuting"],
        "| **False positives (Refuted clean code)** | **%d** |" % s["false_positives"],
        "| Verified SAFE | %d |" % s["verified_safe"],
        "| Abstained | %d |" % s["abstained"],
        "| SAFE coverage | %.3f |" % s["safe_coverage"],
        "",
        "Across the admitted, cleanly-executing random models TensorGuard "
        "produced **%d false positives**. Coverage (the share verified SAFE "
        "rather than abstained) is %.3f, so the zero-false-positive result is "
        "not a vacuous \"always abstain\"."
        % (s["false_positives"], s["safe_coverage"]),
        "",
        "Rank breakdown of admitted models: rank-2 (vector) %d, rank-4 "
        "(image) %d." % (s["rank_breakdown"]["rank2"],
                         s["rank_breakdown"]["rank4"]),
        "",
    ]
    if a["false_positive_models"]:
        lines.append("## False positives")
        lines.append("")
        for fp in a["false_positive_models"]:
            lines.append("- seed %d (%d reported bugs)" % (fp["seed"], fp["bug_count"]))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=N_MODELS)
    args = ap.parse_args()
    a = run(check=args.check, n_models=args.n)
    if args.check:
        print("diff_fuzz.json is up to date")
        return
    s = a["summary"]
    print("Wrote %s and %s" % (os.path.relpath(OUT_JSON, REPO_ROOT),
                               os.path.relpath(OUT_MD, REPO_ROOT)))
    print("  attempted: %d | admitted: %d | false positives: %d | SAFE: %d "
          "| abstain: %d | coverage: %.3f"
          % (s["seeds_attempted"], s["admitted_clean_executing"],
             s["false_positives"], s["verified_safe"], s["abstained"],
             s["safe_coverage"]))
    if a["false_positive_models"]:
        print("  FALSE POSITIVES at seeds:",
              [fp["seed"] for fp in a["false_positive_models"]])


if __name__ == "__main__":
    main()
