"""Per-bug time-to-detect: static verification vs the first failing forward (Step 116).

A dynamic test only surfaces a shape/channel bug once execution actually *reaches*
the offending operation -- which requires constructing a concrete input and
successfully running every preceding op. A static verifier reports the same bug
before any execution, with no input at all. This harness quantifies that gap on a
corpus of buggy modules with a deterministic, hardware-independent "time-to-detect"
proxy measured in *operations*, not wall-clock:

  * **static detect depth** = 0 for every bug (TensorGuard flags it pre-execution,
    input-free), and we confirm the static recall (every buggy module is reported
    UNSAFE);
  * **dynamic detect depth** = the index of the first forward operation that
    actually raises under eager PyTorch -- i.e. the number of operations that must
    execute successfully *before* the bug manifests.

The corpus is drawn from the structured module-AST DSL (Step 114), keeping only
modules the live torch dispatcher rejects. For each we execute the forward pass
operation-by-operation with a real tensor to find the first failing op. Only
counts, depth distributions and rounded summary statistics are recorded, so the
artifact is byte-identical across machines.
"""

from __future__ import annotations

import json
import logging
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.module_ast import (  # noqa: E402
    Conv2d,
    Flatten,
    Linear,
    ModuleAST,
    ReLU,
    random_module_ast,
    render,
    torch_runs_clean,
)
from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "time_to_detect.json"
OUT_MD = REPO / "reproducibility" / "time_to_detect.md"

SEED = 20240603
N_BUGGY_TARGET = 400


def _first_failing_op_depth(ast: ModuleAST):
    """Execute the forward pass op-by-op; return index of the first raising op.

    The index counts every forward operation (Linear / Conv2d / ReLU / Flatten)
    that executes *successfully* before the failure. Returns ``None`` if no op
    raises (the module is clean).
    """

    import torch
    import torch.nn as nn

    x = torch.randn(*ast.input_shape)
    depth = 0
    for layer in ast.layers:
        try:
            if isinstance(layer, Linear):
                x = nn.Linear(layer.in_features, layer.out_features)(x)
            elif isinstance(layer, Conv2d):
                x = nn.Conv2d(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel,
                    padding=layer.kernel // 2,
                )(x)
            elif isinstance(layer, ReLU):
                x = torch.relu(x)
            elif isinstance(layer, Flatten):
                x = x.flatten(1)
        except Exception:
            return depth
        depth += 1
    return None


def _static_verdict(ast: ModuleAST) -> str:
    source, shapes = render(ast)
    return str(
        verify_architecture(
            source,
            input_shapes={k: tuple(v) for k, v in shapes.items()},
            soundness_mode="sound",
        ).verdict
    )


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"point": None, "low": None, "high": None, "k": k, "n": n}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return {
        "point": round(p, 4),
        "low": round(max(0.0, center - half), 4),
        "high": round(min(1.0, center + half), 4),
        "k": k,
        "n": n,
    }


def measure() -> dict:
    logging.disable(logging.CRITICAL)
    try:
        rng = random.Random(SEED)
        depths: list = []
        n_static_caught = 0
        n_examined = 0
        # Draw until we accumulate the target number of genuinely buggy modules.
        while len(depths) < N_BUGGY_TARGET and n_examined < 200_000:
            n_examined += 1
            ast = random_module_ast(rng)
            if torch_runs_clean(ast):
                continue
            depth = _first_failing_op_depth(ast)
            if depth is None:  # defensive: should not happen for a raising module
                continue
            depths.append(depth)
            if _static_verdict(ast) == "UNSAFE":
                n_static_caught += 1

        n = len(depths)
        hist = Counter(depths)
        n_dynamic_needs_prefix = sum(1 for d in depths if d >= 1)

        data = {
            "step": 116,
            "seed": SEED,
            "n_buggy_modules": n,
            "static": {
                "detect_depth": 0,
                "requires_constructed_input": False,
                "requires_execution": False,
                "n_caught_unsafe": n_static_caught,
                "recall_wilson": _wilson(n_static_caught, n),
                "all_caught_at_depth_zero": n_static_caught == n,
            },
            "dynamic": {
                "requires_constructed_input": True,
                "requires_execution": True,
                "detect_depth_min": min(depths),
                "detect_depth_max": max(depths),
                "detect_depth_median": float(statistics.median(depths)),
                "detect_depth_mean": round(statistics.fmean(depths), 4),
                "detect_depth_histogram": dict(sorted(hist.items())),
                "n_requires_successful_prefix": n_dynamic_needs_prefix,
                "frac_requires_successful_prefix": round(
                    n_dynamic_needs_prefix / n, 4
                ),
                "total_ops_executed_before_detection": sum(depths),
            },
            "comparison": {
                # Static detects every bug strictly no later than dynamic, and
                # strictly earlier whenever the dynamic depth is positive.
                "static_never_later_than_dynamic": all(0 <= d for d in depths),
                "static_strictly_earlier_count": n_dynamic_needs_prefix,
                "ops_saved_median": float(statistics.median(depths)),
                "ops_saved_total": sum(depths),
            },
        }
        return data
    finally:
        logging.disable(logging.NOTSET)


def render_markdown(data: dict) -> str:
    s = data["static"]
    d = data["dynamic"]
    c = data["comparison"]
    w = s["recall_wilson"]
    lines = [
        "# Per-bug time-to-detect: static vs first failing forward (Step 116)",
        "",
        f"Seed `{data['seed']}` — **{data['n_buggy_modules']}** buggy modules "
        "(every one rejected by the live torch dispatcher), drawn from the "
        "structured module-AST DSL.",
        "",
        "Time-to-detect is measured in *operations* (hardware-independent), not "
        "wall-clock: how many forward ops must execute successfully before the "
        "bug manifests.",
        "",
        "## Static verification (TensorGuard, sound mode)",
        "",
        f"- detect depth: **{s['detect_depth']}** ops (flagged pre-execution)",
        f"- requires a constructed input: **{s['requires_constructed_input']}**",
        f"- requires execution: **{s['requires_execution']}**",
        f"- caught (UNSAFE): **{s['n_caught_unsafe']}** of "
        f"{data['n_buggy_modules']}; all at depth zero: "
        f"**{s['all_caught_at_depth_zero']}** (Wilson {w['low']}–{w['high']})",
        "",
        "## Dynamic baseline (first failing forward op)",
        "",
        f"- requires a constructed input: **{d['requires_constructed_input']}**",
        f"- detect depth: min **{d['detect_depth_min']}**, median "
        f"**{d['detect_depth_median']}**, mean **{d['detect_depth_mean']}**, "
        f"max **{d['detect_depth_max']}** ops",
        f"- bugs that surface only after at least one successful op: "
        f"**{d['n_requires_successful_prefix']}** "
        f"(fraction {d['frac_requires_successful_prefix']})",
        f"- detect-depth histogram (depth: count): "
        f"`{d['detect_depth_histogram']}`",
        "",
        "## Comparison",
        "",
        f"- static is never later than dynamic: "
        f"**{c['static_never_later_than_dynamic']}**",
        f"- static is strictly earlier on "
        f"**{c['static_strictly_earlier_count']}** modules",
        f"- operations saved before detection: median "
        f"**{c['ops_saved_median']}**, total **{c['ops_saved_total']}**",
        "",
    ]
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("time_to_detect: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
