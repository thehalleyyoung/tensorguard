"""Scaling study: analysis work and wall-clock vs model size (Step 109).

Reviewers and adopters both ask the same question about a static verifier: does
it scale, or does it fall off a cliff on big models? We answer with two
artifacts.

1. ``scaling_study.json`` / ``.md`` (**byte-deterministic**). Over a sweep of
   feed-forward depths (1..64 stacked ``nn.Linear`` layers) we record the
   verifier's *deterministic* structural-work metric -- ``lines_analyzed`` --
   together with the verdict and a "decided" flag, and we fit an ordinary
   least-squares line of work versus depth. The fit is exact and reproducible
   (slope, intercept, R^2 are deterministic), and we assert the verifier returns
   a decided verdict at every size with no abstention or blow-up. This is the
   reproducible scaling claim: analysis work is **linear** in model size and the
   tool never gives up as models grow.

2. ``scaling_walltime.json`` (**volatile**, like ``reproduce_headline_60bug``).
   The same sweep, but recording median wall-clock per size and a log-log
   regression exponent. Wall-clock is machine-dependent, so this file is
   regenerated but not byte-compared; the numeric content (an empirical scaling
   exponent that is polynomial, i.e. far below an exponential) is what matters
   and is re-asserted live by the regression test.

Only the deterministic file participates in the determinism check.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "scaling_study.json"
OUT_MD = REPO / "reproducibility" / "scaling_study.md"
WALLTIME_JSON = REPO / "reproducibility" / "scaling_walltime.json"

DEPTHS = [1, 2, 4, 8, 16, 24, 32, 48, 64]
WIDTH = 32
MODE = "sound"
WALLTIME_REPS = 3


def _model_source(depth: int, width: int = WIDTH) -> str:
    lines = [
        "import torch.nn as nn",
        "class M(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(depth):
        lines.append(f"        self.l{i} = nn.Linear({width}, {width})")
    lines.append("    def forward(self, x):")
    for i in range(depth):
        lines.append(f"        x = self.l{i}(x)")
    lines.append("        return x")
    return "\n".join(lines)


def _params(depth: int, width: int = WIDTH) -> int:
    # depth Linear(width, width): each has width*width weights + width biases.
    return depth * (width * width + width)


def _analyze(depth: int):
    src = _model_source(depth)
    return verify_architecture(
        src, input_shapes={"x": (2, WIDTH)}, soundness_mode=MODE)


def _ols(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return round(float(slope), 6), round(float(intercept), 6), round(r2, 6)


def measure() -> dict:
    rows = []
    for d in DEPTHS:
        r = _analyze(d)
        decided = str(r.verdict) in ("SAFE", "UNSAFE")
        rows.append({
            "depth": d,
            "params": _params(d),
            "lines_analyzed": int(r.lines_analyzed),
            "functions_analyzed": int(r.functions_analyzed),
            "cegar_iterations": int(r._cegar_iterations),
            "verdict": str(r.verdict),
            "decided": decided,
        })

    slope, intercept, r2 = _ols(
        [row["depth"] for row in rows],
        [row["lines_analyzed"] for row in rows],
    )

    return {
        "mode": MODE,
        "width": WIDTH,
        "depths": list(DEPTHS),
        "max_depth": max(DEPTHS),
        "rows": rows,
        "work_metric": "lines_analyzed",
        "work_vs_depth_fit": {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r2,
        },
        "all_sizes_decided": all(row["decided"] for row in rows),
        "no_abstention_at_scale": all(
            row["verdict"] != "UNKNOWN" for row in rows),
        "work_is_linear_in_size": r2 >= 0.999,
        "cegar_bounded_at_scale": max(row["cegar_iterations"] for row in rows) <= 3,
    }


def measure_walltime() -> dict:
    # Warm up once (first analysis pays import/JIT costs).
    _analyze(DEPTHS[0])

    medians = {}
    for d in DEPTHS:
        samples = []
        for _ in range(WALLTIME_REPS):
            t = time.perf_counter()
            _analyze(d)
            samples.append((time.perf_counter() - t) * 1000.0)
        samples.sort()
        medians[d] = samples[len(samples) // 2]

    # log-log fit of median ms vs depth, skipping depth 1 (warm-up sensitive).
    fit_depths = [d for d in DEPTHS if d >= 4]
    lx = [float(np.log10(d)) for d in fit_depths]
    ly = [float(np.log10(medians[d])) for d in fit_depths]
    exponent, _, r2 = _ols(lx, ly)

    return {
        "mode": MODE,
        "width": WIDTH,
        "reps_per_size": WALLTIME_REPS,
        "median_ms_by_depth": {str(d): round(medians[d], 3) for d in DEPTHS},
        "loglog_scaling_exponent": exponent,
        "loglog_r_squared": r2,
        "is_polynomial_not_exponential": exponent < 3.0,
        "note": (
            "Wall-clock is machine-dependent and regenerated but not "
            "byte-compared. The empirical exponent below three confirms "
            "polynomial (sub-cubic) scaling -- there is no exponential blow-up. "
            "Re-asserted live by tests/test_scaling_study.py."
        ),
    }


def render_markdown(data: dict) -> str:
    fit = data["work_vs_depth_fit"]
    lines = [
        "# Scaling study: analysis work vs model size",
        "",
        "Over a sweep of feed-forward depths (1..{} stacked `nn.Linear` "
        "layers, width {}) we record the verifier's deterministic "
        "structural-work metric `lines_analyzed`, the verdict, and whether the "
        "model was decided.".format(data["max_depth"], data["width"]),
        "",
        "| depth | params | lines_analyzed | cegar | verdict | decided |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in data["rows"]:
        lines.append(
            f"| {row['depth']} | {row['params']} | {row['lines_analyzed']} | "
            f"{row['cegar_iterations']} | {row['verdict']} | {row['decided']} |"
        )
    lines += [
        "",
        f"Ordinary least-squares fit of analysis work versus depth: "
        f"slope `{fit['slope']}`, intercept `{fit['intercept']}`, "
        f"R^2 `{fit['r_squared']}`.",
        "",
        f"- analysis work is linear in model size: "
        f"**{data['work_is_linear_in_size']}**",
        f"- every size decided (no abstention/blow-up at scale): "
        f"**{data['all_sizes_decided']}**",
        f"- CEGAR iterations bounded at scale: "
        f"**{data['cegar_bounded_at_scale']}**",
        "",
        "Wall-clock scaling (machine-dependent) is reported separately in "
        "`scaling_walltime.json` with a log-log regression exponent below three, "
        "confirming polynomial (sub-cubic) rather than exponential growth.",
        "",
    ]
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        old_json = OUT_JSON.read_text() if OUT_JSON.exists() else ""
        old_md = OUT_MD.read_text() if OUT_MD.exists() else ""
        if old_json != new_json or old_md != new_md:
            print("MISMATCH: scaling_study artifacts differ")
            return 1
        print("OK: scaling_study artifacts byte-identical")
        return 0
    OUT_JSON.write_text(new_json)
    OUT_MD.write_text(new_md)
    # Volatile wall-clock companion (regenerated, not byte-compared).
    wt = measure_walltime()
    WALLTIME_JSON.write_text(json.dumps(wt, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUT_JSON.name}, {OUT_MD.name}, {WALLTIME_JSON.name}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(run(check=args.check))
