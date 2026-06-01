"""Step 20 -- precision/recall regression dashboard that blocks merges.

This aggregates the headline metrics from every committed evaluation artifact
into a single dashboard and gates them against a frozen, committed baseline
(`evaluation/dashboard_baseline.json`). In `--check` mode it recomputes the
metrics from the current artifacts and exits non-zero on **any regression**,
so it can be wired into CI / a pre-commit hook to *block merges* on a drop in
verifier quality.

Threat model (important)
------------------------
The baseline is a **reviewed regression ratchet**, not a tamper-proof boundary.
It blocks *accidental* metric drops: any intentional drop must show up as a
diff to `dashboard_baseline.json`, which a reviewer must approve. It does not
defend against a committer who edits both the code/artifacts and the baseline
in the same change -- that is what code review + CODEOWNERS on the baseline file
are for. Crucially, the dashboard reads *committed artifacts*; to catch a source
regression that left stale artifacts behind, CI runs each harness in `--check`
mode (byte-identical regeneration) **before** this dashboard gate. The dashboard
is the final aggregation gate, not the freshness gate.

Metric categories
-----------------
* **quality**  -- precision/recall/F1/FP/FN/coverage style metrics. Gated by
  direction: `higher_better` fails if it drops below baseline; `lower_better`
  fails if it rises above baseline. Integer counts are compared exactly; floats
  use a tiny epsilon (1e-9) since they derive from integer counts.
* **integrity** -- corpus sizes (number of benchmarks, fuzz models, faults,
  frozen regressions). These must **never shrink** (a smaller corpus makes the
  quality ratios non-comparable). Treated as `higher_better` with exact
  integer comparison.

Determinism: the baseline and markdown are emitted with `sort_keys=True` and a
trailing newline; `--check` compares byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(HERE, "dashboard_baseline.json")
MD_PATH = os.path.join(HERE, "dashboard.md")

# Tiny tolerance for float metrics that derive from integer counts. Integer
# metrics are compared exactly (see _is_regression).
EPS = 1e-9


@dataclass(frozen=True)
class Metric:
    key: str
    artifact: str          # filename under evaluation/
    path: str              # dotted path into the artifact JSON
    direction: str         # "higher_better" | "lower_better"
    kind: str              # "quality" | "integrity"
    label: str


# The headline metric registry. Each entry is extracted from a committed
# artifact and gated against the baseline.
METRICS: List[Metric] = [
    # --- TensorGuard precision/recall on the real benchmark confusion matrix
    Metric("tg_precision", "confusion_matrices.json",
           "confusion.tensorguard.all.precision", "higher_better", "quality",
           "TG precision (real benchmarks)"),
    Metric("tg_recall", "confusion_matrices.json",
           "confusion.tensorguard.all.recall", "higher_better", "quality",
           "TG recall (real benchmarks)"),
    Metric("tg_f1", "confusion_matrices.json",
           "confusion.tensorguard.all.f1", "higher_better", "quality",
           "TG F1 (real benchmarks)"),
    Metric("tg_false_positives", "confusion_matrices.json",
           "confusion.tensorguard.all.FP", "lower_better", "quality",
           "TG false positives (real benchmarks)"),
    Metric("tg_false_negatives", "confusion_matrices.json",
           "confusion.tensorguard.all.FN", "lower_better", "quality",
           "TG false negatives (real benchmarks)"),
    Metric("benchmark_population", "confusion_matrices.json",
           "confusion.tensorguard.all.N", "higher_better", "integrity",
           "Benchmark population size"),
    # --- Sound-mode false-positive hunt
    Metric("sound_mode_false_positives", "sound_mode_fp.json",
           "summary.false_positives", "lower_better", "quality",
           "Sound-mode false positives (clean models)"),
    Metric("sound_mode_false_positive_rate", "sound_mode_fp.json",
           "summary.false_positive_rate", "lower_better", "quality",
           "Sound-mode false-positive rate"),
    Metric("sound_mode_population", "sound_mode_fp.json",
           "summary.total", "higher_better", "integrity",
           "Sound-mode clean population size"),
    # --- Differential fuzzing
    Metric("diff_fuzz_false_positives", "diff_fuzz.json",
           "summary.false_positives", "lower_better", "quality",
           "Differential-fuzz false positives"),
    Metric("diff_fuzz_safe_coverage", "diff_fuzz.json",
           "summary.safe_coverage", "higher_better", "quality",
           "Differential-fuzz safe coverage"),
    Metric("diff_fuzz_population", "diff_fuzz.json",
           "summary.verified_safe", "higher_better", "integrity",
           "Differential-fuzz verified-safe population"),
    # --- Negative fuzzing
    Metric("neg_fuzz_recall", "neg_fuzz.json",
           "summary.recall", "higher_better", "quality",
           "Negative-fuzz recall on injected faults"),
    Metric("neg_fuzz_false_negatives", "neg_fuzz.json",
           "summary.false_negatives", "lower_better", "quality",
           "Negative-fuzz false negatives"),
    Metric("neg_fuzz_genuine_faults", "neg_fuzz.json",
           "summary.genuine_faults", "higher_better", "integrity",
           "Negative-fuzz genuine-fault population"),
    # --- Hard latent-bug recall vs the strongest dynamic baseline
    Metric("hard_recall_advantage", "hard_recall.json",
           "summary.recall_advantage", "higher_better", "quality",
           "Latent-bug recall advantage over baseline"),
    Metric("hard_recall_tensorguard_recall", "hard_recall.json",
           "summary.tensorguard_recall", "higher_better", "quality",
           "TG latent-bug recall"),
    Metric("hard_recall_tensorguard_misses", "hard_recall.json",
           "summary.tensorguard_misses", "lower_better", "quality",
           "TG latent-bug misses"),
    # --- Triage / frozen regression suite
    Metric("triage_total_disagreements", "triage_regressions.json",
           "disagreement_triage.total_disagreements", "lower_better", "quality",
           "Triage total disagreements"),
    Metric("triage_regression_suite", "triage_regressions.json",
           "regression_suite.count", "higher_better", "integrity",
           "Frozen regression-suite size"),
]


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError("missing path %r at %r" % (dotted, part))
        cur = cur[part]
    return cur


def _load_artifact(name: str) -> Any:
    path = os.path.join(HERE, name)
    with open(path, "r") as fh:
        return json.load(fh)


def compute_metrics() -> Dict[str, float]:
    """Extract every registered metric from the committed artifacts."""
    cache: Dict[str, Any] = {}
    out: Dict[str, float] = {}
    for m in METRICS:
        if m.artifact not in cache:
            cache[m.artifact] = _load_artifact(m.artifact)
        out[m.key] = _dig(cache[m.artifact], m.path)
    return out


def _finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(float(v))


def _is_regression(direction: str, current: Any, baseline: Any) -> bool:
    """True if `current` is worse than `baseline` for the given direction.

    A non-finite / missing current value is always a regression. Integer pairs
    are compared exactly; floats use EPS to absorb representation noise.
    """
    if not _finite_number(current) or not _finite_number(baseline):
        return True
    both_int = isinstance(current, int) and isinstance(baseline, int)
    tol = 0 if both_int else EPS
    if direction == "higher_better":
        return float(current) < float(baseline) - tol
    if direction == "lower_better":
        return float(current) > float(baseline) + tol
    raise ValueError("unknown direction %r" % direction)


@dataclass
class GateResult:
    ok: bool
    regressions: List[str]
    orphans: List[str]          # baseline keys no longer produced
    unregistered: List[str]     # produced keys absent from baseline


def gate(current: Dict[str, Any], baseline: Dict[str, Any]) -> GateResult:
    """Compare current metrics to the baseline, reporting any regression.

    `baseline` maps key -> {"value", "direction", "kind", ...}. Enforces metric
    parity in both directions so a metric cannot be silently dropped or added
    without updating the baseline.
    """
    base_metrics = baseline.get("metrics", {})
    regressions: List[str] = []
    orphans = sorted(set(base_metrics) - set(current))
    unregistered = sorted(set(current) - set(base_metrics))

    for key in sorted(set(current) & set(base_metrics)):
        entry = base_metrics[key]
        direction = entry["direction"]
        base_val = entry["value"]
        cur_val = current[key]
        if _is_regression(direction, cur_val, base_val):
            regressions.append(
                "%s: %s -> %s (%s)"
                % (key, base_val, cur_val, direction))

    ok = not regressions and not orphans and not unregistered
    return GateResult(ok=ok, regressions=regressions,
                      orphans=orphans, unregistered=unregistered)


def build_baseline(current: Dict[str, float]) -> Dict[str, Any]:
    by_key = {m.key: m for m in METRICS}
    metrics = {}
    for key, val in current.items():
        m = by_key[key]
        metrics[key] = {
            "value": val,
            "direction": m.direction,
            "kind": m.kind,
            "artifact": m.artifact,
            "path": m.path,
            "label": m.label,
        }
    return {
        "_doc": ("Frozen regression ratchet for evaluation/dashboard.py. "
                 "Higher_better metrics must not drop; lower_better must not "
                 "rise; integrity (corpus-size) metrics must not shrink. "
                 "Intentional changes appear as a reviewable diff here."),
        "metrics": metrics,
    }


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def load_baseline() -> Dict[str, Any]:
    with open(BASELINE_PATH, "r") as fh:
        return json.load(fh)


def render_markdown(current: Dict[str, Any],
                    baseline: Optional[Dict[str, Any]]) -> str:
    base_metrics = (baseline or {}).get("metrics", {})
    lines = [
        "# TensorGuard precision/recall regression dashboard",
        "",
        ("Aggregated headline metrics from the committed evaluation artifacts, "
         "gated against `dashboard_baseline.json`. Quality metrics are gated by "
         "direction; integrity (corpus-size) metrics must not shrink. See "
         "`evaluation/dashboard.py` for the threat model."),
        "",
        "| Metric | Kind | Direction | Baseline | Current | Status |",
        "|--------|------|-----------|----------|---------|--------|",
    ]
    for m in METRICS:
        cur = current.get(m.key)
        entry = base_metrics.get(m.key, {})
        base_val = entry.get("value", "-")
        if m.key in base_metrics:
            status = "REGRESSED" if _is_regression(
                m.direction, cur, base_val) else "ok"
        else:
            status = "new"
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            m.label, m.kind, m.direction, base_val, cur, status))
    lines.append("")
    return "\n".join(lines)


def run(check: bool = False, update_baseline: bool = False,
        write: bool = True) -> int:
    """Regenerate the dashboard / run the gate.

    Returns a process exit code (0 = ok, non-zero = regression / drift).
    """
    current = compute_metrics()

    if update_baseline:
        baseline = build_baseline(current)
        if write:
            with open(BASELINE_PATH, "w") as fh:
                fh.write(_dumps(baseline))
            with open(MD_PATH, "w") as fh:
                fh.write(render_markdown(current, baseline))
        print("baseline updated: %d metrics" % len(current))
        return 0

    baseline = load_baseline()
    result = gate(current, baseline)
    md = render_markdown(current, baseline)

    if check:
        problems = []
        if result.regressions:
            problems.append("REGRESSIONS:\n  " + "\n  ".join(result.regressions))
        if result.orphans:
            problems.append("baseline metrics no longer produced: "
                            + ", ".join(result.orphans))
        if result.unregistered:
            problems.append("metrics missing from baseline: "
                            + ", ".join(result.unregistered))
        if os.path.exists(MD_PATH):
            with open(MD_PATH, "r") as fh:
                if fh.read() != md:
                    problems.append("dashboard.md is stale; run `make dashboard`")
        else:
            problems.append("dashboard.md missing; run `make dashboard`")
        if problems:
            print("DASHBOARD GATE FAILED")
            for p in problems:
                print(p)
            return 1
        print("DASHBOARD GATE PASSED: %d metrics, no regressions" % len(current))
        return 0

    if write:
        with open(MD_PATH, "w") as fh:
            fh.write(md)
    print(md)
    if not result.ok:
        print("WARNING: gate would fail (regressions/drift present)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Gate against the frozen baseline; non-zero on regression.")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Rewrite dashboard_baseline.json from current artifacts.")
    args = ap.parse_args()
    return run(check=args.check, update_baseline=args.update_baseline)


if __name__ == "__main__":
    sys.exit(main())
