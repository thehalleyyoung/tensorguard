"""Cross-corpus statistical meta-analysis without naive pooling (Step 265).

This artifact summarizes TensorGuard performance across the heterogeneous
evaluation corpora already committed in the repository.  The key design choice is
negative: it deliberately refuses to compute a single pooled success rate across
real-world corpora, generated clean models, mutation tests, fuzzing, and stress
sets.  Those samples are drawn from different distributions, so pooling their raw
case counts would let a large synthetic suite dominate a smaller real suite.

Instead, every source artifact becomes a suite-level Bernoulli estimate
(`successes` / `trials`) tagged by distribution family.  The primary intervals
are deterministic, distribution-stratified cluster bootstraps over suite-level
rates; suites are the resampling unit, not individual cases.  A diagnostic
DerSimonian-Laird-style random-effects summary is reported only when a
distribution has at least two suites, and it is never promoted to a headline
cross-distribution claim.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "reproducibility" / "statistical_meta_analysis.json"
OUT_MD = REPO / "reproducibility" / "statistical_meta_analysis.md"

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 265
CONFIDENCE = 0.95


@dataclass(frozen=True)
class Suite:
    name: str
    metric: str
    distribution: str
    successes: int
    trials: int
    source: str
    evidence: str

    @property
    def rate(self) -> float:
        if self.trials <= 0:
            raise ValueError(f"{self.name} has no trials")
        return self.successes / self.trials


def _load(rel: str) -> Dict[str, Any]:
    return json.loads((REPO / rel).read_text())


def _add_suite(
    suites: List[Suite],
    *,
    name: str,
    metric: str,
    distribution: str,
    successes: int,
    trials: int,
    source: str,
    evidence: str,
) -> None:
    if trials <= 0:
        raise ValueError(f"{name} from {source} has no trials")
    if successes < 0 or successes > trials:
        raise ValueError(f"{name} from {source} has invalid count {successes}/{trials}")
    suites.append(Suite(name, metric, distribution, successes, trials, source, evidence))


def collect_suites() -> List[Suite]:
    suites: List[Suite] = []

    corpus = _load("reproducibility/corpus_extended_score.json")
    sound = corpus["sound"]
    _add_suite(
        suites,
        name="extended-corpus bug recall",
        metric="bug_detection",
        distribution="real_minimized",
        successes=sound["recall_on_all_buggy"]["k"],
        trials=sound["recall_on_all_buggy"]["n"],
        source="reproducibility/corpus_extended_score.json",
        evidence="runtime-validated extended real-bug corpus",
    )
    _add_suite(
        suites,
        name="extended-corpus clean specificity",
        metric="clean_acceptance",
        distribution="real_minimized",
        successes=sound["specificity_on_decided"]["k"],
        trials=sound["specificity_on_decided"]["n"],
        source="reproducibility/corpus_extended_score.json",
        evidence="runtime-validated extended clean corpus",
    )

    hard = _load("evaluation/hard_recall.json")["summary"]
    _add_suite(
        suites,
        name="latent hard-recall bugs",
        metric="bug_detection",
        distribution="real_minimized",
        successes=hard["tensorguard_caught"],
        trials=hard["n_bugs"],
        source="evaluation/hard_recall.json",
        evidence="latent path/phase/silent bugs proven by real execution",
    )

    silent = _load("reproducibility/silent_bug_benchmark.json")["summary"]
    _add_suite(
        suites,
        name="runtime-silent semantic bugs",
        metric="bug_detection",
        distribution="real_semantic",
        successes=silent["gate_caught"],
        trials=silent["total_cases"],
        source="reproducibility/silent_bug_benchmark.json",
        evidence="non-raising PyTorch executions with independent semantic oracle",
    )

    mutation = _load("reproducibility/mutation_clean_models.json")
    _add_suite(
        suites,
        name="clean-model mutation kill rate",
        metric="bug_detection",
        distribution="synthetic_mutation",
        successes=mutation["per_mode"]["sound"]["n_killed"],
        trials=mutation["n_genuine_bug_mutants"],
        source="reproducibility/mutation_clean_models.json",
        evidence="mutants admitted only after real PyTorch runtime failure",
    )

    neg = _load("evaluation/neg_fuzz.json")["summary"]
    _add_suite(
        suites,
        name="negative fuzz injected faults",
        metric="bug_detection",
        distribution="synthetic_fuzz",
        successes=neg["caught"],
        trials=neg["genuine_faults"],
        source="evaluation/neg_fuzz.json",
        evidence="fault-injected random modules checked against eager PyTorch",
    )

    false_unknown = _load("evaluation/false_unknowns.json")["summary"]
    _add_suite(
        suites,
        name="expected-decidable false-UNKNOWN corpus",
        metric="decision",
        distribution="mixed_decidable",
        successes=false_unknown["decided"],
        trials=false_unknown["total"],
        source="evaluation/false_unknowns.json",
        evidence="ground-truthed models users expect sound mode to decide",
    )

    sound_fp = _load("evaluation/sound_mode_fp.json")["summary"]
    _add_suite(
        suites,
        name="sound-mode clean false-positive hunt",
        metric="clean_acceptance",
        distribution="clean_stress",
        successes=sound_fp["verified_safe"] - sound_fp["false_positives"],
        trials=sound_fp["verified_safe"],
        source="evaluation/sound_mode_fp.json",
        evidence="clean executable models in sound mode",
    )

    fp_stress = _load("reproducibility/fp_stress_eval.json")
    _add_suite(
        suites,
        name="100+ clean-model false-alarm stress",
        metric="clean_acceptance",
        distribution="clean_stress",
        successes=fp_stress["n_models"] - fp_stress["per_mode"]["sound"]["n_false_alarms"],
        trials=fp_stress["n_models"],
        source="reproducibility/fp_stress_eval.json",
        evidence="clean-by-construction stress corpus",
    )

    natural = _load("reproducibility/natural_distribution_study.json")
    _add_suite(
        suites,
        name="natural clean public-model sample",
        metric="clean_acceptance",
        distribution="natural_clean",
        successes=natural["n_models"] - natural["per_mode"]["sound"]["n_false_alarms"],
        trials=natural["n_models"],
        source="reproducibility/natural_distribution_study.json",
        evidence="public-repo-style clean model strata",
    )

    diff = _load("evaluation/diff_fuzz.json")["summary"]
    _add_suite(
        suites,
        name="differential clean fuzz",
        metric="clean_acceptance",
        distribution="synthetic_fuzz",
        successes=diff["verified_safe"] - diff["false_positives"],
        trials=diff["verified_safe"],
        source="evaluation/diff_fuzz.json",
        evidence="random valid modules differentially checked against live torch",
    )

    baseline = _load("reproducibility/baseline_head_to_head.json")["tensorguard_full_corpus"]
    _add_suite(
        suites,
        name="same-case baseline full-corpus recall",
        metric="bug_detection",
        distribution="real_minimized",
        successes=baseline["buggy_caught"],
        trials=baseline["buggy_total"],
        source="reproducibility/baseline_head_to_head.json",
        evidence="same-case baseline comparison corpus",
    )
    _add_suite(
        suites,
        name="same-case baseline full-corpus clean specificity",
        metric="clean_acceptance",
        distribution="real_minimized",
        successes=baseline["clean_total"] - baseline["clean_false_alarms"],
        trials=baseline["clean_total"],
        source="reproducibility/baseline_head_to_head.json",
        evidence="same-case baseline comparison clean half",
    )

    return suites


def _round(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _suite_mean_rate(suites: Sequence[Suite]) -> float:
    return sum(s.rate for s in suites) / len(suites)


def robust_suite_bootstrap(suites: Sequence[Suite], seed: int) -> Dict[str, float]:
    """Bootstrap suite-level rates, not individual cases."""
    if not suites:
        return {
            "point": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "confidence": CONFIDENCE,
            "n_resamples": 0,
        }
    rng = random.Random(seed)
    n = len(suites)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [suites[rng.randrange(n)] for _ in range(n)]
        values.append(_suite_mean_rate(sample))
    values.sort()
    alpha = 1.0 - CONFIDENCE
    return {
        "point": _round(_suite_mean_rate(suites)),
        "ci_low": _round(_percentile(values, alpha / 2.0)),
        "ci_high": _round(_percentile(values, 1.0 - alpha / 2.0)),
        "confidence": CONFIDENCE,
        "n_resamples": BOOTSTRAP_RESAMPLES,
    }


def _random_effects_summary(suites: Sequence[Suite]) -> Dict[str, Any] | None:
    """Small DerSimonian-Laird diagnostic over logit-transformed suite rates."""
    if len(suites) < 2:
        return None
    effects = []
    variances = []
    for suite in suites:
        # Continuity correction keeps perfect rates finite without pretending
        # zero failures prove a zero population failure rate.
        k = suite.successes + 0.5
        n = suite.trials + 1.0
        p = k / n
        effects.append(math.log(p / (1.0 - p)))
        variances.append(1.0 / k + 1.0 / (n - k))
    weights = [1.0 / v for v in variances]
    fixed = sum(w * y for w, y in zip(weights, effects)) / sum(weights)
    q = sum(w * (y - fixed) ** 2 for w, y in zip(weights, effects))
    c = sum(weights) - sum(w * w for w in weights) / sum(weights)
    tau2 = max(0.0, (q - (len(effects) - 1)) / c) if c > 0 else 0.0
    re_weights = [1.0 / (v + tau2) for v in variances]
    pooled_logit = sum(w * y for w, y in zip(re_weights, effects)) / sum(re_weights)
    se = math.sqrt(1.0 / sum(re_weights))
    lo = pooled_logit - 1.96 * se
    hi = pooled_logit + 1.96 * se
    expit = lambda x: 1.0 / (1.0 + math.exp(-x))
    i2 = max(0.0, (q - (len(effects) - 1)) / q) if q > 0 else 0.0
    return {
        "method": "DerSimonian-Laird random effects on continuity-corrected logits",
        "diagnostic_only": True,
        "pooled_rate": _round(expit(pooled_logit)),
        "ci_low": _round(expit(lo)),
        "ci_high": _round(expit(hi)),
        "tau2": _round(tau2),
        "i2": _round(i2),
    }


def _group_by(items: Iterable[Suite], key: str) -> Dict[str, List[Suite]]:
    out: Dict[str, List[Suite]] = {}
    for suite in items:
        value = getattr(suite, key)
        out.setdefault(value, []).append(suite)
    return dict(sorted(out.items()))


def _summarize_group(label: str, suites: Sequence[Suite], seed_offset: int) -> Dict[str, Any]:
    successes = sum(s.successes for s in suites)
    trials = sum(s.trials for s in suites)
    return {
        "label": label,
        "n_suites": len(suites),
        "total_successes": successes,
        "total_trials": trials,
        "suite_rate_min": _round(min(s.rate for s in suites)),
        "suite_rate_max": _round(max(s.rate for s in suites)),
        "suite_mean_rate": _round(_suite_mean_rate(suites)),
        "case_weighted_rate_diagnostic_only": _round(successes / trials),
        "robust_suite_bootstrap": robust_suite_bootstrap(
            suites, seed=BOOTSTRAP_SEED + seed_offset),
        "random_effects_logit_diagnostic": _random_effects_summary(suites),
        "suites": [s.name for s in suites],
    }


def measure() -> Dict[str, Any]:
    suites = collect_suites()
    by_distribution = _group_by(suites, "distribution")
    by_metric = _group_by(suites, "metric")

    distribution_summaries = {
        label: _summarize_group(label, group, i)
        for i, (label, group) in enumerate(by_distribution.items())
    }
    metric_summaries = {
        label: _summarize_group(label, group, 100 + i)
        for i, (label, group) in enumerate(by_metric.items())
    }

    return {
        "step": 265,
        "generated_by": "reproducibility/statistical_meta_analysis.py",
        "command": "python3 reproducibility/statistical_meta_analysis.py",
        "method": {
            "primary": "distribution-stratified robust bootstrap over suite-level rates",
            "resampling_unit": "suite, not individual cases",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "naive_pooling_across_distributions_allowed": False,
            "case_weighted_rates_are_diagnostic_only": True,
            "why_no_global_pool": (
                "real bugs, natural clean models, fuzzed modules, mutation tests, "
                "and stress corpora are different sampling distributions; a single "
                "raw pooled denominator would overweight the largest synthetic suite."
            ),
        },
        "n_suites": len(suites),
        "distributions": distribution_summaries,
        "metrics": metric_summaries,
        "suites": [
            {
                "name": s.name,
                "metric": s.metric,
                "distribution": s.distribution,
                "successes": s.successes,
                "trials": s.trials,
                "rate": _round(s.rate),
                "source": s.source,
                "evidence": s.evidence,
            }
            for s in suites
        ],
    }


def render_markdown(data: Dict[str, Any]) -> str:
    lines = [
        "# Cross-corpus statistical meta-analysis (Step 265)",
        "",
        "This artifact summarizes heterogeneous TensorGuard evidence without "
        "naively pooling raw cases across real, synthetic, fuzzed, mutation, and "
        "stress-test distributions. The primary interval is a deterministic "
        "suite-level cluster bootstrap within each distribution.",
        "",
        f"- suites analyzed: **{data['n_suites']}**",
        f"- bootstrap resamples: **{data['method']['bootstrap_resamples']}**",
        f"- naive global pooling allowed: **{data['method']['naive_pooling_across_distributions_allowed']}**",
        f"- resampling unit: **{data['method']['resampling_unit']}**",
        "",
        "## Distribution-stratified summaries",
        "",
        "| distribution | suites | suite mean | bootstrap CI | suite range | case-weighted diagnostic |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for label, summary in data["distributions"].items():
        boot = summary["robust_suite_bootstrap"]
        lines.append(
            f"| {label} | {summary['n_suites']} | {summary['suite_mean_rate']:.3f} "
            f"| [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] "
            f"| [{summary['suite_rate_min']:.3f}, {summary['suite_rate_max']:.3f}] "
            f"| {summary['case_weighted_rate_diagnostic_only']:.3f} |"
        )
    lines += [
        "",
        "## Metric summaries (still suite-level)",
        "",
        "| metric | suites | suite mean | bootstrap CI | case-weighted diagnostic |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for label, summary in data["metrics"].items():
        boot = summary["robust_suite_bootstrap"]
        lines.append(
            f"| {label} | {summary['n_suites']} | {summary['suite_mean_rate']:.3f} "
            f"| [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] "
            f"| {summary['case_weighted_rate_diagnostic_only']:.3f} |"
        )
    lines += [
        "",
        "## Source suites",
        "",
        "| suite | distribution | metric | success/trials | rate | source |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for suite in data["suites"]:
        lines.append(
            f"| {suite['name']} | {suite['distribution']} | {suite['metric']} "
            f"| {suite['successes']}/{suite['trials']} | {suite['rate']:.3f} "
            f"| `{suite['source']}` |"
        )
    lines += [
        "",
        data["method"]["why_no_global_pool"],
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
            print("statistical_meta_analysis: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
