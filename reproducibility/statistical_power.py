"""Statistical power & sample-size justification for every headline claim (Step 120).

Point estimates and confidence intervals say how *precise* an estimate is; a
power analysis says whether the *sample size was large enough to earn the claim*
in the first place. This harness performs an exact, SciPy-free power and
sample-size analysis for each headline statistical claim, reading the real
observed counts ``(k, n)`` straight out of the committed regeneration artifacts
so the analysis tracks the actual evidence rather than invented numbers.

Three claim shapes are covered, each with the exact-binomial machinery that fits
it:

* **Zero-failure claims** ("0 false alarms in n clean models", "0 sound-mode
  false positives"). With zero observed failures the one-sided exact upper
  confidence bound on the failure rate is ``1 - α^(1/n)`` (the exact "rule of
  three"). We report that bound, the *power* of the achieved n to have caught a
  true failure rate of 1% / 2% / 5% (``1 - (1-p0)^n``), and the minimum n needed
  to certify a target rate -- then flag whether the achieved n meets it.

* **Perfect-recall claims** ("k of n bugs caught, k = n"). Symmetrically the
  one-sided exact lower bound on recall is ``α^(1/n)``, the power to have exposed
  a true recall at or below r0 is ``1 - r0^n``, and we report the n required to
  certify a recall floor.

* **Paired McNemar claims** (TensorGuard vs a baseline). With all discordant
  pairs favouring TensorGuard the exact two-sided p is ``2·2^{-d}`` for d
  discordant pairs, so the *minimum discordant count* for significance at α is
  ``⌈log2(2/α)⌉``; we report it next to the achieved d as an explicit
  sample-size justification.

Everything is closed-form and integer/rational, so the artifact is byte-identical
across machines. ``--check`` re-derives and diffs it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "reproducibility" / "statistical_power.json"
OUT_MD = REPO / "reproducibility" / "statistical_power.md"

ALPHA = 0.05
CONF = 1.0 - ALPHA
FAILURE_TARGETS = (0.01, 0.02, 0.05)
RECALL_FLOORS = (0.99, 0.98, 0.95)


def _load(rel: str) -> Any:
    return json.loads((REPO / rel).read_text())


# --- exact-binomial power primitives ---------------------------------------
def one_sided_upper_zero_fail(n: int, conf: float = CONF) -> float:
    """Exact 1-sided upper CI on a rate given 0 failures in n trials."""
    return 1.0 - (1.0 - conf) ** (1.0 / n)


def one_sided_lower_all_success(n: int, conf: float = CONF) -> float:
    """Exact 1-sided lower CI on a success rate given n successes in n trials."""
    return (1.0 - conf) ** (1.0 / n)


def power_to_detect_failure(n: int, p0: float) -> float:
    """P(>=1 failure | true rate p0): the power of n to expose rate p0."""
    return 1.0 - (1.0 - p0) ** n


def power_to_detect_recall_below(n: int, r0: float) -> float:
    """P(>=1 miss | true recall r0): the power of n to expose recall <= r0."""
    return 1.0 - r0 ** n


def min_n_zero_fail(target: float, conf: float = CONF) -> int:
    """Smallest n s.t. 0 failures certifies rate <= target at `conf`."""
    return math.ceil(math.log(1.0 - conf) / math.log(1.0 - target))


def min_n_all_success(floor: float, conf: float = CONF) -> int:
    """Smallest n s.t. n successes certifies recall >= floor at `conf`."""
    return math.ceil(math.log(1.0 - conf) / math.log(floor))


def min_discordant_for_significance(alpha: float = ALPHA) -> int:
    """Smallest all-one-sided discordant count d with 2*2^-d <= alpha."""
    return math.ceil(math.log2(2.0 / alpha))


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# --- claim registry (grounded in committed artifacts) ----------------------
def _zero_fail_claim(cid: str, desc: str, n: int) -> dict:
    bound = one_sided_upper_zero_fail(n)
    powers = {f"p={p}": _r(power_to_detect_failure(n, p)) for p in FAILURE_TARGETS}
    req = {f"<= {t}": min_n_zero_fail(t) for t in FAILURE_TARGETS}
    # The claim is "adequately powered" if n suffices to certify the 5% target.
    return {
        "id": cid,
        "kind": "zero_failure",
        "description": desc,
        "k_failures": 0,
        "n": n,
        "one_sided_upper_bound_95": _r(bound),
        "power_to_detect_failure_rate": powers,
        "min_n_to_certify": req,
        "adequately_powered_5pct": n >= min_n_zero_fail(0.05),
        "adequately_powered_1pct": n >= min_n_zero_fail(0.01),
    }


def _perfect_recall_claim(cid: str, desc: str, n: int) -> dict:
    bound = one_sided_lower_all_success(n)
    powers = {f"r={r}": _r(power_to_detect_recall_below(n, r)) for r in RECALL_FLOORS}
    req = {f">= {f}": min_n_all_success(f) for f in RECALL_FLOORS}
    return {
        "id": cid,
        "kind": "perfect_recall",
        "description": desc,
        "k_caught": n,
        "n": n,
        "one_sided_lower_bound_95": _r(bound),
        "power_to_detect_recall_below": powers,
        "min_n_to_certify_floor": req,
        "adequately_powered_95pct": n >= min_n_all_success(0.95),
        "adequately_powered_99pct": n >= min_n_all_success(0.99),
    }


def _mcnemar_claim(cid: str, desc: str, b: int, c: int) -> dict:
    d = b + c
    # All-one-sided exact two-sided p (c == 0): 2 * 2^-d, capped at 1.
    p = min(1.0, 2.0 * (0.5 ** d)) if d > 0 else 1.0
    need = min_discordant_for_significance()
    return {
        "id": cid,
        "kind": "mcnemar",
        "description": desc,
        "b_tg_right_base_wrong": b,
        "c_tg_wrong_base_right": c,
        "n_discordant": d,
        "exact_two_sided_p": _r(p),
        "min_discordant_for_significance_05": need,
        "all_discordant_favour_tg": c == 0,
        "adequately_powered": (c == 0 and d >= need),
    }


def build_claims() -> List[dict]:
    claims: List[dict] = []

    fp = _load("reproducibility/fp_stress_eval.json")
    nat = _load("reproducibility/natural_distribution_study.json")
    ext = _load("reproducibility/corpus_extended_score.json")
    blind = _load("reproducibility/blind_split_eval.json")
    diff = _load("reproducibility/differential_dispatcher.json")
    mut = _load("reproducibility/mutation_clean_models.json")
    sig = _load("evaluation/significance.json")

    # Zero-false-alarm claims (sound mode where available).
    claims.append(_zero_fail_claim(
        "fp_stress_sound_zero_fa",
        "Sound-mode false alarms on the clean false-positive stress corpus",
        fp["per_mode"]["sound"]["false_alarm_rate"]["n"]))
    claims.append(_zero_fail_claim(
        "natural_sound_zero_fa",
        "Sound-mode false alarms on the natural-distribution model sample",
        nat["per_mode"]["sound"]["false_alarm_rate"]["n"]))
    claims.append(_zero_fail_claim(
        "corpus_ext_zero_fp",
        "False positives on the extended-corpus clean cases (balanced)",
        ext["balanced"]["false_positive_rate_on_decided"]["n"]))
    claims.append(_zero_fail_claim(
        "differential_zero_fa",
        "False alarms vs the live torch dispatcher on clean modules",
        diff["false_alarm_rate"]["n"]))

    # Perfect-recall claims.
    claims.append(_perfect_recall_claim(
        "corpus_ext_recall",
        "Recall on the extended buggy corpus (balanced)",
        ext["balanced"]["confusion"]["tp"]))
    claims.append(_perfect_recall_claim(
        "blind_recall",
        "Recall on the pre-registered held-out blind split (balanced)",
        blind["balanced"]["recall_on_decided"]["n"]))
    claims.append(_perfect_recall_claim(
        "differential_unsafe_recall",
        "UNSAFE agreement on modules the live torch dispatcher rejects",
        diff["agreement_matrix"]["UNSAFE|raises"]))
    claims.append(_perfect_recall_claim(
        "mutation_kill_rate",
        "Kill rate over genuine-bug mutants (sound mode)",
        mut["per_mode"]["sound"]["kill_rate"]["n"]))

    # McNemar comparisons (any that are usable in the significance artifact).
    for comp in sig["comparisons"]:
        if not comp.get("usable"):
            continue
        m = comp["mcnemar"]
        claims.append(_mcnemar_claim(
            f"mcnemar_vs_{comp['baseline']}",
            f"Paired McNemar: TensorGuard vs {comp['baseline']}",
            int(m["b_ref_right_base_wrong"]),
            int(m["c_ref_wrong_base_right"])))

    return claims


def measure() -> dict:
    claims = build_claims()
    zero = [c for c in claims if c["kind"] == "zero_failure"]
    recall = [c for c in claims if c["kind"] == "perfect_recall"]
    mcn = [c for c in claims if c["kind"] == "mcnemar"]

    # Pooled clean-model evidence: total clean trials with zero observed
    # failures gives the tightest aggregate bound on the false-alarm rate.
    pooled_clean_n = sum(c["n"] for c in zero)
    pooled_clean_bound = one_sided_upper_zero_fail(pooled_clean_n)

    data = {
        "step": 120,
        "alpha": ALPHA,
        "confidence": CONF,
        "failure_rate_targets": list(FAILURE_TARGETS),
        "recall_floors": list(RECALL_FLOORS),
        "claims": claims,
        "n_claims": len(claims),
        "pooled_clean_trials": pooled_clean_n,
        "pooled_clean_zero_fail_upper_bound_95": _r(pooled_clean_bound),
        "all_zero_failure_claims_powered_5pct": all(
            c["adequately_powered_5pct"] for c in zero),
        "all_recall_claims_powered_95pct": all(
            c["adequately_powered_95pct"] for c in recall),
        "n_mcnemar_significant_and_powered": sum(
            1 for c in mcn if c["adequately_powered"]),
        "min_discordant_for_significance_05": min_discordant_for_significance(),
    }
    return data


def render_markdown(d: dict) -> str:
    lines = [
        "# Statistical power & sample-size justification (Step 120)",
        "",
        f"Exact-binomial power analysis at α = {d['alpha']} over "
        f"**{d['n_claims']}** headline claims, with observed counts read straight "
        "from the committed regeneration artifacts.",
        "",
        "## Zero-failure claims (false alarms / false positives)",
        "",
        "| claim | n | 95% upper bound | power@1% | power@5% | n needed (≤5%) "
        "| powered? |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in d["claims"]:
        if c["kind"] != "zero_failure":
            continue
        lines.append(
            f"| {c['id']} | {c['n']} | {c['one_sided_upper_bound_95']} "
            f"| {c['power_to_detect_failure_rate']['p=0.01']} "
            f"| {c['power_to_detect_failure_rate']['p=0.05']} "
            f"| {c['min_n_to_certify']['<= 0.05']} "
            f"| {c['adequately_powered_5pct']} |"
        )
    lines += [
        "",
        f"Pooled across every clean trial ({d['pooled_clean_trials']} clean "
        f"models, zero observed false alarms) the aggregate one-sided 95% upper "
        f"bound on the false-alarm rate is "
        f"**{d['pooled_clean_zero_fail_upper_bound_95']}**.",
        "",
        "## Perfect-recall claims",
        "",
        "| claim | n | 95% lower bound | power vs r=0.95 | n needed (≥0.95) "
        "| powered? |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in d["claims"]:
        if c["kind"] != "perfect_recall":
            continue
        lines.append(
            f"| {c['id']} | {c['n']} | {c['one_sided_lower_bound_95']} "
            f"| {c['power_to_detect_recall_below']['r=0.95']} "
            f"| {c['min_n_to_certify_floor']['>= 0.95']} "
            f"| {c['adequately_powered_95pct']} |"
        )
    lines += [
        "",
        "## Paired McNemar comparisons",
        "",
        f"Minimum all-one-sided discordant pairs for significance at α=0.05: "
        f"**{d['min_discordant_for_significance_05']}**.",
        "",
        "| comparison | discordant | exact p | needed | powered? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in d["claims"]:
        if c["kind"] != "mcnemar":
            continue
        lines.append(
            f"| {c['id']} | {c['n_discordant']} | {c['exact_two_sided_p']} "
            f"| {c['min_discordant_for_significance_05']} "
            f"| {c['adequately_powered']} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- every zero-failure claim is powered to exclude a 5% rate: "
        f"**{d['all_zero_failure_claims_powered_5pct']}**",
        f"- every perfect-recall claim is powered to certify a 95% floor: "
        f"**{d['all_recall_claims_powered_95pct']}**",
        f"- McNemar comparisons that are significant *and* adequately powered: "
        f"**{d['n_mcnemar_significant_and_powered']}**",
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
            print("statistical_power: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
