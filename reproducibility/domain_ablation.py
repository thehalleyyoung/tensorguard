"""Reviewer-grade per-domain ablation: contribution of each domain to recall (Step 117).

TensorGuard's reduced product comprises five abstract domains -- shape, dtype,
device, phase and gradient. This harness quantifies how much each domain
contributes to *recall* via a leave-one-domain-out (LODO) study over a labeled,
single-bug-per-module corpus (see ``corpus_extended/domain_ablation_corpus.py``).

Method
------
For every labeled case we record the *full* verdict (all domains on) and, for
each domain ``D``, the verdict with ``D`` ablated:

  * ``device`` and ``gradient`` have genuine runtime toggles
    (``check_devices`` / ``check_gradients``); ablation passes the flag ``False``.
  * ``shape`` and ``dtype`` are part of the always-on base type/shape view and
    have no runtime toggle, so they are ablated at the *report* level: re-run and
    drop every detection whose message tag attributes it to ``D``.
  * ``phase`` is *diagnostic-only* (it registers phase structure but never flips a
    model from SAFE to UNSAFE), recorded honestly as a zero-recall domain.

Two cross-checks keep the ablation method honest:

  1. **Toggle/report agreement** -- for ``device`` and ``gradient`` (which have
     *both* a runtime toggle and a message tag), the runtime-toggle ablation and
     the report-level ablation must yield the *same* caught/missed decision on
     every case. This validates the report-level method used for shape/dtype.
  2. **Orthogonality** -- ablating domain ``D`` must not reduce recall on the
     *other* domains' bugs. The full LODO recall matrix (ablated domain x bug
     domain) should be the identity-complement: zeros on the diagonal of the
     verification domains, full recall off-diagonal.

The headline per-domain contribution is ``recall_full - recall_LODO`` on that
domain's own bugs (Wilson intervals included). Only counts, rounded rates and the
matrix are recorded, so the artifact is byte-identical across machines.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.domain_ablation_corpus import (  # noqa: E402
    DIAGNOSTIC_DOMAINS,
    TAG_TO_DOMAIN,
    VERIFICATION_DOMAINS,
    build_corpus,
)
from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "domain_ablation.json"
OUT_MD = REPO / "reproducibility" / "domain_ablation.md"

SEED = 20240604
N_PER_DOMAIN = 60

_TAG_RE = re.compile(r"\[([A-Z][A-Z-]+)\]")


def _bug_tags(bugs) -> list:
    out = []
    for b in bugs:
        m = _TAG_RE.match(b.message)
        if m:
            out.append(m.group(1))
    return out


def _verify(source, shape, **kwargs):
    return verify_architecture(
        source,
        input_shapes={"x": tuple(shape)},
        soundness_mode="sound",
        max_cegar_iterations=0,
        **kwargs,
    )


def _caught_full(res) -> bool:
    return res.bug_count > 0


def _caught_after_report_ablation(bugs, ablated_domain: str) -> bool:
    """Caught iff at least one detection is *not* attributed to ablated_domain."""

    for b in bugs:
        m = _TAG_RE.match(b.message)
        dom = TAG_TO_DOMAIN.get(m.group(1)) if m else None
        if dom != ablated_domain:
            return True
    return False


_TOGGLE = {
    "device": "check_devices",
    "gradient": "check_gradients",
    "phase": "check_phases",
}


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
        cases = build_corpus(seed=SEED, n_per_domain=N_PER_DOMAIN)
        domains = list(VERIFICATION_DOMAINS) + list(DIAGNOSTIC_DOMAINS)

        # Per (ablated_domain, bug_domain) -> [n_caught, n_total]
        matrix = {a: {d: [0, 0] for d in domains} for a in ["full"] + list(domains)}

        # Cross-check: for device/gradient, runtime toggle vs report ablation.
        toggle_report_disagree = 0
        toggle_report_pairs = 0

        for c in cases:
            full = _verify(c.source, c.input_shape)
            full_caught = _caught_full(full)
            matrix["full"][c.domain][1] += 1
            if full_caught:
                matrix["full"][c.domain][0] += 1

            for ablated in domains:
                matrix[ablated][c.domain][1] += 1
                if ablated in _TOGGLE and ablated in ("device", "gradient"):
                    # Genuine runtime ablation.
                    res = _verify(
                        c.source, c.input_shape, **{_TOGGLE[ablated]: False}
                    )
                    caught_toggle = _caught_full(res)
                    # Report-level on the SAME (full) run for the cross-check.
                    caught_report = _caught_after_report_ablation(
                        full.bugs, ablated
                    )
                    toggle_report_pairs += 1
                    if caught_toggle != caught_report:
                        toggle_report_disagree += 1
                    caught = caught_toggle
                elif ablated == "phase":
                    # Phase toggle off (diagnostic-only domain).
                    res = _verify(
                        c.source, c.input_shape, check_phases=False
                    )
                    caught = _caught_full(res)
                else:
                    # shape / dtype: report-level ablation (no runtime toggle).
                    caught = _caught_after_report_ablation(full.bugs, ablated)
                if caught:
                    matrix[ablated][c.domain][0] += 1

        # Build recall views + per-domain contribution.
        def recall(cell):
            k, n = cell
            return {"k": k, "n": n, **_wilson(k, n)}

        recall_matrix = {
            a: {d: recall(matrix[a][d]) for d in domains}
            for a in ["full"] + list(domains)
        }

        contributions = {}
        for d in VERIFICATION_DOMAINS:
            full_k, full_n = matrix["full"][d]
            lodo_k, lodo_n = matrix[d][d]
            contributions[d] = {
                "recall_full": recall(matrix["full"][d]),
                "recall_lodo": recall(matrix[d][d]),
                "marginal_contribution": round(
                    (full_k / full_n if full_n else 0.0)
                    - (lodo_k / lodo_n if lodo_n else 0.0),
                    4,
                ),
                "lodo_recall_is_zero": lodo_k == 0,
                "full_recall_is_one": full_k == full_n and full_n > 0,
            }

        # Orthogonality: ablating D must not reduce recall on other domains.
        orthogonal = True
        for a in VERIFICATION_DOMAINS:
            for d in VERIFICATION_DOMAINS:
                if a == d:
                    continue
                k, n = matrix[a][d]
                if k != n:
                    orthogonal = False

        phase_diag = {
            "recall_full": recall(matrix["full"]["phase"]),
            "is_diagnostic_only": matrix["full"]["phase"][0] == 0,
        }

        data = {
            "step": 117,
            "seed": SEED,
            "n_per_domain": N_PER_DOMAIN,
            "n_cases": len(cases),
            "domains": domains,
            "verification_domains": list(VERIFICATION_DOMAINS),
            "diagnostic_domains": list(DIAGNOSTIC_DOMAINS),
            "recall_matrix": recall_matrix,
            "contributions": contributions,
            "phase_diagnostic": phase_diag,
            "all_verification_domains_full_recall": all(
                contributions[d]["full_recall_is_one"]
                for d in VERIFICATION_DOMAINS
            ),
            "every_domain_necessary": all(
                contributions[d]["lodo_recall_is_zero"]
                for d in VERIFICATION_DOMAINS
            ),
            "domains_orthogonal": orthogonal,
            "toggle_report_crosscheck": {
                "pairs_compared": toggle_report_pairs,
                "disagreements": toggle_report_disagree,
                "agree": toggle_report_disagree == 0,
            },
        }
        return data
    finally:
        logging.disable(logging.NOTSET)


def render_markdown(data: dict) -> str:
    lines = [
        "# Per-domain ablation: contribution of each abstract domain (Step 117)",
        "",
        f"Seed `{data['seed']}` — **{data['n_cases']}** labeled single-bug "
        f"modules ({data['n_per_domain']} per domain across "
        f"{', '.join(data['domains'])}).",
        "",
        "Leave-one-domain-out (LODO): `device`/`gradient` ablated via genuine "
        "runtime toggles, `shape`/`dtype` ablated at the report level (no runtime "
        "toggle on the always-on base view), `phase` recorded as diagnostic-only.",
        "",
        "## Per-domain contribution to recall",
        "",
        "| domain | full recall | LODO recall | marginal contribution | "
        "necessary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d in data["verification_domains"]:
        c = data["contributions"][d]
        rf = c["recall_full"]
        rl = c["recall_lodo"]
        lines.append(
            f"| {d} | {rf['k']}/{rf['n']} ({rf['point']}) | "
            f"{rl['k']}/{rl['n']} ({rl['point']}) | "
            f"{c['marginal_contribution']} | {c['lodo_recall_is_zero']} |"
        )
    pf = data["phase_diagnostic"]["recall_full"]
    lines += [
        f"| phase (diagnostic-only) | {pf['k']}/{pf['n']} ({pf['point']}) | "
        f"— | — | — |",
        "",
        "## Summary",
        "",
        f"- all verification domains reach full recall: "
        f"**{data['all_verification_domains_full_recall']}**",
        f"- every verification domain is necessary (LODO recall drops to zero on "
        f"its own bugs): **{data['every_domain_necessary']}**",
        f"- domains are orthogonal (ablating one does not reduce recall on the "
        f"others): **{data['domains_orthogonal']}**",
        f"- phase is diagnostic-only (refutes nothing): "
        f"**{data['phase_diagnostic']['is_diagnostic_only']}**",
        f"- toggle/report cross-check agrees on all "
        f"{data['toggle_report_crosscheck']['pairs_compared']} device+gradient "
        f"pairs: **{data['toggle_report_crosscheck']['agree']}**",
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
            print("domain_ablation: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
