#!/usr/bin/env python3.11
"""Full-128 CV assume_M joint satisfiability audit with Clopper-Pearson CI.

This addresses the round-2 reviewer's W3/Q1 obligation: report the
witnessed-ratio on the *full* 128 contract-violation set (rather than a
30-row subsample) plus a Clopper-Pearson 95% CI on that ratio.

Methodology:
  * Bucket "empty"             → assume_M is identically true → trivially
                                 jointly satisfied (26/128 rows).
  * Bucket "no-own-init"        → block has no own __init__; assume_M
                                 contributes no axiom → trivially jointly
                                 satisfied (12/128 rows; vacuous-but-sound).
  * Bucket "symbolic-config-only" → run the conjunction check from
                                 cv_caller_rely_joint_sat.py over each row
                                 (90/128 rows).

Pre-registration:
  * Random sampling: NONE (this audit covers the full 128 set).  The seed
    in the legacy 30-row script is therefore moot for this audit.
  * Config-class resolution and clause-satisfaction logic are imported
    from cv_caller_rely_joint_sat.py without modification (no rule
    edits between freeze and this run).

Output:
  reproducibility/cv_caller_rely_joint_sat_full128.json
  reproducibility/cv_caller_rely_joint_sat_full128.md
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import math
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

CALLER_RELY = os.path.join(ROOT, "reproducibility", "cv_caller_rely.json")
SUBSAMPLE_SCRIPT = os.path.join(
    ROOT, "reproducibility", "cv_caller_rely_joint_sat.py"
)
OUT_JSON = os.path.join(
    ROOT, "reproducibility", "cv_caller_rely_joint_sat_full128.json"
)
OUT_MD = os.path.join(
    ROOT, "reproducibility", "cv_caller_rely_joint_sat_full128.md"
)


def _import_helpers():
    spec = importlib.util.spec_from_file_location(
        "_cv_subsample", SUBSAMPLE_SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Two-sided Clopper-Pearson exact binomial CI."""
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        lo = 0.0
    else:
        # Beta inverse via bisection (no scipy dependency)
        lo = _beta_inv(alpha / 2.0, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = _beta_inv(1 - alpha / 2.0, k + 1, n - k)
    return (lo, hi)


def _beta_cdf(x: float, a: float, b: float) -> float:
    # Regularised incomplete beta via continued fraction (Numerical Recipes).
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(x, a, b) / a
    return 1 - bt * _betacf(1 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 200) -> float:
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-10:
            return h
    return h


def _beta_inv(p: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    helpers = _import_helpers()

    with open(CALLER_RELY) as f:
        cv = json.load(f)
    rows = cv["rows"]
    assert len(rows) == 128, f"expected 128 CV rows, got {len(rows)}"

    per_row: List[Dict[str, Any]] = []
    n_jointly = 0
    n_empty_trivial = 0
    n_noinit_trivial = 0
    n_symbolic_attempted = 0
    n_symbolic_jointly = 0
    n_symbolic_excluded = 0

    for row in rows:
        bucket = row.get("bucket")
        rec: Dict[str, Any] = {
            "id": row["id"],
            "qualified_name": row["qualified_name"],
            "library": row.get("library"),
            "bucket": bucket,
            "jointly_satisfied": False,
            "method": "",
            "note": "",
        }
        if bucket == "empty":
            rec["jointly_satisfied"] = True
            rec["method"] = "vacuous (assume_M ≡ true)"
            n_jointly += 1
            n_empty_trivial += 1
        elif bucket == "no-own-init":
            rec["jointly_satisfied"] = True
            rec["method"] = "vacuous (no own __init__; assume_M contributes no axiom)"
            n_jointly += 1
            n_noinit_trivial += 1
        elif bucket == "symbolic-config-only":
            qname = row["qualified_name"]
            sym_attrs = row.get("sym_attrs", [])
            cfg_name = helpers._config_class_name(qname)
            if not cfg_name:
                rec["method"] = "no-config-class-resolved"
                rec["note"] = "could not resolve *Config class from qualname"
                n_symbolic_excluded += 1
            else:
                cls, err = helpers._try_get_config_class(cfg_name)
                if cls is None:
                    rec["method"] = "config-class-not-importable"
                    rec["note"] = err or ""
                    rec["config_class"] = cfg_name
                    n_symbolic_excluded += 1
                else:
                    cfg, err2 = helpers._instantiate_config(cls)
                    rec["config_class"] = cfg_name
                    if cfg is None:
                        rec["method"] = "config-instantiation-failed"
                        rec["note"] = err2 or ""
                        n_symbolic_excluded += 1
                    else:
                        if not sym_attrs:
                            rec["jointly_satisfied"] = True
                            rec["method"] = "default-config (no sym_attr clauses)"
                            n_jointly += 1
                            n_symbolic_jointly += 1
                            n_symbolic_attempted += 1
                        else:
                            ok, reasons = helpers._check_joint_sat(cfg, sym_attrs)
                            rec["jointly_satisfied"] = ok
                            rec["method"] = (
                                f"default-config ({cfg_name}() with documented defaults)"
                            )
                            rec["clause_results"] = reasons
                            n_symbolic_attempted += 1
                            if ok:
                                n_jointly += 1
                                n_symbolic_jointly += 1
                            else:
                                rec["note"] = "one or more sym_attr clauses failed"
        else:
            rec["method"] = "unknown bucket"
            rec["note"] = f"bucket={bucket} (treated as not-witnessed)"
        per_row.append(rec)

    # Two denominators are reported:
    #   (a) all 128 CVs (the headline the reviewer asked for);
    #   (b) attempted-only (excludes the symbolic rows whose *Config class
    #       could not be resolved/instantiated locally).
    n_total = 128
    lo_a, hi_a = clopper_pearson(n_jointly, n_total)
    n_attempted = (
        n_empty_trivial + n_noinit_trivial + n_symbolic_attempted
    )
    lo_b, hi_b = clopper_pearson(n_jointly, n_attempted)

    output = {
        "_question": (
            "Round-2 W3/Q1: report the witnessed-ratio on the full 128 CV set "
            "with a Clopper-Pearson 95% CI."
        ),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "freeze_commit": "040f6f3",
        "freeze_date": "2026-04-07",
        "denominator_full128": {
            "n": n_total,
            "k_jointly_satisfied": n_jointly,
            "ratio": n_jointly / n_total,
            "clopper_pearson_95": [lo_a, hi_a],
        },
        "denominator_attempted": {
            "n": n_attempted,
            "k_jointly_satisfied": n_jointly,
            "ratio": (n_jointly / n_attempted) if n_attempted else 0.0,
            "clopper_pearson_95": [lo_b, hi_b],
        },
        "bucket_breakdown": {
            "empty_trivially_satisfied": n_empty_trivial,
            "no_own_init_vacuously_satisfied": n_noinit_trivial,
            "symbolic_config_attempted": n_symbolic_attempted,
            "symbolic_config_jointly_satisfied": n_symbolic_jointly,
            "symbolic_config_excluded": n_symbolic_excluded,
        },
        "method": (
            "For bucket=empty (26/128): assume_M is identically true; counted "
            "as jointly satisfied by definition.  For bucket=no-own-init "
            "(12/128): the class inherits __init__ and contributes no axiom; "
            "counted as jointly satisfied (vacuous-but-sound).  For "
            "bucket=symbolic-config-only (90/128): each row's full conjunction "
            "of assume_M sym_attr clauses is checked against a single "
            "default *Config() instantiation using the helpers in "
            "cv_caller_rely_joint_sat.py.  Rows whose *Config class cannot "
            "be resolved or instantiated locally are reported as 'excluded' "
            "and not counted as witnessed; the headline ratio is therefore "
            "a *lower bound* on the true joint-realisability rate."
        ),
        "rows": per_row,
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(output, fh, indent=2)

    md = []
    md.append("# Full-128 CV joint-satisfiability audit (Round 2 W3/Q1)")
    md.append("")
    md.append("## Command")
    md.append("```")
    md.append("python3.11 reproducibility/cv_caller_rely_joint_sat_full128.py")
    md.append("```")
    md.append("")
    md.append("## Headline (full 128 CV denominator)")
    md.append("")
    md.append(f"- Witnessed: **{n_jointly}/{n_total}** ({100.0*n_jointly/n_total:.1f}%)")
    md.append(f"- Clopper-Pearson 95% CI: **[{100.0*lo_a:.1f}%, {100.0*hi_a:.1f}%]**")
    md.append("")
    md.append("## Attempted-only denominator")
    md.append("")
    md.append(
        f"- Witnessed: **{n_jointly}/{n_attempted}** ({100.0*n_jointly/max(1,n_attempted):.1f}%)"
    )
    md.append(f"- Clopper-Pearson 95% CI: **[{100.0*lo_b:.1f}%, {100.0*hi_b:.1f}%]**")
    md.append("")
    md.append("## Bucket breakdown")
    md.append("")
    md.append("| bucket | count | jointly satisfied |")
    md.append("|---|---:|---:|")
    md.append(f"| empty (assume_M ≡ true) | {n_empty_trivial} | {n_empty_trivial} |")
    md.append(
        f"| no-own-init (vacuous) | {n_noinit_trivial} | {n_noinit_trivial} |"
    )
    md.append(
        f"| symbolic-config-only (attempted) | {n_symbolic_attempted} | "
        f"{n_symbolic_jointly} |"
    )
    md.append(
        f"| symbolic-config-only (excluded) | {n_symbolic_excluded} | 0 (excluded) |"
    )
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append(
        "* No random sampling is used; the audit covers all 128 CV rows.  "
        "The 30-row subsample audit (`cv_caller_rely_joint_sat.{py,json,md}`) "
        "is retained for backwards reference."
    )
    md.append(
        "* Excluded rows are counted as *not witnessed* in the full-128 "
        "denominator, so the published ratio is a lower bound."
    )
    md.append(
        "* Clopper-Pearson CI computed analytically (continued-fraction "
        "regularised incomplete beta function); no scipy dependency."
    )
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(
        f"Full-128 CV joint-sat: {n_jointly}/{n_total} "
        f"(CP 95% CI [{lo_a:.3f}, {hi_a:.3f}])"
    )
    print(
        f"Attempted-only:       {n_jointly}/{n_attempted} "
        f"(CP 95% CI [{lo_b:.3f}, {hi_b:.3f}])"
    )
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
