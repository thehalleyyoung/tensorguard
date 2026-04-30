"""Fragment-fair Pytea head-to-head reproducibility artifact.

Single-command reproducibility for the abstract's headline number
"32/34 vs 25/34, McNemar exact p=0.0156" against Pytea on the
``modern subset`` (the bugs whose primary failing op is in Pytea's
2022 operator catalogue).

Inputs (already shipped):
  - experiments_v5/pytea_baseline_results.json
        Pytea 0.1.0 verdicts on all 60 bug repros.
  - experiments_v5/v5_benchmark_results.json
        TensorGuard verdicts on all 60 bug repros.
  - experiments_v5/v8/build_modern_subset.py::BUG_MODERN_MAP
        The 34-row inclusion table (bug_id -> in_modern_subset).
        The classification key is "primary failing op is implemented
        in Pytea's 2022 catalogue (TS handler present in
        packages/pytea/src/ts/index.ts as of commit c536515)".

Output:
  - reproducibility/pytea_fragment_fair.json
        per_bug list with the columns the reviewer asked for:
            { id,
              in_fragment_fair_subset, primary_op, catalogue_note,
              tensorguard_verdict, pytea_verdict, agreement }
        plus a meta block with the 2x2 contingency table and the
        exact McNemar two-sided p-value.

The script is deliberately self-contained and re-derives every cell
of the contingency table from the two baseline JSONs. Running

    python3 reproducibility/pytea_fragment_fair.py

will print the headline counts and (re)write the JSON file.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "experiments_v5", "v8"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_bms_const",
    os.path.join(_REPO, "experiments_v5", "v8", "build_modern_subset.py"),
)
# We only want the BUG_MODERN_MAP constant and not the script's main
# side-effects, so parse the constant out of the source rather than
# executing the module.
_src = open(_spec.origin).read()  # type: ignore[union-attr]
_g: dict = {"__file__": _spec.origin}
exec(compile(_src.split("\n# Load TG and Pytea verdicts")[0], _spec.origin, "exec"), _g)  # type: ignore[union-attr]
BUG_MODERN_MAP = _g["BUG_MODERN_MAP"]

PYTEA_BASELINE = os.path.join(_REPO, "experiments_v5", "pytea_baseline_results.json")
TG_BASELINE = os.path.join(_REPO, "experiments_v5", "v5_benchmark_results.json")
OUT = os.path.join(_REPO, "reproducibility", "pytea_fragment_fair.json")


def _load_pytea_by_short() -> dict:
    with open(PYTEA_BASELINE) as f:
        data = json.load(f)
    out = {}
    for entry in data["bug_corpus"]["per_input"]:
        m = re.match(r"(bug_\d+)", entry["id"])
        if m:
            out[m.group(1)] = entry
    return out


def _load_tg_by_short() -> dict:
    with open(TG_BASELINE) as f:
        data = json.load(f)
    return {entry["id"]: entry for entry in data["bug_corpus"]["per_input"]}


def _tg_verdict(entry: dict | None, primary_op: str, in_subset: bool) -> str:
    """TensorGuard verdict on a bug repro.

    ``bucket`` is one of {Refuted, Verified, ...}; we map to the
    fragment-fair head-to-head categories {Refuted, Verified}. The
    fragment-fair rule (silent-skip correction) only matters for
    out-of-catalogue bugs where TG would otherwise claim a refutation
    on an op Pytea cannot model; on the 34-row in-subset rows it is
    a no-op (we kept this column so the JSON is self-describing).
    """

    if entry is None:
        return "N/A"
    bucket = entry.get("bucket")
    if bucket == "Refuted":
        return "Refuted"
    return "Verified"


def _pytea_verdict(entry: dict | None) -> str:
    if entry is None:
        return "N/A"
    return entry.get("verdict", "Verified")


def _exact_mcnemar_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_one = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_one)


def main() -> None:
    pytea_map = _load_pytea_by_short()
    tg_map = _load_tg_by_short()

    rows = []
    in_subset_count = 0
    a = b = c = d = 0  # both_refute / TG_only / Pytea_only / neither
    for bug_id, (modern, primary_op, note) in sorted(BUG_MODERN_MAP.items()):
        tg_e = tg_map.get(bug_id)
        py_e = pytea_map.get(bug_id)
        tg_v = _tg_verdict(tg_e, primary_op, modern)
        py_v = _pytea_verdict(py_e)
        if modern:
            in_subset_count += 1
            tg_ref = (tg_v == "Refuted")
            py_ref = (py_v == "Refuted")
            if tg_ref and py_ref:
                a += 1
                ag = "both_refute"
            elif tg_ref and not py_ref:
                b += 1
                ag = "tg_only"
            elif (not tg_ref) and py_ref:
                c += 1
                ag = "pytea_only"
            else:
                d += 1
                ag = "neither"
        else:
            ag = "out_of_subset"
        rows.append({
            "id": bug_id,
            "in_fragment_fair_subset": bool(modern),
            "primary_op": primary_op,
            "catalogue_note": note,
            "tensorguard_verdict": tg_v,
            "pytea_verdict": py_v,
            "agreement": ag,
        })

    p_mcnemar = _exact_mcnemar_two_sided(b, c)
    tg_refuted = a + b
    pytea_refuted = a + c
    n_subset = a + b + c + d

    out = {
        "meta": {
            "generated_by": "reproducibility/pytea_fragment_fair.py",
            "command": "python3 reproducibility/pytea_fragment_fair.py",
            "inputs": [
                "experiments_v5/pytea_baseline_results.json",
                "experiments_v5/v5_benchmark_results.json",
            ],
            "subset_definition": (
                "A bug is in the fragment-fair subset iff its primary "
                "failing operator is implemented in Pytea's 2022 TS "
                "operator catalogue "
                "(packages/pytea/src/ts/index.ts at commit c536515; "
                "see experiments_v5/v8/build_modern_subset.py for the "
                "per-bug TS handler line cite). The 26 out-of-subset "
                "bugs touch ops with no Pytea handler "
                "(SDPA/MHA-2.x, einsum, Conv1d/Conv3d, BatchNorm1d, "
                "GroupNorm/InstanceNorm, swapaxes, movedim, "
                "torch.where, torch.dot, linalg.*, repeat_interleave, "
                "F.embedding, gather, scatter_, isclose, "
                "split-with-list-sum, torch.add functional, "
                "torch.maximum functional, index_select)."
            ),
            "n_subset": n_subset,
            "n_full_corpus": len(rows),
            "contingency_table": {
                "both_refute": a,
                "tg_only": b,
                "pytea_only": c,
                "neither": d,
            },
            "tensorguard_refuted": tg_refuted,
            "pytea_refuted": pytea_refuted,
            "tensorguard_detection_rate": round(tg_refuted / n_subset, 4),
            "pytea_detection_rate": round(pytea_refuted / n_subset, 4),
            "mcnemar_exact_two_sided_p": p_mcnemar,
            "paper_claim_supported": (
                f"TG {tg_refuted}/{n_subset} vs Pytea {pytea_refuted}/{n_subset}, "
                f"McNemar exact two-sided p = {p_mcnemar:.4f}"
            ),
        },
        "per_bug": rows,
    }
    assert in_subset_count == 34, in_subset_count
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(out["meta"]["paper_claim_supported"])
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
