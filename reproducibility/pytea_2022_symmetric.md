# Pytea 2022 symmetric catalogue comparison

## Background

The reviewer asks whether the TG vs. Pytea gap survives a symmetric
restriction of both tools to Pytea's 2022 catalogue (commit `cb02a8a`,
2022-04-26).  This is the construction already implemented in the
modern-subset filter; this file makes the symmetry explicit.

## Filter

- Pytea catalogue: operators in `pylib/` as of commit `cb02a8a`
- Inclusion predicate: every `forward()` operator call is in that catalogue
- TG enforcement: non-catalogue handlers masked at verification time
- Pytea silent-skip correction: 3 uninformative Pytea-Verified relabelled

## Result on N=34 symmetric subset

| Tool | Refutes | Rate |
|---|---|---|
| TG (catalogue-masked) | **32/34** | 94.1% |
| Pytea 2022 (silent-skip-corrected) | **22/34** | 64.7% |

- TG-only catches (b): **10**
- Pytea-only catches (c): **0**
- Both refute: **22**
- Gap: **+29.4%**

McNemar exact two-sided p = 0.00195 (b=10, c=0); see
`pytea_modern_mcnemar.json` for the full test.

## Interpretation

The symmetric restriction yields N=34 bugs.  TG 32/34 vs.
Pytea 22/34 on the same catalogue surface.  The gap (+29.4 pp,
McNemar p=0.00195) is present and statistically significant even after
removing the operator-catalogue confound.  The N=34 sample is the
natural denominator for the fair comparison: bugs outside this set would
require either extending Pytea's catalogue or accepting asymmetric coverage.

## Reproduce

    PYTHONPATH=. python3.11 reproducibility/pytea_2022_symmetric.py
