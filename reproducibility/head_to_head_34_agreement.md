# N=34 modern-subset per-bug agreement matrix

## Question

Round 1 W8/Q1: report the per-bug agreement matrix on the N=34
modern-subset head-to-head between TensorGuard and Pytea, and
expose the `b=10 / c=0` strict-subset structure (every Pytea
refute is also a TG refute) explicitly per bug.

## 2×2 agreement matrix

|                    | Pytea: Refute | Pytea: Verified/Abstain |  Row total |
|---:|---:|---:|---:|
| **TG: Refute**             | **22** | **10** | 32 |
| **TG: Verified/Abstain**   | **0**  | **2**  |  2 |
| Column total               | 22     | 12     | 34 |

* Discordant cells: `b = 10` (TG-only), `c = 0` (Pytea-only).
* McNemar exact two-sided p = `0.001953`.
* The structure is a **strict subset**: every Pytea refute is
  also a TG refute.  The 32 vs 22 gap is entirely concentrated in
  the 10 TG-only catches.

## TG-only catches (b=10)

Bug IDs (per `experiments_v5/v8/pytea_modern_subset.json`
ordering):
`bug_010, bug_013, bug_017, bug_022, bug_028, bug_031, bug_036,
bug_041, bug_047, bug_055`.

Categories of the 10 TG-only catches:

| Category | Count |
|---|---:|
| view/reshape total-size with -1 (TG divisibility witness) | 4 |
| broadcasting (Pytea has no handler at modern-subset commit) | 2 |
| einsum dim (Pytea has no handler at modern-subset commit) | 2 |
| transpose axes | 1 |
| linear in/out feature mismatch | 1 |

## Both-refute (22)

22 bugs caught by both tools — predominantly the
`view_reshape_total_size` and `linear_in_out` paths that lie in
both catalogues at the modern-subset commit.

## Neither (2)

`bug_001, bug_002`: both are semantically-aliased view bugs
(buggy and correct view targets agree on total element count for
the input shape that exercises the bug).  No purely
shape-arithmetic tool can fire.  Same residual class as
`rb_001`/`rb_002` in the upstream-faithful real-public-repo
corpus.

## How to verify

```
python3 experiments_v5/v8/pytea_modern_mcnemar.py
```

The script reproduces the McNemar exact two-sided
p-value `0.001953` from the 2×2 matrix above.  The per-bug
verdict log used to construct the matrix is
`experiments_v5/v8/pytea_modern_subset.json`.

## Paper claims cited

* Conservative convention (paper headline): TG 32/34 vs Pytea 25/34,
  McNemar p=0.0156, b=7, c=0.  This treats Pytea N/A
  (operator-dispatch inapplicable) rows as not-refute.  See
  `reproducibility/pytea_mcnemar_per_bug.md` for the per-bug audit.
* Silent-skip-reclassification convention (this file's matrix):
  TG 32/34 vs Pytea 22/34, McNemar p=0.00195, b=10, c=0.  This
  treats Pytea N/A rows as silent misses.  Released as the
  alternative convention so the choice is auditable; not the
  number used in the abstract or body.
