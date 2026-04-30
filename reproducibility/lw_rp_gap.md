# LW-RP gap analysis (round-7 reviewer Q6)

Source: `experiments_v5/v8/lw_rp_gap.py`. Re-run with
`python3.11 experiments_v5/v8/lw_rp_gap.py`.

Total LIBRARY_WARN verdicts on the 488-block corpus: **78**

## Per-bucket breakdown

| Bucket | Count |
|---|---|
| `inheritance_only` | 37 |
| `subclass_inherited_dispatch` | 27 |
| `in_fragment_op_only` | 12 |
| `module_iter` | 2 |

## Aggregate

- **Dispatch-outside-fragment** (legitimately conservative, principled abstain even at the catalogue limit): **66/78**
- **Catalogue-internal** (uses only fragment ops; in principle reachable by RP if internal reasoning were strengthened): **12/78**

## Interpretation

66/78 of the LW verdicts arise from constructs explicitly outside the operator-catalogue fragment (no forward body, dynamic getattr/**kwargs, or for-loop over self-referenced opaque containers); these are principled abstentions for which no realistic catalogue extension would yield an RP. The remaining 12/78 use only fragment ops and could in principle be reached by strengthening the catalogue-internal reasoning; this is the ceiling for converting LW->RP on the 488-block corpus without expanding the operator-catalogue fragment itself.

Practical reading: the headline `0 RP` triple on the 488-block corpus is dominated by
principled abstentions, not by missed reasoning. The `catalogue-internal` slice gives a
concrete upper bound on how many additional RPs strengthening internal reasoning could
produce without enlarging the operator catalogue.
