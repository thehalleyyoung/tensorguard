# Soundness-scope partition of the 6/15 post-freeze RP fires

## Headline

| Category | Count |
|---|---:|
| in-soundness only (all handlers Lean-audited or pen-and-paper) | 0 |
| mixed (at least one tested-only or uncovered handler) | 6 |
| **Total fires** | **6** |

## Per-fire breakdown

| bug_id | TG label | triggered handlers (scope) | scope category |
|---|---|---|---|
| rb_pf_001 | TP | linear(Lean-audited), mul(uncovered) | mixed (tested-only/uncovered) |
| rb_pf_003 | TP | expand(Lean-audited), add(uncovered), einsum(pen-and-paper), unsqueeze(tested-only) | mixed (tested-only/uncovered) |
| rb_pf_004 | TP | linear(Lean-audited), softmax(tested-only) | mixed (tested-only/uncovered) |
| rb_uf_008 | TP | view(Lean-audited), reshape(Lean-audited), mul(uncovered) | mixed (tested-only/uncovered) |
| rb_uf_012 | TP | view(Lean-audited), permute(Lean-audited), conv2d(Lean-audited), mul(uncovered) | mixed (tested-only/uncovered) |
| rb_uf_010 | FP | device_mismatch(tested-only) | mixed (tested-only/uncovered) |

## Interpretation

All 6 post-freeze RP fires traverse the mixed scope: each touches at
least one tested-only or uncovered handler in addition to any Lean-audited
ones.  None fires exclusively through the 35-handler in-soundness footprint
(28 Lean-audited + 7 pen-and-paper).  A reader comparing the headline
6/15 fire count against Theorem thm:ag-sound should note that the
compositional guarantee covers the Lean-audited operators in the trace
but not the mul/add/softmax/device-mismatch handlers that co-fire.

## Paper claim (Q5)

Round-2 Q5 asks what fraction of the 6/15 fires traverse only the
35-handler in-soundness footprint.  This artefact answers: 0/6 (0%)
fire exclusively through in-soundness handlers; all 6/6 touch at least
one tested-only or uncovered handler.  The post-freeze headline therefore
does not directly validate the formal guarantee fragment.
