# Per-block handler scope on the 488-block corpus

Reviewer Q1 (round 1): of the verdicts inside the soundness theorem 
(Verified, Refuted-Proof), how many touch *only* Lean-audited or 
pen-and-paper handlers vs. at least one tested-only handler?

- **Total blocks scanned**: 488
- **Lean-audited handlers**: 36
- **Pen-and-paper handlers**: 13
- **Tested-only handlers**: 34

## Headline regime (with synthesised assume_M): 57 Verified blocks

- only Lean-or-pen-paper handlers: **33**
- touches at least one tested-only handler: **10**
- only uncovered handlers: 9
- no handlers detected: 5

Therefore the **soundness theorem applies tightly to 33/57** Verified verdicts; the remaining 
24 touch handlers covered only by random 
agreement testing.  The paper now reports both numbers in §4.4 
rather than the union under the 'Lean-audited' framing.

## No-assume regime: 34 user-visible Verified blocks

- only Lean-or-pen-paper handlers: **17**
- touches at least one tested-only handler: **10**
- (no handlers / uncovered): 7

## CV (128 blocks) under headline regime — round-3 Q1

- only Lean-or-pen-paper handlers: **95**
- touches at least one tested-only handler: **2**
- only uncovered handlers: 21
- no handlers detected: 10

Therefore on the 128 CV verdicts the soundness theorem applies tightly (entire forward path inside the Lean-or-pen-paper footprint) to **95/128** verdicts; 2 touch at least one of the 48 tested-only handlers.

## Combined V+CV (185 in-soundness verdicts under assume regime)

- only Lean-or-pen-paper handlers: **128/185**
- touches at least one tested-only handler: **12/185**

## Detection methodology

Source-token regex (see HANDLER_TOKENS in `reproducibility/handler_scope_per_block.py`).  Conservative: ambiguous tokens are attributed to the handler.  This *over*-counts tested-only touches, so the reported 'tight Lean coverage' bucket is a lower bound and the 'tested-only' bucket is an upper bound on the true partition.
