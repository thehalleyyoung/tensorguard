● Read round03_reviewer.md
  │ .comet_neurips/round03_reviewer.md
  └ 120 lines read

● List repo and tex files (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ | head -50 && echo "---" && wc -l neurips.tex neurips.txt 2>/dev/null
  └ 42 lines...

● Read neurips.txt
  └ L1:200 (199 lines read)

● Read neurips.txt
  └ L200:600 (401 lines read)

● Read neurips.txt
  └ L600:1100 (501 lines read)

● Read neurips.txt
  └ L1100:1600 (500 lines read)

● Read neurips.txt
  └ L1600:2100 (469 lines read)

● Search (grep)
  │ "Round-2 Q3" (neurips.txt)
  └ No matches found

● Search (grep)
  │ "AST-pattern|AST.pattern.disabled" (neurips.txt)
  └ 1 file found

● Search (grep)
  │ "tied_weights|tied weights" (neurips.txt)
  └ 1 file found

● Search (grep)
  │ "independent AST" (neurips.txt)
  └ 1 file found

● Search (grep)
  │ "unbind" (neurips.txt)
  └ 1 file found

● Find specific tokens in neurips.txt (shell)
  │ grep -n "AST-pattern\|tied_weights\|tied weights\|unbind\|Round-2" /Users/ha
  │ lleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/neurips.txt |
  │ head -30
  └ 12 lines...

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically verifies symbolic shapes and a coarse three-element gradient-flag lattice from class source, without instantiation or tracing. The contribution is a refinement-typed calculus `Tensor{s,g | φ}` with assume/guarantee composition at the class boundary, a Lean 4 audit of 28/79 shape-transfer rules (with 11/11 previously-axiomatic lemmas now closed sorry-free) and a 13-operator mechanised composition DSL, plus a backward verifier targeting three canonical silent-zero-grad bug classes. Headline empirics are 53/60 RP on a curated bug corpus; 32/34 vs Pytea 22/34 (McNemar p=0.00195) on a fragment-fair subset; 5/15 catches vs FakeTensor 2/15 / Pytea 3/15 on a pre-registered post-freeze unfiltered real-PR sample (not separable at α=0.05). On 488 real `nn.Module` blocks the user-visible free-symbolic-config regime returns 0 unconditional RP, calibrated as a fragment-coverage limit rather than a bug-finding number.

## Prior weakness disposition
- [RESOLVED] The "Round-2 Q3" subheader inside §4.3 ("CNN-only restriction (Round-2 Q3). Restricting the aggregate to the 10 fully end-to-end CNN-type subjects ...") narrates the revision process inside the body -- The `(Round-2 Q3)` parenthetical is gone; §4.3 now reads "CNN-only restriction. Restricting the aggregate ..." (no occurrences of "Round-2" anywhere in the paper).
- [PARTIAL] **Theorem 3 is still mechanised only on `{matmul, view, add}`.** The C2 contribution sentence explicitly says "the algorithmic discipline is implemented over the full 79-handler catalogue but is audited end-to-end only on the 3-operator ..." -- The mechanised DSL has been expanded from 3 operators to a 13-operator set (matmul, view, add/broadcast_add, linear, transpose, permute, relu, cat, sum_reduce, mean_reduce, expand, gather, embedding) and `ag_composition_ext` is now the operator-agnostic composition lemma; the gap to the full 79 catalogue remains.
- [UNRESOLVED] **The 105/185 in-soundness verdicts touching a tested-only handler is unchanged.** No handler was promoted into Lean or pen-and-paper soundness this round. -- §4.4 still reports 36/185 in-soundness vs 105/185 touching at least one tested-only handler; load-bearing handlers (view/reshape, broadcast_add) remain in the tested-only bucket.
- [RESOLVED] **Mutation testing is still 3/50 (6%) on the 60-bug corpus only.** The Q3 obligation — rerun the same 50 mutants on the 488-block corpus and on the bug+falsification+stress union, and report the best of those numbers -- §4.3 now reports per-corpus 60-bug 3/50, 488-block 7/50, 25-stress 5/50, and best-of (union) 7/50 = 14.0%, exactly the requested triple plus union.
- [UNRESOLVED] **The AST-pattern-disabled 53/60 reproduction is not in the paper.** §4.1 still attributes the leave-one-out 53/60 invariance to "an independent AST-pattern verification path runs in parallel with the operator dispatch and catches the bu... -- The exact sentence is reproduced verbatim around line 1200; no number is reported for "what is the RP rate when the AST-pattern path is disabled?" so the LOO invariance still rests on an unmeasured fallback.
- [UNRESOLVED] **The 488-block 0-RP number has not been moved.** The 12 named LW→RP candidates remain candidates; none have been implemented. -- The 12-row table is reproduced unchanged with all 12 entries still labelled "Predicted RP"; `Tensor.unbind(dim)` is now explicitly nominated as the smallest-cost candidate but is not implemented; the headline triple is still 57V / 0RP / 206 LW+CV / 225 A.
- [PARTIAL] **Per-feature stress benchmark still presents the two-non-discriminating knobs (CEGAR, phase) as columns of Table 5.** -- L1 (CEGAR) and L3 (phase) rows are kept but explicitly tagged "[shipped, no-op]" in the Feature column with a +0 ∆ entry, the caption flags them as "two zero-delta rows ... listed for completeness as shipped, did not discriminate", and a deletion audit is added; the request "drop OR label clearly" is met by the labeling option, though the columns are not removed.
- [PARTIAL] **The 333/2,908 (11.45%) tied-weights footprint is still not folded into the headline silent-error rate.** The new 8/8-RP runtime harness is reassuring on the gradient-checkpointing axis, but the requested static argument that `tied_weights` ... -- A held-out runtime trainer harness on 8 HF heads with `gradient_checkpointing_enable()` returns 8/8 RP (false-verified rate 0/8); but the static argument that the 333 tied-weight files are not silent-error-positive under the lattice (the specific Q4 ask) is not given — only the disjoint checkpointing slice is closed.

## Strengths
- The 13-operator extension of the Lean-mechanised composition theorem (`ag_composition_ext`) is a substantive structural improvement over the previous 3-operator DSL and is now operator-agnostic in shape, which is the right factoring.
- The mutation-testing measurement is now triangulated across three corpora (60-bug 3/50, 488-block 7/50, 25-stress 5/50, union 7/50 = 14%); reporting per-corpus and best-of together is the right protocol and meaningfully tightens the analyser-robustness floor.
- The TCB fault-injection footprint table (F1–F4 with both *exposure* upper bounds and *measured* RP→V flip counts on the 60-bug corpus) is a clean, falsifiable presentation: exposure 0/0/2/7 vs measured 0/0/0/0, with the gap explained by re-catch on the same path. This is the kind of calibrated implementation-level audit reviewers ask for and rarely get.
- The 8/8-RP runtime trainer harness on HuggingFace heads under `gradient_checkpointing_enable()`, paired with the 1/42 static `examples/pytorch/` audit, gives an end-to-end runtime confirmation that the silent-error regime is correctly forced into Abstain on the construct family it targets.
- Calibrated reporting throughout (Wilson / Clopper-Pearson CIs, McNemar exact, Fisher exact with the explicit non-separation on N=15, pre-registered post-freeze query) is unusually disciplined for a systems-flavoured ML paper and makes the empirical claims load-bearable.

## Weaknesses
- The leave-one-out audit in §4.1 still asserts that disabling per-category handlers "leaves the aggregate RP rate at 53/60, because an independent AST-pattern verification path runs in parallel". This load-bearing claim has no measurement attached anywhere in the paper or in the reproducibility appendix: the AST-pattern-disabled rerun and its bug-by-bug verdict triple should be reported as a number (e.g. RP rate with the AST-pattern path disabled, on the same 60-bug corpus), since the rest of the LOO story is currently unfalsifiable.
- The 488-block 0-unconditional-RP gap is unchanged: all 12 named LW→RP candidates remain in the "Predicted RP" column with no implementation or measurement. The smallest-cost candidate (`Tensor.unbind(dim)` fixed-length tuple, ChannelAttention/ChannelAttentionV2) has been reduced to "approximately 30 source lines of the same form as the existing split/chunk handlers" and the analyser change is described as "mechanical"; please implement it and report the resulting RP-on-488-blocks number (1/488 vs 0/488 would already be falsifiable progress on the headline).
- 105/185 in-soundness verdicts on the 488-block corpus still touch at least one tested-only handler (§4.4). The load-bearing handlers in this bucket — `view/reshape`, `broadcast_add`, `Conv2d`-output — are exactly the ones the F3/F4 fault-injection scan exposes the most blocks to (18 and 19 respectively on the 488 surface). Promoting at least one of these into pen-and-paper soundness (the 3-handler bucket) would directly shrink the 105 number; please report which handler was promoted and the new in-soundness/tested-only split.
- The Lean composition theorem is now mechanised on 13 operators but the 13-operator DSL still excludes the four operators on the bug paths of the post-freeze N=15 catches (`linear` is in; `view/reshape`, `permute`, `conv2d`, `einsum`, `expand` are partially in or out). The C2 contribution sentence as written ("audited end-to-end on this 13-operator fragment") understates the gap to the operators that actually fire on Table 3 catches; either tighten the contribution claim or extend the DSL to cover the union of the bug-firing handlers in Table 3 (specifically `view/reshape` + `conv2d` + `einsum` + `expand`, which would lift the DSL to ~17 operators and cover all 5 of the post-freeze headline catches).
- The 333/2,908 (11.45%) tied-weight footprint is still not directly folded into the headline silent-error rate. The 8/8-RP runtime harness is on `gradient_checkpointing_enable()`, a *different* axis (the 5/2,908 = 0.17% slice of the same sweep). The requested measurement is: of the 333 tied-weight modules, how many does the analyser return Verified vs Abstain vs Refuted on, and on a runtime ground-truth subset of those, what is the false-verified rate? Please report that triple — the 8/8 number on the disjoint checkpointing slice does not generalise to the tied-weights slice without an explicit measurement.
- Theorem 5's empirical surface is still uneven: the 14-module CNN-only restriction (13 SHAPE recompiles, all in-catalogue) is the falsifier-evaluation headline, but the larger 55-module audit "observed 72 in-contract recompiles, all classified as integer / SymInt specialisations (kind INT); the falsification predicate is therefore not exercised". Reporting that "the larger sample contains zero SHAPE/DTYPE/RANK guards" is a *measurement that the falsifier never fires on the surface where it could have* — please add at least one curated module on which a SHAPE guard outside `catalogue(M)` is at least *plausible* (e.g. a custom op that reads a non-catalogue shape bit), so the falsification predicate is shown to be non-vacuously evaluable on something other than the 14-module CNN curation.

## Questions
- What is the RP rate on the 60-bug corpus with the AST-pattern verification path disabled (i.e. operator-dispatch only)? Without this number the 53/60 LOO invariance is unverifiable.
- Implement the `Tensor.unbind(dim)` fixed-length tuple shape rule (the smallest-cost candidate, ~30 LoC). What is the new RP count on the 488-block corpus, and does it flip exactly the two predicted blocks (`timm::ChannelAttention`, `timm::ChannelAttentionV2`) to RP without regression elsewhere?
- Restricted to the 333/2,908 tied-weight modules: what is the analyser verdict triple (V/RP/Abstain) and, on a runtime-instantiable subset, the false-Verified rate against a one-step `loss.backward()` ground truth?
- The 13-operator mechanised DSL covers `linear` and `expand` but not `view/reshape`, `conv2d`, or `einsum`, all of which fire on the Table 3 post-freeze catches. Can you extend `ag_composition_ext` to the operator union of the post-freeze catches and report the new `lake build` status?
- Of the 105/185 in-soundness verdicts touching a tested-only handler, which single handler promotion (Lean-audited or pen-and-paper) would shrink that count the most, and what is the projected new split?
- On the 55-module Dynamo audit where all 72 observed in-contract recompiles were INT-only, can you exhibit at least one module on which the falsification predicate (SHAPE/DTYPE/RANK guard outside `catalogue(M)`) is *capable* of firing, and confirm that the necessary direction still holds there?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 6

## Borderline reasons
The single largest score-moving change would be to convert the 488-block 0-unconditional-RP headline into a non-zero number by implementing the `Tensor.unbind(dim)` rule (the authors have already characterised it as ~30 LoC and isolated the elaborator change), and report the resulting 488-block RP count and the per-block flip witnesses. That single concrete RP on real library source — paired with the AST-pattern-disabled 60-bug rerun to close the unmeasured LOO claim — would push Soundness and Contribution each by one and Overall to 7.


Changes   +0 -0
Requests  7.5 Premium (2m 36s)
Tokens    ↑ 587.5k • ↓ 8.1k • 530.7k (cached)
