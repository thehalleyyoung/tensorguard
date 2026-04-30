● Read round02_reviewer.md
  │ .comet_neurips/round02_reviewer.md
  └ 120 lines read

● word counts (shell)
  │ wc -l /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/neurips.tex /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/te
  │ nsorguard/neurips.txt 2>/dev/null
  └ 4 lines...

● list comet_neurips and root dirs (shell)
  │ ls -la /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorgua
  │ rd/.comet_neurips/ 2>/dev/null | head -30; echo "---"; ls
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/ |
  │ head -50
  └ 50 lines...

● Read round01_reviewer.out.md
  │ .comet_neurips/round01_reviewer.out.md
  └ 82 lines read

● Read round01_improver.out.md
  │ .comet_neurips/round01_improver.out.md
  └ 65 lines read

● Read round01_improver.md
  │ .comet_neurips/round01_improver.md
  └ 1 line read

● peek improver instructions (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -200 .comet_neurips/round01_improver.md
  └ 201 lines...

● view harness state (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat .comet_neurips/history.jsonl .comet_neurips/scores.jsonl
  │ .comet_neurips/trigger_sig.json .comet_neurips/obligations.json 2>/dev/null
  └ 89 lines...

● Read neurips.txt
  └ L1:200 (200 lines read)

● Read neurips.txt
  └ L200:700 (501 lines read)

● Read neurips.txt
  └ L700:1300 (506 lines read)

● Read neurips.txt
  └ L1300:2000 (481 lines read)

● Read neurips.txt
  └ L2000:2700 (701 lines read)

● round-2 family info (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n -i "Q3\|Round-2\|disabled.*pattern\|pattern.*disabled\|stress
  │ benchmark\|three discriminative\|huggingface
  │ family\|Family\|family\|witnessed-ratio\|118/128\|HuggingFace\|architecture
  │ family" neurips.txt | head -50
  └ 17 lines...

● grep round-2 related changes (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n -i "round-2\|Round 2\|round 2\|second wave\|Nnew\|N=15\|N
  │ =15\|tied[ -]weight\|AST-pattern" neurips.txt | head -40
  └ 18 lines...

● grep round-2 mentions (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n -i "Round-2\|Round 2\|second round\|round-2 q\|witness\|Round-2
  │ Q" neurips.txt | head -40
  └ 19 lines...

● verify hard constraints (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -nE "honest|honestly|in the interest|openly" neurips.txt | head -10;
  │ echo --- LW->RP ---; grep -n "LW.*RP\|InvertedResidual\|ASPPPooling"
  │ neurips.txt | head -20
  └ 12 lines...

● grep mechanisation (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -nE "Lean|mechanis|sorry-free|Theorem 3|composition|3-operator|3
  │ operator|five-operator|5-operator|seven-operator|7-operator|nine-operator|9-
  │ operator" neurips.txt | head -30
  └ 31 lines...

● grep stress and mutation (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -nE "stress benchmark|three discriminative|Table
  │ 5|knob|five-feature|five feature|five knob|mutation|mutant" neurips.txt |
  │ head -30
  └ 19 lines...

● grep round-2 changes (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -nE "second
  │ wave|prereg|Nnew|488.+mutation|488-block.+mutation|mutation.+488|Round
  │ 2|round-2|Round-2|round 2 " neurips.txt | head -30
  └ 7 lines...

● grep ablation requests (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -nE "AST-pattern|AST pattern|pattern-matching|disabled\b|alone
  │ path|operator-only|isolation row|per-family|family decomposition|family
  │ breakdown|across families|HF family|architecture family|Llama|Bart|GPT|by
  │ family" neurips.txt | head -30
  └ 8 lines...

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` source that statically verifies tensor shapes and a coarse `requires_grad` lattice via Z3, formalised as a refinement calculus with assume/guarantee composition at the class boundary. The headline numbers are unchanged from the previous round: 53/60 Refuted-Proof on the historical bug corpus, 32/34 vs Pytea 22/34 on the fragment-fair subset (McNemar exact p=0.00195), 5/15 catches vs FakeTensor 2/15 / Pytea 3/15 on the pre-registered post-freeze N=15 sample (now explicitly framed as not statistically separable), and 0 unconditional Refuted-Proof on the 488-block real-source corpus under the user-visible free-symbolic regime. Round 2 adds a CNN-only restriction of the larger Dynamo audit (13/13 SHAPE recompiles in catalogue), a runtime grad-checkpointing harness on 8 HuggingFace heads (8/8 Refuted-Proof, 0/8 silent-verified), and a softer abstract framing of the N=15 result; the Lean composition proof, the LW→RP conversions, the AST-pattern-disabled ablation, and the 488/union mutation rerun are unchanged.

## Prior weakness disposition
- [UNRESOLVED] **Theorem 3 (compositional soundness) is mechanised on a 3-operator DSL only**, while the analyser dispatches over 79 handlers. The paper is upfront about this, but the resulting -- The Lean assume/guarantee proof remains on the matmul/view/add fragment; the Section 4.4 paragraph still names this scope explicitly and labels extension as future work.
- [UNRESOLVED] **48/79 handlers are "tested-only" and outside Theorem 2.** On the 488-block corpus, 105/185 in-soundness verdicts (Section 4.4) touch at least -- The 28+3+48 split and the 105/185 statistic are unchanged; no additional handlers were promoted into Lean or pen-and-paper soundness this round.
- [PARTIAL] **The Dynamo necessary-direction audit on the larger population is empirically empty for the kinds it is supposed to test.** Across 55 successful modul -- A CNN-only restriction (13 SHAPE events, 0 out-of-catalogue) is now the headline for the 14-module audit, but the 0/72 INT-only paragraph for the 55-module population is still printed alongside, and the falsification predicate is still reported on it.
- [UNRESOLVED] **Mutation testing is weak.** 3/50 (6%) mutant kill rate, with the surviving 47 mutants attributed to "arithmetic/comparison handler paths the 60-bug corp -- The mutation paragraph in §4.3 is verbatim 3/50 = 6.0% on the 60-bug regression with no rerun on the 488-block corpus or the bug+falsification+stress union.
- [UNRESOLVED] **The 60-bug corpus has unverifiable handler-development independence.** The authors describe the leave-one-out audit (category-keyword LOO is a no-op b -- The "AST-pattern verification path disabled" 53/60 reproduction asked for in Q4 is not added; the rule-development-holdout paragraph still attributes the 53/60 number to the parallel pattern path without isolating it.
- [PARTIAL] **The N=15 post-freeze headline is not statistically separable from baselines** (Fisher p=0.39 vs FakeTensor, p=0.68 vs Pytea), which the authors acknowled -- Abstract now includes "a directional result on N=15, not statistically separable from either baseline at α=0.05"; the second-wave experiment (Nnew=26/56/77) is still described as a precondition rather than executed.
- [UNRESOLVED] **Per-feature stress benchmark is anti-informative.** Table 5 explicitly notes that the real-corpus ablation is a flat line, that L1 (CEGAR) and L3 (p -- Table 5 and the §4.2 ablation text still present all five knobs (including the two non-discriminating ones); the requested restriction to the three discriminative knobs has not been made.
- [PARTIAL] **The grad-flag silent-error footprint is described as ≤12% of training scripts, but the lattice is first-order and acknowledged-incorrect on parameter-sh -- A new runtime trainer harness on 8 HuggingFace heads with `gradient_checkpointing_enable()` reports 8/8 Refuted-Proof and 0/8 silent-verified, and a separate held-out 1/42 audit is added; the 333/2,908 tied-weights count is still not folded into the headline silent-error rate via either route.

## Strengths
- The new runtime grad-checkpointing harness on 8 HuggingFace head classes (bert/distilbert/roberta/gpt2/t5/bart/vit/wav2vec2) with `gradient_checkpointing_enable()` is the most informative addition this round: 8/8 Refuted-Proof and 0/8 silent-verified is a strict-zero on the trainer-realistic surface where an undetected silent error would be most damaging, and it complements the static AST-grep estimate with a runtime measurement on the same construct family.
- The CNN-only restriction of the 14-module Dynamo audit (13 SHAPE recompiles, all in-catalogue, 0 falsifier events) is the right denominator for Theorem 5's necessary direction and is presented with a clean per-row breakdown.
- The five-way verdict taxonomy (V/RP/CV/LW/A), the 0-RP framing of the 488-block headline, the 12 named LW→RP candidates with per-block predicted single-rule additions, the 28-rule Lean audit (sorry-free, 28,000/28,000 byte-mirror agreement), and the witnessed-ratio 118/128 CV check are still the paper's calibrated-reporting backbone.
- The TCB exposure scan with measured RP→V flips (0/60 across F1–F4 even where exposure is up to 7) remains a useful complement to the rule-side soundness story.

## Weaknesses
- **The "Round-2 Q3" subheader inside §4.3** ("CNN-only restriction (Round-2 Q3). Restricting the aggregate to the 10 fully end-to-end CNN-type subjects ...") narrates the revision process inside the body of the paper. A NeurIPS submission should not contain markers that name reviewer rounds or reviewer questions; please rename the paragraph to something like "CNN-only restriction" with no parenthetical and no rounding marker.
- **Theorem 3 is still mechanised only on `{matmul, view, add}`.** The C2 contribution sentence explicitly says "the algorithmic discipline is implemented over the full 79-handler catalogue but is audited end-to-end only on the 3-operator fragment." Either extend the Lean composition proof to a non-trivial subset of the 28 already-Lean-audited operator rules (the natural next set is `{matmul, view, add, linear, bmm, permute, reshape}` — all already have rule-level Lean lemmas) and report the new mechanised count in §4.4, or remove "assume/guarantee discipline at the class boundary" from C2 and re-state C2 as the per-rule audit only.
- **The 105/185 in-soundness verdicts touching a tested-only handler is unchanged.** No handler was promoted into Lean or pen-and-paper soundness this round. The reviewer-flagged load-bearing handlers (`view/reshape/total_size`, broadcasting, `conv_channel_mismatch`, einsum) are precisely the ones whose preservation lemma is the smallest delta to the existing Lean tree; close at least one this round and report the new in-soundness ratio.
- **Mutation testing is still 3/50 (6%) on the 60-bug corpus only.** The Q3 obligation — rerun the same 50 mutants on the 488-block corpus and on the bug+falsification+stress union, and report the best of those numbers — has not been discharged. Please run it; this is an O(50 × N) sweep on inputs the analyser already processes.
- **The AST-pattern-disabled 53/60 reproduction is not in the paper.** §4.1 still attributes the leave-one-out 53/60 invariance to "an independent AST-pattern verification path runs in parallel with the operator dispatch and catches the bugs even when the per-category handlers are disabled," which is exactly the confound the reviewer asked to isolate. Add a row to the rule-development-holdout subsection reporting the bug-corpus RP rate with the AST-pattern path disabled.
- **The 488-block 0-RP number has not been moved.** The 12 named LW→RP candidates remain candidates; none have been implemented. Implementing any one of `Tensor.unbind(dim)`-fixed-length tuple (timm::ChannelAttention) or `super().forward()→Embedding` (Bart/Whisper PosEmb) is reported as a single missing rule that would unconditionally flip a specified block to RP. Implementing one of these and reporting the resulting 1/488 (or higher) unconditional-RP count would convert the falsifiable prediction into a measured number.
- **Per-feature stress benchmark still presents the two-non-discriminating knobs (CEGAR, phase) as columns of Table 5.** The §4.2 paragraph says only three knobs move verdicts on the real corpus; please either drop the two non-discriminating columns from Table 5 or re-frame the table caption so the three-knob conclusion is the main result and the two zero-delta knobs are an explicit "shipped, did not discriminate" footnote, rather than visually presented as part of a five-knob staircase.
- **The 333/2,908 (11.45%) tied-weights footprint is still not folded into the headline silent-error rate.** The new 8/8-RP runtime harness is reassuring on the gradient-checkpointing axis, but the requested static argument that `tied_weights_keys` / `tie_weights` / `_tie_or_clone_weights` does not produce a renamed-attribute alias the lattice misclassifies is missing. Either add one runtime model (e.g., `bert-base-uncased` with shared input/output embeddings) to the 8-model harness as a tied-weights positive case and report whether the analyser RPs or silently verifies, or include the 333 in the silent-error denominator with the resulting prevalence number.

## Questions
- For Theorem 3, is the obstacle to mechanising assume/guarantee composition on a non-trivial subset (say `{matmul, view, add, linear, bmm}`) Lean-tactic engineering, the operator-rule definitional shape, or pure labour estimate? A specific number (lemmas, days, and which lemmas are blocked on what) would clarify whether C2 can be made mechanised next round or whether it should be re-scoped now.
- What is the bug-corpus RP rate with the AST-pattern verification path disabled? A single-row reproduction would isolate the operator-rule contribution from the pattern-matching path.
- For the 50-mutant sweep, what are the kill rates on (i) the 488-block corpus, (ii) the 25-block falsification corpus, and (iii) the union of all three? The current 6% on the 60-bug corpus alone is a measurement of the corpus, not of the analyser.
- For the witnessed-ratio 118/128 CV result, how do the 10 unwitnessed CVs distribute by HuggingFace family (Llama / Bart / Whisper / OPT / Falcon / GPT-NeoX / other)? Concentration in one family would weaken the user-visible CV column for that family specifically.
- Of the 12 named LW→RP candidates, which one has the smallest implementation cost (estimated lines added to a single handler) and what is the obstacle to implementing it this round?
- The new runtime grad-checkpointing harness reports 8/8 Refuted-Proof. Does the analyser refuse on the *presence* of `gradient_checkpointing_enable()` (a pattern-level RP) or on a specific reverse-reachable parameter analysis? If the former, please describe how the RP is justified under Theorem 2 — pattern-level RPs without a per-parameter witness are usually outside the soundness statement.

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 6

## Borderline reasons
The single change that would push this paper to a clear accept (Overall 7) is implementing one of the 12 named LW→RP candidates and reporting the resulting unconditional-RP count on the 488-block corpus as non-zero — this would convert the paper's central falsifiable prediction into a measured number on the real-source surface, and the 0-RP gap that currently dominates the 488-block headline would no longer be a structural ceiling. A close second would be extending the Lean composition proof beyond the 3-operator DSL onto a non-trivial subset of the already-Lean-audited rules, which would cleanly resolve the C2 scope mismatch flagged here and in the prior round.


Changes   +0 -0
Requests  7.5 Premium (3m 21s)
Tokens    ↑ 1.0m • ↓ 10.1k • 944.1k (cached)
