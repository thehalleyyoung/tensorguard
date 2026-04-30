● Read round01_reviewer.md
  │ .comet_neurips/round01_reviewer.md
  └ 113 lines read

● List files (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ && wc -l neurips.txt neurips.tex 2>/dev/null
  └ 16 lines...

● Read neurips.tex
  └ 90 lines read

● Read neurips.txt
  └ L1:200 (200 lines read)

● Read neurips.txt
  └ L200:600 (401 lines read)

● Read neurips.txt
  └ L600:1100 (501 lines read)

● Read neurips.txt
  └ L1100:1700 (563 lines read)

● Read neurips.txt
  └ L1663:2200 (523 lines read)

● List dirs (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls src/ lean/ tests/ benchmarks/ 2>&1 | head -100
  └ 101 lines...

● List lean and tests (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/TensorGuard/ 2>&1 | head -50; echo ---; ls tests/ 2>&1 | head -40
  └ 47 lines...

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` classes that statically verifies tensor shapes and a coarse `requires_grad` lattice from class source via Z3, without instantiation or tracing. The paper formalises a refinement calculus with assume/guarantee composition at the module boundary, and reports Refuted-Proof on 53/60 historical bugs, 32/34 vs Pytea 22/34 on a fragment-fair subset (McNemar exact p=0.00195), and 5/15 catches vs FakeTensor 2/15 / Pytea 3/15 on a pre-registered post-freeze real-PR sample (no statistical separation at N=15). On a 488-block real-source corpus the user-visible free-symbolic regime yields 0 unconditional Refuted-Proof, which the authors report as a coverage measurement rather than a bug-finding result. A Lean 4 audit closes 11/11 soundness lemmas sorry-free over 28 of 79 shape-transfer rules; the analyser implementation, AST extractor, backward verifier, Z3 dispatch, and assume/guarantee composition (mechanised only on a 3-op DSL) remain in the trusted computing base.

## Prior weakness disposition
(none — first round)

## Strengths
- The five-way verdict taxonomy (V/RP/CV/LW/A) and the explicit separation of soundness coverage from "shipped but not in theorem" is unusually disciplined for an empirical ML-systems paper. The 488-block headline is reported up front as 0 unconditional RP, with the LW→RP ceiling pinned at 12 and named per block — this is falsifiable and a model of calibrated reporting.
- The Lean audit (28 rules, 11/11 sorry-free, exact byte-mirror agreement 28,000/28,000 against torch 2.9.1, plus an off-envelope boundary check on 10 rules) is concrete mechanised work, not a stub.
- The Pytea head-to-head is correctly fragment-fair (operator intersection enforced at verification time, silent-skip-corrected, with the upstream commit log argument that the 2022 catalogue *is* the present catalogue), and the McNemar p=0.00195 with paired bootstrap [+14.7, +44.1] pp is properly computed.
- The pre-registered post-freeze sample (N=15) plus an off-axis RP-fire reported as a false positive against ground truth, and the catalogue freeze SHA pinned, materially reduces retro-fitting concerns.
- The TCB exposure scan with measured RP→V flips on four hand-picked single-edit faults (0/60 measured even where exposure is up to 7) is a useful complement to the Lean-side soundness story.

## Weaknesses
- **Theorem 3 (compositional soundness) is mechanised on a 3-operator DSL only**, while the analyser dispatches over 79 handlers. The paper is upfront about this, but the resulting formal guarantee on actual programs is much weaker than the rule-table audit suggests: even if every leaf module's per-operator rule is Lean-audited, the compositional step that lifts per-rule soundness to the module DAG is itself audited only on `{matmul, view, add}`. The contribution C2 should either (a) extend the Lean composition proof to the full handler set or (b) rephrase the contribution to not include "assume/guarantee discipline at the class boundary" as a mechanised result.
- **48/79 handlers are "tested-only" and outside Theorem 2.** On the 488-block corpus, 105/185 in-soundness verdicts (Section 4.4) touch at least one tested-only handler — i.e. the *majority* of verdicts on the real-source surface do not enjoy the headline soundness statement. Either close the gap on at least the load-bearing handlers (the per-handler attribution names `view/reshape/total_size`, broadcasting, `conv_channel_mismatch`, einsum, etc.) or restrict Theorem 2's stated scope to the 31 in-soundness handlers wherever it is invoked in the empirical sections.
- **The Dynamo necessary-direction audit on the larger population is empirically empty for the kinds it is supposed to test.** Across 55 successful modules it observed 72 in-contract recompiles, *all* of kind INT (Section 4.3). Zero SHAPE/DTYPE/RANK guards were observed, so the falsification predicate ("a SHAPE/DTYPE/RANK guard on a variable outside `catalogue(M)`") evaluates to 0/0 on this surface, not 0/72. The 14-module audit does report 19 SHAPE recompiles, but 4 of those modules use the documented forward-signature surrogate. The right number is the falsifier rate restricted to the 9 fully end-to-end CNN modules (13 SHAPE events); please present that as the headline and drop the 0/72 framing entirely, since it does not test the theorem.
- **Mutation testing is weak.** 3/50 (6%) mutant kill rate, with the surviving 47 mutants attributed to "arithmetic/comparison handler paths the 60-bug corpus does not exercise," is itself an indictment of how representative the 60-bug corpus is for the analyser's actual logic. A 6% mutation score on a verifier whose central claim is soundness should be addressed: rerun against the 488-block corpus and against the bug + falsification + stress union, and report the best of those numbers.
- **The 60-bug corpus has unverifiable handler-development independence.** The authors describe the leave-one-out audit (category-keyword LOO is a no-op by design; handler-class LOO leaves 53/60 unchanged due to a parallel AST-pattern path) but the AST-pattern verification path is itself developed by the same authors with knowledge of the corpus. The 53/60 number cannot be cleanly attributed to the operator handlers vs. the AST-pattern shortcut. Please report 53/60 *with the AST-pattern path disabled* as a separate row, so that the operator-rule contribution is isolated from the pattern-matching contribution.
- **The N=15 post-freeze headline is not statistically separable from baselines** (Fisher p=0.39 vs FakeTensor, p=0.68 vs Pytea), which the authors acknowledge. The pre-registered second wave (Nnew=26 / 56 / 77 depending on target) is described as a precondition, not run. Without it, the real-PR claim is a directional 5/15 vs 2/15 vs 3/15 with overlapping CIs, and the headline should be presented in that frame in the abstract too (the abstract says "5/15 catches versus 2/15 and 3/15" without the confidence intervals).
- **Per-feature stress benchmark is anti-informative.** Table 5 explicitly notes that the real-corpus ablation is a flat line, that L1 (CEGAR) and L3 (phase) are no-ops, and that two of the five "knobs" are dead code shipped with the analyser. The stress benchmark's staircase is therefore a property of how the cases were constructed, not of the analyser. Either remove the stress benchmark entirely or report only the three discriminative knobs; the current presentation invites a misreading.
- **The grad-flag silent-error footprint is described as ≤12% of training scripts, but the lattice is first-order and acknowledged-incorrect on parameter-sharing under renamed attributes.** The 0/2,908 AST-grep on renamed-attribute patterns is reassuring, but the 333/2,908 (11.45%) `tied_weights_keys`/`tie_weights`/`_tie_or_clone_weights` count is a first-order-lattice-breaker that is not folded into the silent-error rate. Please either (a) report the silent-error rate including the 333 tied-weights-using files or (b) provide a static argument that tied-weights via the API does *not* produce a renamed-attribute alias the lattice misclassifies.

## Questions
- For the 12 named LW→RP-conversion candidates (Section 4.1 table), what is the obstacle to implementing at least one of them and reporting the resulting unconditional-RP count on the 488-block corpus? The paper presents this as a falsifiable prediction; running the experiment would convert a prediction into a measured number.
- In the large-population Dynamo audit (Section 4.3), why are zero SHAPE/DTYPE/RANK guards observed across 72 recompiles? Is the harness genuinely exercising shape-varying inputs, or is the SymInt specialisation pre-empting them? Please provide the per-module input-shape variation for the 55 successful modules.
- What is the measured RP→VERIFIED flip rate of the 50 mutants on the 488-block + falsification + 25-block stress union, not just on the 60-bug corpus?
- Can you provide a 53/60 reproduction with the AST-pattern verification path disabled, so the operator-rule contribution is isolated from the pattern-matching contribution?
- For Theorem 3, is there a concrete obstacle to mechanising assume/guarantee composition on more than 3 operators (proof engineering vs Lean-tactic limit vs operator-rule definitional shape), or is it purely a labour estimate?
- How does the witnessed-ratio of 118/128 CV verdicts (Section 4.1) decompose by Hugging Face submodel family — i.e. are the 10 unwitnessed CVs concentrated in one architecture, or spread? This bears directly on whether the user-visible CV column is meaningful in practice.

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 6

## Borderline reasons
The paper would clear the bar to a clear accept (Overall 7) if either (a) at least one of the 12 named LW→RP candidates were implemented and the unconditional-RP count on the 488-block corpus moved off zero, or (b) the assume/guarantee composition rule (Theorem 3) were mechanised on a non-trivial subset of the 79-handler surface — currently the headline soundness claim and the analyser's actual operator surface are separated by a gap the paper acknowledges but does not close.


Changes   +0 -0
Requests  7.5 Premium (2m 3s)
Tokens    ↑ 521.6k • ↓ 6.0k • 444.7k (cached)
