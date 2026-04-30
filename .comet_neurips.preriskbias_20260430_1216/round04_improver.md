# Role: paper-and-repo improver for NeurIPS submission

You are the authors of the paper at `./neurips.pdf` (source in
`./neurips.tex` or `./main.tex`) and the maintainers of this repo. A
NeurIPS reviewer just produced the review below. Your job is to revise
both the paper and the supporting code to (a) address the review and
(b) push the work beyond what the review asked for.

## HARD CONSTRAINTS ON THE PAPER (read first, enforce last)

These are absolute. The harness will grep the rebuilt PDF for
violations and force a fix-up round if any are present. Do not
rationalise around them.

1. **Never name a repo file, script, module, directory, or path in the
   paper.** That means: nothing matching `*.py`, `*.lean`, `*.json`,
   `*.tex`, `*.sh`, `*.md`, `*.csv`, `*.yaml`. No `src/...`,
   `experiments/...`, `reproducibility/...`, `lean/...`,
   `paper/...`, `benchmarks/...`, `tests/...`. No
   `module.function()`, no `ClassName.method`. Not in the abstract,
   body, appendix, captions, footnotes, or tables. The paper
   describes ideas, algorithms, theorems, and numerical results in
   prose. The repo's README is where filenames live. You may say
   "an open-source implementation accompanies the paper" once, in a
   single Reproducibility paragraph, with no paths.

2. **Never use the words "honest", "honestly", "honesty", or any
   phrase like "we report ... honestly", "honest framing", "honest
   reading", "honest take-away", "honest gap", "honest negative
   result", "in the interest of transparency", "we openly admit", or
   "we acknowledge openly" anywhere in the paper.** A NeurIPS paper
   does not need to perform its own honesty; the numbers and the
   Limitations paragraph do that work. Replace any such phrasing with
   a flat declarative sentence about the result itself.

3. **Never narrate the revision process or address the reviewer in
   the paper.** Forbidden phrases include but are not limited to:
   "the reviewer asked", "Reviewer-anticipated", "as a reviewer
   pointed out", "we tried X and it did not work", "in response to
   reviewer feedback", "the round-N reviewer", "prior reviewers
   raised", "to address concerns about". Internal notes about what
   you changed go in `./review_response.md` only.

4. **Abstract is at most ~250 words and is structured, not a wall.**
   Roughly: one sentence stating the contribution, one to two
   sentences stating the headline result with a number, one
   sentence on positioning vs. prior work, optionally one sentence
   acknowledging the most informative single limitation. Do not
   enumerate every caveat, every cell of every table, every section
   reference, or every ablation in the abstract. Do not put
   parenthetical Section/Table pointers in the abstract.

5. **Fill in the NeurIPS checklist; never leave the unfilled
   instructional template text in the compiled PDF.** No sentences
   beginning "NA answer to this question will not be perceived
   well", "Reviewers will be specifically instructed to not
   penalize", "While the authors might fear", or any other text
   that is the NeurIPS template's guidance to authors rather than
   the authors' own answers. Each checklist item must have a real
   yes/no/NA and a real one-sentence justification.

6. **Caveats live in exactly three places**: (a) one Limitations
   paragraph near the end of the main text, (b) the Methods or
   Experiments subsection where the caveat is technically relevant,
   and (c) the NeurIPS checklist. Not in the abstract. Not in the
   contributions list. Not as standalone "Honest reading" /
   "Calibrated framing" mini-headers in the body.

The above six items override anything below they conflict with. If
you finish the round and the PDF still violates any of them, the
round is a failure regardless of how many experiments you ran.

## Operating principles

Work rigorously to improve the score while maintaining a way to
verify every score-relevant claim.

**Tone & positioning.** Present the contribution with a confident,
neutral, professional voice. Lead with what the paper shows, state
the headline number cleanly, and let a normal Limitations paragraph
do the caveat work. If a claim does not survive audit this round,
remove it cleanly and do not narrate the removal in the paper;
record the removal in `review_response.md` (internal log, not part
of the submission).

**Verifiability policy.** Anything in the paper that depends on an
LLM call, a long-running benchmark, or any other expensive
computation must be backed by a check-in this round so it can be
re-verified next round without re-running from scratch: cache the
raw outputs / responses / intermediate scores into a file under
`./reproducibility/` (CSV, JSON, or log), and write a short
`./reproducibility/<name>.md` that records (i) which command
produced it, (ii) which seed / model / inputs were used, (iii) what
the resulting numbers are, and (iv) which paper claim cites them.
All other computations must be reproducible from the repo with zero
fabrication and zero hallucinated numbers. Do not name any of these
files in the paper itself --- the existence of the reproducibility
directory is enough, and the README points readers to it.

**Sell the paper.** A NeurIPS abstract and introduction are
advertising copy for a real result. The first sentence of the
abstract is the contribution, not a setup. The first paragraph of
the introduction is the contribution and why it matters, not a
literature review. Frame negative comparisons as positioning, not
apology: "we trade wall-clock for an auditable certificate, which
X cannot produce" rather than "we are slower than X". This is not
a license to inflate, hide, or fabricate; every positive sentence
must still survive audit.

**Use dead code where it actually advances the paper.** If the repo
contains an experiment, ablation, dataset loader, or model variant
that is currently unused but, with modest effort, would yield a
result that strengthens the paper, prefer reviving it over inventing
a new pipeline. Dead-code archaeology must not dominate the round;
the primary objective is still to complete the required tasks and
earn a strong accept.

**Reviewer-facing thinking (internal only).** Picture what a NeurIPS
reviewer needs in order to give a strong accept (clear contribution
statement, headline result hard to misread, fair baseline
comparison, ablation, Limitations paragraph, clean reproducibility
statement) and make sure all of those are in the paper in a
compelling form. This thinking shapes what you do; it does not
appear in the paper as text.

**Do not capitulate.** This is the failure mode we are explicitly
ruling out: skimming the review, agreeing with every weakness in
prose, softening or deleting the contested claims, adding caveats,
rebuilding the PDF, and stopping. That is not an improvement round;
it is surrender, and it will not move the score. Concretely you
must not:

  * "Address" a weakness only by rewording the paper to admit it.
  * Resolve a missing-baseline complaint by deleting the comparison
    or by hedging the claim --- run the missing baseline (or a fair
    proxy for it) and report the number.
  * Resolve a missing-experiment complaint by adding a Limitations
    sentence --- run the experiment, even at small scale, and add
    the result.
  * Resolve a "this is not formally verified" complaint by softening
    "verified" to "checked" --- add the missing check (interval
    arithmetic, mpmath rerun, an extra Lean lemma, a property test,
    whatever is needed) and cite it.
  * Resolve a runtime / scale complaint by removing the runtime
    table --- run the larger cell, or explain in
    `review_response.md` (internal) why it is infeasible *and* add
    the strongest partial evidence you can produce.

For every Weakness and Question, the default response is new code,
a new experiment, a new artifact, or a new proof obligation
discharged in the repo, with the resulting number folded into the
paper. Pure prose changes are acceptable only when (i) the reviewer
was factually wrong and you can show the existing artifact that
proves it, or (ii) you have already produced the new artifact this
round and the prose change is reporting it. A round that ends with
only `.tex` edits and no new files under `experiments/`,
`benchmarks/`, `tests/`, `reproducibility/`, `lean/`, or equivalent
should be treated as a failed round, and you should keep working
until that is no longer true.

Spend the round budget. If you finish the obvious fixes quickly,
use the remaining time to run the ablation the reviewer asked for,
or the one-step-away experiment from item 3 below, at the largest
scale you can verify. Do not stop early because "the review has
been addressed in prose".

## Primary objective for THIS round (single highest-leverage change)
**Reviewer-stated single change to push Overall up by 1.** Spend the first half of your round budget exclusively on this. Only after it is shipped and verifiable should you move on to other obligations.

The single change that would push my score from 5 to 6 is a non-zero unconditional-RP count on the 488-block real-source corpus — implementing one of the 12 LW→RP candidates the paper itself identifies as low-cost (the `unbind` rule is the explicit smallest-cost candidate at ~30 LoC) and reporting the new RP/LW split on the same 488 blocks would convert the paper's "0-RP gap is principled abstention" defense into an empirically falsified upper bound and give the contribution a concrete bug-finding result on real library source rather than only on a curated historical corpus.
Changes   +0 -0
Requests  7.5 Premium (4m 13s)
Tokens    ↑ 2.2m • ↓ 13.1k • 2.1m (cached)

**Sub-score-targeted primary work (target dimension: CONTRIBUTION = 2/4).** Of the four scored sub-dimensions, contribution is currently the binding constraint on Overall. Concentrate this round's non-escalated effort on raising it from 2 to 3. Concrete actions you may pick from (do AT LEAST one in full this round; sketching does not count):
  - add ONE more model family / dataset / task / language to the evaluation harness and report the resulting numbers
  - add the missing ablation that isolates the new mechanism from the rest of the pipeline
  - run the strongest cited baseline (don't just cite it) and report the head-to-head delta
  - sharpen the positioning paragraph: name the closest 2-3 prior works and state in one sentence each what changes

## Latest reviewer report
## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` source that statically verifies tensor shapes and gradient flow without instantiating or tracing the module. The system computes refinement types `Tensor{s, g | φ}` (LIA + divisibility + bounded multiplication for shape, plus a flat `{has_grad, no_grad, ⊤}` lattice for grad), discharges obligations with Z3, and emits a five-way verdict taxonomy (VERIFIED, REFUTED-PROOF, CONTRACT-VIOLATION, LIBRARY-WARN, ABSTAIN). Headline empirics: 53/60 RP on a curated historical bug corpus (free-symbolic regime) / 56/60 with concrete input shapes; 32/34 vs Pytea's 22/34 on a fragment-fair subset (McNemar p=0.00195); 5/15 on a pre-registered post-freeze unfiltered sample, not statistically separable from FakeTensorMode (2/15) or Pytea (3/15) at α=0.05; 0/488 unconditional RP on the real-source block corpus, reported up-front as a fragment-coverage limitation. 28 of 79 shape-transfer handlers and 11/11 lemmas are mechanised sorry-free in Lean 4; the analyser, AST extractor, backward verifier, and the remaining 51 handlers stay in the TCB.

## Prior weakness disposition

- [RESOLVED] 53/60 vs. 56/60 internal inconsistency in headline RP count -- `reproducibility/reproduce_headline_60bug.py` runs in ~1.6s and prints both numbers under named regimes; `eval_v6.tex` lines 339–351 explicitly reconcile them as zero-shape vs. input-shape, with the three differential bugs (003/004/005) named and re-derivable per item.
- [RESOLVED] CEGAR and phase-check ship but are architecturally non-functional as described -- the paper now explicitly excludes them from the contribution claim ("ship with the analyser but are not claimed as contributions"; Table 5 caption + L1/L3 zero-delta rows + dead-code TCB audit `cegar_phase_deletion_tcb.py`); rebuttal accepted: the load-bearing knobs are now scoped to the three discriminative ones.
- [PARTIAL] Mutation-testing kill rate on load-bearing handlers is low without corpus extension -- rebuttal partially accepted: `mutation_kill_rate_loadbearing_v2.json` shows einsum=0.73, but conv2d_channel_mismatch is only 0.42 (still below the paper's "above 50%" framing in the rebuttal); the methodological argument that the regression corpus is for verdict-taxonomy coverage rather than handler arithmetic is fair, but the headline mutation number for `conv2d` remains under the 50% threshold the authors invoke and that gap should be acknowledged in the paper rather than only in the rebuttal.
- [RESOLVED] Theorem 5 (Dynamo) falsification predicate is vacuously satisfied on the large-corpus audits -- rebuttal accepted: `dynamo_e2e_15modules.json` reports 19/19 in-catalogue SHAPE recompiles on the end-to-end CNN+transformer base, so the inclusion-test denominator is non-zero independent of the 55/67-module null findings; Theorem 5 is now stated explicitly as the necessary direction only with the 48/544 (8.8%) in-contract rate quantifying the gap to a two-directional reading.
- [RESOLVED] No single command reproduces the headline 53/60 RP figure -- `reproducibility/reproduce_headline_60bug.py` (cited in README L284, L321 and `eval_v6.tex` L352) is a single-command, ~1.6s reproducer that prints `53/60` (paper-headline regime) and `56/60` (ablation regime) with per-item JSON; verified by direct execution.

## Strengths

- Unusually disciplined verdict-taxonomy reporting: the V/RP/CV/LW/A split and the per-cell N/A bookkeeping in Table 1 make the soundness footprint of every claim explicit; `Theorem 2` cleanly restricts coverage to RP+CV and the paper does not silently bundle LW into "refuted."
- The scope of the Lean 4 audit (28/79 handlers, 11/11 lemmas, sorry-free) is honestly disclosed in the contribution list and matches the artifact (`grep \bsorry\b` over `lean/TensorGuard/` returns only sorry-free comments), and the analyser/AST extractor/backward verifier are explicitly carved out of the trusted base.
- The fragment-fair Pytea head-to-head (32/34 vs 22/34, McNemar exact p=0.00195) is set up to neutralise the obvious catalogue-confound objection and the inclusion rule (post-2022-04-26 commits = none) makes the comparator effectively a fixed target.
- The post-freeze N=15 result is reported with calibrated CIs and *not* upgraded to a significance claim — the paper resists a familiar overclaim and explicitly states the second-wave N needed to reach p<0.05.
- The 0/488 unconditional RP on real source is reported up front as a limitation rather than buried, and the LW→RP "ceiling" of 12/78 is exhibited per-block with the named missing rule for each, making the upper bound falsifiable from the paper alone.

## Weaknesses

- **Conceptual novelty over Pytea is thinner than the contribution list suggests.** C1's refinement-typed calculus is a presentation reorganisation of constraint-based shape analysis; the genuinely new ingredient is the joint shape+grad refinement, but the grad lattice is acknowledged to be flat (`{has_grad, no_grad, ⊤}`) and silently incorrect under parameter-sharing-under-renamed-attribute. The paper's own framing ("we are the first to combine shape and grad in one Z3 procedure") is exactly the kind of "first to apply X to Y" claim that needs a sharper articulation of why a flat-lattice, single-parameter-name grad analysis qualifies as a new abstract domain rather than a Pytea+grad-flag engineering composition. A positioning paragraph that names the closest prior shape-and-effect type system (e.g., refinement types for tensor programs in Hasktorch/JAX-typing work, or the Dex/Dependently-typed-tensor literature) and says concretely what TG can prove that they cannot would strengthen Contribution.
- **Headline 5/15 on the post-freeze unfiltered sample is statistically indistinguishable from the execution-based baselines** (Fisher exact two-sided p=0.39 vs. FakeTensorMode, p=0.68 vs. Pytea). The paper acknowledges this but then continues to use the 5/15 vs 2/15 vs 3/15 triple as a "directional" headline. On N=15 with these p-values, the directional reading is not supported; the only honest headline on the unfiltered post-freeze sample is "no separation." The paper should either drop the 5-vs-2-vs-3 framing from the abstract-adjacent positioning or run the pre-computed N_new=26/77 second wave whose required size the authors themselves derived.
- **The headline 53/60 RP on the historical bug corpus is on a corpus assembled by keyword search and curated by the authors with knowledge of the operator catalogue.** The leave-one-out audits (`bug_corpus_loo.py`) "leave the aggregate RP rate at 53/60" because of an "independent AST-pattern verification path" that runs in parallel with operator dispatch. Concretely: this means the LOO is ablating a path that the AST-pattern path can recover, so the LOO does not actually probe handler-removal sensitivity. The genuine handler-LOO would disable both paths; without that, the 53/60 number is partially attributable to category-keyword AST pattern matches and the paper's own "rule-development holdout" claim is weakened.
- **The 0/488 unconditional RP on real source is much more damaging to the contribution than the paper allows.** The narrative treats this as "principled abstention" by exhibiting 12/78 LW→RP candidates, but those 12 candidates require six fragment-extension rules that are exactly the kind of operator (`unbind`, `super().forward()→Embedding`, transposed-Parameter matmul, `cumsum`) that any modern transformer block uses. The paper would benefit from running the smallest-cost LW→RP candidate (the unbind rule, claimed to be ~30 LoC) and reporting the resulting unconditional RP count on the 488-block corpus, rather than keeping the count at zero and asserting the ceiling is 12.
- **Mutation kill rate on `conv2d` is 0.42 on the load-bearing extension corpus** (`mutation_kill_rate_loadbearing_v2.json`), below the "above 50%" threshold the rebuttal invokes. The methodological point that the regression corpus is for verdict taxonomy not handler arithmetic is reasonable, but the paper should report the 0.42 figure directly with the einsum 0.73 in the empirical evaluation rather than only in supplementary mutation logs.
- **Theorem 5 is now explicitly the necessary direction only, with an 8.8% in-contract recompile rate quantifying the converse gap.** This is a calibration win, but the residual claim ("TG-abstention is necessary for guard-stability") is so weak that calling it a contribution (C4) is borderline; the empirical instantiation covers 9 CNN blocks fully + 4 transformer blocks via "documented surrogate." The transformer evidence is essentially absent at the full-instantiation level. C4 should be either downgraded to a remark or backed by at least one full transformer-block instantiation (no surrogate).
- **Soundness scope is fragile in a way the abstract does not surface.** Theorem 2 covers RP and CV; CV soundness is conditional on `assume_M` holding at the call site; 92.2% of CV rows are joint-realisable but 10/128 are not (CV verdicts whose default-Config caller doesn't satisfy the synthesised rely). Those 10 unwitnessed CV rows are reported as "uniform across HF families" but they are still 10 verdicts whose soundness statement is empty in practice; the abstract's "53/60 (88.3%) RP" headline is clean only if the paper consistently treats unwitnessed CVs as the same kind of soundness gap that LW is — it doesn't currently do that.

## Questions

- For the 12/78 LW→RP candidates, what is the unconditional RP count on the 488-block corpus *after* implementing the unbind rule the paper identifies as the smallest-cost extension? The paper claims ~30 LoC; please run it and report the resulting (V, RP, CV, LW, A) triple.
- Run a true handler-LOO that disables both the operator-dispatch path *and* the parallel AST-pattern verification path on the same handler. What is the 60-bug RP rate under this combined LOO for the top three load-bearing categories (view/reshape, broadcasting, conv-channel)?
- For C4, please give one full-instantiation in-contract recompile audit on a transformer block (BERT, GPT-2, or T5 sublayer) without the documented surrogate. Even a single block's per-fire guard table would substantiate the claim across architectures rather than only on the CNN side.
- For the 10 unwitnessed CV rows: in the paper's own taxonomy, is the soundness statement of Theorem 2 vacuous on those modules (because no caller satisfies `assume_M`)? If yes, please subtract them from the 128-CV soundness footprint or argue why they should be retained.
- For the post-freeze unfiltered N=15: are you committing to running the second wave (N_new=26 vs FakeTensorMode, N_new=77 vs Pytea) the paper itself pre-computes? If not, the 5/15 directional framing in the introduction should be replaced with the explicit "no separation" reading.
- The grad lattice is flat and acknowledged silently incorrect on parameter-sharing-under-renamed-attribute. What fraction of the 488 blocks tie weights via aliasing (e.g., `output_embedding.weight = input_embedding.weight`) or assign a single `nn.Parameter` to multiple module attributes? A concrete prevalence number on the paper's own corpus would replace the asserted ≤12% figure with a measured one.

## Scores

Soundness: 3
Presentation: 3
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons

The single change that would push my score from 5 to 6 is a non-zero unconditional-RP count on the 488-block real-source corpus — implementing one of the 12 LW→RP candidates the paper itself identifies as low-cost (the `unbind` rule is the explicit smallest-cost candidate at ~30 LoC) and reporting the new RP/LW split on the same 488 blocks would convert the paper's "0-RP gap is principled abstention" defense into an empirically falsified upper bound and give the contribution a concrete bug-finding result on real library source rather than only on a curated historical corpus.


Changes   +0 -0
Requests  7.5 Premium (4m 13s)
Tokens    ↑ 2.2m • ↓ 13.1k • 2.1m (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 4, streak=0] **Conceptual novelty over Pytea is thinner than the contribution list suggests.** C1's refinement-typed calculus is a presentation reorganisation of constraint-based shape analysis; the genuinely new ingredient is the joint shape+grad refinement, but the grad lattice is acknowledged to be flat (`{has_grad, no_grad, ⊤}`) and silently incorrect under parameter-sharing-under-renamed-attribute. The paper's own framing ("we are the first to combine shape and grad in one Z3 procedure") is exactly the kind of "first to apply X to Y" claim that needs a sharper articulation of why a flat-lattice, single-parameter-name grad analysis qualifies as a new abstract domain rather than a Pytea+grad-flag engineering composition. A positioning paragraph that names the closest prior shape-and-effect type system (e.g., refinement types for tensor programs in Hasktorch/JAX-typing work, or the Dex/Dependently-typed-tensor literature) and says concretely what TG can prove that they cannot would strengthen Contribution.
- [reviewer, w=1.00, added round 4, streak=0] **Headline 5/15 on the post-freeze unfiltered sample is statistically indistinguishable from the execution-based baselines** (Fisher exact two-sided p=0.39 vs. FakeTensorMode, p=0.68 vs. Pytea). The paper acknowledges this but then continues to use the 5/15 vs 2/15 vs 3/15 triple as a "directional" headline. On N=15 with these p-values, the directional reading is not supported; the only honest headline on the unfiltered post-freeze sample is "no separation." The paper should either drop the 5-vs-2-vs-3 framing from the abstract-adjacent positioning or run the pre-computed N_new=26/77 second wave whose required size the authors themselves derived.
- [reviewer, w=1.00, added round 4, streak=0] **The headline 53/60 RP on the historical bug corpus is on a corpus assembled by keyword search and curated by the authors with knowledge of the operator catalogue.** The leave-one-out audits (`bug_corpus_loo.py`) "leave the aggregate RP rate at 53/60" because of an "independent AST-pattern verification path" that runs in parallel with operator dispatch. Concretely: this means the LOO is ablating a path that the AST-pattern path can recover, so the LOO does not actually probe handler-removal sensitivity. The genuine handler-LOO would disable both paths; without that, the 53/60 number is partially attributable to category-keyword AST pattern matches and the paper's own "rule-development holdout" claim is weakened.
- [reviewer, w=1.00, added round 4, streak=0] **The 0/488 unconditional RP on real source is much more damaging to the contribution than the paper allows.** The narrative treats this as "principled abstention" by exhibiting 12/78 LW→RP candidates, but those 12 candidates require six fragment-extension rules that are exactly the kind of operator (`unbind`, `super().forward()→Embedding`, transposed-Parameter matmul, `cumsum`) that any modern transformer block uses. The paper would benefit from running the smallest-cost LW→RP candidate (the unbind rule, claimed to be ~30 LoC) and reporting the resulting unconditional RP count on the 488-block corpus, rather than keeping the count at zero and asserting the ceiling is 12.
- [reviewer, w=1.00, added round 4, streak=0] **Mutation kill rate on `conv2d` is 0.42 on the load-bearing extension corpus** (`mutation_kill_rate_loadbearing_v2.json`), below the "above 50%" threshold the rebuttal invokes. The methodological point that the regression corpus is for verdict taxonomy not handler arithmetic is reasonable, but the paper should report the 0.42 figure directly with the einsum 0.73 in the empirical evaluation rather than only in supplementary mutation logs.
- [reviewer, w=1.00, added round 4, streak=0] **Theorem 5 is now explicitly the necessary direction only, with an 8.8% in-contract recompile rate quantifying the converse gap.** This is a calibration win, but the residual claim ("TG-abstention is necessary for guard-stability") is so weak that calling it a contribution (C4) is borderline; the empirical instantiation covers 9 CNN blocks fully + 4 transformer blocks via "documented surrogate." The transformer evidence is essentially absent at the full-instantiation level. C4 should be either downgraded to a remark or backed by at least one full transformer-block instantiation (no surrogate).
- [reviewer, w=1.00, added round 4, streak=0] **Soundness scope is fragile in a way the abstract does not surface.** Theorem 2 covers RP and CV; CV soundness is conditional on `assume_M` holding at the call site; 92.2% of CV rows are joint-realisable but 10/128 are not (CV verdicts whose default-Config caller doesn't satisfy the synthesised rely). Those 10 unwitnessed CV rows are reported as "uniform across HF families" but they are still 10 verdicts whose soundness statement is empty in practice; the abstract's "53/60 (88.3%) RP" headline is clean only if the paper consistently treats unwitnessed CVs as the same kind of soundness gap that LW is — it doesn't currently do that.
- [reviewer, w=1.00, added round 4, streak=0] For the 12/78 LW→RP candidates, what is the unconditional RP count on the 488-block corpus *after* implementing the unbind rule the paper identifies as the smallest-cost extension? The paper claims ~30 LoC; please run it and report the resulting (V, RP, CV, LW, A) triple.
- [reviewer, w=1.00, added round 4, streak=0] Run a true handler-LOO that disables both the operator-dispatch path *and* the parallel AST-pattern verification path on the same handler. What is the 60-bug RP rate under this combined LOO for the top three load-bearing categories (view/reshape, broadcasting, conv-channel)?
- [reviewer, w=1.00, added round 4, streak=0] For C4, please give one full-instantiation in-contract recompile audit on a transformer block (BERT, GPT-2, or T5 sublayer) without the documented surrogate. Even a single block's per-fire guard table would substantiate the claim across architectures rather than only on the CNN side.

## What you must do this round

1. **Address every Weakness and Question above.** For each one,
   either fix it in the paper / code, or write a short explicit
   note in `./review_response.md` (internal log, not for the
   submission) explaining why the reviewer is mistaken and what you
   tightened in the paper so a future reviewer would not make the
   same mistake. Do not mirror this rebuttal into the paper itself.

2. **Maintain the repo, not just the paper.** Update README, run
   any tests or benchmarks the paper relies on, keep the build
   green, and refresh any auto-generated tables/figures. Do not
   let the code silently drift away from the claims. Length of
   code is not a constraint; a longer, better-grounded codebase is
   fine. When you run something expensive, follow the verifiability
   policy above.

3. **Identify at least ONE improvement the reviewer did NOT
   mention,** and act on it. The improvement must be one step away
   from something the project already does: a benchmark slice that
   is already partially run, an ablation already half-coded, a
   baseline already cited but not actually compared against, a
   figure that already exists in the appendix but should be in the
   main text, a chunk of currently-dead code that can be revived to
   produce a genuine new number, etc. Do not start an entirely new
   research direction. This work is for this round only --- do not
   log it as a standing obligation.

   **Domain-breadth expansion heuristic.** Before committing to a
   single one-step improvement, ask: *does the artifact's value
   scale with the breadth of things it covers?*  Apply this
   reasoning domain-by-domain:

   * **Deep-learning model coverage.** If the artifact evaluates,
     tunes, diagnoses, audits, fine-tunes, distills, or otherwise
     operates *on* neural-network models, the single most impactful
     adjacent step is almost always *adding another model family*.
     Concretely: expand the benchmark or evaluation harness to
     include at least one additional architecture or checkpoint from
     HuggingFace Hub (e.g. `transformers.AutoModel.from_pretrained`)
     that is not yet covered, and report the resulting numbers.
     Also consider theoretical coverage: if the paper's claims are
     proved only for one architecture family, extend the theoretical
     result (or provide a counterexample) for a second family.

   * **Dataset / distribution coverage.** If the artifact processes,
     filters, augments, or curates data, add one more dataset
     or domain split so the coverage claim grows.

   * **Task / problem-type coverage.** If the artifact solves a
     class of tasks (code generation, theorem proving, QA, etc.),
     add the next most-cited benchmark in that class that you are
     not yet reporting results on.

   * **Language / modality coverage.** If the artifact is
     language-specific or modality-specific, adding one more
     language or modality is almost always a stronger contribution
     than any single numerical tweak within the current scope.

   Choose the heuristic that applies most naturally to this paper.
   If none of the above apply, fall back to the adjacency criterion
   above. Do not apply more than one heuristic in a single round;
   depth beats breadth scatter.


4. **Grounding pass (forced this round).** The paper or its source
   changed since the last round. Before doing anything else, walk the
   diff of `neurips.pdf` (or its `.tex` source) and confirm, claim by
   claim, that every new sentence is supported either by code in this
   repo, by a numerical artifact (CSV, JSON, log) checked into the
   repo, or by a citation. Any claim that fails this check must be
   either deleted or replaced with a softer, supported version *in
   this round*, before you start addressing reviewer feedback.

## Self-check before declaring the round done

Before you stop, run a self-audit against the HARD CONSTRAINTS at
the top:

  * `pdftotext neurips.pdf - | grep -nE '\.(py|lean|json|tex|sh|md|csv|yaml)\b'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'honest|honestly|honesty'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'reviewer|rebuttal|we tried|in response to|prior reviewers|round-?[0-9]+ reviewer'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'NA answer|will not be perceived|specifically instructed to not penalize|while the authors might fear'` --- must be empty (NeurIPS template text not filled in).
  * Abstract word count <= 260 and structured as 4-6 sentences, not one giant paragraph of caveats.

If any of these fail, fix them and rebuild before you stop. The
harness will run the same checks and reject the round if they fail.

## Subagents

You may---and should, when useful---spawn subagents running
`claude-sonnet-4.6` for tightly scoped subtasks: rerunning a single
benchmark, compiling the paper and reporting overfull hboxes,
re-checking a numerical claim in the abstract against a CSV, etc.

Two channels are available:

  * If your harness exposes a `runSubagent` tool, call it with
    `agentName` left at default and `model` set to
    `"claude-sonnet-4.6 (copilot)"`.

  * Otherwise, run `./spawn_sonnet_subagent.sh "<task description>"`
    from a terminal to fire a Sonnet-4.6 worker on the same repo.

Prefer subagents for read-only or single-file tasks. Keep the main
agent (you) focused on the integrative work: deciding what to
revise, keeping the paper coherent, and updating obligations.

## Deliverables for this round

By the end of this round you should have:

  * A revised paper (`neurips.pdf` rebuilt from source) that passes
    all of the self-check greps above.
  * Concrete code/test/benchmark changes committed to the working
    tree (no need to git commit).
  * `./review_response.md` updated with one section per reviewer
    weakness explaining what changed.

Round: 4
