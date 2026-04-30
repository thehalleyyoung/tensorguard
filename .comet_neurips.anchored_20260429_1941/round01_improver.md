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

## Latest reviewer report
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

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 1] **Theorem 3 (compositional soundness) is mechanised on a 3-operator DSL only**, while the analyser dispatches over 79 handlers. The paper is upfront about this, but the resulting formal guarantee on actual programs is much weaker than the rule-table audit suggests: even if every leaf module's per-operator rule is Lean-audited, the compositional step that lifts per-rule soundness to the module DAG is itself audited only on `{matmul, view, add}`. The contribution C2 should either (a) extend the Lean composition proof to the full handler set or (b) rephrase the contribution to not include "assume/guarantee discipline at the class boundary" as a mechanised result.
- [reviewer, w=1.00, added round 1] **48/79 handlers are "tested-only" and outside Theorem 2.** On the 488-block corpus, 105/185 in-soundness verdicts (Section 4.4) touch at least one tested-only handler — i.e. the *majority* of verdicts on the real-source surface do not enjoy the headline soundness statement. Either close the gap on at least the load-bearing handlers (the per-handler attribution names `view/reshape/total_size`, broadcasting, `conv_channel_mismatch`, einsum, etc.) or restrict Theorem 2's stated scope to the 31 in-soundness handlers wherever it is invoked in the empirical sections.
- [reviewer, w=1.00, added round 1] **The Dynamo necessary-direction audit on the larger population is empirically empty for the kinds it is supposed to test.** Across 55 successful modules it observed 72 in-contract recompiles, *all* of kind INT (Section 4.3). Zero SHAPE/DTYPE/RANK guards were observed, so the falsification predicate ("a SHAPE/DTYPE/RANK guard on a variable outside `catalogue(M)`") evaluates to 0/0 on this surface, not 0/72. The 14-module audit does report 19 SHAPE recompiles, but 4 of those modules use the documented forward-signature surrogate. The right number is the falsifier rate restricted to the 9 fully end-to-end CNN modules (13 SHAPE events); please present that as the headline and drop the 0/72 framing entirely, since it does not test the theorem.
- [reviewer, w=1.00, added round 1] **Mutation testing is weak.** 3/50 (6%) mutant kill rate, with the surviving 47 mutants attributed to "arithmetic/comparison handler paths the 60-bug corpus does not exercise," is itself an indictment of how representative the 60-bug corpus is for the analyser's actual logic. A 6% mutation score on a verifier whose central claim is soundness should be addressed: rerun against the 488-block corpus and against the bug + falsification + stress union, and report the best of those numbers.
- [reviewer, w=1.00, added round 1] **The 60-bug corpus has unverifiable handler-development independence.** The authors describe the leave-one-out audit (category-keyword LOO is a no-op by design; handler-class LOO leaves 53/60 unchanged due to a parallel AST-pattern path) but the AST-pattern verification path is itself developed by the same authors with knowledge of the corpus. The 53/60 number cannot be cleanly attributed to the operator handlers vs. the AST-pattern shortcut. Please report 53/60 *with the AST-pattern path disabled* as a separate row, so that the operator-rule contribution is isolated from the pattern-matching contribution.
- [reviewer, w=1.00, added round 1] **The N=15 post-freeze headline is not statistically separable from baselines** (Fisher p=0.39 vs FakeTensor, p=0.68 vs Pytea), which the authors acknowledge. The pre-registered second wave (Nnew=26 / 56 / 77 depending on target) is described as a precondition, not run. Without it, the real-PR claim is a directional 5/15 vs 2/15 vs 3/15 with overlapping CIs, and the headline should be presented in that frame in the abstract too (the abstract says "5/15 catches versus 2/15 and 3/15" without the confidence intervals).
- [reviewer, w=1.00, added round 1] **Per-feature stress benchmark is anti-informative.** Table 5 explicitly notes that the real-corpus ablation is a flat line, that L1 (CEGAR) and L3 (phase) are no-ops, and that two of the five "knobs" are dead code shipped with the analyser. The stress benchmark's staircase is therefore a property of how the cases were constructed, not of the analyser. Either remove the stress benchmark entirely or report only the three discriminative knobs; the current presentation invites a misreading.
- [reviewer, w=1.00, added round 1] **The grad-flag silent-error footprint is described as ≤12% of training scripts, but the lattice is first-order and acknowledged-incorrect on parameter-sharing under renamed attributes.** The 0/2,908 AST-grep on renamed-attribute patterns is reassuring, but the 333/2,908 (11.45%) `tied_weights_keys`/`tie_weights`/`_tie_or_clone_weights` count is a first-order-lattice-breaker that is not folded into the silent-error rate. Please either (a) report the silent-error rate including the 333 tied-weights-using files or (b) provide a static argument that tied-weights via the API does *not* produce a renamed-attribute alias the lattice misclassifies.
- [reviewer, w=1.00, added round 1] For the 12 named LW→RP-conversion candidates (Section 4.1 table), what is the obstacle to implementing at least one of them and reporting the resulting unconditional-RP count on the 488-block corpus? The paper presents this as a falsifiable prediction; running the experiment would convert a prediction into a measured number.
- [reviewer, w=1.00, added round 1] In the large-population Dynamo audit (Section 4.3), why are zero SHAPE/DTYPE/RANK guards observed across 72 recompiles? Is the harness genuinely exercising shape-varying inputs, or is the SymInt specialisation pre-empting them? Please provide the per-module input-shape variation for the 55 successful modules.
- [reviewer, w=1.00, added round 1] What is the measured RP→VERIFIED flip rate of the 50 mutants on the 488-block + falsification + 25-block stress union, not just on the 60-bug corpus?
- [reviewer, w=1.00, added round 1] Can you provide a 53/60 reproduction with the AST-pattern verification path disabled, so the operator-rule contribution is isolated from the pattern-matching contribution?
- [reviewer, w=1.00, added round 1] For Theorem 3, is there a concrete obstacle to mechanising assume/guarantee composition on more than 3 operators (proof engineering vs Lean-tactic limit vs operator-rule definitional shape), or is it purely a labour estimate?
- [reviewer, w=1.00, added round 1] How does the witnessed-ratio of 118/128 CV verdicts (Section 4.1) decompose by Hugging Face submodel family — i.e. are the 10 unwitnessed CVs concentrated in one architecture, or spread? This bears directly on whether the user-visible CV column is meaningful in practice.

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

Round: 1
