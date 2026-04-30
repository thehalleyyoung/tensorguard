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
TensorGuard is a no-execution refinement-type checker for `nn.Module` forward methods that statically verifies tensor shapes and a coarse `requires_grad` flag from class source, without instantiating or tracing the module. It introduces a Tensor{shape, grad | φ} calculus discharged by Z3, an assume/guarantee discipline at the module boundary (mechanised in Lean 4 on a 13-operator DSL), a backward-pass verifier for three silent-zero-grad classes, and a one-directional inclusion lemma against TorchDynamo's guards. The shape-transfer rule table is mechanised in Lean for 28 of 79 handlers (sorry-free, with 28k/28k torch-parity samples). Empirically the paper reports 53/60 RP on a curated historical bug corpus, 32/34 vs Pytea 22/34 on a fragment-fair subset (McNemar p=0.00195), 5/15 catches vs FakeTensorMode 2/15 / Pytea 3/15 on a pre-registered post-freeze unfiltered sample (not statistically separable), and 0 unconditional RP on the 488-block real-source corpus under the user-visible free-symbolic regime.

## Prior weakness disposition
(none — first round)

## Strengths
- Genuinely calibrated reporting: a five-way verdict taxonomy (V/RP/CV/LW/A), an explicit pre-registered post-freeze unfiltered N=15 sample with confidence intervals and Fisher-exact p-values that the paper does *not* spin into a significance claim, and an upfront acknowledgment that the user-visible 488-block headline produces 0 unconditional RP.
- Lean 4 mechanisation is real and auditable: `lean/TensorGuard/{Extended,Soundness,Parity,V5OperatorRules}.lean` build sorry-free in the tree (verified by grep), 28 shape-transfer rules with 11/11 previously-axiomatic lemmas closed, and a JSON registry export gating Python/Lean drift.
- The fragment-fair Pytea head-to-head is rigorously isolated from the catalogue confound (Pytea repo has no commits after 2022-04-26, mechanically restricted to common ops at verification time, silent-skip-corrected), giving a genuine 32/34 vs 22/34.
- The hybrid-mode N=25 stress contingency table (Table 4) is a clean falsification of the "TG and FakeTensor refute the same things on importable code" reading: 20 TG-only / 5 FT-only / 0 overlap.
- Useful negative result on `nn.Module` execution baselines (FakeTensorMode/torch.fx/torch.export N/A on 481/488 blocks) due to constructor-argument requirements — a structurally meaningful gap.

## Weaknesses
- The headline "soundness theorem" surface is much narrower than the paper's prose suggests. Theorem 2 covers only 28 Lean-audited + 3 pen-and-paper of 79 handlers (Section 4.4); on the 488-block corpus only 36/185 verified+CV verdicts touch *exclusively* in-soundness handlers, while 105/185 traverse at least one tested-only handler. The composition rule (Theorem 3) is mechanised on 13 operators of 79. The user is therefore being asked to trust an analyser whose Lean guarantee covers a minority of verdict paths, and the paper does not quantify how many of the 53/60 historical RPs land entirely inside the in-soundness footprint.
- The user-visible real-source result is a non-result. Under the free-symbolic-config regime — the only regime that does not assume a synthesised, possibly vacuous caller-rely — TG returns 0 unconditional RP and 34/0/206/248 on 488 blocks. The 12/78 LW→RP "ceiling" is a prediction, not a measurement; the paper should report the actual count after implementing at least the smallest-cost rule (`Tensor.unbind(dim)`) it identifies, since that conversion is described as ~30 LoC.
- The N=15 pre-registered post-freeze sample is the best generalization evidence, and it is not statistically separable from either baseline (Fisher p=0.39 vs FakeTensorMode, p=0.68 vs Pytea). 2/5 of the catches are attributed solely to the round-6 envelope additions and 1 of 6 RP fires (rb_uf_010) is an off-axis false positive against ground truth. The honest reading is "directional only on N=15"; the abstract still places "5/15 catches versus FakeTensorMode 2/15 and Pytea 3/15" as a contribution-bullet without making the non-separation salient at that altitude.
- The 60-bug corpus has substantial curator latitude: 1,087 keyword-search hits filtered to 60 by four hand-defined exclusion rules, including ~113 "config-attribute bugs" excluded under rule (iv) on which TG returns 0/113 RP. The 88.3% headline is therefore conditional on the same team's exclusion criteria; the leave-one-out audit retains 53/60 only because an independent AST-pattern path catches them, which is precisely the kind of redundancy that masks rule-table overfitting.
- Theorem 5 is empirical for the transformer case. 4/13 modules in the extended audit (Sec. 4.3) and 16/17 in the original audit are evaluated through a "documented forward-signature surrogate" because full instantiation exceeds constraint solving. The denominator audit on 55 successful modules contributes 0 SHAPE/DTYPE/RANK guards (all 72 recompiles are INT specialisations), so the falsifier is *not exercised* on the larger population — it is exercised only on 13 CNN events. This is a much narrower correspondence than the abstract's "TorchDynamo guards become the runtime shadow of these refinements."
- Mutation testing is weak: 3/50 kill rate on the 60-bug corpus, 7/50 union across three corpora (14%). For an analyser whose sales pitch is soundness, a 86% mutant-survival rate is a meaningful negative signal about test sensitivity, and the paper presents this as a complement to the four hand-picked TCB faults rather than as the limitation it is.
- The grad-flag contribution is bounded by a 12% prevalence ceiling for parameter-sharing-under-renamed-attribute and `torch.utils.checkpoint`, but the 500/500 random-module agreement is on a generated grammar and the 10-model real-world sweep is on models the paper itself notes do not exercise either failure mode. The 8/8 trainer harness with `gradient_checkpointing_enable()` returns RP because the construct flips out of the lattice — this measures detection of the construct, not soundness of the lattice on it.
- Per-feature ablation on the real corpora is a flat line for all five knobs (Sec. 4.2), and CEGAR/phase are explicitly "shipped, did not discriminate." Combined with the 0-RP free-symbolic regime, this means three of the six contributions (C5's CEGAR, the phase encoder, the localisation tracer) make no difference to the headline numbers; the contribution list overstates what is actually load-bearing.

## Questions
- Of the 53/60 RP on the historical corpus and the 7+1/10 RP on the upstream-faithful corpus, how many bugs are caught entirely along handler paths that are Lean-audited or pen-and-paper-audited (i.e. inside Theorem 2's footprint)? A per-bug scope column would tell the reader how much of the "88.3%" is actually backed by the mechanised guarantee.
- After implementing the `Tensor.unbind(dim)` rule that the paper itself identifies as ~30 LoC, what is the actual converted RP count on the 12 residual LW blocks? The paper currently reports a falsifiable prediction (12 RPs) but not the measurement.
- For the N=15 post-freeze sample, what is the per-PR ground-truth label (shape vs. dtype vs. distributed vs. autograd) and the per-tool TP/FP/FN matrix? With one off-axis fire already documented (rb_uf_010), the relevant comparison is precision at fixed recall, not a raw catch count.
- Theorem 5's transformer surrogate replaces full instantiation with the documented forward signature. On at least one transformer block (say `timm ViT Block`), can you exhibit a single in-contract input on which the surrogate's predicted guard set and the actually-installed Dynamo guard set agree, and quantify the gap if not?
- The 1,087→60 funnel relies on four hand-curated exclusion categories; for category (iv) ("config-attribute bugs") the paper reports 0/113 RP. Could an independent third party reconstruct the inclusion list from the published query, and what fraction of the 60 would survive a held-out reviewer's re-application of the same rules without seeing the analyser's behaviour on each candidate?
- The mutation-kill rate is 14% at union of three corpora; what is the kill rate on the in-soundness-handler subset, and would adding the standard mutation operators (statement deletion, condition negation in branch guards) on `src/typing_rules.py` and `src/backward/` materially change the estimate?

## Scores
Soundness: 3
Presentation: 3
Contribution: 2
Confidence: 3
Overall: 5

## Borderline reasons
The single change that would push this to a 6 is closing the user-visible real-source headline: implement at least one of the named LW→RP rules (`Tensor.unbind(dim)` is described as ~30 LoC) and report the resulting unconditional-RP count on the 488-block corpus under the free-symbolic-config regime, so that the paper has at least one positive number on the surface its abstract is being read against rather than only on curated/historical corpora.


Changes   +0 -0
Requests  7.5 Premium (3m 49s)
Tokens    ↑ 1.1m • ↓ 7.0k • 1.1m (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 1] The headline "soundness theorem" surface is much narrower than the paper's prose suggests. Theorem 2 covers only 28 Lean-audited + 3 pen-and-paper of 79 handlers (Section 4.4); on the 488-block corpus only 36/185 verified+CV verdicts touch *exclusively* in-soundness handlers, while 105/185 traverse at least one tested-only handler. The composition rule (Theorem 3) is mechanised on 13 operators of 79. The user is therefore being asked to trust an analyser whose Lean guarantee covers a minority of verdict paths, and the paper does not quantify how many of the 53/60 historical RPs land entirely inside the in-soundness footprint.
- [reviewer, w=1.00, added round 1] The user-visible real-source result is a non-result. Under the free-symbolic-config regime — the only regime that does not assume a synthesised, possibly vacuous caller-rely — TG returns 0 unconditional RP and 34/0/206/248 on 488 blocks. The 12/78 LW→RP "ceiling" is a prediction, not a measurement; the paper should report the actual count after implementing at least the smallest-cost rule (`Tensor.unbind(dim)`) it identifies, since that conversion is described as ~30 LoC.
- [reviewer, w=1.00, added round 1] The N=15 pre-registered post-freeze sample is the best generalization evidence, and it is not statistically separable from either baseline (Fisher p=0.39 vs FakeTensorMode, p=0.68 vs Pytea). 2/5 of the catches are attributed solely to the round-6 envelope additions and 1 of 6 RP fires (rb_uf_010) is an off-axis false positive against ground truth. The honest reading is "directional only on N=15"; the abstract still places "5/15 catches versus FakeTensorMode 2/15 and Pytea 3/15" as a contribution-bullet without making the non-separation salient at that altitude.
- [reviewer, w=1.00, added round 1] The 60-bug corpus has substantial curator latitude: 1,087 keyword-search hits filtered to 60 by four hand-defined exclusion rules, including ~113 "config-attribute bugs" excluded under rule (iv) on which TG returns 0/113 RP. The 88.3% headline is therefore conditional on the same team's exclusion criteria; the leave-one-out audit retains 53/60 only because an independent AST-pattern path catches them, which is precisely the kind of redundancy that masks rule-table overfitting.
- [reviewer, w=1.00, added round 1] Theorem 5 is empirical for the transformer case. 4/13 modules in the extended audit (Sec. 4.3) and 16/17 in the original audit are evaluated through a "documented forward-signature surrogate" because full instantiation exceeds constraint solving. The denominator audit on 55 successful modules contributes 0 SHAPE/DTYPE/RANK guards (all 72 recompiles are INT specialisations), so the falsifier is *not exercised* on the larger population — it is exercised only on 13 CNN events. This is a much narrower correspondence than the abstract's "TorchDynamo guards become the runtime shadow of these refinements."
- [reviewer, w=1.00, added round 1] Mutation testing is weak: 3/50 kill rate on the 60-bug corpus, 7/50 union across three corpora (14%). For an analyser whose sales pitch is soundness, a 86% mutant-survival rate is a meaningful negative signal about test sensitivity, and the paper presents this as a complement to the four hand-picked TCB faults rather than as the limitation it is.
- [reviewer, w=1.00, added round 1] The grad-flag contribution is bounded by a 12% prevalence ceiling for parameter-sharing-under-renamed-attribute and `torch.utils.checkpoint`, but the 500/500 random-module agreement is on a generated grammar and the 10-model real-world sweep is on models the paper itself notes do not exercise either failure mode. The 8/8 trainer harness with `gradient_checkpointing_enable()` returns RP because the construct flips out of the lattice — this measures detection of the construct, not soundness of the lattice on it.
- [reviewer, w=1.00, added round 1] Per-feature ablation on the real corpora is a flat line for all five knobs (Sec. 4.2), and CEGAR/phase are explicitly "shipped, did not discriminate." Combined with the 0-RP free-symbolic regime, this means three of the six contributions (C5's CEGAR, the phase encoder, the localisation tracer) make no difference to the headline numbers; the contribution list overstates what is actually load-bearing.
- [reviewer, w=1.00, added round 1] Of the 53/60 RP on the historical corpus and the 7+1/10 RP on the upstream-faithful corpus, how many bugs are caught entirely along handler paths that are Lean-audited or pen-and-paper-audited (i.e. inside Theorem 2's footprint)? A per-bug scope column would tell the reader how much of the "88.3%" is actually backed by the mechanised guarantee.
- [reviewer, w=1.00, added round 1] After implementing the `Tensor.unbind(dim)` rule that the paper itself identifies as ~30 LoC, what is the actual converted RP count on the 12 residual LW blocks? The paper currently reports a falsifiable prediction (12 RPs) but not the measurement.
- [reviewer, w=1.00, added round 1] For the N=15 post-freeze sample, what is the per-PR ground-truth label (shape vs. dtype vs. distributed vs. autograd) and the per-tool TP/FP/FN matrix? With one off-axis fire already documented (rb_uf_010), the relevant comparison is precision at fixed recall, not a raw catch count.
- [reviewer, w=1.00, added round 1] Theorem 5's transformer surrogate replaces full instantiation with the documented forward signature. On at least one transformer block (say `timm ViT Block`), can you exhibit a single in-contract input on which the surrogate's predicted guard set and the actually-installed Dynamo guard set agree, and quantify the gap if not?
- [reviewer, w=1.00, added round 1] The 1,087→60 funnel relies on four hand-curated exclusion categories; for category (iv) ("config-attribute bugs") the paper reports 0/113 RP. Could an independent third party reconstruct the inclusion list from the published query, and what fraction of the 60 would survive a held-out reviewer's re-application of the same rules without seeing the analyser's behaviour on each candidate?
- [reviewer, w=1.00, added round 1] The mutation-kill rate is 14% at union of three corpora; what is the kill rate on the in-soundness-handler subset, and would adding the standard mutation operators (statement deletion, condition negation in branch guards) on `src/typing_rules.py` and `src/backward/` materially change the estimate?

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
