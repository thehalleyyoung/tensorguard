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
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` classes that statically infers tensor shape and `requires_grad` refinements from class source. The system formalises a refinement-type calculus `Tensor[τ, φ_shape, φ_grad]` over an `nn.Module` fragment F_TG, with assume/guarantee composition at the class boundary, a backward-pass verifier for three silent-zero-grad bug classes, and a necessary-direction correspondence to TorchDynamo guards. A 28-rule subset of the 79-handler shape-transfer table is mechanised in Lean 4 with 11/11 soundness lemmas closed sorry-free. Empirically the paper reports 53/60 RP on a curated historical bug corpus, 32/34 vs Pytea 22/34 on a fragment-fair subset (McNemar p=0.00195), 5/15 catches on a pre-registered post-freeze unfiltered real-PR sample (vs FakeTensorMode 2/15 and Pytea 3/15, not statistically separable), and 0 unconditional RP on the 488-block real-source corpus under the free-symbolic-config regime.

## Prior weakness disposition
(none — first round)

## Strengths
- Unusually calibrated reporting: the paper does not hide that on the 488-block real corpus it issues 0 unconditional RP under the free-symbolic-config regime; verdicts are split RP/CV/LW/Abstain and only RP+CV are claimed sound (Sec 4.1, Table 1). This is rare in the ML-tooling literature.
- The Lean 4 audit is real and inspectable (`lean/TensorGuard/`, the tree builds sorry-free per `grep`-able comments and source); the byte-mirror cross-check against torch 2.9.1 on 28,000 in-fragment samples is a meaningful operator-rule sanity check.
- The fragment-fair head-to-head with Pytea (N=34, McNemar exact p=0.00195, paired-bootstrap CI [+14.7, +44.1] pp) is a methodologically sound way of isolating the catalogue-confound from the comparison.
- Pre-registration of the post-freeze GitHub-search query (2026-04-08) and the freeze-hash protocol (`040f6f3`) credibly rules out retro-fitting handlers to the headline corpus.
- A genuine ablation/falsification surface exists: Table 4 shows TG and FakeTensorMode are complementary (20 vs 5) on importable inputs, and Table 5 honestly reports that two of five claimed knobs (CEGAR, phase) are no-ops in the current implementation.

## Weaknesses
- The headline post-freeze unfiltered comparison (Table 3: TG 5/15 vs FakeTensor 2/15 vs Pytea 3/15) is not statistically separable at α=0.05 (Fisher p=0.39 and 0.68; the authors say so), and the Bayes factors (8.1, 3.6) do not clear "strong evidence". Yet the abstract still presents this as "point estimate above both baselines". On N=15 with these p-values, the substantive evidence for TG > baselines on out-of-distribution PRs is weak; a larger pre-registered draw (N≥50, same query) is needed before this can be the third headline number.
- Theorem 2's soundness scope (Lean-audited + pen-and-paper handlers, i.e. 28+3=31/79) covers only 36/185 in-soundness verdicts on the 488-block corpus per the paper itself (line ~498); the remaining 105/185 verdicts touch at least one tested-only handler and are therefore outside the formal soundness theorem. The contribution as advertised in the abstract ("Lean-audited operator-rule table") understates that 48/79 handlers are tested-only and most real-corpus verdicts depend on them.
- Theorem 3 (compositional soundness of the assume/guarantee discipline) is mechanised only on a 3-operator DSL (matmul/view/add), yet C2 is sold as a primary contribution. The gap between "assume/guarantee at the `nn.Module` boundary" as a research artifact and what is actually proven sound is substantial; please either mechanise the composition rule on a non-trivial subset of the 79-handler surface or restate C2 to scope the formal claim to the 3-operator DSL throughout the contribution list, not just in the theorem statement.
- The bug-corpus exclusion pipeline removes ~237/1087 (~22%) of keyword hits under categories (iii) distributed-shape and (iv) config-attribute, where the latter is precisely the silent-miss class flagged in §6 and produces 0/113 RP in the reported audit. The 53/60 (88.3%) headline therefore measures performance conditional on the fragment, not on shape-bug PRs in the wild. A complementary "headline-on-all-1087" or "headline-on-(60+113)" denominator would let readers price in the scope cap.
- The Theorem 5 instantiation uses a "documented forward-signature surrogate" on 4 of the 14 modules (and on 16 of the 17 modules in the prior audit because "full instantiation exceeds end-to-end constraint solving"). The empirical falsifier (zero in-contract recompiles outside the catalogue) is therefore evaluated against a hand-written contract, not against the contract TG would synthesise end-to-end on a transformer. Either run the audit end-to-end on at least one transformer block, or drop "Dynamo-guard correspondence" from the contributions and frame Theorem 5 as a CNN-block result with an open transformer obligation.
- The localisation result (Sec 4.2: tracer ±5 lines on 14/17 caught bugs) has a self-selecting numerator: the 13/30 cases where TG did not refute or returned `location.line=0` are exactly the cases where localisation is needed and missing. The reported 82% should be paired with an end-to-end "fraction of the 30 bugs localised within ±5 lines" (i.e. 14/30 = 47%) so the headline is comparable to a runtime tracer that always fires.
- The L1 (CEGAR) and L3 (phase) features ship in the analyser but are no-ops (Table 5 caption), which is honest but raises a TCB concern: dead-but-present code paths can still affect verdict computation. Please confirm (e.g. by running the test suite with these modules deleted, not just disabled) that the unused CEGAR loop and the always-satisfiable phase encoder cannot influence any RP/CV verdict on the headline corpora.
- The grad-flag silent-misclassification regime (parameter sharing under renamed attributes; ≤12% prevalence) is acknowledged but not reflected in the headline. Since C3 explicitly claims "0/50 false positives" on the backward verifier, please report the false-negative rate of the silent-misclassification regime on a corpus drawn from the ≤12% slice (e.g. 50 training scripts that exhibit tied weights), so C3's calibration matches its denominator.

## Questions
- On the post-freeze N=15 sample, what is the smallest N at which the pre-registered query would give Fisher-exact p<0.05 vs both baselines under the observed 5/15, 2/15, 3/15 rates? Is a pre-registered N=50 draw feasible before camera-ready?
- How many of the 53/60 RPs on the historical bug corpus depend on at least one handler outside the 31 Lean-audited + pen-and-paper soundness scope? A per-bug attribution would let readers compute the in-soundness RP count directly.
- For Theorem 5's necessary-direction claim, does the 8.8% (48/544) in-contract recompile rate include any guards on shape/dtype/rank variables that are in `catalogue(M)` syntactically but whose specialiser bit is read on a different aten op than the one TG's rule fires on? The falsification predicate as stated would not catch this aliasing.
- Can the assume/guarantee composition rule (Theorem 3) be mechanised on at least the matmul/view/add/broadcast/conv2d/reshape subset, or is there a structural obstacle (e.g. handler representation in Python that does not lift to Lean)? If so, please name it.
- The 32/34 vs 22/34 fragment-fair Pytea comparison enforces the modern subset by a static AST predicate over Pytea's commit `cb02a8a` catalogue; what is the verdict triple if this predicate is also enforced on TG's *handler dispatch trace* (not just the AST), to ensure no TG handler outside Pytea's catalogue contributed to a refutation indirectly via Z3?
- For the 2/10 silent verifieds on `rb_001`/`rb_002` (semantically-aliased view bugs), is there an einops-style typed-target extension that the existing rule table can encode, or does this require a new judgement form? A one-paragraph sketch in the limitations would clarify whether this is a future-rule or future-fragment problem.

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 6

## Borderline reasons
The single change that would push my Overall to 7 is a pre-registered N≥50 post-freeze draw under the same GitHub-search query showing TG > FakeTensor and TG > Pytea at Fisher-exact p<0.05; that would convert the third headline number from a directional point estimate to a real claim, and the rest of the paper (Lean audit, fragment-fair Pytea win, calibrated reporting) is already strong enough to carry an accept at that point.

Round: 1


Changes   +0 -0
Requests  7.5 Premium (2m 5s)
Tokens    ↑ 564.6k • ↓ 6.2k • 515.4k (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 1] The headline post-freeze unfiltered comparison (Table 3: TG 5/15 vs FakeTensor 2/15 vs Pytea 3/15) is not statistically separable at α=0.05 (Fisher p=0.39 and 0.68; the authors say so), and the Bayes factors (8.1, 3.6) do not clear "strong evidence". Yet the abstract still presents this as "point estimate above both baselines". On N=15 with these p-values, the substantive evidence for TG > baselines on out-of-distribution PRs is weak; a larger pre-registered draw (N≥50, same query) is needed before this can be the third headline number.
- [reviewer, w=1.00, added round 1] Theorem 2's soundness scope (Lean-audited + pen-and-paper handlers, i.e. 28+3=31/79) covers only 36/185 in-soundness verdicts on the 488-block corpus per the paper itself (line ~498); the remaining 105/185 verdicts touch at least one tested-only handler and are therefore outside the formal soundness theorem. The contribution as advertised in the abstract ("Lean-audited operator-rule table") understates that 48/79 handlers are tested-only and most real-corpus verdicts depend on them.
- [reviewer, w=1.00, added round 1] Theorem 3 (compositional soundness of the assume/guarantee discipline) is mechanised only on a 3-operator DSL (matmul/view/add), yet C2 is sold as a primary contribution. The gap between "assume/guarantee at the `nn.Module` boundary" as a research artifact and what is actually proven sound is substantial; please either mechanise the composition rule on a non-trivial subset of the 79-handler surface or restate C2 to scope the formal claim to the 3-operator DSL throughout the contribution list, not just in the theorem statement.
- [reviewer, w=1.00, added round 1] The bug-corpus exclusion pipeline removes ~237/1087 (~22%) of keyword hits under categories (iii) distributed-shape and (iv) config-attribute, where the latter is precisely the silent-miss class flagged in §6 and produces 0/113 RP in the reported audit. The 53/60 (88.3%) headline therefore measures performance conditional on the fragment, not on shape-bug PRs in the wild. A complementary "headline-on-all-1087" or "headline-on-(60+113)" denominator would let readers price in the scope cap.
- [reviewer, w=1.00, added round 1] The Theorem 5 instantiation uses a "documented forward-signature surrogate" on 4 of the 14 modules (and on 16 of the 17 modules in the prior audit because "full instantiation exceeds end-to-end constraint solving"). The empirical falsifier (zero in-contract recompiles outside the catalogue) is therefore evaluated against a hand-written contract, not against the contract TG would synthesise end-to-end on a transformer. Either run the audit end-to-end on at least one transformer block, or drop "Dynamo-guard correspondence" from the contributions and frame Theorem 5 as a CNN-block result with an open transformer obligation.
- [reviewer, w=1.00, added round 1] The localisation result (Sec 4.2: tracer ±5 lines on 14/17 caught bugs) has a self-selecting numerator: the 13/30 cases where TG did not refute or returned `location.line=0` are exactly the cases where localisation is needed and missing. The reported 82% should be paired with an end-to-end "fraction of the 30 bugs localised within ±5 lines" (i.e. 14/30 = 47%) so the headline is comparable to a runtime tracer that always fires.
- [reviewer, w=1.00, added round 1] The L1 (CEGAR) and L3 (phase) features ship in the analyser but are no-ops (Table 5 caption), which is honest but raises a TCB concern: dead-but-present code paths can still affect verdict computation. Please confirm (e.g. by running the test suite with these modules deleted, not just disabled) that the unused CEGAR loop and the always-satisfiable phase encoder cannot influence any RP/CV verdict on the headline corpora.
- [reviewer, w=1.00, added round 1] The grad-flag silent-misclassification regime (parameter sharing under renamed attributes; ≤12% prevalence) is acknowledged but not reflected in the headline. Since C3 explicitly claims "0/50 false positives" on the backward verifier, please report the false-negative rate of the silent-misclassification regime on a corpus drawn from the ≤12% slice (e.g. 50 training scripts that exhibit tied weights), so C3's calibration matches its denominator.
- [reviewer, w=1.00, added round 1] On the post-freeze N=15 sample, what is the smallest N at which the pre-registered query would give Fisher-exact p<0.05 vs both baselines under the observed 5/15, 2/15, 3/15 rates? Is a pre-registered N=50 draw feasible before camera-ready?
- [reviewer, w=1.00, added round 1] How many of the 53/60 RPs on the historical bug corpus depend on at least one handler outside the 31 Lean-audited + pen-and-paper soundness scope? A per-bug attribution would let readers compute the in-soundness RP count directly.
- [reviewer, w=1.00, added round 1] For Theorem 5's necessary-direction claim, does the 8.8% (48/544) in-contract recompile rate include any guards on shape/dtype/rank variables that are in `catalogue(M)` syntactically but whose specialiser bit is read on a different aten op than the one TG's rule fires on? The falsification predicate as stated would not catch this aliasing.
- [reviewer, w=1.00, added round 1] Can the assume/guarantee composition rule (Theorem 3) be mechanised on at least the matmul/view/add/broadcast/conv2d/reshape subset, or is there a structural obstacle (e.g. handler representation in Python that does not lift to Lean)? If so, please name it.
- [reviewer, w=1.00, added round 1] The 32/34 vs 22/34 fragment-fair Pytea comparison enforces the modern subset by a static AST predicate over Pytea's commit `cb02a8a` catalogue; what is the verdict triple if this predicate is also enforced on TG's *handler dispatch trace* (not just the AST), to ensure no TG handler outside Pytea's catalogue contributed to a refutation indirectly via Z3?
- [reviewer, w=1.00, added round 1] For the 2/10 silent verifieds on `rb_001`/`rb_002` (semantically-aliased view bugs), is there an einops-style typed-target extension that the existing rule table can encode, or does this require a new judgement form? A one-paragraph sketch in the limitations would clarify whether this is a future-rule or future-fragment problem.

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
