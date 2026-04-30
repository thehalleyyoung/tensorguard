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

The single change that would push this to a 6 is removing the C5 contradiction in the Introduction *and* embedding the per-bug Pytea contingency table plus the AST-extractor oracle-validation numbers directly in the compiled PDF — those are three small surgical edits that would convert claims currently parked in `reproducibility/` into evidence a reviewer can inspect inside the paper. A second change that would push it further is a non-surrogate Theorem 5 audit on ≥ 10 transformer blocks; without that, C4 should be downgraded from "preliminary result" to "exploratory."
Round: 4
Changes   +0 -0
Requests  7.5 Premium (4m 58s)
Tokens    ↑ 2.0m • ↓ 15.4k • 1.9m (cached)

## Latest reviewer report
## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` source that statically verifies tensor shapes and a static `requires_grad` flag without instantiating or tracing the module. The paper formalises a calculus `Tensor{s, g | φ}` over Z3-decidable shape predicates plus a flat grad lattice, gives an assume/guarantee discipline at the `nn.Module` boundary, and mechanises a 13-operator composition lemma plus 11/11 previously-axiomatic soundness lemmas (sorry-free) on a 28/79-handler subset in Lean 4. Empirically, it claims 53/60 Refuted-Proof on a curated historical bug corpus, 32/34 vs Pytea 22/34 on a fragment-fair modern subset (McNemar exact p=0.00195), and a directional 5/15 vs FakeTensorMode 2/15 vs Pytea 3/15 on a pre-registered post-freeze real-PR sample (explicitly reported as not separable at α=0.05). On a 488-block real-source corpus the user-visible free-symbolic regime returns 0 unconditional RP, framed as a fragment-coverage measurement, with 36/185 in-soundness verdicts touching only Lean-or-pen-paper handlers. A preliminary Dynamo-guard inclusion lemma (necessary direction only) is empirically audited on 14 modules.

## Prior weakness disposition
- [UNRESOLVED] C5 wording in the Introduction contradicts the paper's own ablation data... -- C5 (lines 113–117) still reads "a per-feature stress benchmark on 25 targeted cases (Table 5) reporting that, **on the real corpora**, only three knobs ... move verdicts," while §4.2 (lines 1308–1311) explicitly says "On the 488-block + 60-bug aggregate corpus the per-feature ablation is a flat line"; the contradiction is now between intro and §4.2 instead of within §4.2, but it is still in the paper.
- [PARTIAL] Pytea per-bug table claimed but absent from the compiled artifact... -- the 34-row contingency table now exists at `reproducibility/pytea_mcnemar_per_bug.md` (25 both / 7 TG-only / 0 Pytea-only / 2 neither), but it is still not embedded in the PDF's appendix; the paper only refers to "the reproducibility appendix" without reproducing the table where the McNemar/CI claim is made.
- [PARTIAL] Inconsistent "unwitnessed" terminology... -- §4.1 (lines 706–711) now distinguishes "Zero `assume_M` is unwitnessed" (no contradictory empty contract) from "10 unwitnessed/non-witnessed rows" (default `*Config()` omits one declared sym-attr) by adding the joint-realisability framing, but the word "unwitnessed" is still reused for two different predicates within the same paragraph.
- [RESOLVED] Lean footprint covers a small fraction of deployed verdicts... -- the paper now states the footprint explicitly ("11 of the 57 Verified verdicts and 25 of the 128 CV verdicts touch only handlers in this set (36/185 in-soundness verdicts in total), while 33+72=105/185 touch at least one of the 48 tested-only handlers", lines 1684–1690) and labels the remainder a "TCB obligation rather than as an in-theorem result"; this is an honest disclosure, not a fix to coverage.
- [PARTIAL] AST extractor untrusted and unvalidated against an independent oracle... -- `reproducibility/ast_extractor_oracle_validation.py` now exists and reports `symbolic_config_attrs ⊆ oracle config refs` at 140/140 across the in-repo corpora, but the paper itself never mentions this oracle, never reports its numbers, and continues to label the AST extractor as "not audited" (line 124).

## Strengths
- The verdict taxonomy is genuinely calibrated. The paper consistently distinguishes RP vs CV vs LW vs Abstain vs Verified, and the headline in the abstract that the 488-block free-symbolic regime gives "0 unconditional RP" is reported up front as a fragment-coverage measurement rather than buried.
- The Lean side is honest about its scope. Closing 11/11 previously-axiomatic soundness lemmas sorry-free, exporting the operator registry as JSON so the analyser cannot silently reference an undeclared op, and printing per-handler {Lean / pen-and-paper / tested-only} in Table 7 is a meaningful artifact compared to most papers in this area.
- The matched-pair Pytea comparison on the modern subset is set up correctly: identical operator-catalogue restriction enforced on both tools at verification time, and the McNemar exact statistic with a paired-bootstrap CI lower bound at +14.7 pp is the right test for the design (b=10, c=0).
- The TCB fault-injection scan (F1–F4) with both exposure ceilings and measured RP→Verified flips is a more credible robustness check than any of the comparable PL/ML papers cited.

## Weaknesses
- The C5 contradiction flagged a round ago is still in the introduction. C5 (lines 112–117) attributes the "only three knobs move verdicts" claim to "the real corpora", but §4.2 (lines 1308–1311) says the real-corpus per-feature ablation is *flat*, with the three-knob result coming from a hand-built 25-case stress benchmark that was "constructed so that each feature would discriminate." A reader of the contributions list will form a false impression of what was demonstrated on real code; the correct fix is to localise C5 to the stress benchmark.
- The Pytea matched-pair claim still rests on a 34-row table that is not in the compiled PDF. The paper writes "the membership table is committed at ." (line 982) and "the 32/34 figure is reproduced at verification time ... by\n\nwhich AST-screens each repro" (lines 971–972) — both have empty `\href`/`\cite` targets in the rendered text. A reviewer cannot inspect the per-bug breakdown without leaving the paper, and the broken cross-references undermine the artifact in its own right.
- The AST extractor — explicitly identified as the component synthesising `assume_M` and as the load-bearing TCB component for *every* CV verdict — is now actually validated against an independent simple-AST oracle in `reproducibility/ast_extractor_oracle_validation.py` (140/140 ⊆-config agreement), but the paper does not surface this. The validation result, the corpora it covers (113-fixture + 10-real-public + 6 post-freeze + 9 unfiltered = 140 classes), and its scope (soundness direction only, scalar-attrs only 63/140 ⊆) belong in the body where the AST extractor is named in the TCB list (lines 22, 124, 1673, 1917, 2024). Without that, the audit exists but does not retire the prior reviewer's concern at the level of the paper.
- The headline real-source bug-finding evidence is weak. The 488-block result is 0 unconditional RP (now framed as a fragment measurement); the 5/15 post-freeze unfiltered comparison is "not statistically separable from either baseline at α=0.05" (line 18). What is left is a curated 60-bug corpus and a fragment-fair Pytea subset that, by construction, biases toward operators TensorGuard's catalogue covers. The claim that this checker provides a useful real-world bug-finding signal beyond curated examples is therefore not yet demonstrated; the paper is honest about this but the evidence base does not support a strong contribution claim under C3/C5.
- Theorem 5 (the Dynamo-guard inclusion lemma) is "preliminary, necessary-direction only" with empirical audit on 14 modules of which 4 transformer blocks use a "documented forward-signature surrogate because their full instantiation exceeds constraint solving at this scale" (lines 109–111). For a result that ties the contribution to a widely-deployed PyTorch component, "9 CNN blocks fully + 4 transformer surrogates" is a small base; the paper should either expand this audit (e.g. ≥ 30 modules, no surrogate on ≥ 10 transformer blocks) or downgrade the framing of C4 below "preliminary" to "exploratory."
- The grad-flag lattice's silent-error caveat (Section 6) is bounded by an AST-grep sweep returning 0/2,908 renamed-attribute hits, 1/42 silent-error-positive on training scripts, and 0/8 false-verified on a runtime trainer harness — this is good. But the held-out runtime sweep is on 8 well-known HF heads, and the 2,908-file sweep uses a fixed regex union; an adversarial author who renames `self.X = self.Y.weight` through a helper function evades the regex. The "≤ 12% prevalence" ceiling comes from this regex, not from a semantic alias check, and the paper should either widen the regex or weaken the ceiling claim.

## Questions
- Will you rewrite C5 in the Introduction to scope the "three knobs move verdicts" claim to the *stress benchmark*, removing the phrase "on the real corpora," so it agrees with the §4.2 statement that the real-corpus ablation is flat?
- Can you embed the 34-row `pytea_mcnemar_per_bug` contingency table directly in the appendix (or as a single table figure), and fix the empty `\href`/`\cite` targets at lines 972 and 982?
- Can you add a paragraph to §4.4 reporting the AST-extractor oracle-validation numbers (`extractor ⊆ oracle` 140/140 on `symbolic_config_attrs`, the 113+10+6+9 corpus breakdown, and the explicit scalar-attrs gap of 63/140 ⊆) and explain what the residual scalar-attr non-coverage implies for the soundness of the 128 CV verdicts?
- Can you report a Theorem 5 audit that does *not* rely on the forward-signature surrogate for any transformer block, and quote how many of the audited modules are non-surrogate, before keeping C4 in the contributions list?
- The grad-flag silent-error AST-grep covers five literal patterns; can you give a worst-case bound on what fraction of renamed-attribute aliasing escapes a literal-pattern regex (e.g. by comparing the regex hit count against a small dynamic per-parameter `.data_ptr()` aliasing check on the 8-model runtime harness)?
- On the post-freeze real-PR sample (5/15 vs FakeTensorMode 2/15 vs Pytea 3/15): can you state the Fisher exact p-values explicitly in the body alongside the counts, and report whether the directional gap survives a Benjamini–Hochberg correction across the three pairwise tests?

## Scores
Soundness: 3
Presentation: 2
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would push this to a 6 is removing the C5 contradiction in the Introduction *and* embedding the per-bug Pytea contingency table plus the AST-extractor oracle-validation numbers directly in the compiled PDF — those are three small surgical edits that would convert claims currently parked in `reproducibility/` into evidence a reviewer can inspect inside the paper. A second change that would push it further is a non-surrogate Theorem 5 audit on ≥ 10 transformer blocks; without that, C4 should be downgraded from "preliminary result" to "exploratory."

Round: 4


Changes   +0 -0
Requests  7.5 Premium (4m 58s)
Tokens    ↑ 2.0m • ↓ 15.4k • 1.9m (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 4, streak=0] The C5 contradiction flagged a round ago is still in the introduction. C5 (lines 112–117) attributes the "only three knobs move verdicts" claim to "the real corpora", but §4.2 (lines 1308–1311) says the real-corpus per-feature ablation is *flat*, with the three-knob result coming from a hand-built 25-case stress benchmark that was "constructed so that each feature would discriminate." A reader of the contributions list will form a false impression of what was demonstrated on real code; the correct fix is to localise C5 to the stress benchmark.
- [reviewer, w=1.00, added round 4, streak=0] The Pytea matched-pair claim still rests on a 34-row table that is not in the compiled PDF. The paper writes "the membership table is committed at ." (line 982) and "the 32/34 figure is reproduced at verification time ... by\n\nwhich AST-screens each repro" (lines 971–972) — both have empty `\href`/`\cite` targets in the rendered text. A reviewer cannot inspect the per-bug breakdown without leaving the paper, and the broken cross-references undermine the artifact in its own right.
- [reviewer, w=1.00, added round 4, streak=0] The AST extractor — explicitly identified as the component synthesising `assume_M` and as the load-bearing TCB component for *every* CV verdict — is now actually validated against an independent simple-AST oracle in `reproducibility/ast_extractor_oracle_validation.py` (140/140 ⊆-config agreement), but the paper does not surface this. The validation result, the corpora it covers (113-fixture + 10-real-public + 6 post-freeze + 9 unfiltered = 140 classes), and its scope (soundness direction only, scalar-attrs only 63/140 ⊆) belong in the body where the AST extractor is named in the TCB list (lines 22, 124, 1673, 1917, 2024). Without that, the audit exists but does not retire the prior reviewer's concern at the level of the paper.
- [reviewer, w=1.00, added round 4, streak=0] The headline real-source bug-finding evidence is weak. The 488-block result is 0 unconditional RP (now framed as a fragment measurement); the 5/15 post-freeze unfiltered comparison is "not statistically separable from either baseline at α=0.05" (line 18). What is left is a curated 60-bug corpus and a fragment-fair Pytea subset that, by construction, biases toward operators TensorGuard's catalogue covers. The claim that this checker provides a useful real-world bug-finding signal beyond curated examples is therefore not yet demonstrated; the paper is honest about this but the evidence base does not support a strong contribution claim under C3/C5.
- [reviewer, w=1.00, added round 4, streak=0] Theorem 5 (the Dynamo-guard inclusion lemma) is "preliminary, necessary-direction only" with empirical audit on 14 modules of which 4 transformer blocks use a "documented forward-signature surrogate because their full instantiation exceeds constraint solving at this scale" (lines 109–111). For a result that ties the contribution to a widely-deployed PyTorch component, "9 CNN blocks fully + 4 transformer surrogates" is a small base; the paper should either expand this audit (e.g. ≥ 30 modules, no surrogate on ≥ 10 transformer blocks) or downgrade the framing of C4 below "preliminary" to "exploratory."
- [reviewer, w=1.00, added round 4, streak=0] The grad-flag lattice's silent-error caveat (Section 6) is bounded by an AST-grep sweep returning 0/2,908 renamed-attribute hits, 1/42 silent-error-positive on training scripts, and 0/8 false-verified on a runtime trainer harness — this is good. But the held-out runtime sweep is on 8 well-known HF heads, and the 2,908-file sweep uses a fixed regex union; an adversarial author who renames `self.X = self.Y.weight` through a helper function evades the regex. The "≤ 12% prevalence" ceiling comes from this regex, not from a semantic alias check, and the paper should either widen the regex or weaken the ceiling claim.
- [reviewer, w=1.00, added round 4, streak=0] Will you rewrite C5 in the Introduction to scope the "three knobs move verdicts" claim to the *stress benchmark*, removing the phrase "on the real corpora," so it agrees with the §4.2 statement that the real-corpus ablation is flat?
- [reviewer, w=1.00, added round 4, streak=0] Can you embed the 34-row `pytea_mcnemar_per_bug` contingency table directly in the appendix (or as a single table figure), and fix the empty `\href`/`\cite` targets at lines 972 and 982?
- [reviewer, w=1.00, added round 4, streak=0] Can you add a paragraph to §4.4 reporting the AST-extractor oracle-validation numbers (`extractor ⊆ oracle` 140/140 on `symbolic_config_attrs`, the 113+10+6+9 corpus breakdown, and the explicit scalar-attrs gap of 63/140 ⊆) and explain what the residual scalar-attr non-coverage implies for the soundness of the 128 CV verdicts?
- [reviewer, w=1.00, added round 4, streak=0] Can you report a Theorem 5 audit that does *not* rely on the forward-signature surrogate for any transformer block, and quote how many of the audited modules are non-surrogate, before keeping C4 in the contributions list?

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
