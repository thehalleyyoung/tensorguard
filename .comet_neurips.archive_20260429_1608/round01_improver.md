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
TensorGuard is a static, no-execution refinement-type checker for PyTorch `nn.Module` forward methods that reasons about symbolic shapes and a flat first-order grad-flag lattice (`{has_grad, no_grad, ⊤}`), discharging side conditions to Z3. The system reports a five-way verdict taxonomy (`Verified / Refuted-Proof / Contract-Violation / Library-Warn / Abstain`); only RP and CV are covered by the soundness theorem. Empirical claims include 53/60 RP on a curated historical bug corpus, 32/34 vs. Pytea 22/34 on a fragment-fair modern subset (McNemar p=0.00195), 5/15 catches on an unfiltered pre-registered post-freeze N=15 PR sample (vs. FakeTensorMode 2/15 and Pytea 3/15, not statistically separable), 0 unconditional RP on the 488 real-source blocks under a free-symbolic regime, and a Lean 4 audit covering 28 of 79 shape-transfer handlers with 11/11 previously-axiomatic soundness lemmas closed sorry-free. A necessary-direction Dynamo-guard correspondence (Thm. 5) is stated and empirically audited on 17 modules.

## Prior weakness disposition
(none — first round)

## Strengths
- Calibration discipline is unusually high for an applied-PL paper: verdicts are partitioned into `RP / CV / LW / Abstain`, the soundness theorem (Thm. 2) is explicitly scoped to RP+CV, the 488-block corpus is honestly reported as 0 unconditional RP under the free-symbolic regime, and the Pytea modern-subset comparison enforces fragment fairness at verification time (`experiments_v5/v8/verify_modern_subset_enforced.py`).
- The Lean 4 audit (`lean/TensorGuard/`) is concrete and inspectable: 28 operator rules with 11 soundness lemmas closed sorry-free, plus a 28,000/28,000 byte-mirror cross-check against `torch 2.9.1`. The honest restatement of `permList_compose` to its in-range form (Sec. I) is the right move, not papered over.
- The pre-registered post-freeze N=15 PR sample, with the freeze hash recorded and a query frozen one day after the catalogue freeze, is a genuine generalisation test rather than a retro-fit, and the off-axis fire (`rb_uf_010`) is correctly counted as a false positive.
- The semantic-aliased view-bug residual (`rb_001`, `rb_002`) is diagnosed precisely (buggy/correct view targets agree on element count for the supplied shape) rather than waved away, and the constructor-bound integer-attribute envelope is reported as a residual, not closed by hand-waving.

## Weaknesses
- The unconditional-RP headline rests almost entirely on a curated 60-bug historical corpus (53/60). On the 488-block real-source corpus the user-visible regime returns **zero** unconditional RP verdicts; on the unfiltered post-freeze N=15 sample the catch rate is 5/15 with Wilson CI [15.2%, 58.3%] and Fisher p=0.39 vs. FakeTensorMode. The headline claim "catches real PyTorch shape bugs that execution-based tools cannot" is therefore not statistically separable from the baselines on the only unfiltered, pre-registered evaluation. Provide either a substantially larger pre-registered post-freeze sample (e.g. N≥60) so the head-to-head Fisher comparison clears α=0.05, or restate the contribution as a calibrated-coverage result rather than an empirical superiority claim.
- The title advertises "Sound Static Verification … with a 28/79-Handler Lean-Audited … Calculus", but only 28/79 ≈ 35% of handlers are Lean-audited, the analyser implementation, AST extractor, backward verifier, Z3 dispatch, and assume/guarantee composition (Thm. 3 is mechanised on a 3-operator DSL only) are *not* mechanised. The phrase "sound static verification" in §1 and the abstract is therefore overloaded relative to what the artefact actually certifies. Either reduce the soundness claim in the title/abstract to the rule-table layer, or extend the Lean audit to the remaining 51 handlers and the assume/guarantee rule on the full operator surface.
- The 488-block "0 unconditional RP / 57 Verified" headline depends critically on the synthesised caller-rely `assume_M`. The CV-witness audit cites 26/128 empty assumes, 90/128 reducing to documented config attributes, and 12/128 PreTrainedModel stubs, but does not show that any of the 90 documented-config assumes is *jointly* satisfied by a real published-checkpoint config (only that the symbols exist in some config). Show, for the full 128-CV set or a uniformly random subsample with a stated CI, that each `assume_M` is satisfied by at least one concrete published-checkpoint instantiation, with the resulting ratio reported.
- Theorem 5 is necessary-direction-only and the empirical audit reports an 8.8% in-contract recompile rate; the falsification predicate (`{SHAPE,DTYPE,RANK} guards outside catalogue`) is exercised on only 17 modules and 48 in-contract recompiles, all of which fall in the `INT` bucket the theorem already excludes. This is not strong evidence for the theorem — it is a "no falsifier observed" result on a small sample. Run the falsifier on a substantially larger module set (e.g. ≥100 timm/HF blocks) and report the resulting fraction; if zero out-of-catalogue SHAPE/DTYPE/RANK guards are seen, that bound is meaningful, otherwise the theorem statement needs to be tightened.
- The per-feature ablation on the real corpus is "a flat line": CEGAR contract discovery and the train/eval phase check are honestly reported as no-ops (`ShapeCEGARLoop` predicates never reach the verdict pipeline; phase encodes `Or(TRAIN, EVAL)` which is always satisfiable). The paper should either remove these from the contribution list (currently still implicitly part of C5) or wire them through to actually fire on at least one real bug.
- The first-order grad-flag lattice `{has_grad, no_grad, ⊤}` is silently incorrect on parameter-sharing-under-renamed-attribute; the prevalence is bounded at ≤12% of training scripts by a self-conducted GitHub sweep but no independent corroboration is given. Either run the backward verifier on a held-out set of HF training scripts containing this construct and report the false-verified rate, or restrict the grad-flow contribution (C3) to non-shared-parameter modules.
- The 33/33 within-±5-line localisation result is, by the paper's own admission, a "consistency check, not a precision claim" because the AST-walk strategy and the heuristic ground truth share information; only 3/3 within-1 on the marker-only items is independent. The marker-only audit needs to be at least the ≥30-item size the paper itself names before any localisation result enters the contributions.
- Presentation: the paper is *exceptionally* dense with rebuttal-style apparatus (round-2 Q4, round-3 Q6, round-5 Q3, round-7 W5, etc.) embedded in the body. The abstract alone is ~22 sentences and mixes the headline with five caveats; Section 4.1 reads as a stack of patches rather than a result section. Restructure §4 so the five "round-N" responses are condensed into a single calibration paragraph; move the per-round disposition to an appendix.

## Questions
- For the N=15 unfiltered post-freeze sample, what is the planned next-round denominator, and will the pre-registration query be the same? On a doubled sample, does the TG-vs-FakeTensorMode Fisher gap clear α=0.05?
- Of the 90/128 "documented-config" CV verdicts, how many are simultaneously satisfied by *the same* published checkpoint config (i.e. the conjunction is realisable in one instantiation), as opposed to each clause being satisfied by some config?
- On the N=10 upstream-faithful re-extracts, what fraction of the verdict path on `rb_003`, `rb_004`, `rb_006`, `rb_010` flows through the round-6 envelope synthesiser additions vs. the legacy v4 path? An ablation that disables only the new constructor-bound integer-attribute envelope on these 4 cases would isolate the round-6 contribution.
- For Theorem 5: in the 17-module audit, were any modules excluded for catalogue mismatch before the recompile counts were taken? If so, what is the recompile rate on the un-pruned set?
- For the 25-block hybrid stress set (Table 4), what is the construction protocol — were the 20 TG-only catches and 5 FT-only catches selected to maximise complementarity? A pre-registration or random-sample protocol would strengthen the "complementary, not coincident" claim.
- The Lean operator-rule table covers 28 of 79 handlers. Which 51 are tested-only, and what is the soundness risk profile (e.g., are the un-audited handlers concentrated in low-frequency operators, or do they include load-bearing rules like SDPA/LayerNorm)?

## Scores
Soundness: 3
Presentation: 2
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would move my overall from 5 to 6 is a larger pre-registered post-freeze sample (N≥60) on which the TG-vs-FakeTensorMode Fisher comparison clears α=0.05 — i.e., turning the calibrated-confidence result of Table 3 into a statistically separated head-to-head on an unfiltered surface. Alternatively, extending the Lean audit to the full 79-handler surface (or to the assume/guarantee rule on more than the 3-operator DSL) would justify the "Sound Static Verification" framing of the title and similarly raise the score.


Changes   +0 -0
Requests  7.5 Premium (1m 53s)
Tokens    ↑ 542.0k • ↓ 5.3k • 474.1k (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 1] The unconditional-RP headline rests almost entirely on a curated 60-bug historical corpus (53/60). On the 488-block real-source corpus the user-visible regime returns **zero** unconditional RP verdicts; on the unfiltered post-freeze N=15 sample the catch rate is 5/15 with Wilson CI [15.2%, 58.3%] and Fisher p=0.39 vs. FakeTensorMode. The headline claim "catches real PyTorch shape bugs that execution-based tools cannot" is therefore not statistically separable from the baselines on the only unfiltered, pre-registered evaluation. Provide either a substantially larger pre-registered post-freeze sample (e.g. N≥60) so the head-to-head Fisher comparison clears α=0.05, or restate the contribution as a calibrated-coverage result rather than an empirical superiority claim.
- [reviewer, w=1.00, added round 1] The title advertises "Sound Static Verification … with a 28/79-Handler Lean-Audited … Calculus", but only 28/79 ≈ 35% of handlers are Lean-audited, the analyser implementation, AST extractor, backward verifier, Z3 dispatch, and assume/guarantee composition (Thm. 3 is mechanised on a 3-operator DSL only) are *not* mechanised. The phrase "sound static verification" in §1 and the abstract is therefore overloaded relative to what the artefact actually certifies. Either reduce the soundness claim in the title/abstract to the rule-table layer, or extend the Lean audit to the remaining 51 handlers and the assume/guarantee rule on the full operator surface.
- [reviewer, w=1.00, added round 1] The 488-block "0 unconditional RP / 57 Verified" headline depends critically on the synthesised caller-rely `assume_M`. The CV-witness audit cites 26/128 empty assumes, 90/128 reducing to documented config attributes, and 12/128 PreTrainedModel stubs, but does not show that any of the 90 documented-config assumes is *jointly* satisfied by a real published-checkpoint config (only that the symbols exist in some config). Show, for the full 128-CV set or a uniformly random subsample with a stated CI, that each `assume_M` is satisfied by at least one concrete published-checkpoint instantiation, with the resulting ratio reported.
- [reviewer, w=1.00, added round 1] Theorem 5 is necessary-direction-only and the empirical audit reports an 8.8% in-contract recompile rate; the falsification predicate (`{SHAPE,DTYPE,RANK} guards outside catalogue`) is exercised on only 17 modules and 48 in-contract recompiles, all of which fall in the `INT` bucket the theorem already excludes. This is not strong evidence for the theorem — it is a "no falsifier observed" result on a small sample. Run the falsifier on a substantially larger module set (e.g. ≥100 timm/HF blocks) and report the resulting fraction; if zero out-of-catalogue SHAPE/DTYPE/RANK guards are seen, that bound is meaningful, otherwise the theorem statement needs to be tightened.
- [reviewer, w=1.00, added round 1] The per-feature ablation on the real corpus is "a flat line": CEGAR contract discovery and the train/eval phase check are honestly reported as no-ops (`ShapeCEGARLoop` predicates never reach the verdict pipeline; phase encodes `Or(TRAIN, EVAL)` which is always satisfiable). The paper should either remove these from the contribution list (currently still implicitly part of C5) or wire them through to actually fire on at least one real bug.
- [reviewer, w=1.00, added round 1] The first-order grad-flag lattice `{has_grad, no_grad, ⊤}` is silently incorrect on parameter-sharing-under-renamed-attribute; the prevalence is bounded at ≤12% of training scripts by a self-conducted GitHub sweep but no independent corroboration is given. Either run the backward verifier on a held-out set of HF training scripts containing this construct and report the false-verified rate, or restrict the grad-flow contribution (C3) to non-shared-parameter modules.
- [reviewer, w=1.00, added round 1] The 33/33 within-±5-line localisation result is, by the paper's own admission, a "consistency check, not a precision claim" because the AST-walk strategy and the heuristic ground truth share information; only 3/3 within-1 on the marker-only items is independent. The marker-only audit needs to be at least the ≥30-item size the paper itself names before any localisation result enters the contributions.
- [reviewer, w=1.00, added round 1] Presentation: the paper is *exceptionally* dense with rebuttal-style apparatus (round-2 Q4, round-3 Q6, round-5 Q3, round-7 W5, etc.) embedded in the body. The abstract alone is ~22 sentences and mixes the headline with five caveats; Section 4.1 reads as a stack of patches rather than a result section. Restructure §4 so the five "round-N" responses are condensed into a single calibration paragraph; move the per-round disposition to an appendix.
- [reviewer, w=1.00, added round 1] For the N=15 unfiltered post-freeze sample, what is the planned next-round denominator, and will the pre-registration query be the same? On a doubled sample, does the TG-vs-FakeTensorMode Fisher gap clear α=0.05?
- [reviewer, w=1.00, added round 1] Of the 90/128 "documented-config" CV verdicts, how many are simultaneously satisfied by *the same* published checkpoint config (i.e. the conjunction is realisable in one instantiation), as opposed to each clause being satisfied by some config?

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


## Self-check before declaring the round done

Before you stop, run a self-audit against the HARD CONSTRAINTS at
the top:

  * `pdftotext neurips.pdf - | grep -nE '\.(py|lean|json|tex|sh|md|csv|yaml)\b'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'honest|honestly|honesty'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'reviewer|rebuttal|we tried|in response to|prior reviewers|round-?[0-9]+ reviewer'` --- must be empty.
  * `pdftotext neurips.pdf - | grep -niE 'NA answer|will not be perceived|specifically instructed to not penalize|while the authors might fear'` --- must be empty (NeurIPS template text not filled in).
  * Abstract word count <= 260 and structured as 4-6 sentences, not one giant paragraph of caveats.

If any of these fail, fix them and rebuild before you stop. A
one-time cleanup pass already cleared the paper of these
violations before round 1; do not re-introduce them.

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
