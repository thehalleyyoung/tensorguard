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

The paper presents \textsc{TensorGuard}, a no-execution refinement-type checker for PyTorch \texttt{nn.Module} classes. Tensor types $\mathsf{Tensor}\{s,g\mid\varphi\}$ carry a symbolic shape, a flat grad-flag lattice, and a Z3-decidable side condition; assume/guarantee contracts compose at the class boundary; a backward verifier flags three silent-zero-grad bug classes; and a Dynamo-guard "necessary direction" inclusion lemma is claimed (Thm. 5). Empirically the authors report a 488-block real-source corpus on which TG returns 57 Verified / 0 unconditional Refuted-Proof / 128 Contract-Violation / 78 Library-Warn / 225 Abstain (vs. ≥435/488 N/A for the four PyTorch baselines and Pytea), a 60-bug historical corpus (53/60 RP), 10 upstream-faithful re-extracts (7/10 RP@0.99 + 1/10 @0.80), a held-out unfiltered N=15 post-freeze sample (5/15 catches vs. FakeTensorMode 2/15, Pytea 3/15), and a Lean 4 audit that closes 11/11 previously-axiomatic soundness lemmas sorry-free for 28 of 79 shape-transfer rules and agrees with torch 2.9.1 on 28 000/28 000 in-fragment cases.

## Strengths

- The reporting is unusually calibrated for this style of paper: the headline triple is split into RP / CV / LW; only RP and CV are claimed to be covered by Thm. 1; LW and Abstain make no soundness claim; and the user-visible (free-symbolic-config) recomputation 34V/0RP/206LW/248A is reported alongside the assume-envelope number.
- The Lean artefact is real and non-trivial: \texttt{lean/TensorGuard/} builds sorry-free under \texttt{lake build} (verified: 0 \texttt{sorry} occurrences in proofs across \texttt{Soundness.lean}, \texttt{AssumeGuarantee.lean}, \texttt{V5OperatorRules.lean}; the textual \texttt{sorry} hits in those files are inside comments). The 28 000-sample byte-mirror differential against torch 2.9.1 plus the off-envelope boundary check on ~2 400 mutators is the right shape of evidence.
- The post-freeze protocol (catalogue freeze \texttt{040f6f3} on 2026-04-07, pre-registered query 2026-04-08) and the explicit retention of out-of-fragment cases as honest abstentions in the N=15 unfiltered sample is a methodologically clean way to bound the fragment-fit confound.
- The handler-soundness scope table (Tab. \ref{tab:handler-soundness}) and the per-block scope accounting (\texttt{reproducibility/handler\_scope\_per\_block.md}) explicitly partition verdicts by which audit class their handlers fall into — a level of bookkeeping most refinement-type tool papers skip.

## Weaknesses

- **The paper's central headline is structurally fragile.** On the 488-block real-source corpus the verifier returns 0 unconditional Refuted-Proof verdicts, and the user-visible (no synthesised \texttt{config} envelope) recomputation collapses Verified from 57 to 34 and CV+RP to 0/0. The "find real bugs in real source" claim is therefore carried entirely by the 60 historical bugs (selected for shape-error keywords), the 10 upstream-faithful re-extracts, and the N=15 post-freeze sample. The 488-block evaluation is, on its own terms, a coverage-and-abstention measurement rather than a bug-finding result, and the abstract should not be allowed to read as if the 488 number is doing bug-finding work.
- **The "TG ≫ baselines" reading on the 488-block corpus is corpus-baked.** ≥435/488 N/A for FakeTensorMode/torch.fx/torch.export/torch.export/Pytea is a structural fact about constructor-argument requirements on a corpus that was selected to consist of \texttt{nn.Module} classes that need a non-trivial \texttt{config}. The paper acknowledges this once (line 200–204 of \texttt{eval\_v6.tex}) and then continues to use the 488-block triple as the headline framing in the abstract and conclusion. Either the 488 number is fairness-compatible with the N/A baselines (it isn't) or it should be demoted to "applicability" rather than presented next to the bug-finding numbers.
- **The fair head-to-head N=34 vs. Pytea is a constructed subset.** The "modern subset" is defined by the static AST predicate "every operator called from \texttt{forward} appears in Pytea's 2022 \texttt{pylib/}". Since TG is restricted symmetrically (Sec. \ref{sec:eval-benchmark}, "Pytea modern-subset filter" paragraph), the comparison is internally consistent, but the selection still excludes precisely the bugs Pytea cannot reach by construction. Please report (i) the size of the symmetric handler intersection vs. each tool's full catalogue, and (ii) the corresponding TG-only / Pytea-only refute counts on the *complement* (the 26 bugs that fall out of the modern subset) so the reader can judge whether the 32/34 vs 22/34 gap is a fragment-coverage artefact.
- **The post-freeze N=15 result is presented as the deciding generalisation evidence but the head-to-head is not statistically separable.** TG 5/15 vs. FakeTensorMode 2/15 (Fisher-exact two-sided p=0.39) and vs. Pytea 3/15 (p=0.68). The paper concedes this once and then in the abstract still asserts TG is "point-strictly above" the baselines; on N=15 a difference of 2–3 catches is well inside binomial noise and should not be in the abstract as a separation result. Please either grow the unfiltered sample to a size where Wilson intervals on the rate difference exclude zero, or remove the comparative claim from the abstract.
- **Thm. 5 (Dynamo-guard correspondence, necessary direction) is close to vacuous as stated.** "Every shape/dtype/rank refinement bit Dynamo reads on the trace at any input $x$ with $A(x)$ true is a refinement variable in $\varphi$ for some rule in the catalogue" is *true by construction* whenever the catalogue is the set of metadata bits TG inspects, because TG's catalogue is also defined off Dynamo's specialiser bits. The empirical falsifier (Sec. \ref{sec:eval-dynamo}, "Falsification predicate" paragraph) reports {SHAPE:0, DTYPE:0, RANK:0, INT:48} — i.e., the 48 in-contract recompiles all fall in the bucket the theorem already excludes. A non-vacuous test would be: a catalogue-extension experiment where you intentionally widen TG's refinement language to include one Dynamo guard kind not currently in the catalogue (e.g., one of the integer or list-length specialisers), measure whether a recompile in that kind has ever fallen on an in-contract input, and report whether the theorem still holds at the wider catalogue. The current audit cannot distinguish "Dynamo guards are a superset of TG refinements" (a tautology) from "Dynamo's shape/dtype/rank guards are exactly TG's refinements" (the substantive claim).
- **The Lean audit covers the rule table, not the analyser.** This is stated honestly in §\ref{sec:eval-lean}, but the consequence for the reader is sharper than the paper admits: only 28/79 shape handlers are Lean-audited, the AST extractor and Python analyser are not, and the assume/guarantee composition theorem (Thm. 3) is mechanised on a 3-operator DSL only. Per the per-block scope accounting (line 845–852 of \texttt{eval\_v6.tex}), only 36/185 in-soundness verdicts on the 488-block corpus touch only Lean-audited or pen-and-paper handlers — meaning the soundness theorem applies tightly to roughly one-fifth of the headline verdicts. The "sound static verification" framing in the title is therefore stronger than the underlying mechanisation.
- **Two of the per-feature ablation knobs are conceded to be no-ops in the implementation.** Sec. \ref{sec:eval-loc-hybrid} reports CEGAR contract discovery and the train/eval phase check as "honest no-ops" on real data; CEGAR predicates are stored as metadata but never surfaced as \texttt{Bug} objects. These are listed alongside the active knobs in Tab. \ref{tab:ablation} and in the contributions list, even after the abstract concedes the empirical contribution is now scoped to three features (device, grad, low-conf gating). Please remove the inactive features from the implementation/contribution claims rather than carrying them as engineering "ships".
- **The leave-one-category-out result (53/60 → 53/60) is a non-result that the paper presents as robustness.** When LOO leaves the rate unchanged across every category drop, the most parsimonious explanation is that the LOO is hitting orchestration code rather than the load-bearing rules. The paper acknowledges this and pivots to a "true rule-class LOO" that *also* yields 53/60 → 53/60, attributing the constancy to an independent AST-pattern path. If catches survive simultaneous removal of (a) the per-category operator handlers and (b) the AST-pattern intent-bug analyser, then a substantial fraction of the 53/60 is attributable to a constraint-based shape backend that is not what the paper's calculus and Lean audit are about. Please report the catch count when *all three* paths (per-operator handlers, intent-bug AST patterns, constraint-based backend) are disabled, and the catch count of the constraint backend alone, so the reader can attribute the 53/60 to a specific component.
- **Several rb_* "catches" were engineered into the system inside the development window.** Lines 172–192 of \texttt{eval\_v6.tex} note that the rb_003/rb_004/rb_006/rb_010 conversions from silent-verified to RP@0.99 came from a "three-stage constructor-bound integer-attribute envelope synthesiser" and a "per-forward local-scalar map" added within the round-4 to round-6 window. The pre-registered post-freeze split (N=6 + N=15) is the only segment of the real-bug evaluation that is uncontaminated by handler additions designed against specific bugs; on that split the catch rate is 3/6 + 5/15. The headline 7/10 should be reported with a clear footnote that ≥4 of the 7 catches were enabled by handler edits made after the bugs were inspected.

## Questions

- On the 488-block corpus, what is the user-visible (no-assume) catch rate on a *bug-injected* subset — i.e., if you mutate one shape-arithmetic axis in each of, say, 100 of the 488 blocks, how many does TG flag as RP (not CV) without any synthesised \texttt{config} envelope? This isolates the calculus's bug-finding power from the catalogue-coverage and assume-synthesis layers.
- For the N=15 unfiltered post-freeze sample, please report the per-PR overlap matrix (which PRs each tool catches), not just the marginal counts. With N=15 and 5/2/3 catches the joint distribution is the only way to tell whether TG's catches *contain* the FakeTensor/Pytea catches or are disjoint from them.
- Can you provide a falsifier for Thm. 5 that does not reduce to the catalogue's own definition? E.g., an audit where TG's refinement language is *extended* by one dtype-promotion guard kind currently outside the catalogue, and the in-contract recompile bucket for that kind is reported on the same 17-module audit.
- The backward verifier's $500/500$ static↔runtime agreement and 8/8 canonical-bug catch are reported on synthetic small modules and 8 hand-curated cases respectively; the 10-real-model sweep explicitly excludes \texttt{torch.utils.checkpoint} and parameter sharing. What is the agreement rate on a sample of the (≤12% prevalence) training scripts that *do* use these constructs?
- On the joint LOO that disables per-category handlers and the AST-pattern intent-bug analyser simultaneously, you still report 53/60. What is the catch count when you additionally disable the constraint-based shape back-end (the "third path")? If it is 53/60 still, what catches the bugs?
- For the modern-subset Pytea comparison, what is the size of the *complement* (60-block bugs that fall outside the symmetric handler intersection) and the per-tool catch counts on it?

## Scores

Soundness: 2
Presentation: 3
Contribution: 2
Confidence: 4
Overall: 4

## Borderline reasons

The single change that would push the overall score from 4 to 5 is to grow the pre-registered, post-freeze unfiltered real-PR sample from N=15 to N≥40 with the same protocol, so that the head-to-head Fisher-exact comparison against \texttt{FakeTensorMode} and Pytea can either separate at α=0.05 or be honestly reported as null. As it stands, the unfiltered post-freeze sample is the only bug-finding evaluation segment that is uncontaminated by either corpus-baked baseline-inapplicability (488 blocks) or in-development handler edits (rb_001–rb_010), and at N=15 it cannot carry the comparative-bug-finding headline the abstract leans on.


Changes   +0 -0
Requests  7.5 Premium (5m 40s)
Tokens    ↑ 1.8m • ↓ 10.3k • 1.7m (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 1] **The paper's central headline is structurally fragile.** On the 488-block real-source corpus the verifier returns 0 unconditional Refuted-Proof verdicts, and the user-visible (no synthesised \texttt{config} envelope) recomputation collapses Verified from 57 to 34 and CV+RP to 0/0. The "find real bugs in real source" claim is therefore carried entirely by the 60 historical bugs (selected for shape-error keywords), the 10 upstream-faithful re-extracts, and the N=15 post-freeze sample. The 488-block evaluation is, on its own terms, a coverage-and-abstention measurement rather than a bug-finding result, and the abstract should not be allowed to read as if the 488 number is doing bug-finding work.
- [reviewer, w=1.00, added round 1] **The "TG ≫ baselines" reading on the 488-block corpus is corpus-baked.** ≥435/488 N/A for FakeTensorMode/torch.fx/torch.export/torch.export/Pytea is a structural fact about constructor-argument requirements on a corpus that was selected to consist of \texttt{nn.Module} classes that need a non-trivial \texttt{config}. The paper acknowledges this once (line 200–204 of \texttt{eval\_v6.tex}) and then continues to use the 488-block triple as the headline framing in the abstract and conclusion. Either the 488 number is fairness-compatible with the N/A baselines (it isn't) or it should be demoted to "applicability" rather than presented next to the bug-finding numbers.
- [reviewer, w=1.00, added round 1] **The fair head-to-head N=34 vs. Pytea is a constructed subset.** The "modern subset" is defined by the static AST predicate "every operator called from \texttt{forward} appears in Pytea's 2022 \texttt{pylib/}". Since TG is restricted symmetrically (Sec. \ref{sec:eval-benchmark}, "Pytea modern-subset filter" paragraph), the comparison is internally consistent, but the selection still excludes precisely the bugs Pytea cannot reach by construction. Please report (i) the size of the symmetric handler intersection vs. each tool's full catalogue, and (ii) the corresponding TG-only / Pytea-only refute counts on the *complement* (the 26 bugs that fall out of the modern subset) so the reader can judge whether the 32/34 vs 22/34 gap is a fragment-coverage artefact.
- [reviewer, w=1.00, added round 1] **The post-freeze N=15 result is presented as the deciding generalisation evidence but the head-to-head is not statistically separable.** TG 5/15 vs. FakeTensorMode 2/15 (Fisher-exact two-sided p=0.39) and vs. Pytea 3/15 (p=0.68). The paper concedes this once and then in the abstract still asserts TG is "point-strictly above" the baselines; on N=15 a difference of 2–3 catches is well inside binomial noise and should not be in the abstract as a separation result. Please either grow the unfiltered sample to a size where Wilson intervals on the rate difference exclude zero, or remove the comparative claim from the abstract.
- [reviewer, w=1.00, added round 1] **Thm. 5 (Dynamo-guard correspondence, necessary direction) is close to vacuous as stated.** "Every shape/dtype/rank refinement bit Dynamo reads on the trace at any input $x$ with $A(x)$ true is a refinement variable in $\varphi$ for some rule in the catalogue" is *true by construction* whenever the catalogue is the set of metadata bits TG inspects, because TG's catalogue is also defined off Dynamo's specialiser bits. The empirical falsifier (Sec. \ref{sec:eval-dynamo}, "Falsification predicate" paragraph) reports {SHAPE:0, DTYPE:0, RANK:0, INT:48} — i.e., the 48 in-contract recompiles all fall in the bucket the theorem already excludes. A non-vacuous test would be: a catalogue-extension experiment where you intentionally widen TG's refinement language to include one Dynamo guard kind not currently in the catalogue (e.g., one of the integer or list-length specialisers), measure whether a recompile in that kind has ever fallen on an in-contract input, and report whether the theorem still holds at the wider catalogue. The current audit cannot distinguish "Dynamo guards are a superset of TG refinements" (a tautology) from "Dynamo's shape/dtype/rank guards are exactly TG's refinements" (the substantive claim).
- [reviewer, w=1.00, added round 1] **The Lean audit covers the rule table, not the analyser.** This is stated honestly in §\ref{sec:eval-lean}, but the consequence for the reader is sharper than the paper admits: only 28/79 shape handlers are Lean-audited, the AST extractor and Python analyser are not, and the assume/guarantee composition theorem (Thm. 3) is mechanised on a 3-operator DSL only. Per the per-block scope accounting (line 845–852 of \texttt{eval\_v6.tex}), only 36/185 in-soundness verdicts on the 488-block corpus touch only Lean-audited or pen-and-paper handlers — meaning the soundness theorem applies tightly to roughly one-fifth of the headline verdicts. The "sound static verification" framing in the title is therefore stronger than the underlying mechanisation.
- [reviewer, w=1.00, added round 1] **Two of the per-feature ablation knobs are conceded to be no-ops in the implementation.** Sec. \ref{sec:eval-loc-hybrid} reports CEGAR contract discovery and the train/eval phase check as "honest no-ops" on real data; CEGAR predicates are stored as metadata but never surfaced as \texttt{Bug} objects. These are listed alongside the active knobs in Tab. \ref{tab:ablation} and in the contributions list, even after the abstract concedes the empirical contribution is now scoped to three features (device, grad, low-conf gating). Please remove the inactive features from the implementation/contribution claims rather than carrying them as engineering "ships".
- [reviewer, w=1.00, added round 1] **The leave-one-category-out result (53/60 → 53/60) is a non-result that the paper presents as robustness.** When LOO leaves the rate unchanged across every category drop, the most parsimonious explanation is that the LOO is hitting orchestration code rather than the load-bearing rules. The paper acknowledges this and pivots to a "true rule-class LOO" that *also* yields 53/60 → 53/60, attributing the constancy to an independent AST-pattern path. If catches survive simultaneous removal of (a) the per-category operator handlers and (b) the AST-pattern intent-bug analyser, then a substantial fraction of the 53/60 is attributable to a constraint-based shape backend that is not what the paper's calculus and Lean audit are about. Please report the catch count when *all three* paths (per-operator handlers, intent-bug AST patterns, constraint-based backend) are disabled, and the catch count of the constraint backend alone, so the reader can attribute the 53/60 to a specific component.
- [reviewer, w=1.00, added round 1] **Several rb_* "catches" were engineered into the system inside the development window.** Lines 172–192 of \texttt{eval\_v6.tex} note that the rb_003/rb_004/rb_006/rb_010 conversions from silent-verified to RP@0.99 came from a "three-stage constructor-bound integer-attribute envelope synthesiser" and a "per-forward local-scalar map" added within the round-4 to round-6 window. The pre-registered post-freeze split (N=6 + N=15) is the only segment of the real-bug evaluation that is uncontaminated by handler additions designed against specific bugs; on that split the catch rate is 3/6 + 5/15. The headline 7/10 should be reported with a clear footnote that ≥4 of the 7 catches were enabled by handler edits made after the bugs were inspected.
- [reviewer, w=1.00, added round 1] On the 488-block corpus, what is the user-visible (no-assume) catch rate on a *bug-injected* subset — i.e., if you mutate one shape-arithmetic axis in each of, say, 100 of the 488 blocks, how many does TG flag as RP (not CV) without any synthesised \texttt{config} envelope? This isolates the calculus's bug-finding power from the catalogue-coverage and assume-synthesis layers.
- [reviewer, w=1.00, added round 1] For the N=15 unfiltered post-freeze sample, please report the per-PR overlap matrix (which PRs each tool catches), not just the marginal counts. With N=15 and 5/2/3 catches the joint distribution is the only way to tell whether TG's catches *contain* the FakeTensor/Pytea catches or are disjoint from them.
- [reviewer, w=1.00, added round 1] Can you provide a falsifier for Thm. 5 that does not reduce to the catalogue's own definition? E.g., an audit where TG's refinement language is *extended* by one dtype-promotion guard kind currently outside the catalogue, and the in-contract recompile bucket for that kind is reported on the same 17-module audit.
- [reviewer, w=1.00, added round 1] The backward verifier's $500/500$ static↔runtime agreement and 8/8 canonical-bug catch are reported on synthetic small modules and 8 hand-curated cases respectively; the 10-real-model sweep explicitly excludes \texttt{torch.utils.checkpoint} and parameter sharing. What is the agreement rate on a sample of the (≤12% prevalence) training scripts that *do* use these constructs?
- [reviewer, w=1.00, added round 1] On the joint LOO that disables per-category handlers and the AST-pattern intent-bug analyser simultaneously, you still report 53/60. What is the catch count when you additionally disable the constraint-based shape back-end (the "third path")? If it is 53/60 still, what catches the bugs?
- [reviewer, w=1.00, added round 1] For the modern-subset Pytea comparison, what is the size of the *complement* (60-block bugs that fall outside the symmetric handler intersection) and the per-tool catch counts on it?
- [reviewer, w=0.71, added round 4] **The unconditional-RP claim on real library code is zero.** Section 4.1 / Table 1 show 0 RP on the 488-block corpus; all 128 CV verdicts are sound only under a *synthesised* caller-rely `assume_M`. The 84.6% LW rate is principled abstention, but "0 RP on 488 blocks" is the user-visible truth for anyone who does not buy the synthesised contract envelope, and 90/128 of the CV witnesses depend on `transformers` `*Config` defaults rather than independently-checkable preconditions (`reproducibility/cv_caller_rely_witnesses.{json,md}`). The paper carries the unconditional-RP claim entirely on hand-curated bug corpora.
- [reviewer, w=0.71, added round 4] **The N=15 post-freeze unfiltered evaluation does not separate from baselines.** Table 3 reports TG 5/15 vs FakeTensor 2/15 vs Pytea 3/15 with two-sided Fisher p=0.39 and 0.68. The paper concedes this is not statistically separable at α=0.05, yet it is the only evaluation that actually controls for retro-fitting, and it is framed as a calibrated-confidence positive result. With p=0.39 vs FakeTensor and a 6/15 RP-fire rate (one of which is admitted to be a false positive against ground truth), the post-freeze evidence cannot bear the weight the abstract puts on it.
- [reviewer, w=0.71, added round 4] **The handler-class LOO is a no-op (53/60→53/60) by the paper's own admission**, because catches are duplicated through an "AST-pattern intent-bug analyser" and a constraint-based shape back-end that operate independently of the operator dispatch (Section 4.1 round-3 Q6 paragraph; `reproducibility/bug_corpus_loo_joint.{json,md}`). This means the bug-corpus number measures the union of three pattern-matchers, not the soundness of the typing rules; the keyword-attribution numbers (7/7/6/...) rest on a manual mapping of bugs to handler categories, which is exactly the kind of post-hoc accounting the LOO was supposed to rule out.
- [reviewer, w=0.71, added round 4] **Theorem 3 (compositional soundness) is mechanised on the same 3-operator DSL (matmul/view/add) as Theorem 1, not on the 79-handler operator surface** (`lean/TensorGuard/AssumeGuarantee.lean`). The paper states this in §4.4, but the contribution C2 ("assume/guarantee discipline at the nn.Module class boundary") is then only mechanised on a toy DSL. The gap between Lean-mechanised composition and the Python composition that ships in `src/composition_soundness.py` is a TCB obligation indistinguishable from "we asserted it".
- [reviewer, w=0.71, added round 4] **The 500/500 backward-pass agreement and 8/8 canonical-bug catch are weak evidence**. The 500 modules are random small `nn.Module`s drawn from a grammar (Section 3.2), and the 8 canonical bugs are hand-curated. Together with 0/50 false positives on a clean grammar-sampled set, this is in-distribution agreement, not evidence on the parameter-sharing-under-renamed-attribute case the paper itself flags as silently-incorrect at ≤12% prevalence (Section 6).
- [reviewer, w=0.71, added round 4] **Localisation 33/33 is admittedly a consistency check, not a precision claim** (Section 4.2 caveat: heuristic ground truth and AST-walk strategy share information; only 3/3 marker-only items are independently scored). The improvement from v4's 9% to "33/33" is therefore not interpretable, and an N=3 marker-only audit cannot support the claimed gain.
- [reviewer, w=0.71, added round 4] **The Dynamo-falsification audit lands {SHAPE:0, DTYPE:0, RANK:0, INT:48}**, i.e. all 48 in-contract recompiles fall into the bucket Theorem 5 *already excludes* (Section 4.3). The paper notes this is "consistent with Theorem 5 without proving it"; in practice the audit can never falsify the theorem because the theorem's claim has been narrowed to exclude exactly the metadata Dynamo specialises on. This is a near-vacuous empirical instantiation.
- [reviewer, w=0.71, added round 4] **Per-feature stress benchmark is hand-engineered to discriminate** (Table 5), and on the 488-block + 60-bug *real* aggregate the ablation is admittedly flat across all five knobs. The paper itself scopes the empirical contribution to 3 features (device-consistency, gradient-flow, low-confidence gating) rather than 5, and labels CEGAR contract discovery and the train/eval phase check as "engineering no-ops". This significantly deflates the implementation contribution.
- [reviewer, w=0.71, added round 4] For the 128 CV verdicts: how many of the synthesised `assume_M` predicates are entailed by the *actual* preconditions of an instantiated `*PreTrainedModel.forward` call as exercised in a published example script (not a config-default existence check)? The 12 randomly-sampled witnesses in `cv_caller_rely_witnesses.md` are evidence of nonemptiness, not of typical-caller validity.
- [reviewer, w=0.71, added round 4] On the rb_uf_010 off-axis fire (device-mismatch where the upstream PR fixes a dtype bug): how many of the other 5 RP fires in the N=15 post-freeze sample have been independently audited for "right answer for the right reason" rather than ground-truth axis match alone?
- [reviewer, w=0.71, added round 4] The Lean precondition-boundary check is on ~2,400 off-envelope samples across only 10 of 28 audited rules. Can the authors run the boundary check on all 28 rules, and report whether any rule has a precondition strictly narrower than torch's permissive behaviour? The "10-rule" partial coverage leaves 18/28 boundaries unchecked.
- [reviewer, w=0.71, added round 4] For Theorem 5's necessary-direction empirical instantiation: on the five end-to-end TG-verified blocks (BasicBlock, Bottleneck, InvertedResidual, Fire, Block), what is the breakdown of post-compile recompiles by guard kind? "0–2" recompiles "all attributable to standard dynamic-shape graph specialisation" is an informal claim and should be a `r.guard_kind ∈ {SHAPE,DTYPE,RANK}` count.
- [reviewer, w=0.71, added round 4] The "AST-pattern intent-bug analyser" plus "constraint-based shape back-end" duplicate operator-handler catches (joint LOO = 53/60). Could the authors report the 60-bug RP count *without* the operator handlers and *without* the AST-pattern analyser — i.e., relying only on the typing rules whose soundness is in Theorem 2? That is the number that actually corresponds to the soundness theorem.
- [reviewer, w=0.71, added round 4] The first-order grad lattice's ≤12%-of-training-scripts caveat: was this prevalence measured by AST-grep over `from torch.utils.checkpoint import` and tied-weight keys? If so, "renamed-attribute parameter sharing" is not detectable by such a filter and the 12% is an underestimate; please clarify the measurement protocol.
- [reviewer, w=0.50, added round 3] **The headline thesis is still not supported by the headline corpus.** On the 488-block corpus TG produces **0 unconditional RP** in both regimes (`experiments_v5/v8/user_visible_rp.json`). The 128 CVs are sound only under a TG-synthesised `assume_M`; the abstract still leads with "206 refutations" before disclosing the decomposition. The unconditional-RP story rests on a 60-bug curated catalogue and a 10-bug + N=15 post-freeze sample — small numbers for a paper whose thesis is dominance over execution-based tools on real code.
- [reviewer, w=0.50, added round 3] **The Dynamo-correspondence audit is materially unchanged from rounds 1 and 2 and remains near-vacuous as an empirical instantiation of Theorem 5.** `experiments_v5/dynamo_correspondence_v5.json` still contains `signature-trusted` on **16 of 17** modules. The 8.8% in-contract recompile and 97.9% OOS-violation rates therefore measure how well a documented forward signature predicts Dynamo guards, not how well TG's analyser does. Round 2 asked for an end-to-end subset; the repo shows no follow-up commit.
- [reviewer, w=0.50, added round 3] **The mechanised soundness footprint is much smaller than the "Lean-audited" framing.** Theorems 1 and 3 are mechanised on a 3-operator DSL (matmul/view/add); the rule table covers 28 of 79 handlers in Lean, 3 pen-and-paper, and **48 tested-only**. The paper itself states the parser, AST extractor, analyser implementation, backward verifier, and Z3 dispatch are not mechanised. Theorem 2's soundness therefore does not transfer to any forward path that touches a tested-only handler, which is most of the 488-block corpus. The paper does not report what fraction of V/CV verdicts on the 488 corpus actually have an end-to-end Lean-covered path; this is the single number that would let a reader calibrate the soundness rhetoric, and it is missing.
- [reviewer, w=0.50, added round 3] **The 28k/28k random-agreement check is still an in-fragment uniform-sample test, not a precondition-discovery test** (acknowledged in §4.4: "implementation-agreement check, not a precondition-discovery check"). A wrong precondition envelope on a hand-written rule is exactly the failure mode the harness cannot detect, and is the most plausible source of unsoundness for a 79-rule table.
- [reviewer, w=0.50, added round 3] **The paper text on Lean status is internally inconsistent with the source as of submission.** The abstract and §4.4 say "11/11 previously-axiomatic soundness lemmas closed sorry-free," yet downstream prose elsewhere in the v6 sections (carried over from earlier versions; flagged in round 2) still refers to a remaining `permList_compose` `sorry`. `grep '^\s*sorry' lean/TensorGuard/*.lean` returns nothing. Reviewers should not have to chase which prose is current.
- [reviewer, w=0.50, added round 3] **The leave-one-category-out remains a literal no-op (53/60 → 53/60).** This is the only quantitative ablation with a counterfactual structure; the v5 modules disabled are not on the handler path of any of the 60 bugs, so the LOO does not actually ablate anything. Reporting this as a holdout is closer to evidence of the absence of a holdout. The per-feature ablation on the 10 upstream-faithful PR repros is similarly flat.
- [reviewer, w=0.50, added round 3] **The 10/10 "shape-pattern coverage" headline (Table 2) is still mis-leading at first reading.** `experiments_v5/v8/real_bugs/rb_*.py` are by construction one-line `forward(self,x): return x.view(...)` distillations — TG's most favourable case. The honest "real-class" number is 7/10 RP@≥0.99 + 1/10 @0.80 with 2/10 silent verifieds; the constructor-bound integer-attribute envelope flagged in §6 is a substantive scope limit, not a polish item.
- [reviewer, w=0.50, added round 3] **The post-freeze N=15 result, while pre-registered, is fragile under multiple-comparisons honesty.** With 5/15 vs 2/15 / 3/15, a Fisher exact comparable test against the strongest baseline (Pytea 3/15) does not approach significance (p ≈ 0.43). The paper presents the gap as "strictly above" without any uncertainty interval; for a paper whose contribution is calibration this should be a CI or a Bayes factor, not a point estimate.
- [reviewer, w=0.50, added round 3] **The CV regime needs an ecological-validity check.** For the 128 CV refutations, the paper does not exhibit even a small sample where the synthesised `assume_M` is satisfied by an actual caller in `torchvision`/`timm`/`transformers`. Without that, a CV verdict on a leaf module is consistent with TG having invented an unsatisfiable precondition and then refuting the caller for failing to satisfy it.
- [reviewer, w=0.35, added round 2] **The headline thesis is still not supported by the headline corpus.** On the 488-block real-source corpus TG produces **0 unconditional RP** in both regimes (`experiments_v5/v8/user_visible_rp.json`). The 128 "CV" refutations are sound only under a TG-synthesised `assume_M`, i.e. TG invents the precondition that makes its own refutation hold, then refutes the caller for violating it. The paper acknowledges this but still leads with "206 refutations". The unconditional-RP claim rests entirely on the 60-bug historical corpus and 3/10 (or 6/10 at 0.80) of a 10-bug corpus.
- [reviewer, w=0.35, added round 2] **The Dynamo-correspondence audit is materially unchanged from round 1 and remains near-vacuous.** `experiments_v5/dynamo_correspondence_v5.json` still records `signature-trusted` for **16 of 17** modules ("Module source too large for end-to-end constraint solving; shape contract is taken from the documented forward signature, which is the same artefact TG would emit after CEGAR"). The 8.8% in-contract recompile and 97.9% OOS-violation rates therefore measure how well a documented signature predicts Dynamo guards, not how well TG's verifier does. Theorem 5's empirical instantiation in §4.3 is detached from the analyser on 16/17 modules.
- [reviewer, w=0.35, added round 2] **Mechanised soundness footprint is much smaller than "Lean-audited" framing suggests.** Table 6 / §4.4 admit 28 Lean-audited + 3 pen-and-paper + **48 tested-only** handlers out of 79, and "the parser, the AST extractor, the analyser implementation, the assume/guarantee composition rule [for the full handler set], and the backward verifier are not mechanised". Theorem 2's soundness theorem therefore does not transfer to any verdict whose path touches one of the 48 tested-only handlers (which is most of the 488-block corpus). The paper continues to lean on "soundness theorem" rhetoric for verdicts the theorem does not in fact cover.
- [reviewer, w=0.35, added round 2] **The 28k/28k Lean–torch agreement remains an in-fragment random sample, not a precondition-discovery test.** §4.4 still concedes: "inputs are sampled uniformly within the rule's declared precondition envelope; the test is therefore an implementation-agreement check, not a precondition-discovery check." A wrong precondition envelope — the most plausible source of unsoundness in a hand-written rule table — is exactly what the harness cannot catch.
- [reviewer, w=0.35, added round 2] **Lean status disclosure is internally inconsistent.** The paper still writes "10/11 previously-axiomatic soundness lemmas closed sorry-free … the one remaining sorry (`permList_compose`) is documented with a counterexample showing the original statement is false at the boundary" (lines 358–363). But (a) `grep '^\s*sorry' lean/TensorGuard/*.lean` returns nothing, and (b) `V5OperatorRules.lean` lines 20–24 explicitly state the tree is sorry-free including `permList_compose` (the in-range restatement is closed). The "one remaining sorry" is no longer present in the source — the paper's framing predates the most recent commit (`3471faf Eliminate 10 sorry sites …`) and should be reconciled, particularly because the round-1 reviewer flagged this exact contradiction.
- [reviewer, w=0.35, added round 2] **The leave-one-category-out is still reported as a literal no-op** (53/60 → 53/60; "the v5 orchestration modules disabled by the LOO are not where operator handlers live"). This is the only quantitative holdout in the paper, and it does not disable any handler on the path of the bugs it was meant to ablate. The "real" holdout reduces to author attestation about catalogue freeze order, with no externally checkable artefact summarised in the paper. The per-feature ablation on the upstream-faithful 10-bug corpus is also a flat line (§306–311), which the paper reports honestly but which further weakens the empirical story.
- [reviewer, w=0.35, added round 2] **The 10/10 "shape-pattern coverage suite" remains misleading as headlined.** `experiments_v5/v8/real_bugs/rb_*.py` is by construction a hand-distilled `forward(self,x): return x.view(...)` one-liner per bug — exactly TG's most favourable case. Table 2 still bolds the 10/10 column; the paper concedes this is a "catalogue-coverage check, not a live-class run", but the bold typography and abstract continue to invite the wrong reading. The genuinely real-class number is 3/10 RP@0.99 + 3/10 @0.80, with 4/10 silent verifieds — the constructor-bound-integer gap discussed in §6 is a substantive scope limitation, not just an implementation polish item.
- [reviewer, w=0.35, added round 2] **Backward-verifier evaluation excludes precisely the regimes §6 admits TG misclassifies.** The 10-real-model sweep in `experiments_v5/v8/backward_real/` explicitly excludes `torch.utils.checkpoint` and parameter sharing; the "≤12% / ≤4%" prevalence estimate is based on filename keyword filtering of an unspecified 5,000 HF training scripts (no script list shipped, no commit SHA frozen). 8/8 canonical bugs and 0/50 false positives are on hand-curated and randomly-generated grammars TG itself helped specify; the realistic frequency at which the silent-misclassification regime fires in production is unmeasured.
- [reviewer, w=0.35, added round 2] **The hybrid-mode falsification (Table 3) is a 25-block hand-built stress set, not a corpus measurement.** It is honest evidence that TG and FakeTensor surfaces are complementary on importable code — but it is not, and is not claimed to be, a falsification of the "0 RP on 488 blocks" headline; on that corpus the hybrid contributes "zero additional resolutions". The paper should not let Table 3 quietly soften the §4.1 result.
- [reviewer, w=0.25, added round 1] **The headline number does not support the thesis.** On the 488-block real-source corpus, TG produces **0 unconditional RP** (line 219–225 of the paper, confirmed in `experiments_v5/v8/user_visible_rp.json`: `Refuted_Proof: 0` in both regimes). The 206 "refutations" are 128 CV (sound only under a tool-synthesised caller-rely envelope, i.e. the tool is allowed to invent the precondition that makes its own refutation hold) plus 78 LW (explicitly outside the soundness theorem). Calling 128 CV verdicts "sound only under the synthesised caller-rely assume" is close to circular: TG synthesises `assume_M`, then refutes the caller for violating it. The paper acknowledges this but still leads with the 488-block corpus; the unconditional-RP claim rests entirely on the 60-bug historical corpus and 3/10 (or 6/10 at 0.80) on the 10-bug upstream corpus.
- [reviewer, w=0.25, added round 1] **The Dynamo correspondence "audit" is not what it appears.** `experiments_v5/dynamo_correspondence_v5.json` shows `"signature-trusted"` for 16 of 17 modules with the candid annotation: *"Module source too large for end-to-end constraint solving; shape contract is taken from the documented forward signature, which is the same artefact TG would emit after CEGAR."* In other words, TG did not actually verify the modules in §4.3; the contracts were taken from documentation, then Dynamo was run against those contracts. The 8.8% in-contract recompile and 97.9% OOS-violation numbers therefore measure how well a hand-written signature predicts Dynamo guards, not how well TG's verifier does. Theorem 5's empirical instantiation is essentially vacuous on 16/17 modules.
- [reviewer, w=0.25, added round 1] **Lean claim and Lean source disagree about `sorry`-freedom.** §1 / §4.4 state "10/11 previously-axiomatic sites" closed with one outstanding `sorry` for `permList_compose`. But `lean/TensorGuard/V5OperatorRules.lean` lines 20–24 say *"the entire `lean/TensorGuard/` tree is also `sorry`-free: the … unconditional `permList_compose`) is now closed sorry-free"*. Either the paper or the source is stale. `grep -n sorry lean/TensorGuard/*.lean` returns matches only inside comments/docstrings, but the in-paper count of 28/79 handlers Lean-audited (and 48 tested-only) is the more important — and uncomforting — number: **48 of 79 operator handlers are covered only by random agreement testing**, and §C/§Theorem 1's soundness sketch explicitly excludes them. The actual mechanised soundness footprint is far smaller than the "Lean-audited" framing suggests.
- [reviewer, w=0.25, added round 1] **The 28,000/28,000 Lean–torch agreement is an in-fragment random sample, not a precondition-discovery test.** §4.4 admits this ("inputs are sampled uniformly within the rule's declared precondition envelope; the test is therefore an implementation-agreement check, not a precondition-discovery check"). So a wrong precondition envelope cannot be caught by the harness — the most plausible source of unsoundness in a hand-written rule table is exactly the one the test is blind to.
- [reviewer, w=0.25, added round 1] **The leave-one-category-out holdout is reported as a literal no-op** ("aggregate RP rate falls from 53/60 to 53/60 ... the v5 orchestration modules disabled by the LOO are not where operator handlers live"). This means the only "holdout" evidence presented is a process claim (catalogue enumerated from `torch.nn.modules` before bug forwards were inspected), which is not independently verifiable from the artifact. The per-feature ablation on the real-public 10-bug corpus is also a flat line (§306–311). The empirical case for catalogue-not-overfit therefore reduces to author attestation.
- [reviewer, w=0.25, added round 1] **The 10-bug "shape-pattern coverage suite" 10/10 result is misleading as presented.** The hand-distilled `forward(self,x): return x.view(...)` one-liners (`experiments_v5/v8/real_bugs/`) are by construction inside FTG's most favourable case (literal-int divisibility view), so 10/10 is a self-test of TG's view rule, not evidence on real public-repo code. The paper does eventually concede this ("catalogue-coverage check, not that TG runs on the upstream class") but Table 2's bolded 10/10 column risks being read as a real headline.
- [reviewer, w=0.25, added round 1] **Backward verifier "500/500 agreement" is on random small modules drawn from a generative grammar** TG itself helped specify; the 10-real-model sweep in `experiments_v5/v8/backward_real/` explicitly excludes parameter sharing and `torch.utils.checkpoint`, which §6 admits is the regime where TG silently misclassifies. The "≤12% / ≤4%" prevalence estimate is uncited and based on filename keyword filtering of "5,000 HuggingFace training scripts" (no script list shipped).
- [reviewer, w=0.25, added round 1] **Theorem 2 has a reach gap relative to the implementation.** §2.1 ("the analyser implementation … is not mechanised"), §6 ("Lean does not mechanise the AST extractor, the analyser implementation, the assume/guarantee composition rule [for the full handler set], or the backward verifier") — combined with 48/79 tested-only handlers — means the soundness theorem applies to the rule table on paper, not to what `verify(M)` actually computes for any module containing a tested-only handler. The paper should not be allowed to leverage "soundness theorem" rhetoric for verdicts that involve any of those 48 handlers.

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
