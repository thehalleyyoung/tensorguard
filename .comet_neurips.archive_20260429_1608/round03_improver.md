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
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically reasons about symbolic shapes and a flat first-order grad-flag lattice (`{has_grad, no_grad, ⊤}`), discharging side conditions to Z3 and reporting a five-way verdict taxonomy. Empirics: 53/60 RP on a curated historical corpus; 32/34 vs. Pytea 22/34 on a fragment-fair modern subset (McNemar p=0.00195); 5/15 catches on the unfiltered pre-registered post-freeze N=15 PR sample (vs. FakeTensorMode 2/15, Pytea 3/15, Fisher non-separable; Bayesian BF₁₀=8.1 vs. FT, 3.6 vs. Pytea); 0 unconditional RP on the 488-block real-source corpus under the free-symbolic regime. A Lean 4 audit covers 28/79 shape-transfer handlers with 11/11 axiomatic lemmas closed sorry-free and 28,000/28,000 byte-mirror cases agreeing with torch 2.9.1. The Dynamo-guard correspondence (Thm. 5) is necessary-direction only and audited on ~31 modules total (17 original + 14 extended), with 19/19 recompiles classified `{SHAPE:19}` and zero out-of-catalogue guards.

## Prior weakness disposition
- [UNRESOLVED] the unfiltered pre-registered post-freeze evaluation is still N=15. The Bayesian supplement (BF₁₀=8.1 vs. FT, 3.6 vs. Pytea) does not exceed the conventional "st... -- Sample size remains N=15 with two-sided Fisher p=0.39 / 0.68; the Bayesian BFs sit in the "moderate" band, below the strong-evidence ≥10 threshold the prior round flagged.
- [PARTIAL] the 12-CV joint-realisability audit is a sample of 12/128 (~9.4%) selected as "12 randomly-sampled CV verdicts". The prior round asked for either the full 128-set or a uniformly... -- The 12-of-128 random sample with named-checkpoint pairings is the only joint-realisability evidence in the body; no full-128 ratio with CI and no scaled-up uniform subsample is reported.
- [PARTIAL] the Dynamo-falsification corpus is now ~31 modules (17 original + 14 extended), still well short of the ≥100 timm/HF blocks the prior round asked for, and 4 of the 14 exte... -- Total instantiated coverage remains ~31 modules; 4 of the 14 extended transformer blocks still rely on the forward-signature surrogate, so end-to-end ≥100-module evidence for Theorem 5 is still missing.
- [PARTIAL] the grad-flag silent-error audit ("0/16 `torch.utils.checkpoint`, 0/16 renamed-attribute parameter sharing") is a same-author pattern check on the same 17-module Theorem 5 fixture,... -- The audit fixture is unchanged; no held-out HF training-script false-verified-rate measurement appears, so the ≤12% prevalence claim still rests on the same-author sweep.
- [UNRESOLVED] The catalogue-coverage residual 12/78 "could-in-principle convert to RP" upper bound on the LW→RP gap is asserted in §4.1 but not exhibited at the per-block level... -- §4.1 still asserts the 12/78 residual without a per-block enumeration of which blocks would convert and under which missing rule.

## Strengths
- Calibration discipline remains unusually high: the 0 unconditional RP on the 488-block free-symbolic surface is reported as the headline; the 5/15 post-freeze catch is reported with explicit Wilson CI [15.2%, 58.3%] and explicit Fisher non-separability rather than as a separation claim; the off-axis fire on `rb_uf_010` is accounted as a false positive against ground truth and excluded from the headline catch count.
- The Lean audit is described accurately for what it certifies: 28/79 handlers, 11/11 axiomatic lemmas closed sorry-free, lake build sorry-free, and 28,000/28,000 byte-mirror against torch 2.9.1, with the analyser/AST extractor/backward verifier/Z3 dispatch explicitly held out as TCB. The `permList_compose_inrange` correction (replacing the originally false unconditional statement) and the boundary-mutator off-envelope check (no silent-through on ~2,400 samples across 10 rules) are non-trivial soundness work.
- C5 has been honestly narrowed: only the three discriminative knobs (device-consistency, gradient-flow, low-confidence gating) are claimed in the per-feature ablation; the unused CEGAR loop and always-satisfiable phase encoder are explicitly disclaimed as non-contributions, and the localisation tracer is relegated to engineering. This is the kind of negative framing reviewers usually have to drag out of authors.
- The fragment-fair Pytea head-to-head is methodologically tight: the 32/34 vs. 22/34 split on the modern subset is reproduced at verification time (TG restricted to the 2022 catalogue intersection at run time, AST-screen on each repro, forensics scan of `Bug.message`), the Pytea silent-skip correction is explicit, and the McNemar exact two-sided p=0.00195 with paired-bootstrap 95% CI [+14.7 pp, +44.1 pp] is a defensible statistical claim on this N.

## Weaknesses
- The post-freeze unfiltered evaluation is still N=15 (Section 4.1, Table 3). The BF₁₀=8.1 vs. FakeTensorMode and BF₁₀=3.6 vs. Pytea both sit in the "moderate" Jeffreys band, well below the conventional ≥10 strong-evidence threshold the authors themselves cite, and the frequentist Fisher tests (p=0.39 and p=0.68) do not separate. The empirical-superiority claim over execution-based baselines on the only unfiltered pre-registered surface is therefore still a point estimate, not a separation. Either extend the pre-registered query to N≥40 (which would, on the observed point estimates, push at least the TG-vs-FakeTensorMode comparison toward Fisher significance and BF₁₀≥10) and report the resulting numbers, or restate the headline as "point-above on N=15, not statistically separable" without leaning on the Bayesian supplement.
- The 488-block CV joint-realisability evidence (§4.1) is still the 12-of-128 random-sample audit (~9.4%) with named `*Config`-default instantiations and published checkpoints. The prior round explicitly asked for either all 128 or a substantially scaled-up uniform subsample with a confidence interval on the joint-realisability ratio. Without that, the "0 unconditional RP / 57 Verified" surface continues to depend on a `assume_M` whose *aggregate* realisability is estimated from <10% of the corpus. Please report the joint-realisability ratio over either all 128 CVs or a uniform random subsample of ≥60 with a Wilson/Clopper-Pearson CI on the realisable fraction.
- The Theorem 5 empirical audit is still ~31 modules total (17 original + 14 extended; §4.3 / "Extended end-to-end audit"). Of the 14 extended blocks, 4 transformer blocks are audited via the documented forward-signature surrogate, not a full instantiation, so the end-to-end count is closer to 9 CNN blocks + 1 ResNet50 layer + previously-audited cases. Theorem 5 is the central PL-side claim about Dynamo, and it is still being instantiated on roughly a third of the ≥100-block target the prior round named. Please run the falsifier on a further ≥70 importable timm/HF blocks (not the surrogate path) and report `{SHAPE, DTYPE, RANK, INT}` recompile counts and any out-of-catalogue guard hits.
- The grad-flag silent-error audit (§6) reports `0/16 torch.utils.checkpoint` and `0/16` renamed-attribute parameter sharing on the 16 importable Track-E modules — the same fixture used elsewhere in the paper. This is the only quantitative support for the headline "≤12% prevalence" caveat on the grad lattice's known unsoundness, and it is a same-author pattern check on a same-author fixture. Please run a held-out audit on a different population — e.g. the top-N HuggingFace `Trainer`/`accelerate` example training scripts — and report a false-verified-rate against runtime grad equality (not against a manually authored `# BUG` marker). A corpus around 30–50 training scripts would already discriminate ≤12% from, say, 25%.
- The "12/78 catalogue-coverage residual" bound on the LW→RP gap (§4.1) is asserted as an upper bound but not exhibited per-block. Without a list of which 12 of the 78 LW blocks would convert to RP under which specific missing rule, the bound is unfalsifiable from the paper alone — a future reviewer cannot tell whether the residual is realistic or whether several "in-principle convertible" entries are themselves blocked by the very catalogue gap that produced LW. Please add a small per-block table (block id → missing rule → predicted converted verdict) for the 12 cases.
- Theorem 1 (fragment-level soundness) and Theorems 10/11 (Preservation/Progress) are pen-and-paper, while Theorem 3 (compositional/assume-guarantee) is mechanised only on a 3-operator DSL via `lemma ag_composition`. The paper concedes this, but the TCB statement in §4.4 ("the analyser implementation, AST extractor, backward verifier, and Z3 dispatch remain in TCB") therefore covers the *entire* operational soundness story for the user-facing tool. A short explicit accounting of what survives if a TCB component is wrong (e.g. AST extractor mis-binds a starred view, backward verifier mis-classifies an in-place op) would calibrate what the 53/60 RP headline actually means about the artefact vs. about the calculus.

## Questions
- On the post-freeze unfiltered surface, what is the smallest N at which the observed point estimates (TG 1/3 vs. FT 2/15, Pytea 1/5) would yield BF₁₀≥10 and Fisher p<0.05 on at least one of the two pairwise comparisons? If extending the pre-registered query to that N is feasible, do so and report.
- For the 488-block corpus, what is the joint-realisability ratio of `assume_M` on either the full N=128 CV set or a uniform random subsample of N≥60, with a 95% CI? How many of the 12-of-128 named `*Config`-default instantiations remain joint-realisable when the published checkpoint's runtime input distribution (rather than the config defaults alone) is taken as the rely?
- Please tabulate the Theorem 5 falsification predicate over a further ≥70 end-to-end-instantiated importable blocks (not via forward-signature surrogate). What is the `{SHAPE, DTYPE, RANK, INT}` breakdown, and does any single block produce a SHAPE/DTYPE/RANK guard whose `guard_var` lies outside `catalogue(M)`?
- For the grad-flag lattice unsoundness on parameter-sharing-under-renamed-attribute, what is the false-verified-rate against runtime-observed `p.grad ≠ None` on a held-out corpus (e.g. ≥30 HF Trainer/accelerate example training scripts) as opposed to the same 16-module pattern sweep?
- Please give a per-block enumeration of the 12/78 LW→RP residual: for each of the 12 blocks, the missing operator handler whose addition would (in isolation) flip the verdict to unconditional RP, and the witnessing input shape.
- For the TCB components held out of the Lean audit (analyser, AST extractor, backward verifier, Z3 dispatch), what is the largest verdict-flip from a single deliberate fault-injection (e.g. AST mis-binding of `view(*new_shape)` star-expansion, backward verifier mis-classifying `Tensor.add_` as out-of-place) on the 60-bug and 488-block corpora?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons
A single change would lift my score: extending either the unfiltered post-freeze sample to N≥40 (so that the BF₁₀≥10 / Fisher-significant separation claim against FakeTensorMode and Pytea becomes statistical rather than point-only), or the Theorem 5 end-to-end audit to ≥100 fully-instantiated importable blocks with the falsification predicate explicitly evaluated. Either would convert the most consequential PARTIAL/UNRESOLVED items into RESOLVED at the next round.

Round: 3


Changes   +0 -0
Requests  7.5 Premium (1m 52s)
Tokens    ↑ 273.4k • ↓ 5.5k • 243.6k (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 3] W4 (PARTIAL): the Dynamo-falsification corpus is now $\sim 31$ modules ($17$ original + $14$ extended), still well short of the $\ge 100$ timm/HF blocks the prior round asked for, and 4 of the 14 extended modules use the documented forward-signature surrogate rather than full instantiation. The "zero out-of-catalogue SHAPE/DTYPE/RANK guards" result on $19$ aggregate recompile events is therefore based on a small denominator, and the surrogate split (9 CNN end-to-end vs. 4 transformer surrogate) is not separated in the headline. Either run the falsifier on $\ge 100$ end-to-end modules or restrict Theorem 5's empirical claim to the CNN-block subset.
- [reviewer, w=1.00, added round 3] The catalogue-coverage residual $12/78$ "could-in-principle convert to RP" upper bound on the LW→RP gap is asserted in §4.1 but not exhibited at the per-block level inside the body. Provide a per-block list of the $12$ "fragment-only forward bodies" so the upper bound can be independently checked, or reduce the claim to "≤12/78" without the structural decomposition.
- [reviewer, w=1.00, added round 3] The post-freeze unfiltered evaluation is still N=15 (Section 4.1, Table 3). The BF₁₀=8.1 vs. FakeTensorMode and BF₁₀=3.6 vs. Pytea both sit in the "moderate" Jeffreys band, well below the conventional ≥10 strong-evidence threshold the authors themselves cite, and the frequentist Fisher tests (p=0.39 and p=0.68) do not separate. The empirical-superiority claim over execution-based baselines on the only unfiltered pre-registered surface is therefore still a point estimate, not a separation. Either extend the pre-registered query to N≥40 (which would, on the observed point estimates, push at least the TG-vs-FakeTensorMode comparison toward Fisher significance and BF₁₀≥10) and report the resulting numbers, or restate the headline as "point-above on N=15, not statistically separable" without leaning on the Bayesian supplement.
- [reviewer, w=1.00, added round 3] The 488-block CV joint-realisability evidence (§4.1) is still the 12-of-128 random-sample audit (~9.4%) with named `*Config`-default instantiations and published checkpoints. The prior round explicitly asked for either all 128 or a substantially scaled-up uniform subsample with a confidence interval on the joint-realisability ratio. Without that, the "0 unconditional RP / 57 Verified" surface continues to depend on a `assume_M` whose *aggregate* realisability is estimated from <10% of the corpus. Please report the joint-realisability ratio over either all 128 CVs or a uniform random subsample of ≥60 with a Wilson/Clopper-Pearson CI on the realisable fraction.
- [reviewer, w=1.00, added round 3] The grad-flag silent-error audit (§6) reports `0/16 torch.utils.checkpoint` and `0/16` renamed-attribute parameter sharing on the 16 importable Track-E modules — the same fixture used elsewhere in the paper. This is the only quantitative support for the headline "≤12% prevalence" caveat on the grad lattice's known unsoundness, and it is a same-author pattern check on a same-author fixture. Please run a held-out audit on a different population — e.g. the top-N HuggingFace `Trainer`/`accelerate` example training scripts — and report a false-verified-rate against runtime grad equality (not against a manually authored `# BUG` marker). A corpus around 30–50 training scripts would already discriminate ≤12% from, say, 25%.
- [reviewer, w=1.00, added round 3] The "12/78 catalogue-coverage residual" bound on the LW→RP gap (§4.1) is asserted as an upper bound but not exhibited per-block. Without a list of which 12 of the 78 LW blocks would convert to RP under which specific missing rule, the bound is unfalsifiable from the paper alone — a future reviewer cannot tell whether the residual is realistic or whether several "in-principle convertible" entries are themselves blocked by the very catalogue gap that produced LW. Please add a small per-block table (block id → missing rule → predicted converted verdict) for the 12 cases.
- [reviewer, w=1.00, added round 3] Theorem 1 (fragment-level soundness) and Theorems 10/11 (Preservation/Progress) are pen-and-paper, while Theorem 3 (compositional/assume-guarantee) is mechanised only on a 3-operator DSL via `lemma ag_composition`. The paper concedes this, but the TCB statement in §4.4 ("the analyser implementation, AST extractor, backward verifier, and Z3 dispatch remain in TCB") therefore covers the *entire* operational soundness story for the user-facing tool. A short explicit accounting of what survives if a TCB component is wrong (e.g. AST extractor mis-binds a starred view, backward verifier mis-classifies an in-place op) would calibrate what the 53/60 RP headline actually means about the artefact vs. about the calculus.
- [reviewer, w=1.00, added round 3] On the post-freeze unfiltered surface, what is the smallest N at which the observed point estimates (TG 1/3 vs. FT 2/15, Pytea 1/5) would yield BF₁₀≥10 and Fisher p<0.05 on at least one of the two pairwise comparisons? If extending the pre-registered query to that N is feasible, do so and report.
- [reviewer, w=1.00, added round 3] For the 488-block corpus, what is the joint-realisability ratio of `assume_M` on either the full N=128 CV set or a uniform random subsample of N≥60, with a 95% CI? How many of the 12-of-128 named `*Config`-default instantiations remain joint-realisable when the published checkpoint's runtime input distribution (rather than the config defaults alone) is taken as the rely?
- [reviewer, w=1.00, added round 3] Please tabulate the Theorem 5 falsification predicate over a further ≥70 end-to-end-instantiated importable blocks (not via forward-signature surrogate). What is the `{SHAPE, DTYPE, RANK, INT}` breakdown, and does any single block produce a SHAPE/DTYPE/RANK guard whose `guard_var` lies outside `catalogue(M)`?

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

Round: 3
