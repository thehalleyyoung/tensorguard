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
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically verifies symbolic shapes and a coarse three-element gradient-flag lattice from class source. The contribution is the refinement-typed calculus `Tensor{s,g | φ}` with assume/guarantee composition at the class boundary, a Lean 4 audit of 28/79 shape-transfer rules (sorry-free) plus a 13-operator mechanised composition DSL via `ag_composition_ext`, and a backward verifier targeting silent-zero-grad bug classes. Headline empirics are 53/60 RP on a curated bug corpus, 32/34 vs Pytea 22/34 on a fragment-fair subset (McNemar p=0.00195), 5/15 catches vs FakeTensor 2/15 / Pytea 3/15 on a pre-registered post-freeze unfiltered real-PR sample (not separable at α=0.05), and on 488 real `nn.Module` blocks the user-visible free-symbolic-config regime returns 0 unconditional RP, calibrated as a fragment-coverage limit. New this round: a 356/488 no-synthesised-assume subset surfaces 26 unconditional refutations (Wilson 95% CI [5.0%, 10.5%]).

## Prior weakness disposition
- [PARTIAL] The leave-one-out audit in §4.1 still asserts that disabling per-category handlers "leaves the aggregate RP rate at 53/60, because an independent AST-pattern v -- The AST-pattern-disabled rerun now exists at `reproducibility/ast_pattern_disabled_60bug.md` (53/60 with the AST-path off; operator-dispatch alone is sufficient) and a triple-LOO at `reproducibility/triple_path_loo_60bug.md` (P1+P2+P3 off → 53/60); but §4.1 still attributes invariance to the AST-pattern parallel path, which the new repro contradicts (operator-dispatch alone gets 53/60), so the body text has not been reconciled with its own appendix.
- [PARTIAL] The 488-block 0-unconditional-RP gap is unchanged: all 12 named LW→RP candidates remain in the "Predicted RP" column with no implementation or measure -- A `Tensor.unbind(dim)` UNBIND OpKind and elaborator branch are now in `src/model_checker.py:293,2994,3168,8096`, and a runner script `reproducibility/run_unbind_488.py` exists, but no `unbind_handler_488_run.{json,md}` artefact was produced and §4.1's table still lists `timm::ChannelAttention*` as Predicted RP; the new no-assume subset (`reproducibility/no_assume_subset_488.md`, 26/356 RP, CI [5.0%, 10.5%]) is a genuine reframing that surfaces non-zero unconditional refutations, but the originally requested "1/488 vs 0/488 falsifiable progress on the headline" was not produced.
- [UNRESOLVED] 105/185 in-soundness verdicts on the 488-block corpus still touch at least one tested-only handler (§4.4). The load-bearing handlers in this bucket — `view/r -- §4.4 still reports 36/185 in-soundness vs 33+72=105/185 touching tested-only handlers; no handler was promoted into Lean or pen-and-paper soundness this round (no new entry in Table 7).
- [UNRESOLVED] The Lean composition theorem is now mechanised on 13 operators but the 13-operator DSL still excludes the four operators on the bug paths of the post-freeze N=15 c -- The contribution sentence still reads "audited end-to-end on this 13-operator fragment"; `view/reshape`, `conv2d`, `einsum`, `expand` (modulo whatever `expand`-ish entry is in the existing 13) were not added; no new Lean operator soundness lemmas were committed and `ag_composition_ext` operates over the same 13 operators.
- [UNRESOLVED] The 333/2,908 (11.45%) tied-weight footprint is still not directly folded into the headline silent-error rate. The 8/8-RP runtime harness is on `gradient_check -- A new `reproducibility/backward_param_sharing_audit.md` with 6 hand-built tied-weight repros (BERT/GPT-2/T5/BART/RoBERTa + minimal) is added, but it returns SAFE_NO_BUGS (not ABSTAIN) on 6/6 — and the §6 paper text claims "the verifier returns 6/6 ABSTAIN and 0/6 silently-incorrect Verified", which directly contradicts the appendix table; the requested measurement (V/RP/Abstain triple over the 333 tied-weight modules with a runtime false-Verified subset) is not provided.
- [UNRESOLVED] Theorem 5's empirical surface is still uneven: the 14-module CNN-only restriction (13 SHAPE recompiles, all in-catalogue) is the falsifier-evaluation -- §4.3 still presents the same 14-module CNN-only headline and the 55-module 72-INT-only audit; no curated module on which a SHAPE/DTYPE/RANK guard outside `catalogue(M)` is *plausible* has been added, so the falsification predicate remains evaluated only on a surface where it cannot fire.

## Strengths
- The new `reproducibility/ast_pattern_disabled_60bug.md` and `triple_path_loo_60bug.md` artefacts finally give measured numbers for the LOO invariance: operator-dispatch alone gets 53/60, AST-pattern alone gets 53/60, and even the triple-disabled (P1+P2+P3 globally off) reportedly returns 53/60. This is the right protocol and is the first time the LOO claim is independently checkable.
- The no-synthesised-assume 488-block subset (`no_assume_subset_488.md`: V=27, RP=26, CV=0, LW=78, A=225 over 356 blocks, Wilson 95% CI [5.0%, 10.5%]) is a clean reframing that exposes the 26/128 empty-assume CV slice as a non-zero unconditional-refutation count with a calibrated CI, which is the correct way to surface the "0 unconditional RP" caveat.
- The unbind handler implementation (UNBIND OpKind, dedicated tuple-elaborator path, `src/model_checker.py:8096–8120`) is a real code-side delivery of the smallest-cost LW→RP candidate even though the corresponding 488-block measurement was not produced.
- Calibrated reporting throughout (Wilson / Clopper-Pearson CIs, McNemar exact, Fisher exact with explicit non-separation on N=15, pre-registered post-freeze query, fault-injection exposure-vs-measured table) remains unusually disciplined and load-bearable.

## Weaknesses
- §4.1's prose still says the LOO invariance holds "because an independent AST-pattern verification path runs in parallel with the operator dispatch and catches the bugs even when the per-category handlers are disabled." The new appendix (`ast_pattern_disabled_60bug.md`) shows operator-dispatch alone *also* gets 53/60, i.e. the AST-pattern path is not what is carrying the LOO invariance — it is over-determined on both sides. The body text and the appendix now disagree; please update §4.1 to state the operator-dispatch-only and AST-pattern-disabled rates explicitly (53/60 each) and remove the "because" attribution. Additionally, the triple-LOO (P1+P2+P3 all disabled) reporting 53/60 is *internally suspicious*: if all three refute paths are off and the rate is unchanged, what is the fourth path? Either name it in §4.1 or correct the appendix.
- The 488-block "0 unconditional RP" headline triple in §4.1 is unchanged. The unbind handler is in code (`UNBIND` OpKind, elaborator at `src/model_checker.py:8096`) and `reproducibility/run_unbind_488.py` is the runner, but the script's expected output `reproducibility/unbind_handler_488_run.{json,md}` does not exist; the table at line ~803 still lists `timm::ChannelAttention/ChannelAttentionV2` as "Predicted RP" rather than as a measured flip. Run the script and report the new 488-block triple and per-block flip witnesses; even a 0-flip outcome with an autopsy ("unbind handler does not flip these blocks because …") would close the falsifiability ask.
- The tied-weights audit at `reproducibility/backward_param_sharing_audit.md` directly contradicts the §6 prose. The appendix table records *6/6 SAFE_NO_BUGS* (zero ABSTAINs) on tied-weight HF heads with `false_verified=YES⚠` for every row and a "False-verified rate 6/6 = 1.000" headline; the paper text at line ~1795 claims the same harness returns "6/6 ABSTAIN and 0/6 silently-incorrect Verified". Both cannot be right — please reconcile (and, if the appendix is correct, the §6 silent-error envelope claim flips sign and needs to be redrafted accordingly). The originally asked measurement — V/RP/Abstain triple over the 333 tied-weight modules in the population, plus a runtime false-Verified subset — is still missing.
- The Lean composition DSL is unchanged at 13 operators. The four operators on the Table 3 post-freeze bug paths (`view/reshape`, `conv2d`, `einsum`, plus the genuinely-out-of-DSL part of `expand`) are still outside `ag_composition_ext`, so the C2 contribution sentence ("audited end-to-end on this 13-operator fragment") still understates the gap between mechanised composition and the operators that fire on the headline catches. Either extend `ag_composition_ext` to the union of operators in Table 3 (the 5 post-freeze catches), or tighten C2 to claim composition only on the operators that the empirical headline actually uses.
- §4.4's 105/185 split is unchanged (36 in-soundness vs 105 touching tested-only). No handler was promoted from tested-only to pen-and-paper or Lean this round. The smallest-delta candidate (`view/reshape/total_size`, exposed on 18 of 488 blocks per the F3 fault-injection scan) is still tested-only despite being the most load-bearing handler in both the LW→RP attribution and the F3 exposure footprint. Promote one and report the new 36+k / 185 split.
- §4.3's Theorem 5 surface is unchanged. The falsification predicate (SHAPE/DTYPE/RANK guard outside `catalogue(M)`) has still only been evaluated on the 14-module CNN-only restriction (where, by construction, all 13 SHAPE recompiles are in-catalogue), and the 55-module population still observes 72-of-72 INT-only recompiles, so the falsifier is evaluated on 0 events of the right kind. Add at least one curated module with a custom op that reads a non-catalogue shape bit so the falsification predicate is non-vacuously evaluable on at least one event of kind ≠ INT.

## Questions
- The triple-LOO appendix (`triple_path_loo_60bug.md`) reports that disabling all three refute paths globally still yields 53/60 RP. Which fourth refute path is producing those verdicts, and can the LOO be re-run with that fourth path also disabled, so the operator-rule contribution is genuinely isolated?
- Please run `reproducibility/run_unbind_488.py` and report the resulting 488-block verdict triple and the per-block flip witnesses for `timm::ChannelAttention` and `timm::ChannelAttentionV2`. Does the headline 0/488 RP move to ≥1/488?
- Of the 333 tied-weight modules in the population sweep, what is the analyser verdict triple (V/RP/Abstain) and, on a runtime-instantiable subset, the false-Verified rate against a one-step `loss.backward()` ground truth? (The 6 hand-built repros in `backward_param_sharing_audit.md` do not address this.)
- Why does `backward_param_sharing_audit.md` record 6/6 SAFE_NO_BUGS while the §6 paper text claims 6/6 ABSTAIN on the same harness? Which is correct?
- Of the 105/185 in-soundness verdicts touching a tested-only handler on the 488-block corpus, which single handler promotion (Lean-audited or pen-and-paper) would shrink that count the most, and what is the projected new split?
- On the 55-module Dynamo audit where all 72 observed in-contract recompiles were INT-only, can you exhibit at least one module on which the falsification predicate (SHAPE/DTYPE/RANK guard outside `catalogue(M)`) is *capable* of firing, and confirm the necessary direction holds there?

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 6

## Borderline reasons
The single largest score-moving change would be to actually execute `reproducibility/run_unbind_488.py`, report the resulting 488-block RP count and per-block flip witnesses, and reconcile the §6 tied-weights prose with `backward_param_sharing_audit.md` (which currently directly contradicts it). Either of those — paired with a §4.1 rewrite that uses the now-measured AST-pattern-disabled and operator-dispatch-only numbers (53/60 each) instead of the unfalsifiable "because" attribution — would push Soundness and Contribution each by one and Overall to 7.


Changes   +0 -0
Requests  7.5 Premium (4m 21s)
Tokens    ↑ 1.0m • ↓ 14.3k • 945.2k (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 4] §4.1's prose still says the LOO invariance holds "because an independent AST-pattern verification path runs in parallel with the operator dispatch and catches the bugs even when the per-category handlers are disabled." The new appendix (`ast_pattern_disabled_60bug.md`) shows operator-dispatch alone *also* gets 53/60, i.e. the AST-pattern path is not what is carrying the LOO invariance — it is over-determined on both sides. The body text and the appendix now disagree; please update §4.1 to state the operator-dispatch-only and AST-pattern-disabled rates explicitly (53/60 each) and remove the "because" attribution. Additionally, the triple-LOO (P1+P2+P3 all disabled) reporting 53/60 is *internally suspicious*: if all three refute paths are off and the rate is unchanged, what is the fourth path? Either name it in §4.1 or correct the appendix.
- [reviewer, w=1.00, added round 4] The 488-block "0 unconditional RP" headline triple in §4.1 is unchanged. The unbind handler is in code (`UNBIND` OpKind, elaborator at `src/model_checker.py:8096`) and `reproducibility/run_unbind_488.py` is the runner, but the script's expected output `reproducibility/unbind_handler_488_run.{json,md}` does not exist; the table at line ~803 still lists `timm::ChannelAttention/ChannelAttentionV2` as "Predicted RP" rather than as a measured flip. Run the script and report the new 488-block triple and per-block flip witnesses; even a 0-flip outcome with an autopsy ("unbind handler does not flip these blocks because …") would close the falsifiability ask.
- [reviewer, w=1.00, added round 4] The tied-weights audit at `reproducibility/backward_param_sharing_audit.md` directly contradicts the §6 prose. The appendix table records *6/6 SAFE_NO_BUGS* (zero ABSTAINs) on tied-weight HF heads with `false_verified=YES⚠` for every row and a "False-verified rate 6/6 = 1.000" headline; the paper text at line ~1795 claims the same harness returns "6/6 ABSTAIN and 0/6 silently-incorrect Verified". Both cannot be right — please reconcile (and, if the appendix is correct, the §6 silent-error envelope claim flips sign and needs to be redrafted accordingly). The originally asked measurement — V/RP/Abstain triple over the 333 tied-weight modules in the population, plus a runtime false-Verified subset — is still missing.
- [reviewer, w=1.00, added round 4] The Lean composition DSL is unchanged at 13 operators. The four operators on the Table 3 post-freeze bug paths (`view/reshape`, `conv2d`, `einsum`, plus the genuinely-out-of-DSL part of `expand`) are still outside `ag_composition_ext`, so the C2 contribution sentence ("audited end-to-end on this 13-operator fragment") still understates the gap between mechanised composition and the operators that fire on the headline catches. Either extend `ag_composition_ext` to the union of operators in Table 3 (the 5 post-freeze catches), or tighten C2 to claim composition only on the operators that the empirical headline actually uses.
- [reviewer, w=1.00, added round 4] §4.4's 105/185 split is unchanged (36 in-soundness vs 105 touching tested-only). No handler was promoted from tested-only to pen-and-paper or Lean this round. The smallest-delta candidate (`view/reshape/total_size`, exposed on 18 of 488 blocks per the F3 fault-injection scan) is still tested-only despite being the most load-bearing handler in both the LW→RP attribution and the F3 exposure footprint. Promote one and report the new 36+k / 185 split.
- [reviewer, w=1.00, added round 4] §4.3's Theorem 5 surface is unchanged. The falsification predicate (SHAPE/DTYPE/RANK guard outside `catalogue(M)`) has still only been evaluated on the 14-module CNN-only restriction (where, by construction, all 13 SHAPE recompiles are in-catalogue), and the 55-module population still observes 72-of-72 INT-only recompiles, so the falsifier is evaluated on 0 events of the right kind. Add at least one curated module with a custom op that reads a non-catalogue shape bit so the falsification predicate is non-vacuously evaluable on at least one event of kind ≠ INT.
- [reviewer, w=1.00, added round 4] The triple-LOO appendix (`triple_path_loo_60bug.md`) reports that disabling all three refute paths globally still yields 53/60 RP. Which fourth refute path is producing those verdicts, and can the LOO be re-run with that fourth path also disabled, so the operator-rule contribution is genuinely isolated?
- [reviewer, w=1.00, added round 4] Please run `reproducibility/run_unbind_488.py` and report the resulting 488-block verdict triple and the per-block flip witnesses for `timm::ChannelAttention` and `timm::ChannelAttentionV2`. Does the headline 0/488 RP move to ≥1/488?
- [reviewer, w=1.00, added round 4] Of the 333 tied-weight modules in the population sweep, what is the analyser verdict triple (V/RP/Abstain) and, on a runtime-instantiable subset, the false-Verified rate against a one-step `loss.backward()` ground truth? (The 6 hand-built repros in `backward_param_sharing_audit.md` do not address this.)
- [reviewer, w=1.00, added round 4] Why does `backward_param_sharing_audit.md` record 6/6 SAFE_NO_BUGS while the §6 paper text claims 6/6 ABSTAIN on the same harness? Which is correct?
- [reviewer, w=1.00, added round 4] Of the 105/185 in-soundness verdicts touching a tested-only handler on the 488-block corpus, which single handler promotion (Lean-audited or pen-and-paper) would shrink that count the most, and what is the projected new split?
- [reviewer, w=1.00, added round 4] On the 55-module Dynamo audit where all 72 observed in-contract recompiles were INT-only, can you exhibit at least one module on which the falsification predicate (SHAPE/DTYPE/RANK guard outside `catalogue(M)`) is *capable* of firing, and confirm the necessary direction holds there?
- [reviewer, w=0.71, added round 3] The leave-one-out audit in §4.1 still asserts that disabling per-category handlers "leaves the aggregate RP rate at 53/60, because an independent AST-pattern verification path runs in parallel". This load-bearing claim has no measurement attached anywhere in the paper or in the reproducibility appendix: the AST-pattern-disabled rerun and its bug-by-bug verdict triple should be reported as a number (e.g. RP rate with the AST-pattern path disabled, on the same 60-bug corpus), since the rest of the LOO story is currently unfalsifiable.
- [reviewer, w=0.71, added round 3] The 488-block 0-unconditional-RP gap is unchanged: all 12 named LW→RP candidates remain in the "Predicted RP" column with no implementation or measurement. The smallest-cost candidate (`Tensor.unbind(dim)` fixed-length tuple, ChannelAttention/ChannelAttentionV2) has been reduced to "approximately 30 source lines of the same form as the existing split/chunk handlers" and the analyser change is described as "mechanical"; please implement it and report the resulting RP-on-488-blocks number (1/488 vs 0/488 would already be falsifiable progress on the headline).
- [reviewer, w=0.71, added round 3] 105/185 in-soundness verdicts on the 488-block corpus still touch at least one tested-only handler (§4.4). The load-bearing handlers in this bucket — `view/reshape`, `broadcast_add`, `Conv2d`-output — are exactly the ones the F3/F4 fault-injection scan exposes the most blocks to (18 and 19 respectively on the 488 surface). Promoting at least one of these into pen-and-paper soundness (the 3-handler bucket) would directly shrink the 105 number; please report which handler was promoted and the new in-soundness/tested-only split.
- [reviewer, w=0.71, added round 3] The Lean composition theorem is now mechanised on 13 operators but the 13-operator DSL still excludes the four operators on the bug paths of the post-freeze N=15 catches (`linear` is in; `view/reshape`, `permute`, `conv2d`, `einsum`, `expand` are partially in or out). The C2 contribution sentence as written ("audited end-to-end on this 13-operator fragment") understates the gap to the operators that actually fire on Table 3 catches; either tighten the contribution claim or extend the DSL to cover the union of the bug-firing handlers in Table 3 (specifically `view/reshape` + `conv2d` + `einsum` + `expand`, which would lift the DSL to ~17 operators and cover all 5 of the post-freeze headline catches).
- [reviewer, w=0.71, added round 3] The 333/2,908 (11.45%) tied-weight footprint is still not directly folded into the headline silent-error rate. The 8/8-RP runtime harness is on `gradient_checkpointing_enable()`, a *different* axis (the 5/2,908 = 0.17% slice of the same sweep). The requested measurement is: of the 333 tied-weight modules, how many does the analyser return Verified vs Abstain vs Refuted on, and on a runtime ground-truth subset of those, what is the false-verified rate? Please report that triple — the 8/8 number on the disjoint checkpointing slice does not generalise to the tied-weights slice without an explicit measurement.
- [reviewer, w=0.71, added round 3] Theorem 5's empirical surface is still uneven: the 14-module CNN-only restriction (13 SHAPE recompiles, all in-catalogue) is the falsifier-evaluation headline, but the larger 55-module audit "observed 72 in-contract recompiles, all classified as integer / SymInt specialisations (kind INT); the falsification predicate is therefore not exercised". Reporting that "the larger sample contains zero SHAPE/DTYPE/RANK guards" is a *measurement that the falsifier never fires on the surface where it could have* — please add at least one curated module on which a SHAPE guard outside `catalogue(M)` is at least *plausible* (e.g. a custom op that reads a non-catalogue shape bit), so the falsification predicate is shown to be non-vacuously evaluable on something other than the 14-module CNN curation.
- [reviewer, w=0.71, added round 3] What is the RP rate on the 60-bug corpus with the AST-pattern verification path disabled (i.e. operator-dispatch only)? Without this number the 53/60 LOO invariance is unverifiable.
- [reviewer, w=0.71, added round 3] Implement the `Tensor.unbind(dim)` fixed-length tuple shape rule (the smallest-cost candidate, ~30 LoC). What is the new RP count on the 488-block corpus, and does it flip exactly the two predicted blocks (`timm::ChannelAttention`, `timm::ChannelAttentionV2`) to RP without regression elsewhere?
- [reviewer, w=0.71, added round 3] Restricted to the 333/2,908 tied-weight modules: what is the analyser verdict triple (V/RP/Abstain) and, on a runtime-instantiable subset, the false-Verified rate against a one-step `loss.backward()` ground truth?
- [reviewer, w=0.71, added round 3] The 13-operator mechanised DSL covers `linear` and `expand` but not `view/reshape`, `conv2d`, or `einsum`, all of which fire on the Table 3 post-freeze catches. Can you extend `ag_composition_ext` to the operator union of the post-freeze catches and report the new `lake build` status?
- [reviewer, w=0.71, added round 3] Of the 105/185 in-soundness verdicts touching a tested-only handler, which single handler promotion (Lean-audited or pen-and-paper) would shrink that count the most, and what is the projected new split?
- [reviewer, w=0.71, added round 3] On the 55-module Dynamo audit where all 72 observed in-contract recompiles were INT-only, can you exhibit at least one module on which the falsification predicate (SHAPE/DTYPE/RANK guard outside `catalogue(M)`) is *capable* of firing, and confirm that the necessary direction still holds there?
- [reviewer, w=0.50, added round 2] **The "Round-2 Q3" subheader inside §4.3** ("CNN-only restriction (Round-2 Q3). Restricting the aggregate to the 10 fully end-to-end CNN-type subjects ...") narrates the revision process inside the body of the paper. A NeurIPS submission should not contain markers that name reviewer rounds or reviewer questions; please rename the paragraph to something like "CNN-only restriction" with no parenthetical and no rounding marker.
- [reviewer, w=0.50, added round 2] **Theorem 3 is still mechanised only on `{matmul, view, add}`.** The C2 contribution sentence explicitly says "the algorithmic discipline is implemented over the full 79-handler catalogue but is audited end-to-end only on the 3-operator fragment." Either extend the Lean composition proof to a non-trivial subset of the 28 already-Lean-audited operator rules (the natural next set is `{matmul, view, add, linear, bmm, permute, reshape}` — all already have rule-level Lean lemmas) and report the new mechanised count in §4.4, or remove "assume/guarantee discipline at the class boundary" from C2 and re-state C2 as the per-rule audit only.
- [reviewer, w=0.50, added round 2] **The 105/185 in-soundness verdicts touching a tested-only handler is unchanged.** No handler was promoted into Lean or pen-and-paper soundness this round. The reviewer-flagged load-bearing handlers (`view/reshape/total_size`, broadcasting, `conv_channel_mismatch`, einsum) are precisely the ones whose preservation lemma is the smallest delta to the existing Lean tree; close at least one this round and report the new in-soundness ratio.
- [reviewer, w=0.50, added round 2] **Mutation testing is still 3/50 (6%) on the 60-bug corpus only.** The Q3 obligation — rerun the same 50 mutants on the 488-block corpus and on the bug+falsification+stress union, and report the best of those numbers — has not been discharged. Please run it; this is an O(50 × N) sweep on inputs the analyser already processes.
- [reviewer, w=0.50, added round 2] **The AST-pattern-disabled 53/60 reproduction is not in the paper.** §4.1 still attributes the leave-one-out 53/60 invariance to "an independent AST-pattern verification path runs in parallel with the operator dispatch and catches the bugs even when the per-category handlers are disabled," which is exactly the confound the reviewer asked to isolate. Add a row to the rule-development-holdout subsection reporting the bug-corpus RP rate with the AST-pattern path disabled.
- [reviewer, w=0.50, added round 2] **The 488-block 0-RP number has not been moved.** The 12 named LW→RP candidates remain candidates; none have been implemented. Implementing any one of `Tensor.unbind(dim)`-fixed-length tuple (timm::ChannelAttention) or `super().forward()→Embedding` (Bart/Whisper PosEmb) is reported as a single missing rule that would unconditionally flip a specified block to RP. Implementing one of these and reporting the resulting 1/488 (or higher) unconditional-RP count would convert the falsifiable prediction into a measured number.
- [reviewer, w=0.50, added round 2] **Per-feature stress benchmark still presents the two-non-discriminating knobs (CEGAR, phase) as columns of Table 5.** The §4.2 paragraph says only three knobs move verdicts on the real corpus; please either drop the two non-discriminating columns from Table 5 or re-frame the table caption so the three-knob conclusion is the main result and the two zero-delta knobs are an explicit "shipped, did not discriminate" footnote, rather than visually presented as part of a five-knob staircase.
- [reviewer, w=0.50, added round 2] **The 333/2,908 (11.45%) tied-weights footprint is still not folded into the headline silent-error rate.** The new 8/8-RP runtime harness is reassuring on the gradient-checkpointing axis, but the requested static argument that `tied_weights_keys` / `tie_weights` / `_tie_or_clone_weights` does not produce a renamed-attribute alias the lattice misclassifies is missing. Either add one runtime model (e.g., `bert-base-uncased` with shared input/output embeddings) to the 8-model harness as a tied-weights positive case and report whether the analyser RPs or silently verifies, or include the 333 in the silent-error denominator with the resulting prevalence number.
- [reviewer, w=0.50, added round 2] For Theorem 3, is the obstacle to mechanising assume/guarantee composition on a non-trivial subset (say `{matmul, view, add, linear, bmm}`) Lean-tactic engineering, the operator-rule definitional shape, or pure labour estimate? A specific number (lemmas, days, and which lemmas are blocked on what) would clarify whether C2 can be made mechanised next round or whether it should be re-scoped now.
- [reviewer, w=0.50, added round 2] What is the bug-corpus RP rate with the AST-pattern verification path disabled? A single-row reproduction would isolate the operator-rule contribution from the pattern-matching path.
- [reviewer, w=0.50, added round 2] For the 50-mutant sweep, what are the kill rates on (i) the 488-block corpus, (ii) the 25-block falsification corpus, and (iii) the union of all three? The current 6% on the 60-bug corpus alone is a measurement of the corpus, not of the analyser.
- [reviewer, w=0.50, added round 2] For the witnessed-ratio 118/128 CV result, how do the 10 unwitnessed CVs distribute by HuggingFace family (Llama / Bart / Whisper / OPT / Falcon / GPT-NeoX / other)? Concentration in one family would weaken the user-visible CV column for that family specifically.
- [reviewer, w=0.50, added round 2] Of the 12 named LW→RP candidates, which one has the smallest implementation cost (estimated lines added to a single handler) and what is the obstacle to implementing it this round?
- [reviewer, w=0.50, added round 2] The new runtime grad-checkpointing harness reports 8/8 Refuted-Proof. Does the analyser refuse on the *presence* of `gradient_checkpointing_enable()` (a pattern-level RP) or on a specific reverse-reachable parameter analysis? If the former, please describe how the RP is justified under Theorem 2 — pattern-level RPs without a per-parameter witness are usually outside the soundness statement.
- [reviewer, w=0.35, added round 1] **Theorem 3 (compositional soundness) is mechanised on a 3-operator DSL only**, while the analyser dispatches over 79 handlers. The paper is upfront about this, but the resulting formal guarantee on actual programs is much weaker than the rule-table audit suggests: even if every leaf module's per-operator rule is Lean-audited, the compositional step that lifts per-rule soundness to the module DAG is itself audited only on `{matmul, view, add}`. The contribution C2 should either (a) extend the Lean composition proof to the full handler set or (b) rephrase the contribution to not include "assume/guarantee discipline at the class boundary" as a mechanised result.
- [reviewer, w=0.35, added round 1] **48/79 handlers are "tested-only" and outside Theorem 2.** On the 488-block corpus, 105/185 in-soundness verdicts (Section 4.4) touch at least one tested-only handler — i.e. the *majority* of verdicts on the real-source surface do not enjoy the headline soundness statement. Either close the gap on at least the load-bearing handlers (the per-handler attribution names `view/reshape/total_size`, broadcasting, `conv_channel_mismatch`, einsum, etc.) or restrict Theorem 2's stated scope to the 31 in-soundness handlers wherever it is invoked in the empirical sections.
- [reviewer, w=0.35, added round 1] **The Dynamo necessary-direction audit on the larger population is empirically empty for the kinds it is supposed to test.** Across 55 successful modules it observed 72 in-contract recompiles, *all* of kind INT (Section 4.3). Zero SHAPE/DTYPE/RANK guards were observed, so the falsification predicate ("a SHAPE/DTYPE/RANK guard on a variable outside `catalogue(M)`") evaluates to 0/0 on this surface, not 0/72. The 14-module audit does report 19 SHAPE recompiles, but 4 of those modules use the documented forward-signature surrogate. The right number is the falsifier rate restricted to the 9 fully end-to-end CNN modules (13 SHAPE events); please present that as the headline and drop the 0/72 framing entirely, since it does not test the theorem.
- [reviewer, w=0.35, added round 1] **Mutation testing is weak.** 3/50 (6%) mutant kill rate, with the surviving 47 mutants attributed to "arithmetic/comparison handler paths the 60-bug corpus does not exercise," is itself an indictment of how representative the 60-bug corpus is for the analyser's actual logic. A 6% mutation score on a verifier whose central claim is soundness should be addressed: rerun against the 488-block corpus and against the bug + falsification + stress union, and report the best of those numbers.
- [reviewer, w=0.35, added round 1] **The 60-bug corpus has unverifiable handler-development independence.** The authors describe the leave-one-out audit (category-keyword LOO is a no-op by design; handler-class LOO leaves 53/60 unchanged due to a parallel AST-pattern path) but the AST-pattern verification path is itself developed by the same authors with knowledge of the corpus. The 53/60 number cannot be cleanly attributed to the operator handlers vs. the AST-pattern shortcut. Please report 53/60 *with the AST-pattern path disabled* as a separate row, so that the operator-rule contribution is isolated from the pattern-matching contribution.
- [reviewer, w=0.35, added round 1] **The N=15 post-freeze headline is not statistically separable from baselines** (Fisher p=0.39 vs FakeTensor, p=0.68 vs Pytea), which the authors acknowledge. The pre-registered second wave (Nnew=26 / 56 / 77 depending on target) is described as a precondition, not run. Without it, the real-PR claim is a directional 5/15 vs 2/15 vs 3/15 with overlapping CIs, and the headline should be presented in that frame in the abstract too (the abstract says "5/15 catches versus 2/15 and 3/15" without the confidence intervals).
- [reviewer, w=0.35, added round 1] **Per-feature stress benchmark is anti-informative.** Table 5 explicitly notes that the real-corpus ablation is a flat line, that L1 (CEGAR) and L3 (phase) are no-ops, and that two of the five "knobs" are dead code shipped with the analyser. The stress benchmark's staircase is therefore a property of how the cases were constructed, not of the analyser. Either remove the stress benchmark entirely or report only the three discriminative knobs; the current presentation invites a misreading.
- [reviewer, w=0.35, added round 1] **The grad-flag silent-error footprint is described as ≤12% of training scripts, but the lattice is first-order and acknowledged-incorrect on parameter-sharing under renamed attributes.** The 0/2,908 AST-grep on renamed-attribute patterns is reassuring, but the 333/2,908 (11.45%) `tied_weights_keys`/`tie_weights`/`_tie_or_clone_weights` count is a first-order-lattice-breaker that is not folded into the silent-error rate. Please either (a) report the silent-error rate including the 333 tied-weights-using files or (b) provide a static argument that tied-weights via the API does *not* produce a renamed-attribute alias the lattice misclassifies.
- [reviewer, w=0.35, added round 1] For the 12 named LW→RP-conversion candidates (Section 4.1 table), what is the obstacle to implementing at least one of them and reporting the resulting unconditional-RP count on the 488-block corpus? The paper presents this as a falsifiable prediction; running the experiment would convert a prediction into a measured number.
- [reviewer, w=0.35, added round 1] In the large-population Dynamo audit (Section 4.3), why are zero SHAPE/DTYPE/RANK guards observed across 72 recompiles? Is the harness genuinely exercising shape-varying inputs, or is the SymInt specialisation pre-empting them? Please provide the per-module input-shape variation for the 55 successful modules.
- [reviewer, w=0.35, added round 1] What is the measured RP→VERIFIED flip rate of the 50 mutants on the 488-block + falsification + 25-block stress union, not just on the 60-bug corpus?
- [reviewer, w=0.35, added round 1] Can you provide a 53/60 reproduction with the AST-pattern verification path disabled, so the operator-rule contribution is isolated from the pattern-matching contribution?
- [reviewer, w=0.35, added round 1] For Theorem 3, is there a concrete obstacle to mechanising assume/guarantee composition on more than 3 operators (proof engineering vs Lean-tactic limit vs operator-rule definitional shape), or is it purely a labour estimate?
- [reviewer, w=0.35, added round 1] How does the witnessed-ratio of 118/128 CV verdicts (Section 4.1) decompose by Hugging Face submodel family — i.e. are the 10 unwitnessed CVs concentrated in one architecture, or spread? This bears directly on whether the user-visible CV column is meaningful in practice.

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

Round: 4
