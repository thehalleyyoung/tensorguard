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
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically reasons about symbolic shapes and a flat first-order grad-flag lattice (`{has_grad, no_grad, ⊤}`), discharges side conditions to Z3, and emits a five-way verdict taxonomy. Headline empirics (unchanged from R3): 53/60 RP on the historical corpus; 32/34 vs. Pytea 22/34 on the fragment-fair modern subset (McNemar exact p=0.00195); 5/15 catches on the unfiltered post-freeze N=15 PR sample (vs. FT 2/15, Pytea 3/15, Fisher non-separable; the headline now disclaims rather than leans on the Bayesian supplement); 0 unconditional RP on the 488-block real-source corpus under free-symbolic configs. The R4 revision adds: full-128 joint-realisability check 118/128 (92.2%, Clopper-Pearson [86.1%, 96.2%]); a per-block table for all 12/78 LW→RP residuals naming the single missing rule per block; a held-out HF `examples/pytorch/` Trainer audit at 1/42 (2.4%) silent-error positives; and a TCB single-fault verdict-flip exposure scan (F1 0/60, F2 0/60, F3 2/60, F4 7/60). The Theorem 5 end-to-end Dynamo audit, however, remains at ~31 modules; the n=100 attempt shipped in `reproducibility/dynamo_theorem5_n100.py` reports 0 successful modules / 112 excluded, so no ≥100-block instantiation lands in the paper.

## Prior weakness disposition
- [RESOLVED] The post-freeze unfiltered evaluation is still N=15 (Section 4.1, Table 3). The BF₁₀=8.1 vs. FakeTensorMode and BF₁₀=3.6 vs. Pytea both sit in the "moderate"... -- The Bayesian supplement is no longer leaned on for the headline; §4.1 now explicitly states "we do not rely on a Bayesian supplement to upgrade the claim" and reports the comparison as "point above, not statistically separable at α=0.05 on N=15", which is exactly the prior reviewer's offered alternative path.
- [RESOLVED] The 488-block CV joint-realisability evidence (§4.1) is still the 12-of-128 random-sample audit (~9.4%) with named `*Config`-default instantiations and published checkpoints... -- §4.1 now reports the joint-realisability check on the full N=128 CV set: each row's full `assume_M` conjunction is evaluated against a default `*Config()` of its natural caller, yielding 118/128 (92.2%) witnessed with Clopper-Pearson 95% CI [86.1%, 96.2%], with the 10 non-witnessed rows characterised as `*PreTrainedModel` stubs / aliasing-only constructors rather than actual contradictions.
- [UNRESOLVED] The Theorem 5 empirical audit is still ~31 modules total (17 original + 14 extended; §4.3 / "Extended end-to-end audit"). Of the 14 extended blocks, 4 transformer blocks are audited via the documented... -- §4.3 still reports the same 17-module + 14-module corpus (with 4 of 14 transformer blocks on the forward-signature surrogate); the `dynamo_theorem5_n100.py` artifact ran 112 candidates and produced 0 successful modules (all excluded on build/warmup/timeout), so no ≥100-block end-to-end falsifier evaluation appears in the paper.
- [RESOLVED] The grad-flag silent-error audit (§6) reports `0/16 torch.utils.checkpoint` and `0/16` renamed-attribute parameter sharing on the 16 importable Track-E modules — the same fixture used elsewhere in the... -- A held-out audit on a disjoint population of 42 PyTorch training scripts under `examples/pytorch/` of `huggingface/transformers` reports 1/42 (2.4%) silent-error positives (`torch.utils.checkpoint`, `gradient_checkpointing_enable`, or renamed-attribute sharing), well within the ≤12% ceiling and folded into both §4.1 and §6.
- [RESOLVED] The "12/78 catalogue-coverage residual" bound on the LW→RP gap (§4.1) is asserted as an upper bound but not exhibited per-block. Without a list of which 12 of the 78 LW blocks would convert to RP unde... -- §4.1 now contains a 12-row table enumerating each residual block (e.g. `tv::InvertedResidual`, `tv::LayerNorm2d`, `timm::ChannelAttention`, `tx::WhisperPosEmb`, `tx::FalconLinear`, ...) paired with the single missing operator-rule whose addition would (in isolation) flip its verdict to unconditional RP, making the 12/78 ceiling falsifiable from the paper alone.
- [RESOLVED] Theorem 1 (fragment-level soundness) and Theorems 10/11 (Preservation/Progress) are pen-and-paper, while Theorem 3 (compositional/assume-guarantee) is mechanised only on a 3-operator DSL via `lemma ag... -- §4.4 / eval now contains a TCB fault-injection footprint that bounds the verdict-flip a single deliberate fault in any held-out TCB component could induce on the headline corpora; the audited single faults give 0/60 (F1 view-star), 0/60 (F2 add_), 2/60 (F3 cat-dim), 7/60 (F4 Conv2d) on the 60-bug corpus, calibrating what the 53/60 RP headline actually depends on at the implementation layer.

## Strengths
- The Round-4 revision is the first round in this loop where the symmetric-scoring criterion is clearly satisfied: five of six prior weaknesses are addressed with non-trivial new measurement (full-128 joint-realisability with Clopper-Pearson CI, per-block 12/78 table with named missing rules, held-out 1/42 HF Trainer audit, TCB fault-injection footprint with per-fault exposure on both headline corpora), and the one item not addressed (W3) is conceded by simply not advancing the corresponding paper text rather than being papered over.
- The N=15 retreat from the Bayesian supplement is the methodologically right move: §4.1 now reads "we do not rely on a Bayesian supplement to upgrade the claim" and explicitly carries "TG strictly above ... not statistically separable at α=0.05 on N=15" as the headline. This converts the comparison from a soft Bayesian over-reach into a calibrated point-above claim, which is what the corpus actually licenses.
- The TCB fault-injection footprint is a substantive addition to the soundness story: it makes the prior abstract caveat ("the analyser implementation, AST extractor, backward verifier, and Z3 dispatch are not mechanised") quantitatively bounded — under the worst single audited fault (F4, Conv2d off-by-one) the 53/60 headline could degrade by at most 7 to 46/60, and only on the conv-channel-mismatch slice. This is the kind of fault-locality calibration that is usually missing in static-analysis papers.
- The full-128 joint-realisability witness (118/128, 92.2%, Clopper-Pearson [86.1%, 96.2%], with the 10 non-witnessed rows characterised as `*PreTrainedModel` stubs / aliasing-only constructors rather than contradictions) closes the largest remaining hole in the 488-block story. The CV bucket can no longer be dismissed as "synthesised assumes that might never fire".
- The per-block 12/78 LW→RP table — concrete blocks paired with the single missing operator-rule (`tv::InvertedResidual` ← const-attr-gated branch; `timm::ChannelAttention` ← `unbind(dim)` tuple-shape; `tx::WhisperPosEmb` ← slice-getitem; etc.) — turns an unfalsifiable upper bound into a checkable engineering roadmap.

## Weaknesses
- The Theorem 5 end-to-end audit remains at ~31 modules total (17 + 14, with 4 of 14 still on the forward-signature surrogate). The `dynamo_theorem5_n100.py` script attempts a 112-candidate run but the artefact (`dynamo_theorem5_n100.{json,md}`) records "Successful modules: 0 / Excluded (build/warmup/timeout): 112"; no ≥100-block evaluation of the falsifier predicate (`r.guard_kind ∈ {SHAPE, DTYPE, RANK} ∧ r.guard_var ∉ catalogue(M)`) lands in the paper. Theorem 5 is the central PL-side claim about Dynamo and it is still being validated on roughly a third of the ≥100-module target. Please diagnose the build/warmup/timeout exclusion path on `dynamo_theorem5_n100.py` (the run aborts on every candidate, so this is not a corpus problem — likely a contract-construction or warmup-budget bug in the harness), then ship the `{SHAPE, DTYPE, RANK, INT}` recompile breakdown and the falsifier rate on a fully-instantiated ≥70 additional importable timm/HF blocks.
- Footnote on the held-out HF Trainer audit: 1/42 = 2.4% is a clean number, but it conflates "construct present" with "silent verdict-flip on a class TG would otherwise verify". The audit measures script-level construct exposure (G1∨G2∨G6), not "TG verifies a module to which one of these constructs applies and that verdict is wrong against runtime `p.grad ≠ None`". The ≤12% ceiling in §6 is about prevalence of the construct, so the audit is internally consistent, but the *false-verified-rate* the prior round asked for — measured against runtime `p.grad ≠ None` on the held-out scripts — is not what 2.4% measures. Please either (i) re-run the held-out script subset against runtime grad equality and report the false-verified-rate directly, or (ii) clarify in §6 that the held-out audit is a held-out *construct-prevalence* check (which it is) and not a held-out false-verified-rate.
- The N=15 post-freeze surface is now correctly described, but the structural problem — only N=15 unfiltered post-freeze observations exist, of which 1 is an off-axis false positive — caps how much weight any reader should put on the "TG point-above" framing. The pre-registration query (frozen 2026-04-08) is reusable: extending the same query window forward another month would generate ~10–15 fresh items at the rate the original sample suggests. A pre-registered N≥30 second wave would convert the "above-but-not-separable" claim into a (likely) Fisher-significant separation against FT and would not require rebuilding the methodology. The paper does not say whether such an extension is being run.
- The TCB fault-injection footprint is a conservative upper bound (exposure ≥ flip), not a measured flip rate. Under F4 the bound is "≤7 RP could flip to silent V on the 60-bug corpus" — but the paper does not exhibit the actual flip count on a deliberately-injected F4 build. A 30-line patch that changes the Conv2d output formula by ±1 in the analyser handler, re-runs the 60-bug corpus, and reports the *measured* RP→V flip, would convert the F4 line from "≤7" to a tight number and would close the gap between exposure and flip. The same applies to F1–F3.
- The 28-of-79 Lean handler audit (Table 7) is unchanged, and the explicit TCB list ("the analyser implementation, AST extractor, backward verifier, and Z3 dispatch") still covers the user-facing path on every block. The TCB fault-injection footprint partially calibrates the consequences (above), but only for four hand-chosen fault classes; an automated mutation-testing sweep (e.g. AST-rewrite-driven mutation of `model_checker.py` followed by 60-bug regression) on, say, 50 mutants would give a kill-rate that actually quantifies analyser-level robustness rather than four anecdotal faults. This is not asked for as a blocker, but it is the natural next instrument.

## Questions
- The `dynamo_theorem5_n100.py` artefact reports 0/112 successful modules (all excluded on build/warmup/timeout). What is the per-stage exclusion breakdown (import error, contract-construction failure, warmup OOM, compile timeout, recompile-event collection failure)? Is the harness actually measuring 0 successes because none of 112 candidates survive warmup, or because of a single shared bug in contract construction?
- For the held-out HF Trainer audit (1/42 = 2.4%), what is the false-verified-rate on the same 42 scripts when measured against runtime `p.grad ≠ None` after one optimiser step (rather than against script-level construct presence)? Even if the answer is also small, it is the directly comparable number to the ≤12% ceiling.
- Of the 12 LW→RP residual blocks listed, how many of the named single missing rules are already in the catalogue *roadmap* (i.e. would be added in a hypothetical v2 of TG), and how many require fragment-extension work outside the current well-typed-operator-rule discipline?
- Under the F4 fault (Conv2d off-by-one), what is the *measured* RP→V flip on the 60-bug corpus, and does it equal the 7/60 exposure upper bound or fall below it? A measured-flip number (not just the exposure ceiling) would close the soundness-implementation gap quantitatively.
- For the N=15 post-freeze surface, is a pre-registered N≥30 second wave under the same GitHub-search query feasible before camera-ready, and on the observed point estimates (TG 5/15, FT 2/15, Pytea 3/15) what is the smallest second-wave N at which the union (N=15 + new N) yields Fisher p<0.05 on at least one of the two pairwise comparisons?
- The `Bug.message` forensics scan and the 2022 Pytea-catalogue restriction make the modern-subset 32/34 vs 22/34 head-to-head fragment-fair. Do you have a mirror-experiment in which TG is restricted to the *2024* Pytea-catalogue intersection on the same modern subset, to verify that the +29.4 pp gap is not specifically a 2022-catalogue artefact?

## Scores
Soundness: 3
Presentation: 4
Contribution: 3
Confidence: 4
Overall: 7

## Borderline reasons
The remaining gap that would lift this to 8 is the same one that has been on the queue for two rounds: a working Theorem 5 end-to-end audit on ≥100 fully-instantiated importable blocks (the `dynamo_theorem5_n100.py` script needs to be fixed so it produces non-zero successful modules, and the falsifier rate plus `{SHAPE, DTYPE, RANK, INT}` recompile breakdown reported in the paper). With that single addition the paper would have closed every Round-3 weakness with measurement rather than concession.


Changes   +0 -0
Requests  7.5 Premium (4m 30s)
Tokens    ↑ 1.3m • ↓ 13.7k • 1.3m (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 4] On the post-freeze unfiltered surface, what is the smallest N at which the observed point estimates (TG 1/3 vs. FT 2/15, Pytea 1/5) would yield BF₁₀≥10 and Fisher p<0.05 on at least one of the two pairwise comparisons? If extending the pre-registered query to that N is feasible, do so and report.
- [reviewer, w=1.00, added round 4] The Theorem 5 end-to-end audit remains at ~31 modules total (17 + 14, with 4 of 14 still on the forward-signature surrogate). The `dynamo_theorem5_n100.py` script attempts a 112-candidate run but the artefact (`dynamo_theorem5_n100.{json,md}`) records "Successful modules: 0 / Excluded (build/warmup/timeout): 112"; no ≥100-block evaluation of the falsifier predicate (`r.guard_kind ∈ {SHAPE, DTYPE, RANK} ∧ r.guard_var ∉ catalogue(M)`) lands in the paper. Theorem 5 is the central PL-side claim about Dynamo and it is still being validated on roughly a third of the ≥100-module target. Please diagnose the build/warmup/timeout exclusion path on `dynamo_theorem5_n100.py` (the run aborts on every candidate, so this is not a corpus problem — likely a contract-construction or warmup-budget bug in the harness), then ship the `{SHAPE, DTYPE, RANK, INT}` recompile breakdown and the falsifier rate on a fully-instantiated ≥70 additional importable timm/HF blocks.
- [reviewer, w=1.00, added round 4] Footnote on the held-out HF Trainer audit: 1/42 = 2.4% is a clean number, but it conflates "construct present" with "silent verdict-flip on a class TG would otherwise verify". The audit measures script-level construct exposure (G1∨G2∨G6), not "TG verifies a module to which one of these constructs applies and that verdict is wrong against runtime `p.grad ≠ None`". The ≤12% ceiling in §6 is about prevalence of the construct, so the audit is internally consistent, but the *false-verified-rate* the prior round asked for — measured against runtime `p.grad ≠ None` on the held-out scripts — is not what 2.4% measures. Please either (i) re-run the held-out script subset against runtime grad equality and report the false-verified-rate directly, or (ii) clarify in §6 that the held-out audit is a held-out *construct-prevalence* check (which it is) and not a held-out false-verified-rate.
- [reviewer, w=1.00, added round 4] The N=15 post-freeze surface is now correctly described, but the structural problem — only N=15 unfiltered post-freeze observations exist, of which 1 is an off-axis false positive — caps how much weight any reader should put on the "TG point-above" framing. The pre-registration query (frozen 2026-04-08) is reusable: extending the same query window forward another month would generate ~10–15 fresh items at the rate the original sample suggests. A pre-registered N≥30 second wave would convert the "above-but-not-separable" claim into a (likely) Fisher-significant separation against FT and would not require rebuilding the methodology. The paper does not say whether such an extension is being run.
- [reviewer, w=1.00, added round 4] The TCB fault-injection footprint is a conservative upper bound (exposure ≥ flip), not a measured flip rate. Under F4 the bound is "≤7 RP could flip to silent V on the 60-bug corpus" — but the paper does not exhibit the actual flip count on a deliberately-injected F4 build. A 30-line patch that changes the Conv2d output formula by ±1 in the analyser handler, re-runs the 60-bug corpus, and reports the *measured* RP→V flip, would convert the F4 line from "≤7" to a tight number and would close the gap between exposure and flip. The same applies to F1–F3.
- [reviewer, w=1.00, added round 4] The 28-of-79 Lean handler audit (Table 7) is unchanged, and the explicit TCB list ("the analyser implementation, AST extractor, backward verifier, and Z3 dispatch") still covers the user-facing path on every block. The TCB fault-injection footprint partially calibrates the consequences (above), but only for four hand-chosen fault classes; an automated mutation-testing sweep (e.g. AST-rewrite-driven mutation of `model_checker.py` followed by 60-bug regression) on, say, 50 mutants would give a kill-rate that actually quantifies analyser-level robustness rather than four anecdotal faults. This is not asked for as a blocker, but it is the natural next instrument.
- [reviewer, w=1.00, added round 4] The `dynamo_theorem5_n100.py` artefact reports 0/112 successful modules (all excluded on build/warmup/timeout). What is the per-stage exclusion breakdown (import error, contract-construction failure, warmup OOM, compile timeout, recompile-event collection failure)? Is the harness actually measuring 0 successes because none of 112 candidates survive warmup, or because of a single shared bug in contract construction?
- [reviewer, w=1.00, added round 4] For the held-out HF Trainer audit (1/42 = 2.4%), what is the false-verified-rate on the same 42 scripts when measured against runtime `p.grad ≠ None` after one optimiser step (rather than against script-level construct presence)? Even if the answer is also small, it is the directly comparable number to the ≤12% ceiling.
- [reviewer, w=1.00, added round 4] Of the 12 LW→RP residual blocks listed, how many of the named single missing rules are already in the catalogue *roadmap* (i.e. would be added in a hypothetical v2 of TG), and how many require fragment-extension work outside the current well-typed-operator-rule discipline?
- [reviewer, w=1.00, added round 4] Under the F4 fault (Conv2d off-by-one), what is the *measured* RP→V flip on the 60-bug corpus, and does it equal the 7/60 exposure upper bound or fall below it? A measured-flip number (not just the exposure ceiling) would close the soundness-implementation gap quantitatively.

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

Round: 4
