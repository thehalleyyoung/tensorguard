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

Adding a live end-to-end demonstration that `check_devices`, `check_phases`, and `check_gradients` actually flip verdicts on at least one committed real-source example — backed by a JSON artifact — would convert C5's "5-theory product domain" from a documented no-op on real corpora to a demonstrated contribution; this single change would push the overall score to 6 by substantiating the tool's architectural narrative with evidence rather than stress-benchmark proxies.
Changes   +0 -0
Requests  1 Premium (3m 27s)
Tokens    ↑ 322.6k • ↓ 9.1k • 273.8k (cached)

**Sub-score-targeted primary work (target dimension: SOUNDNESS = 2/4).** Of the four scored sub-dimensions, soundness is currently the binding constraint on Overall. Concentrate this round's non-escalated effort on raising it from 2 to 3. Concrete actions you may pick from (do AT LEAST one in full this round; sketching does not count):
  - tighten / formalize a key theorem; if a Lean / Coq / Mathematica skeleton exists, close one open lemma in this round
  - replace a proof-by-figure or sketch with a numbered theorem + proof
  - state every regularity/assumption explicitly and verify the constants in code
  - run one extra experimental seed and report the variance to defuse 'might be cherry-picked' concerns

## Speculative wins from last round's exploration phase
Last round's exploration phase produced these speculative wins. The next reviewer will see the paper after they have been folded in. Treat each as a planned, intentional extension of the work and integrate it into the paper naturally — into the contributions list, the experiments section, the related work, or the appendix as appropriate. Do NOT mark them as 'experimental', 'tentative', or 'in exploration'. They are part of the paper now.

  - Forward device/phase/gradient checks through the public API and CLI — device/phase/gradient check flags are now pinned by 25 tests covering API signatures, runtime behaviour, CLI flag presence, and end-to-end subprocess invocations with exit-code validation
  - Mechanize the broadcast_add operator lemma in Lean to retire one operator-agnostic axiom — applyOpExt_sound_broadcast_add is mechanised sorry-free in Lean, retiring the operator-agnostic axiom for broadcast_add so only matmul remains under the agnostic witness

## Risk bias (how to spend the round budget)

The harness runs an isolated **exploration phase** after this main
improver pass. In that phase, you (via subagents) brainstorm and
attempt 2+ bold extensions on git-snapshotted scratch state; failures
are reverted silently and the reviewer never sees them. So here, in
the main improver pass, you should:

  * **Default to ambitious bets, not polish.** A single bold,
    successful extension (a new model family added to the harness, a
    new ablation cell, a new theorem closed in Lean, a new dataset
    benchmarked) typically moves Soundness or Contribution by a full
    point. Three rounds of paragraph polish move Presentation by at
    most 1 and never move Soundness or Contribution.

  * **Treat the exploration phase as expected EV-positive.** Even
    if half the bets fail, the harness reverts them with no cost to
    the visible paper or repo. You only ever ship the wins.
    Therefore the rational policy is to PROPOSE risky bets, not to
    pre-self-censor them.

  * **Do not try to "balance the round" between safe and bold.**
    The mandatory per-round work that addresses reviewer weaknesses
    is the safe baseline; everything *additional* should be a
    deliberate stretch. If you find yourself reaching for a small
    cosmetic improvement, instead reach for the largest extension
    that could plausibly land in 10 minutes of subagent work.

  * **Report in the paper only the wins.** Failed attempts must NOT
    appear in the .tex source, in the abstract, in the limitations,
    in the appendix, or in any reviewer-visible artifact. Internal
    notes on what was tried-and-reverted go in the harness's
    exploration log under `.comet_neurips/` only --- the reviewer
    will never read it. (If the speculative-wins block above is
    non-empty, fold those wins into the paper as if they were
    planned all along.)

## Latest reviewer report
## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that verifies tensor shapes and gradient flow statically from class source, without instantiating or tracing the module. The paper formalises a refinement-type calculus Tensor{s,g|φ} unifying shape and gradient-flag analysis under Z3, proves a soundness theorem over a 44-handler sub-catalogue Catsound (17 operators mechanised in Lean 4, 17/17 sorry-free lemmas, 16 pen-and-paper), and reports 53/60 REFUTED-PROOF on a curated 60-bug corpus, 32/34 vs. Pytea (McNemar p=0.0156) on a fragment-fair subset, 9/9 naturally-occurring cross-family HuggingFace bugs, and 15/488 (3.07%) unconditional REFUTED-PROOF on a 488-block real-source corpus. The paper is explicit that the analyser implementation, AST extractor, and backward verifier sit in a trusted computing base outside the Lean audit. The primary claimed contribution over execution-based tools is applicability to constructor-argument-dependent class source that cannot be instantiated without a full configuration.

## Prior weakness disposition

- [RESOLVED] On the fairest directly comparable bug subset, the strongest maintained baseline is actually `torch.compile`, which catches 34/34 while TG catches 32/34 -- Rebuttal accepted: the paper clearly delineates the two regimes (execution vs. no-execution); `torch.compile` requires concrete inputs and a traceable module, TG operates on un-instantiated class source; the 32/34 vs. 25/34 head-to-head explicitly excludes `torch.compile` because it belongs to the different regime; the regime asymmetry is now documented in the abstract.
- [PARTIAL] The user-visible real-source result remains weak: on the 488-block corpus the free-symbolic regime yields 0 unconditional RP -- The paper now reports 26/356 unconditional RP on the zero-contract-obligation subset and 15/488 overall, which is non-zero, but 3.07% RP rate with 418/488 abstains still means the tool provides almost no signal on the bulk of real-source code; the improvement from 0 to 15–26 is real but the practical yield remains very low.
- [UNRESOLVED] The main 53/60 number is still driven by a historically mined and filtered corpus; the newer pre-registered unfiltered post-freeze sample is only 5/15, with wide intervals and no statistically separable advantage over FakeTensorMode or Pytea -- The 5/15 result (Wilson 95% CI [15.2%, 58.3%]) is still reported in the paper with explicit acknowledgment that the CIs overlap with baselines (FakeTensorMode/torch.compile at 2/15); the paper states "not a separation"; no additional real-PR corpus evidence is offered.
- [PARTIAL] The soundness footprint on real-source verdicts is still limited: only 62/185 in-soundness verdicts touch handlers entirely inside the Lean-or-pen-paper audited footprint -- Rebuttal partially accepted: the floor/ceiling argument is methodologically sound (any verdict touching one tested-only handler drops out entirely), and the rebuttal notes Catsound concentrates on high-frequency operators; however, the 62/185 number has not moved and the claimed "propagation of audited-handler coverage across composition" is not demonstrated empirically on the 123 remaining verdicts; the gap between 44/79 audited handlers and the 57/185 verdicts touching only outside-any-scope handlers remains a concrete concern.
- [PARTIAL] The public artifact surface still looks immature relative to the paper's architectural narrative: the README states that `check_devices`, `check_phases`, and `check_gradients` are currently not forwarded by the public API/CLI -- The README now explicitly documents this limitation under "Known limitations." However, the feature_ablation.json metadata notes "check_devices, check_phases, check_gradients are accepted by the API but NOT forwarded to verify_model in the current implementation; L2/L3/L4 rows therefore replicate L1 verdict counts," meaning the paper's advertised "5-theory product domain" (Shape × Device × Phase × Stride × Permutation) produces no device, phase, or gradient-specific verdicts on either real corpus; the per-feature ablation confirms this is not just a reporting choice but an implementation gap.

## Strengths

- **Genuine regime contribution**: The no-execution, un-instantiated class-source regime is a real gap in the ecosystem. The ≥ 435/488 mechanical N/A of all execution-based baselines on the 488-block corpus is a structural fact that strongly motivates the approach.
- **Carefully scoped theoretical claims**: The paper gives explicit TCB declarations, partitions verdicts by handler-audit tier (Lean/pen-and-paper/tested-only), restricts the soundness theorem to Catsound, and reports ABSTAIN as a first-class outcome. This level of epistemic hygiene is uncommon and commendable.
- **Compelling naturally-occurring bug evidence**: 9/9 REFUTED-PROOF on real HuggingFace PR/issue repros across five decoder families (Llama, Qwen2, Mistral, Phi-3, Gemma 2), with per-PR citations, is the most externally-valid result in the paper.
- **Reproducible artifacts**: `verify_neurips_revision.py` runs to completion and corroborates the revision headlines; benchmark JSON artifacts are committed; the Lean development builds sorry-free under `lake build`.
- **Honest calibration**: The paper reports 418/488 abstains, explicit CI intervals on every headline, the qkv known false-positive, and the silent-verified gap on 2/10 upstream-faithful bugs. These are not buried in appendices.

## Weaknesses

- **The CEGAR contribution (C5) is effectively unimplemented on the real corpora.** `feature_ablation.json` explicitly documents: "CEGAR predicates are stored as metadata only (not fed back as Bug objects). check_devices, check_phases, check_gradients are accepted by the API but NOT forwarded to verify_model in the current implementation; L2/L3/L4 rows therefore replicate L1 verdict counts." The 25-case stress benchmark activates these knobs, but the paper's own ablation on the real corpora produces a flat line. C5 as stated ("CEGAR predicate discovery … discovers shape predicates automatically") and the claimed "5-theory product domain" are misleading characterisations of what the shipped tool actually does on any real-world input, and the paper's disclosure of this (README Known Limitations section) is too understated relative to the abstract's claims.
- **The pre-registered unfiltered corpus (Table 3, 5/15) provides no statistical separation from baselines.** The paper states this explicitly ("not a separation") and provides a power calculation, but then the abstract and Section 4.1 headline the 53/60 curated figure without a comparable disclaimer. The curated corpus was constructed by historical triage of known bugs; the 5/15 result on the one corpus collected without that foreknowledge is the cleanest unbiased estimate of real-world utility, and its Wilson 95% CI [15.2%, 58.3%] overlaps completely with the execution-based baselines' 2/15 (CI [3.8%, 40.7%]).
- **test_config_qkv_upgrade.py is a known-failing test that must be explicitly ignored.** The prior round's experiment log shows this test was skipped with `--ignore=tests/test_config_qkv_upgrade.py` to get a passing suite. A reproducible artifact should not require ignoring a test that presumably validates a real analysis behaviour. Neither the paper nor the README explains what this test exercises or why it fails.
- **The backward verifier's gradient-flow analysis (C3) is claimed "8/8 canonical bugs caught, 0/50 false positives" but the eval corpus is entirely synthetic.** The 8 canonical bug classes and 50 clean scripts are author-authored; there is no third-party or natural-occurrence validation analogous to the 9/9 cross-family HF result for shape bugs. The ≤ 3.0% false-Verified bound on "regex-screened training-script population" depends on a regex screen that is not defined or validated in the paper's main body.
- **The 57/185 verdicts touching only handlers outside any soundness scope (Table 8 bottom row) are not discussed defensively.** These are not "tested-only" — they are outside all three tiers (Lean, pen-and-paper, tested-only). This is acknowledged with "TCB obligation explicitly tracked" but the paper does not report whether these 57 verdicts are concentrated in particular operator families, which would let the reader gauge whether the theoretical apparatus covers the majority of practical fires.

## Questions

- The feature ablation JSON metadata states that `check_devices`, `check_phases`, and `check_gradients` "are accepted by the API but NOT forwarded to verify_model." Does the paper intend to claim device, phase, and gradient analysis as live, callable contributions, or are these planned features? If the former, which specific commits implement the forwarding, and which benchmark script exercises them end-to-end?
- What does `test_config_qkv_upgrade.py` test, why is it currently failing, and can the authors either fix it or remove it and explain what real behaviour it was meant to validate?
- On the 5/15 pre-registered corpus (Table 3), the paper cites a power calculation "conditioning on 5/15 for TG, 2/15 for baselines." What sample size N would be required to achieve 80% power to reject H₀ under these observed proportions, and is there a concrete plan to collect it?
- The abstract claims "26 unconditional REFUTED-PROOF verdicts on the 356-block subset whose contract obligation is empty." Is there a single command (without cached benchmarks) that regenerates this 26/356 number from the committed corpus, analogous to `python3 verify_neurips_revision.py`?

## Scores

Soundness: 2
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 5

## Borderline reasons

Adding a live end-to-end demonstration that `check_devices`, `check_phases`, and `check_gradients` actually flip verdicts on at least one committed real-source example — backed by a JSON artifact — would convert C5's "5-theory product domain" from a documented no-op on real corpora to a demonstrated contribution; this single change would push the overall score to 6 by substantiating the tool's architectural narrative with evidence rather than stress-benchmark proxies.


Changes   +0 -0
Requests  1 Premium (3m 27s)
Tokens    ↑ 322.6k • ↓ 9.1k • 273.8k (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 2, streak=1] The main 53/60 number is still driven by a historically mined and filtered corpus; the newer pre-registered unfiltered post-freeze sample is only 5/15, with wide intervals and no statistically separable advantage over FakeTensorMode or Pytea.
- [reviewer, w=1.00, added round 3, streak=0] **The CEGAR contribution (C5) is effectively unimplemented on the real corpora.** `feature_ablation.json` explicitly documents: "CEGAR predicates are stored as metadata only (not fed back as Bug objects). check_devices, check_phases, check_gradients are accepted by the API but NOT forwarded to verify_model in the current implementation; L2/L3/L4 rows therefore replicate L1 verdict counts." The 25-case stress benchmark activates these knobs, but the paper's own ablation on the real corpora produces a flat line. C5 as stated ("CEGAR predicate discovery … discovers shape predicates automatically") and the claimed "5-theory product domain" are misleading characterisations of what the shipped tool actually does on any real-world input, and the paper's disclosure of this (README Known Limitations section) is too understated relative to the abstract's claims.
- [reviewer, w=1.00, added round 3, streak=0] **The pre-registered unfiltered corpus (Table 3, 5/15) provides no statistical separation from baselines.** The paper states this explicitly ("not a separation") and provides a power calculation, but then the abstract and Section 4.1 headline the 53/60 curated figure without a comparable disclaimer. The curated corpus was constructed by historical triage of known bugs; the 5/15 result on the one corpus collected without that foreknowledge is the cleanest unbiased estimate of real-world utility, and its Wilson 95% CI [15.2%, 58.3%] overlaps completely with the execution-based baselines' 2/15 (CI [3.8%, 40.7%]).
- [reviewer, w=1.00, added round 3, streak=0] **test_config_qkv_upgrade.py is a known-failing test that must be explicitly ignored.** The prior round's experiment log shows this test was skipped with `--ignore=tests/test_config_qkv_upgrade.py` to get a passing suite. A reproducible artifact should not require ignoring a test that presumably validates a real analysis behaviour. Neither the paper nor the README explains what this test exercises or why it fails.
- [reviewer, w=1.00, added round 3, streak=0] **The backward verifier's gradient-flow analysis (C3) is claimed "8/8 canonical bugs caught, 0/50 false positives" but the eval corpus is entirely synthetic.** The 8 canonical bug classes and 50 clean scripts are author-authored; there is no third-party or natural-occurrence validation analogous to the 9/9 cross-family HF result for shape bugs. The ≤ 3.0% false-Verified bound on "regex-screened training-script population" depends on a regex screen that is not defined or validated in the paper's main body.
- [reviewer, w=1.00, added round 3, streak=0] **The 57/185 verdicts touching only handlers outside any soundness scope (Table 8 bottom row) are not discussed defensively.** These are not "tested-only" — they are outside all three tiers (Lean, pen-and-paper, tested-only). This is acknowledged with "TCB obligation explicitly tracked" but the paper does not report whether these 57 verdicts are concentrated in particular operator families, which would let the reader gauge whether the theoretical apparatus covers the majority of practical fires.
- [reviewer, w=1.00, added round 3, streak=0] The feature ablation JSON metadata states that `check_devices`, `check_phases`, and `check_gradients` "are accepted by the API but NOT forwarded to verify_model." Does the paper intend to claim device, phase, and gradient analysis as live, callable contributions, or are these planned features? If the former, which specific commits implement the forwarding, and which benchmark script exercises them end-to-end?
- [reviewer, w=1.00, added round 3, streak=0] What does `test_config_qkv_upgrade.py` test, why is it currently failing, and can the authors either fix it or remove it and explain what real behaviour it was meant to validate?
- [reviewer, w=1.00, added round 3, streak=0] On the 5/15 pre-registered corpus (Table 3), the paper cites a power calculation "conditioning on 5/15 for TG, 2/15 for baselines." What sample size N would be required to achieve 80% power to reject H₀ under these observed proportions, and is there a concrete plan to collect it?
- [reviewer, w=1.00, added round 3, streak=0] The abstract claims "26 unconditional REFUTED-PROOF verdicts on the 356-block subset whose contract obligation is empty." Is there a single command (without cached benchmarks) that regenerates this 26/356 number from the committed corpus, analogous to `python3 verify_neurips_revision.py`?

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

Round: 3
