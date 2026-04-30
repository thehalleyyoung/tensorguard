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
**ESCALATED OBLIGATIONS (highest priority).** The reviewer has rejected your last 2+ attempts on the following items. You may not paraphrase or hand-wave further. For each one, either ship the missing artifact (code, baseline, ablation, Lean theorem, or empirical result) THIS round, OR remove the disputed claim from the abstract and contributions list. Pick one per item. Do not let a third round pass with the same item still PARTIAL/UNRESOLVED.

  - (streak=2) Last round you only edited paper-side files (.tex/.bib/.md) and shipped no code, no new data artifact, no Lean theorem, no new figure regenerated from a script. The reviewer's weaknesses require substantive evidence, not paraphrase. This round, EITHER ship the requested artifact via spawn_sonnet_subagent.sh (preferred), OR remove the disputed claim from the abstract+contributions list.

**Reviewer-stated single change to push Overall up by 1.** Spend the first half of your round budget exclusively on this. Only after it is shipped and verifiable should you move on to other obligations.

Adding second-wave post-freeze data (wave-2, N_new ≥ 26) with at least one statistically separable comparison at α=0.05 would push the score to 7: the core formal contributions and calibrated methodology already meet the NeurIPS bar, but the post-freeze real-world evaluation is the only experiment on an unbiased naturalistic sample and its current non-significant N=15 result weakens the claim that TG is a practically useful bug-finder on real code rather than a demonstrably sound but abstention-heavy analyser.
Changes   +0 -0
Requests  1 Premium (3m 1s)
Tokens    ↑ 435.1k • ↓ 7.8k • 360.2k (cached)

## Latest reviewer report
## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically verifies tensor shapes and gradient flow without instantiating or tracing the module. Its type system is `Tensor{s, g | φ}` where `φ` is a Z3-decidable predicate over symbolic shape variables and a flat three-value grad lattice. The system contributes: (C1) a refinement calculus with classical Preservation/Progress; (C2) an assume/guarantee composition discipline mechanised in Lean 4 on a 17-operator DSL with 11/11 soundness lemmas closed sorry-free; (C3) a backward verifier with 500/500 static-runtime agreement; (C4) a necessary-direction Dynamo-guard correspondence (Theorem 5); and (C5–C6) a 488-block + 60-bug benchmark with calibrated five-verdict taxonomy. On the 60-bug historical corpus TG refutes 53/60 (88.3%); on the fragment-fair Pytea head-to-head (N=34) it achieves 32/34 vs. 25/34 (McNemar p=0.0156); on a pre-registered N=15 post-freeze real-PR sample it catches 5/15 vs. 2/15 (FakeTensorMode) and 3/15 (Pytea), a directional but statistically non-separable result.

## Prior weakness disposition

- [RESOLVED] **Section 6 currently disagrees with the shipped artifact on the backward-pass limitation.** -- The paper now reports 2/8 = 25.0% false-verified rate on the worst-case construct family using self-contained parseable modules, with specificity 2/2 on negative controls, retiring the vacuous 0/8 readout.
- [RESOLVED] **The Theorem 5 reproducibility story is internally inconsistent.** -- The paper now explicitly discusses both audits (107 candidates/55 successful/72 INT recompiles and 146 candidates/67 successful/0 INT recompiles) and reconciles them, explaining the INT-recompile density difference and confirming both audits agree on the only Theorem 5–relevant quantity (zero out-of-catalogue SHAPE/DTYPE/RANK guards).
- [PARTIAL] **The hybrid-mode complementarity claim is still stress-set-only.** -- The paper has made the caveat more explicit ("We do not claim general complementarity on a natural distribution") and provides the 488-block zero-gain result, but the complementarity evidence remains confined to the 25-block hand-designed importable corpus and no naturalistic sample is provided.
- [PARTIAL] **The real-public successes still do not sit cleanly inside the theorem-backed footprint.** -- The four primary operators for post-freeze catches (view/reshape, conv2d, einsum, unbind) are now in the Lean-audited fragment with sorry-free lemmas; however, `postfreeze_5catches_handler_scope.md` still classifies all 5 as "mixed" because each catch additionally touches at least one uncovered handler (`mul`, `add`, `softmax`, `unsqueeze`).
- [PARTIAL] **The post-freeze baseline comparison remains underpowered.** -- The paper honestly reports non-separability at α=0.05 and ships a pre-registered second wave (wave 2 window 2026-04-08 to 2026-08-31, N_new ≥ 26 target), but wave-2 data are not yet collected and the submission still rests on N=15 alone.

## Strengths

- **Calibrated, honest reporting.** The paper refuses to over-claim: it uses Wilson intervals throughout, reports ABSTAIN as a first-class verdict, acknowledges the 0-unconditional-RP ceiling on real library source up front, and quantifies the grad-flag false-verified rate at 2/8 = 25% on the worst-case construct family.
- **Mechanised rule audit with practical impact.** Eleven previously-axiomatic soundness lemmas are closed sorry-free under `lake build`, and the sorry-free operators (view/reshape, conv2d, einsum, unbind) cover the operators that fire on the post-freeze real catches, giving the main empirical results a Lean backing that is rare in static-analysis papers.
- **Statistically significant head-to-head with the closest comparator.** The McNemar exact p=0.0156 on the fragment-fair N=34 Pytea comparison is a clean causal isolation — the Pytea inapplicability rows are conservatively counted as not-refute, and the full per-bug contingency table is auditable.
- **Comprehensive reproducibility infrastructure.** The repository ships one `.py` + `.json` artefact per empirical claim, with explicit falsification predicates for Theorem 5, per-handler mutation-kill data, and a pre-registered second wave. The artifact allows any claim to be independently reproduced or falsified.

## Weaknesses

- **The aggregate mutation kill rate is 7/50 = 14% at union across three corpora.** Forty-three surviving mutants sit on handler paths not exercised by any of the 60-bug corpus, the 488-block sample, or the 25-case stress benchmark. For a tool where soundness rests on the analyser implementation (which is not Lean-audited), a 14% union kill rate implies that large regions of the handler code could be silently broken without any corpus detecting it. The targeted extension raises the load-bearing view/reshape and broadcasting handlers above 30–40%, but conv2d reaches 53% and einsum 100% only on an 18-case hand-built extension corpus, not the naturalistic benchmarks.
- **Zero unconditional Refuted-Proof on real library source without a user-supplied contract.** Every meaningful refutation on the 488-block corpus falls in the CV or LW category; under the user-visible free-symbolic-config regime the RP count is 0/488. The tool's practical value for real library code therefore depends entirely on the quality of the synthesised `assume_M`, which sits in the trusted computing base (not Lean-audited). The AST-extractor cross-validation shows 0 over-extractions on 140 classes, but those 140 classes are the same ones used to develop the extractor, not a held-out set drawn from a different library distribution.
- **Theorem 5 is empirically grounded primarily on CNN-type modules; transformer results rely on forward-signature surrogates.** The falsification predicate produces zero out-of-catalogue events on 9 CNN blocks and 3 HuggingFace sublayer modules, but the 4 transformer blocks (ViT, Swin, MLP-Mixer, EncoderBlock) use the documented surrogate because full instantiation "exceeds end-to-end constraint solving at this scale." The necessary direction is thus not empirically confirmed end-to-end for transformer architectures, which are the dominant deployment target of the HuggingFace corpus the paper evaluates on.
- **Backward verifier false-verified rate is 2/8 = 25% on the worst-case construct family (tied/renamed-attribute parameter sharing).** The paper correctly reports this and bounds prevalence at ≤ 12% via regex sweep. However, the regex sweep over 2,908 files queries only literal-surface aliasing patterns; a renamed-attribute alias routed through a helper function evades the regex. The 12% ceiling is therefore a lower bound on prevalence, and the actual false-verified exposure could be higher.
- **The hybrid-mode complementarity result is demonstration-only.** The 25-block stress corpus (Table 4) is hand-designed to produce complementary verdicts by construction. On the natural-distribution 488-block corpus, hybrid mode returns the same verdict triple as TG alone (zero gain). The paper correctly labels this an "existence demonstration," but the stress-set result receives near-equal billing to the null real-corpus result in Section 4.2, creating a risk of misreading for readers who process tables before prose.
- **CEGAR loop ships in the implementation but never fires.** An architectural gap (ShapeCEGARLoop predicates are never surfaced as Bug objects) means the CEGAR loop is dead code in all reported experiments. The paper acknowledges this, but the gap between the claimed system and the evaluated system is a concern: readers citing the method should be aware that the implemented CEGAR path has not been evaluated on any corpus.

## Questions

- The AST-extractor cross-validation uses 140 classes drawn from the same repository corpora on which the extractor was developed (config-attribute fixtures, upstream-faithful real corpus, post-freeze real corpus). Can the authors report extractor accuracy on a held-out library not in any of the development corpora, e.g., a randomly sampled set of HuggingFace models from a family not present in the 488-block corpus?
- The 43 surviving mutants all sit on handler paths not exercised by any of the three test corpora. Can the authors characterise which handler families these paths belong to (e.g., which of the 44 tested-only handlers, or which TCB component), and whether they include any handler that could produce a false Refuted-Proof verdict rather than a missed refutation?
- The post-freeze second wave is pre-registered with a window extending to 2026-08-31. Is any partial second-wave data available that could be included as a supplementary table at camera-ready time? Even N_new=10 additional unbiased PRs would substantially narrow the confidence interval on the catch-rate estimate.
- For the backward verifier, the 2/8 = 25% false-verified rate is on a deliberately oversampled worst-case harness. On the naturalistic HuggingFace modules from the Theorem 5 audit (Track-E fixtures), all 16 are clean. Can the authors report the false-verified rate on the 42 held-out HuggingFace training scripts from `examples/pytorch/`, where 1/42 contains a silent-error-positive construct?

## Scores

Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons

Adding second-wave post-freeze data (wave-2, N_new ≥ 26) with at least one statistically separable comparison at α=0.05 would push the score to 7: the core formal contributions and calibrated methodology already meet the NeurIPS bar, but the post-freeze real-world evaluation is the only experiment on an unbiased naturalistic sample and its current non-significant N=15 result weakens the claim that TG is a practically useful bug-finder on real code rather than a demonstrably sound but abstention-heavy analyser.


Changes   +0 -0
Requests  1 Premium (3m 1s)
Tokens    ↑ 435.1k • ↓ 7.8k • 360.2k (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 5, streak=2] **[ESCALATED — reviewer rejected your last 2 attempts; you must EITHER ship the missing artifact this round OR remove the disputed claim from the abstract+contributions list, no further paraphrase is allowed]**
  Last round you only edited paper-side files (.tex/.bib/.md) and shipped no code, no new data artifact, no Lean theorem, no new figure regenerated from a script. The reviewer's weaknesses require substantive evidence, not paraphrase. This round, EITHER ship the requested artifact via spawn_sonnet_subagent.sh (preferred), OR remove the disputed claim from the abstract+contributions list.
- [reviewer, w=1.00, added round 9, streak=0] **The aggregate mutation kill rate is 7/50 = 14% at union across three corpora.** Forty-three surviving mutants sit on handler paths not exercised by any of the 60-bug corpus, the 488-block sample, or the 25-case stress benchmark. For a tool where soundness rests on the analyser implementation (which is not Lean-audited), a 14% union kill rate implies that large regions of the handler code could be silently broken without any corpus detecting it. The targeted extension raises the load-bearing view/reshape and broadcasting handlers above 30–40%, but conv2d reaches 53% and einsum 100% only on an 18-case hand-built extension corpus, not the naturalistic benchmarks.
- [reviewer, w=1.00, added round 9, streak=0] **Zero unconditional Refuted-Proof on real library source without a user-supplied contract.** Every meaningful refutation on the 488-block corpus falls in the CV or LW category; under the user-visible free-symbolic-config regime the RP count is 0/488. The tool's practical value for real library code therefore depends entirely on the quality of the synthesised `assume_M`, which sits in the trusted computing base (not Lean-audited). The AST-extractor cross-validation shows 0 over-extractions on 140 classes, but those 140 classes are the same ones used to develop the extractor, not a held-out set drawn from a different library distribution.
- [reviewer, w=1.00, added round 9, streak=0] **Theorem 5 is empirically grounded primarily on CNN-type modules; transformer results rely on forward-signature surrogates.** The falsification predicate produces zero out-of-catalogue events on 9 CNN blocks and 3 HuggingFace sublayer modules, but the 4 transformer blocks (ViT, Swin, MLP-Mixer, EncoderBlock) use the documented surrogate because full instantiation "exceeds end-to-end constraint solving at this scale." The necessary direction is thus not empirically confirmed end-to-end for transformer architectures, which are the dominant deployment target of the HuggingFace corpus the paper evaluates on.
- [reviewer, w=1.00, added round 9, streak=0] **Backward verifier false-verified rate is 2/8 = 25% on the worst-case construct family (tied/renamed-attribute parameter sharing).** The paper correctly reports this and bounds prevalence at ≤ 12% via regex sweep. However, the regex sweep over 2,908 files queries only literal-surface aliasing patterns; a renamed-attribute alias routed through a helper function evades the regex. The 12% ceiling is therefore a lower bound on prevalence, and the actual false-verified exposure could be higher.
- [reviewer, w=1.00, added round 9, streak=0] **The hybrid-mode complementarity result is demonstration-only.** The 25-block stress corpus (Table 4) is hand-designed to produce complementary verdicts by construction. On the natural-distribution 488-block corpus, hybrid mode returns the same verdict triple as TG alone (zero gain). The paper correctly labels this an "existence demonstration," but the stress-set result receives near-equal billing to the null real-corpus result in Section 4.2, creating a risk of misreading for readers who process tables before prose.
- [reviewer, w=1.00, added round 9, streak=0] **CEGAR loop ships in the implementation but never fires.** An architectural gap (ShapeCEGARLoop predicates are never surfaced as Bug objects) means the CEGAR loop is dead code in all reported experiments. The paper acknowledges this, but the gap between the claimed system and the evaluated system is a concern: readers citing the method should be aware that the implemented CEGAR path has not been evaluated on any corpus.
- [reviewer, w=1.00, added round 9, streak=0] The AST-extractor cross-validation uses 140 classes drawn from the same repository corpora on which the extractor was developed (config-attribute fixtures, upstream-faithful real corpus, post-freeze real corpus). Can the authors report extractor accuracy on a held-out library not in any of the development corpora, e.g., a randomly sampled set of HuggingFace models from a family not present in the 488-block corpus?
- [reviewer, w=1.00, added round 9, streak=0] The 43 surviving mutants all sit on handler paths not exercised by any of the three test corpora. Can the authors characterise which handler families these paths belong to (e.g., which of the 44 tested-only handlers, or which TCB component), and whether they include any handler that could produce a false Refuted-Proof verdict rather than a missed refutation?
- [reviewer, w=1.00, added round 9, streak=0] The post-freeze second wave is pre-registered with a window extending to 2026-08-31. Is any partial second-wave data available that could be included as a supplementary table at camera-ready time? Even N_new=10 additional unbiased PRs would substantially narrow the confidence interval on the catch-rate estimate.

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

Round: 9
