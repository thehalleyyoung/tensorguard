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

Resolving the 53/56 internal inconsistency with a single authoritative re-run of the 60-bug corpus that prints both the raw refute count and the RP sub-count—ideally invocable via one command that the README points to—would address the most concrete credibility gap; if accompanied by a regenerated sorry-free `lake build` log, these two documentation fixes would push the score to 6 without requiring any new experiments.
Changes   +0 -0
Requests  1 Premium (8m 58s)
Tokens    ↑ 958.7k • ↓ 25.0k • 875.0k (cached)

**Sub-score-targeted primary work (target dimension: SOUNDNESS = 2/4).** Of the four scored sub-dimensions, soundness is currently the binding constraint on Overall. Concentrate this round's non-escalated effort on raising it from 2 to 3. Concrete actions you may pick from (do AT LEAST one in full this round; sketching does not count):
  - tighten / formalize a key theorem; if a Lean / Coq / Mathematica skeleton exists, close one open lemma in this round
  - replace a proof-by-figure or sketch with a numbered theorem + proof
  - state every regularity/assumption explicitly and verify the constants in code
  - run one extra experimental seed and report the variance to defuse 'might be cherry-picked' concerns

## Latest reviewer report
## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` shapes and gradient flow. The core contributions are: (C1) a refinement-type calculus `Tensor{s, g | φ}` with Z3-backed shape/grad entailment; (C2) an assume/guarantee composition discipline with a Lean 4–mechanised soundness proof on a 17-operator DSL (28 shape-transfer rules, 11 soundness lemmas closed sorry-free); (C3) an autograd backward verifier; (C4) an exploratory Dynamo-guard inclusion lemma (necessary direction only); and (C5–C6) a 488-block + 60-bug benchmark with a five-way verdict taxonomy. On the curated 60-bug historical corpus TG claims 53/60 RP (88.3%); on the 488-block real-source corpus the user-visible regime produces 0 unconditional RP, acknowledged from the abstract. The paper is notable for its calibration discipline: every limitation—flat ablation on natural workloads, 0-RP ceiling on real source, non-functional CEGAR/phase-check knobs—is reported openly rather than buried.

## Prior weakness disposition

- [PARTIAL] The main practical limitation remains central: in the user-visible regime on the unreduced **488-block real-source corpus**, TensorGuard still reports **0/488 unconditional RP**... -- The paper now provides a much more thorough characterisation: a per-block LW→RP gap table with 12 named candidates, 3/12 measured-flipped this round, and a rerun showing 15/488 under an input-shape-contract regime. The 0/488 fact in the user-visible regime is unchanged; the gap is now more fully characterised, hence PARTIAL.

- [PARTIAL] The real-bug evidence is still **small-N**: the upstream-faithful table is `7/10` at `>=0.99` plus `1/10` at `0.80`, and the unfiltered post-freeze result is `5/15`... -- The paper extends to a pre-registered N=15 unfiltered post-freeze corpus with explicit power calculation; Fisher-exact two-sided p=0.39 (TG vs. FakeTensorMode) and 0.68 (TG vs. Pytea). Still not statistically separable at α=0.05 and a power calculation shows N_new=26–77 would be required. The evidence is broader but the N is still small, hence PARTIAL.

- [UNRESOLVED] The ablation story is weak on natural workloads: Section 4.4 states that the five-knob ablation on the `488+60` corpora is a **flat line**... -- This round the paper further confirms that the 10-bug real corpus ablation is also a flat line. The flatness is now more firmly established empirically, not improved. Feature ablation JSON L0–L5 on the 60-bug corpus shows all levels produce identical refute counts (56 at each level per the checked-in artifact). Marking UNRESOLVED: no natural-workload knob discriminates; the discriminative evidence remains restricted to the 25-case hand-designed stress benchmark.

- [PARTIAL] The Dynamo section is better framed now, but much of the evidence is still **signature-trusted or audit-by-inspection**... -- Rebuttal accepted in part: the paper now presents 9 fully end-to-end CNN blocks and 3 T5/BERT sublayers end-to-end (total 12), with per-module recompile tables materialized in dynamo_e2e_results.json. However, 4/14 transformer blocks in the extended audit still use the "forward-signature surrogate," and the large-corpus audits (55 and 67 modules) are vacuously consistent with Theorem 5 because they observe zero SHAPE/DTYPE/RANK guards (only INT specialisation fires). The end-to-end base is thin for the transformer claim, hence PARTIAL.

- [PARTIAL] The released artifact still has at least one **stale internal inconsistency**: `experiments_v5/v8/lean_sorry_elim_report.json` reports one remaining `sorry`... -- The Lean sources are genuinely sorry-free by direct inspection: a word-boundary grep (`\bsorry\b`) over `lean/TensorGuard/Extended.lean` returns only two docstring-comment lines (15, 92). However, the checked-in `lean_build_v8.log`—cited by the authors as the canonical sorry-free evidence—still contains `warning: ././././TensorGuard/Extended.lean:92:8: declaration uses 'sorry'`. In Lean 4, "declaration uses 'sorry'" is only triggered by a proof-position `sorry`, not by the substring in a comment; this warning therefore reflects the state of the source at log-generation time, which predates the docstring edit. The build log is stale relative to the current source. The paper's claim is grounded in the live source (correct), but the cited artifact still contradicts it. A regenerated log is needed to close this.

## Strengths

- **Honest, calibrated reporting.** The paper leads with its own limitations (0/488 user-visible RP; non-functional CEGAR and phase-check knobs; flat ablation on natural workloads) rather than burying them, which is rare and commendable.
- **Genuine Lean 4 mechanisation.** The 28-rule operator audit and 11 sorry-free soundness lemmas are verifiable from the checked-in source; the `\bsorry\b` grep confirms no proof-position sorry in the live tree. This goes meaningfully beyond most ML-systems papers.
- **Statistically clean head-to-head with Pytea.** On the N=34 fragment-fair modern subset, 32/34 vs. 25/34 (McNemar exact p=0.0156, bootstrap CI [+8.8 pp, +35.3 pp]) is a real result with proper paired analysis, per-bug contingency tables, and fragment-fairness enforcement at verification time.
- **Extensive reproducibility artifacts.** Checked-in JSONs for virtually every table (dynamo_e2e_results, feature_ablation, lean_parity_v5_results, lw_rp_gap, per_block_user_visible_rp, etc.) give reviewers direct access to the underlying data without re-running the full pipeline.
- **Transparent TCB scope.** The paper explicitly states what is and is not Lean-audited (28+7=35 in-soundness handlers vs. 44 tested-only), and the per-fire soundness classification on the post-freeze catches is traceable to the handler-level.

## Weaknesses

- **53/60 vs. 56/60 internal inconsistency in headline RP count.** The abstract and the body both state "REFUTED-PROOF on 53/60 (88.3%)." Table 1's caption states "all 56 refutations are REFUTED-PROOF." The checked-in `experiments_v5/feature_ablation.json` shows `bug_corpus.refuted = 56` and `silent_miss = 4` at every feature level L0–L5, totalling 60. The Wilson CI given in the abstract (77.8%, 94.2%) is consistent with 53/60, not 56/60. If 56 is the correct current count the CI is stale; if 53 is correct the table caption and the ablation JSON are inconsistent. Neither reconciliation is provided anywhere in the paper or appendices, and the repository's `verify_neurips.py` does not run the 60-bug corpus (it runs seven synthetic models), so the discrepancy cannot be resolved by running the shipped validation scripts.

- **CEGAR and phase-check ship but are architecturally non-functional as described.** The feature ablation JSON meta note explicitly states: "check_devices, check_phases, check_gradients are accepted by the API but NOT forwarded to verify_model in the current implementation; L2/L3/L4 rows therefore replicate L1 verdict counts." CEGAR predicates are computed but "stored as metadata only (not fed back as Bug objects)." Yet the README advertises "CEGAR loop—counterexample-guided abstraction refinement discovers shape predicates automatically" and "Multi-phase train/eval analysis—detects BatchNorm/Dropout misuse." These are materially misleading claims for a shipped tool, not just limitations of the contribution-scoped evaluation.

- **Mutation-testing kill rate on load-bearing handlers is low without corpus extension.** On the 60-bug regression corpus alone, conv2d and einsum kill rates are both 0/10. A special 18-case targeted extension is needed to lift them above 50%. The union kill rate across three corpora is 7/50 = 14%. This means the standard regression corpus does not exercise the arithmetic paths of the two most important handlers, leaving a meaningful test-oracle gap for the soundness claim.

- **Theorem 5 (Dynamo) falsification predicate is vacuously satisfied on the large-corpus audits.** The 55-module and 67-module audits find zero SHAPE/DTYPE/RANK in-contract recompile guards (only INT specialisation fires). The paper explicitly reports this as "the falsification predicate is therefore not exercised on this population (denominator 0 SHAPE/DTYPE/RANK guards)." The predicate evaluating to false on a vacuous corpus is not positive evidence for the necessary direction; it merely confirms that INT-dominant modules don't falsify a shape-inclusion theorem, which is uninformative. The substantive evidence for Theorem 5 is therefore the 9 CNN blocks in the 14-module end-to-end audit plus the 3 T5/BERT sublayers—a combined population too small to extrapolate broadly.

- **No single command reproduces the headline 53/60 RP figure.** The README references `experiments_v5/run_v5_benchmark.py` as the reproducibility script. The shipped `verify_neurips.py` runs seven synthetic models only. A reader attempting end-to-end reproduction of the headline must assemble the 60-bug corpus runner from scratch; no top-level `make reproduce` or equivalent is shipped.

## Questions

- **Reconcile 53 vs. 56.** Which is the correct RP count on the 60-bug corpus: the 53 reported in the abstract and body, the 56 in the Table 1 caption, or the 56 in `feature_ablation.json`? If the current code gives 56, what accounts for the three additional catches relative to the 53 cited everywhere in the text? If 53 is correct, what explains the Table 1 caption and the ablation artifact?

- **Regenerate `lean_build_v8.log`.** The cited canonical sorry-free artifact still contains `warning: ././././TensorGuard/Extended.lean:92:8: declaration uses 'sorry'`. Please supply a build log generated from the current source (post docstring-edit) to replace the stale artifact.

- **API truth vs. README advertising.** The README claims CEGAR and multi-phase train/eval analysis as key features. The implementation forwards neither to the verdict pipeline on the shipped API surface. Will the tool be corrected to match the README, or will the README be corrected to match the tool?

- **Largest-scale Theorem 5 test.** Given that the 55- and 67-module large-scale audits observe zero SHAPE/DTYPE/RANK guards (vacuous falsification predicate), what is the smallest importable transformer module for which Theorem 5's falsification predicate is non-vacuously satisfied (i.e., at least one SHAPE/DTYPE/RANK in-contract recompile is observed) without using a forward-signature surrogate?

- **Post-freeze N=15 power gap.** The power calculation puts the required second-wave sample at N_new=26 (TG vs FakeTensorMode one-sided) and N_new=77 (TG vs Pytea). Is there a plan to run that second wave, and has the pre-registered query been frozen for that extension?

- **Backward verifier tied-weight limitation.** Section 6 flags parameter-sharing-under-renamed-attribute as silently incorrect in ≤12% of training scripts. The reproducibility directory contains `tied_weight_stubmock_runtime.json`. Does running the stub-mock harness on the 60-bug corpus or the post-freeze corpus reveal any false-RP bugs attributable specifically to this class?

## Scores

Soundness: 2
Presentation: 3
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons

Resolving the 53/56 internal inconsistency with a single authoritative re-run of the 60-bug corpus that prints both the raw refute count and the RP sub-count—ideally invocable via one command that the README points to—would address the most concrete credibility gap; if accompanied by a regenerated sorry-free `lake build` log, these two documentation fixes would push the score to 6 without requiring any new experiments.


Changes   +0 -0
Requests  1 Premium (8m 58s)
Tokens    ↑ 958.7k • ↓ 25.0k • 875.0k (cached)

## Active obligations (decayed across rounds)
These are the open items, ordered by current weight. Items with low
weight are stale and may be quietly dropped if no longer relevant.
- [reviewer, w=1.00, added round 2, streak=1] The ablation story is weak on natural workloads: Section 4.4 states that the five-knob ablation on the `488+60` corpora is a **flat line**, and the discriminative evidence comes only from a hand-designed **25-case stress benchmark**.
- [reviewer, w=1.00, added round 3, streak=0] **53/60 vs. 56/60 internal inconsistency in headline RP count.** The abstract and the body both state "REFUTED-PROOF on 53/60 (88.3%)." Table 1's caption states "all 56 refutations are REFUTED-PROOF." The checked-in `experiments_v5/feature_ablation.json` shows `bug_corpus.refuted = 56` and `silent_miss = 4` at every feature level L0–L5, totalling 60. The Wilson CI given in the abstract (77.8%, 94.2%) is consistent with 53/60, not 56/60. If 56 is the correct current count the CI is stale; if 53 is correct the table caption and the ablation JSON are inconsistent. Neither reconciliation is provided anywhere in the paper or appendices, and the repository's `verify_neurips.py` does not run the 60-bug corpus (it runs seven synthetic models), so the discrepancy cannot be resolved by running the shipped validation scripts.
- [reviewer, w=1.00, added round 3, streak=0] **CEGAR and phase-check ship but are architecturally non-functional as described.** The feature ablation JSON meta note explicitly states: "check_devices, check_phases, check_gradients are accepted by the API but NOT forwarded to verify_model in the current implementation; L2/L3/L4 rows therefore replicate L1 verdict counts." CEGAR predicates are computed but "stored as metadata only (not fed back as Bug objects)." Yet the README advertises "CEGAR loop—counterexample-guided abstraction refinement discovers shape predicates automatically" and "Multi-phase train/eval analysis—detects BatchNorm/Dropout misuse." These are materially misleading claims for a shipped tool, not just limitations of the contribution-scoped evaluation.
- [reviewer, w=1.00, added round 3, streak=0] **Mutation-testing kill rate on load-bearing handlers is low without corpus extension.** On the 60-bug regression corpus alone, conv2d and einsum kill rates are both 0/10. A special 18-case targeted extension is needed to lift them above 50%. The union kill rate across three corpora is 7/50 = 14%. This means the standard regression corpus does not exercise the arithmetic paths of the two most important handlers, leaving a meaningful test-oracle gap for the soundness claim.
- [reviewer, w=1.00, added round 3, streak=0] **Theorem 5 (Dynamo) falsification predicate is vacuously satisfied on the large-corpus audits.** The 55-module and 67-module audits find zero SHAPE/DTYPE/RANK in-contract recompile guards (only INT specialisation fires). The paper explicitly reports this as "the falsification predicate is therefore not exercised on this population (denominator 0 SHAPE/DTYPE/RANK guards)." The predicate evaluating to false on a vacuous corpus is not positive evidence for the necessary direction; it merely confirms that INT-dominant modules don't falsify a shape-inclusion theorem, which is uninformative. The substantive evidence for Theorem 5 is therefore the 9 CNN blocks in the 14-module end-to-end audit plus the 3 T5/BERT sublayers—a combined population too small to extrapolate broadly.
- [reviewer, w=1.00, added round 3, streak=0] **No single command reproduces the headline 53/60 RP figure.** The README references `experiments_v5/run_v5_benchmark.py` as the reproducibility script. The shipped `verify_neurips.py` runs seven synthetic models only. A reader attempting end-to-end reproduction of the headline must assemble the 60-bug corpus runner from scratch; no top-level `make reproduce` or equivalent is shipped.
- [reviewer, w=1.00, added round 3, streak=0] **Reconcile 53 vs. 56.** Which is the correct RP count on the 60-bug corpus: the 53 reported in the abstract and body, the 56 in the Table 1 caption, or the 56 in `feature_ablation.json`? If the current code gives 56, what accounts for the three additional catches relative to the 53 cited everywhere in the text? If 53 is correct, what explains the Table 1 caption and the ablation artifact?
- [reviewer, w=1.00, added round 3, streak=0] **Regenerate `lean_build_v8.log`.** The cited canonical sorry-free artifact still contains `warning: ././././TensorGuard/Extended.lean:92:8: declaration uses 'sorry'`. Please supply a build log generated from the current source (post docstring-edit) to replace the stale artifact.
- [reviewer, w=1.00, added round 3, streak=0] **API truth vs. README advertising.** The README claims CEGAR and multi-phase train/eval analysis as key features. The implementation forwards neither to the verdict pipeline on the shipped API surface. Will the tool be corrected to match the README, or will the README be corrected to match the tool?
- [reviewer, w=1.00, added round 3, streak=0] **Largest-scale Theorem 5 test.** Given that the 55- and 67-module large-scale audits observe zero SHAPE/DTYPE/RANK guards (vacuous falsification predicate), what is the smallest importable transformer module for which Theorem 5's falsification predicate is non-vacuously satisfied (i.e., at least one SHAPE/DTYPE/RANK in-contract recompile is observed) without using a forward-signature surrogate?

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

Round: 3
