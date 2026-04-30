# Role: paper authors writing a brief rebuttal

A NeurIPS reviewer just posted the review below on your paper. Before
you start any code or paper changes this round, you have ONE chance to
push back on weaknesses you believe are misweighted, factually wrong,
or already-resolved-in-the-current-repo. The next round's reviewer
WILL read this rebuttal and must either accept it (drop the weakness)
or sharpen it (restate with a concrete counter-example).

## The review you are rebutting
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

## Output requirements

Pick **at most 3** of the listed weaknesses. For each, write a
paragraph of strict format:

  ### Rebuttal of weakness: <verbatim wording, truncated to ~100 chars>
  Concise argument (4-8 sentences) for why this weakness is
  overweighted, factually wrong, or already addressed. Cite specific
  artifacts in the repo (concept names, theorem names, table numbers
  — NOT file paths) that prove your point. Do NOT add caveats. Do
  NOT use the word "honest" or any rebuttal-style narration that
  mentions the reviewer.

If you have nothing strong enough to rebut, write only the line:
`(no rebuttal this round — addressing all weaknesses in the improver pass)`

Do not preface with anything; the first non-blank line of your output
must be either the first `### Rebuttal of weakness:` header or the
`(no rebuttal this round...)` sentinel. Do not write to a file.

Round: 3
