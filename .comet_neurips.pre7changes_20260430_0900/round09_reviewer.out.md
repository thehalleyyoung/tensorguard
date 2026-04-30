● Read round09_reviewer.md
  │ .comet_neurips/round09_reviewer.md
  └ 118 lines read

● List reproducibility and test files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/ && ls tests/ 2>/dev/null | head -30
  └ 162 lines...

● Extract paper text (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "import pdfplumber; pdf = pdfplumber.open('neurips.pdf'); text
  │ = ''; [text := text + p.extract_text() + '\n' for p in pdf.pages[:6]];
  │ print(text[:8000])" 2>/dev/null || pdftotext neurips.pdf - 2>/dev/null |
  │ head -200
  └ 201 lines...

● Read grad lattice runtime holdout (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/grad_lattice_runtime_holdout.md
  └ 43 lines...

● Read postfreeze catches and power analysis (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/postfreeze_5catches_handler_scope.md && echo "---" &&
  │ cat reproducibility/postfreeze_power_analysis.md
  └ 43 lines...

● Read paper middle sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | sed -n '200,500p'
  └ 302 lines...

● Read Theorem 5 reproducibility files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/dynamo_theorem5_n100.md | head -60 && echo "===" &&
  │ cat reproducibility/dynamo_theorem5_n200.md | head -60
  └ 80 lines...

● Read paper evaluation sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | sed -n '500,800p'
  └ 302 lines...

● Read paper Theorem 5 and backward verifier sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | sed -n '1500,1850p'
  └ 352 lines...

● Read paper Section 4 continued (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | sed -n '800,1100p'
  └ 302 lines...

● Read second wave postfreeze results (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/postfreeze_second_wave_results.md && echo "===" &&
  │ cat reproducibility/postfreeze_second_wave_power.md
  └ 83 lines...

● Read Section 6 backward pass limitation text (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | sed -n '1750,1900p'
  └ 152 lines...

● Check paper Section 6 backward pass claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | grep -n
  │ "false.verif\|false-verif\|0/8\|2/8\|backward.*lattice\|lattice.*backward\|t
  │ rainer.*grad\|grad.*trainer\|Section 6\|section 6" | head -40
  └ 22 lines...

● Read paper Section 6 limitations (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | sed -n '1800,1950p'
  └ 152 lines...

● Read end of Section 6 and discussion (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | sed -n '2200,2450p'
  └ 252 lines...

● Read Section 6 in full (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | sed -n '2100,2200p'
  └ 102 lines...

Now I have enough information to write the full review.

● Check mutation testing claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | grep -n
  │ "mutation.*kill\|kill.*rate\|14%\|7/50\|43.*surviv\|surviv.*43" | head -20
  └ 8 lines...

● Check CEGAR dead code claim (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && pdftotext neurips.pdf - 2>/dev/null | grep -n
  │ "CEGAR\|cegar\|dead.code\|dead code\|non-contribution" | head -20
  └ 11 lines...

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
