● Read round06_reviewer.md
  │ .comet_neurips/round06_reviewer.md
  └ 118 lines read

● List directory .
  └ 55 files found

● Read paper source (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.tex 2>/dev/null | head -300 || echo "No tex file, checking for pdf
  │ text extraction"
  └ 95 lines...

● List reproducibility folder (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/r
  │ eproducibility/
  └ 127 lines...

● List paper sections (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/s
  │ ections_v5/
  └ 2 lines...

● Read paper text (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.txt | head -500
  └ 501 lines...

● Find paper sections (shell)
  │ find /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ -name "sections_v5" -type d 2>/dev/null || find
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ -name "*.tex" | grep -v neurips.tex | head -20
  └ 2 lines...

● Read more of paper text (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.txt | sed -n '189,500p'
  └ 313 lines...

● Check grad lattice holdout (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ reproducibility/grad_lattice_runtime_holdout.md
  └ 43 lines...

● Read evaluation section of paper (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.txt | sed -n '500,900p'
  └ 402 lines...

● Check Dynamo Theorem 5 evidence (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ reproducibility/dynamo_theorem5_n100.md 2>/dev/null | head -100
  └ 19 lines...

● Read later sections of paper (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.txt | sed -n '900,1300p'
  └ 402 lines...

● Check post-freeze second wave results (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ reproducibility/postfreeze_second_wave_results.md 2>/dev/null | head -80
  └ 57 lines...

● Check grad lattice holdout script (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ reproducibility/grad_lattice_runtime_holdout.py | head -100
  └ 101 lines...

● Read more of paper - Dynamo and theorem sections (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.txt | sed -n '1300,1700p'
  └ 402 lines...

● Find Dynamo mentions in paper (shell)
  │ grep -n "TorchDynamo\|Dynamo" /Users/halleyyoung/Documents/div/mathdivergenc
  │ e/halley-labs/tensorguard/neurips.txt | head -30
  └ 27 lines...

● Read intro Dynamo framing (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.txt | sed -n '35,45p'
  └ 12 lines...

● Read intro section on Dynamo framing specifically (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.txt | sed -n '33,93p'
  └ 62 lines...

● Check AST pattern disabled result (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ reproducibility/ast_pattern_disabled_60bug.md 2>/dev/null | head -60
  └ 20 lines...

● Read later paper sections - limitations (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.txt | sed -n '1700,2000p'
  └ 302 lines...

● Check bug corpus decomposition (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ reproducibility/bug_corpus_no_parser_marker.md 2>/dev/null | head -60
  └ 61 lines...

● Check abstract vs body Pytea numbers (shell)
  │ grep -n "25/34\|22/34\|0\.0156\|0\.00195\|McNemar" /Users/halleyyoung/Docume
  │ nts/div/mathdivergence/halley-labs/tensorguard/neurips.tex | head -20
  └ 2 lines...

● Check Pytea McNemar calculation (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ reproducibility/pytea_mcnemar_per_bug.md 2>/dev/null | head -40
  └ 41 lines...

● Check Pytea modern mcnemar artifact (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ reproducibility/pytea_modern_mcnemar.md 2>/dev/null | head -50
  └ 51 lines...

● Read limitations section (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ neurips.txt | sed -n '770,900p'
  └ 132 lines...

## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that verifies tensor shapes and gradient flow statically from class source without instantiating the module. The central contribution is a refinement-typed calculus (symbolic shape + flat grad-flag, Z3-discharged) with an assume/guarantee composition discipline at the module boundary, and a Lean 4 audit of 28 of 79 operator shape-transfer rules with 11/11 soundness lemmas closed sorry-free. On a curated 60-bug historical corpus the system returns Refuted-Proof on 53/60 (88.3%); on a pre-registered unfiltered post-freeze N=15 real-PR sample, TG catches 5/15 versus FakeTensorMode 2/15 and Pytea 3/15 (directional, non-statistically-separable at α=0.05). On the 488-block real-source corpus the user-visible free-symbolic regime produces 0 unconditional Refuted-Proof verdicts, acknowledged upfront as a fragment-coverage measurement.

## Prior weakness disposition

- [PARTIAL] The opening framing still overstates the Dynamo result relative to both the theorem and the empirical evidence. In the introduction, "TorchDynamo guards become the runtime shadow of these refinements" and "Abstain ... marks exactly the s... -- Contribution C4 is now scoped as "preliminary result" and Theorem 5 is labelled "necessary direction only," but the intro sentence "TensorGuard's ABSTAIN verdict marks exactly the subgraphs on which Dynamo would have broken the graph" (lines 37–38 of the extracted text) still asserts a bidirectional correspondence that Theorem 5 does not prove; removing the word "exactly" alone would bring the intro into alignment with the stated theorem.

- [PARTIAL] The headline real-world bug-finding case remains weak. On the user-visible 488-block corpus the paper reports `0/488` unconditional RP, and on the unfiltered post-freeze sample the main comparable number is `5/15` versus `2/15` and `3/15`... -- The paper now reports CI [15.2%, 58.3%], Fisher-exact p=0.39/0.68 (non-separable), and a pre-registered wave-2 power calculation; calibration is much improved, but the fundamental weakness (N=15, non-significant, 0 unconditional RP on real library source) persists as a substantive limitation on the empirical case.

- [RESOLVED] The grad-lattice runtime holdout appears artifact-inconsistent. §4.4 says the analyser returns Refuted-Proof on `8/8` checkpoint-enabled scripts because its detector flips them out of the first-order lattice, but `reproducibility/grad_lattice_runtime_holdout...` -- The holdout script has been fully rewritten with self-contained parseable `nn.Module` subjects; it now reports a non-vacuous 6/8 RP and 2/8 false-verified on tied-weight subjects, and the paper discloses the 2/8 false-verified explicitly, with the 8/8 figure now referring to a separate checkpoint-specific experiment on 8 HuggingFace head classes.

- [PARTIAL] The theorem-to-evidence bridge for Theorem 5 is still limited. The paper's own strongest end-to-end audit is only 9 CNN blocks without surrogate, 4 transformer blocks still use surrogates, and the larger-module audit reports zero SHAPE/DTYPE/RANK... -- The 55-module population audit was added, but those 55 modules yield 72 in-contract recompiles classified entirely as INT (0 SHAPE/DTYPE/RANK), meaning the SHAPE-correspondence falsification predicate is never exercised in the larger population; the CNN-only 9-block end-to-end result remains the sole test of the predicate and is unchanged.

- [PARTIAL] The curated 60-bug benchmark seems insufficiently diagnostic of the intended reasoning contribution. The paper states that with operator dispatch disabled the AST-pattern path alone still gets `53/60`, and with AST patterns disabled the... -- `reproducibility/bug_corpus_no_parser_marker.md` now explicitly reports that the symbolic calculus operator rules alone catch **0/60** bugs once both the AST-pattern path and the parser-failure marker are excluded; the paper acknowledges this but the abstract still leads with the 88.3% figure without flagging its source.

## Strengths

- The paper is unusually forthcoming about its own limitations: 0 unconditional RP on real library source, non-significant N=15 real-PR result, and one-directionality of Theorem 5 are all stated quantitatively and upfront rather than buried in an appendix.
- The Lean 4 operator-rule audit (28 rules, 11/11 sorry-free lemmas, 28,000/28,000 torch parity samples) is a meaningful mechanisation contribution—small in scope but executed carefully with boundary checks and an exported JSON registry that prevents silent drift.
- The fragment-fair Pytea head-to-head on the N=34 modern subset (32/34 vs. 22/34, McNemar p=0.00195, bootstrap CI [+14.7 pp, +44.1 pp]) is a well-controlled comparison that isolates the catalogue-gap confound.
- The per-module CV caller-rely witnessing (118/128, Clopper-Pearson CI [86.1%, 96.2%]) with explicit diagnosis of the 10 unwitnessed cases is methodologically solid and goes beyond what most static-analysis papers report.

## Weaknesses

- **Abstract/body numerical inconsistency on the Pytea head-to-head.** The abstract (neurips.tex line 45) states "32/34 vs. 25/34 (McNemar exact p=0.0156)" while the body says "32/34 vs. 22/34 (McNemar exact p=0.00195)." `reproducibility/pytea_mcnemar_per_bug.md` records b=7 (corresponding to Pytea=25/34, p=0.0156), while `reproducibility/pytea_modern_mcnemar.md` records b=10 (Pytea=22/34, p=0.00195). These cannot both be correct; the abstract reports pre-silent-skip-correction numbers while the body reports post-correction. One of the two must be fixed.

- **The symbolic calculus contributes 0/60 detections on the headline benchmark.** `reproducibility/bug_corpus_no_parser_marker.md` shows configuration (C): rule-driven symbolic reasoning only catches 0/60 bugs. The 53/60 RP headline is fully attributable to the "parser-failure marker" path. The paper states "the calculus is the correctness substrate that justifies which catches are sound, but the recognition of a buggy fragment routinely goes through one of the other two paths," but this framing is confusing: if detection is via structural pattern matching and soundness is via the rule table, readers need a concrete worked example showing exactly where the rule-based inference contributes something the pattern match could not do alone. The headline number measured on a corpus where the calculus fires 0 times does not support Section 1's framing of the contribution.

- **The Dynamo ABSTAIN claim in the introduction is bidirectional; Theorem 5 is unidirectional.** "TensorGuard's ABSTAIN verdict marks *exactly* the subgraphs on which Dynamo would have broken the graph" (intro, lines 37–38) asserts a necessary-and-sufficient correspondence. Theorem 5 proves only the necessary direction. The empirical audit on the 55-module population finds 0 SHAPE guards (only INT), providing no additional support for the SHAPE direction. This claim should be corrected to match the theorem.

- **The 55-module Dynamo population fails to exercise the SHAPE-correspondence falsification predicate.** The 72 in-contract recompiles are all classified INT; the paper correctly notes that this makes it a "denominator audit, not a falsifier evaluation," but it also means the only test of the SHAPE falsification predicate is the unchanged CNN-only 9-block audit from prior rounds. Presenting the 55-module result as expanding the evidence base for Theorem 5 is misleading if it exercises a different guard kind altogether.

- **The mutation kill rate on the 60-bug corpus (3/50 = 6.0%) is low enough to be concerning.** Even with the three-corpus union (7/50 = 14%), 43 of 50 single-edit mutations survive undetected. For a system whose correctness story depends on the analyser implementation, a 14% mutation kill rate means the corpora leave most implementation faults undetected. The paper bounds this from below, but no attempt is made to characterise whether the surviving mutants are structurally unreachable by the published corpora or represent real coverage gaps.

## Questions

1. The abstract says "32/34 vs. 25/34 (McNemar exact p=0.0156)" and `reproducibility/pytea_mcnemar_per_bug.md` reports b=7 matching those numbers, yet the body says "32/34 vs. 22/34 (McNemar exact p=0.00195)" matching `pytea_modern_mcnemar.md`. Which number is the intended headline, and will the abstract be corrected to match?

2. What exactly is the "parser-failure marker" path that drives 53/60 of the curated corpus detections? Is it a pattern-match on syntactic bug tokens (e.g., mismatched dim literals) rather than the symbolic calculus? If the calculus fires 0 times on the 60-bug corpus, can the authors exhibit even one bug from the *real-PR* set (Table 3) where the Z3 solver — rather than structural AST pattern matching — is the component that generates the refutation witness?

3. The 55-module Dynamo population yields 0 SHAPE/DTYPE/RANK guards (all INT). Can the authors identify any module in the 55 that one would *expect* to produce a SHAPE guard under `torch.compile(dynamic=True)` and explain why it does not? If none of the 55 is expected to exhibit SHAPE guards, the population is not a test of Theorem 5 at all.

4. `grad_lattice_runtime_holdout.md` reports 2/8 false-verified on tied-weight subjects, while the paper also states "six hand-built tied-weight + torch.utils.checkpoint positive modules, the verifier returns 6/6 ABSTAIN." Are these two different test sets, and if so, why does the holdout artifact show false-verified on tied-weight subjects rather than ABSTAIN?

## Scores

Soundness: 3
Presentation: 2
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons

The single change that would do most to push the score toward 6 is fixing the abstract/body Pytea inconsistency and providing one concrete end-to-end example — tracing the Z3 query, the refinement-type derivation, and the Refuted-Proof witness — for a bug in the unfiltered post-freeze set (Table 3) where the symbolic calculus rather than a pattern-matching path is the active component; this would convert the current disconnect between the formal contribution and the empirical headline from a structural ambiguity into a falsifiable claim.


Changes   +0 -0
Requests  1 Premium (5m 36s)
Tokens    ↑ 649.6k • ↓ 15.7k • 560.6k (cached)
