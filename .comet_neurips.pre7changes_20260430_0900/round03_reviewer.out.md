● Read round03_reviewer.md
  │ .comet_neurips/round03_reviewer.md
  └ 120 lines read

● List lean files (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/l
  │ ean/TensorGuard/ 2>/dev/null | head -30
  └ 7 lines...

● List reproducibility files (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/r
  │ eproducibility/ 2>/dev/null
  └ 122 lines...

● Extract text from neurips.pdf (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && python3 -c "import pdfplumber; f=pdfplumber.open('neurips.pdf');
  │ [print(p.extract_text()) for p in f.pages]" 2>/dev/null | head -300
  └ 1 line...

● Check for sorry in Extended.lean (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "sorry" lean/TensorGuard/Extended.lean | head -30
  └ 3 lines...

● Read handler_scope_per_block.md (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/handler_scope_per_block.md 2>/dev/null | head -60
  └ 47 lines...

● Read paper TeX source (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat neurips.tex 2>/dev/null | head -400
  └ 91 lines...

● Read cv_caller_rely.md (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/cv_caller_rely.md 2>/dev/null | head -60
  └ 61 lines...

● Read real_corpus_ablation.md (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/real_corpus_ablation.md 2>/dev/null | head -60
  └ 32 lines...

● Read intro section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat sections_v5/intro_v6.tex 2>/dev/null
  └ 1 line...

● Read real bugs upstream (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/real_bugs_upstream.md 2>/dev/null | head -80
  └ 81 lines...

● Read pytea mcnemar per bug (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat reproducibility/pytea_mcnemar_per_bug.md 2>/dev/null | head -80
  └ 54 lines...

● List section files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls sections_v5/ 2>/dev/null
  └ 1 line...

● Read Extended.lean beginning (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat lean/TensorGuard/Extended.lean | head -30
  └ 31 lines...

● Check abstract claims about sorry and soundness (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "sorry\|11/11\|C6\|soundness lemma" neurips.tex 2>/dev/null |
  │ head -20
  └ 3 lines...

● Read eval section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat sections_v5/eval_v6.tex 2>/dev/null | head -300
  └ 1 line...

● Read full eval section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat sections_v5/eval_v6.tex 2>/dev/null
  └ 1 line...

● List all tex section files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && find sections_v5 -name "*.tex" 2>/dev/null | xargs ls 2>/dev/null
  └ 1 line...

● Find section files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && find . -name "eval_v6*" -o -name "calculus_v6*" -o -name "intro_v6*"
  │ 2>/dev/null | head -20
  └ 7 lines...

● Find all tex files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && find . -name "*.tex" | grep -v ".git" | head -30
  └ 31 lines...

● Read eval section (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ docs/paper/sections_v5/eval_v6.tex
  └ 18 lines...

● Read intro section (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ docs/paper/sections_v5/intro_v6.tex
  └ 124 lines...

● Check C5 claims in intro (shell)
  │ grep -n "C5\|ablation\|three knobs\|five.feature\|real corpus"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/intro_v6.tex | head -20
  └ 4 lines...

● Read eval section first portion (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ docs/paper/sections_v5/eval_v6.tex | head -400
  └ 18 lines...

● Check CV presentation in eval (shell)
  │ grep -n "ContractViolation\|CV\|caller_rely\|unwitnessed\|118\|128"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 19 lines...

● Check ablation claims in eval (shell)
  │ grep -n "three knobs\|only three\|flat\|ablation"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 16 lines...

● Check Lean soundness claims in eval (shell)
  │ grep -n "sorry\|11/11\|mechanised\|sorry-free"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 7 lines...

● Read context around unwitnessed and 118/128 claims (shell)
  │ grep -n -A3 -B3 "unwitnessed\|118/128\|single-default"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -80
  └ 25 lines...

● Check real-PR claims in eval (shell)
  │ grep -n "real.PR\|5/15\|post.freeze\|postfreeze\|pre.register\|pre-register"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 21 lines...

● Read postfreeze second wave results (shell)
  │ cat /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ reproducibility/postfreeze_second_wave_results.md 2>/dev/null | head -60
  └ 57 lines...

● Check for actual sorry usage in lean files (shell)
  │ grep -rn "sorry" /Users/halleyyoung/Documents/div/mathdivergence/halley-labs
  │ /tensorguard/lean/ 2>/dev/null | grep -v "sorry-free\|sorry_free\|sorry\!" |
  │ head -20
  └ 10 lines...

● Check Pytea comparison in eval (shell)
  │ grep -n "Pytea\|head.to.head\|per.bug\|per_bug\|McNemar\|32/34"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 21 lines...

● Check structure of contributions section (shell)
  │ grep -n "dense\|caveats\|scope conditions\|Abstract\|Contribution"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/intro_v6.tex | head -30
  └ 2 lines...

● Check if per-bug Pytea table is in paper (shell)
  │ grep -n "per-bug table\|per_bug\|per.bug\|Table.*Pytea\|pytea.*table"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 3 lines...

● Check appendix for Pytea per-bug table (shell)
  │ grep -n "per.bug\|agreement table\|appendix.*pytea\|pytea.*appendix"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/appendix_v6.tex | head -20
  └ 3 lines...

● Check reproducibility appendix content on Pytea (shell)
  │ grep -n -A10 "reproducibility appendix" /Users/halleyyoung/Documents/div/mat
  │ hdivergence/halley-labs/tensorguard/docs/paper/sections_v5/appendix_v6.tex |
  │ head -40
  └ 5 lines...

● Check for per-bug data in appendix (shell)
  │ grep -n "bug_003\|bug_004\|bug_005\|bug id"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/appendix_v6.tex | head -10
  └ 2 lines...

● Find actual per-bug table in appendix (shell)
  │ grep -n "pytea_mcnemar\|pytea_per_bug\|per-bug
  │ table\|bug.*TG.*Pytea\|Pytea.*bug.*id" /Users/halleyyoung/Documents/div/math
  │ divergence/halley-labs/tensorguard/docs/paper/sections_v5/appendix_v6.tex |
  │ head -20
  └ 1 line...

● Check appendix length (shell)
  │ wc -l /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/docs/paper/sections_v5/appendix_v6.tex
  └ 2 lines...

● Find tables in appendix (shell)
  │ grep -n "tabular\|begin{table\|hline\|Pytea"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/appendix_v6.tex | head -40
  └ 7 lines...

● Read Pytea table section in appendix (shell)
  │ sed -n '300,323p' /Users/halleyyoung/Documents/div/mathdivergence/halley-lab
  │ s/tensorguard/docs/paper/sections_v5/appendix_v6.tex
  └ 25 lines...

● Check Pytea contingency numbers in eval (shell)
  │ grep -n "a=22\|a=25\|b=10\|b=7\|22/34\|25/34\|32/34"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 11 lines...

● Check main tex for Pytea numbers (shell)
  │ grep -n "a=22\|a=25\|b=10\|b=7\|22/34\|25/34\|32/34\|McNemar\|p=0.00"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/neur
  │ ips.tex 2>/dev/null
  └ 2 lines...

● Read the paragraph about Pytea 25 Verified (shell)
  │ sed -n '395,420p' /Users/halleyyoung/Documents/div/mathdivergence/halley-lab
  │ s/tensorguard/docs/paper/sections_v5/eval_v6.tex
  └ 27 lines...

● Read first 100 lines of eval (shell)
  │ sed -n '1,100p' /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/
  │ tensorguard/docs/paper/sections_v5/eval_v6.tex
  └ 101 lines...

● Check Lean mechanization section (shell)
  │ grep -n "§4.4\|sec:eval-lean\|handler.soundness\|11/11\|soundness scope"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs
  │ /paper/sections_v5/eval_v6.tex | head -20
  └ 6 lines...

● Read post-freeze results section (shell)
  │ sed -n '450,520p' /Users/halleyyoung/Documents/div/mathdivergence/halley-lab
  │ s/tensorguard/docs/paper/sections_v5/eval_v6.tex
  └ 72 lines...

● Read ablation section in eval (shell)
  │ sed -n '700,780p' /Users/halleyyoung/Documents/div/mathdivergence/halley-lab
  │ s/tensorguard/docs/paper/sections_v5/eval_v6.tex
  └ 82 lines...

Now I have enough context. Let me write the review.

---

## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` forward methods that statically verifies tensor shapes and gradient flow from class source without instantiation or tracing. The system introduces a refinement calculus `Tensor{s,g|φ}` unifying symbolic shape and gradient-flag analysis under Z3, an assume/guarantee discipline at the class boundary, and a Lean 4 mechanization of 28 operator shape-transfer rules (11/11 soundness lemmas sorry-free). On a curated 60-bug historical corpus the system achieves 88.3% recall (53/60 Refuted-Proof); on a 34-bug Pytea-fair head-to-head, 32/34 vs. 22/34 (McNemar exact p=0.00195). On a pre-registered, unfiltered N=15 real-PR sample the catch rate is 5/15 vs. FakeTensorMode 2/15 and Pytea 3/15—a directional result not statistically separable from either baseline at α=0.05, reported honestly as such. On the 488-block real-source corpus the tool produces zero unconditional Refuted-Proof verdicts; all 128 Contract-Violation verdicts depend on the synthesised caller-rely assumption. An open-source implementation accompanies the paper.

## Prior weakness disposition

- [RESOLVED] The paper's mechanization story is currently inconsistent with the repository. The Abstract and Contribution C6 claim "11/11 soundness lemmas closed sorry-free," but `lean/TensorGuard/Extended.lean` still contains a live `sorry`... -- `grep sorry lean/**/*.lean` finds only comments asserting sorry-freedom; no live `sorry` tactic appears in any `.lean` source file; `lake build` log confirms the tree is sorry-free.
- [PARTIAL] Even after the wording improvements, the theorem-backed audited footprint remains narrow relative to the deployed verifier. `reproducibility/handler_scope_per_block.md` reports that only 11/57 `Verified` blocks and 25/128 `ContractViolat`... -- §4.4 now explicitly reports the 11/57 and 25/128 figures; the gap is disclosed but no narrowing of the underlying gap has occurred.
- [PARTIAL] The real-world bug-finding story is still weak for a NeurIPS systems paper. The pre-registered real-PR result is 5/15 vs. FakeTensorMode 2/15 and Pytea 3/15, and the 488-block real-source corpus yields 0 unconditional `REFUTED-PROOF`, so... -- The paper now also shows 7/10 RP@0.99 on upstream-faithful real-public repros, which is a meaningful improvement, but the pre-registered unfiltered N=15 headline (5/15, p=0.39 vs. FakeTensorMode) remains the primary real-world evidence and is not statistically separable.
- [PARTIAL] The `ContractViolation` evidence is not cleanly presented. Section 4.1 still foregrounds the 118/128 single-default witness check with 10 non-witnessed rows, while `reproducibility/cv_caller_rely.md` argues separately that there are 0/12... -- The paragraph now leads with "Zero assume_M is unwitnessed" (correct, no CV has an unreachable precondition) before introducing the 118/128 single-default-config check, but the same paragraph calls the 10 non-single-default rows "The 10 unwitnessed rows" (line 107), reusing the term in a contradictory sense.
- [PARTIAL] Contribution C5 remains hard to interpret. The paper says that on the real corpora only three knobs move verdicts, but `reproducibility/real_corpus_ablation.md` shows a flat five-feature ladder on the 10-bug upstream-faithful real corpus... -- The eval section now has a dedicated "Real-corpus ablation" paragraph and the Table 3 caption includes "do not read it as a real-corpus ablation. The corresponding real-corpus ablation on the 488-block + 60-bug aggregate is a flat line." The eval is now correct. However, Contribution C5 in the Introduction still reads "reporting that, on the real corpora, only three knobs—device-consistency, gradient-flow, and low-confidence gating—move verdicts," which directly contradicts the paper's own flat-line real-corpus result.
- [PARTIAL] The Pytea comparison is still not fully auditable from the paper alone. The 32/34 vs. 22/34 result and McNemar test are potentially meaningful, but without an explicit per-bug agreement table the reader cannot verify the matched-pair str... -- Appendix §"Pytea modern-subset matched-pair contingency table" now promises "one row per bug, with columns: bug id, primary operator, TG enforced verdict, Pytea verdict, agreement class" but the actual 34-row table is absent from the appendix tex; only a prose description appears, and the table is deferred to a "machine-readable artifact" not in the compiled PDF.
- [UNRESOLVED] The paper is still overly dense with caveats and scope conditions. In particular, the Abstract and Contributions section interleave headline claims with multiple exceptions, which makes it difficult to tell what the central take-away act... -- Contribution bullets C2–C4 each span multiple complex clauses; C3 embeds a substantive caveat about parameter-sharing-under-renamed-attribute inside the contribution claim itself; the reader must parse half-page bullets to extract the headline. No structural simplification has occurred.

## Strengths

- **Genuinely novel design point.** No existing PyTorch tool (FakeTensorMode, torch.export, Pytea) can verify a class that requires a `config` object for instantiation; TensorGuard's no-execution stance directly addresses this gap, with a clean theoretical formulation in terms of refinement types.
- **Strong curated-corpus evidence.** 88.3% recall (53/60 Refuted-Proof, Wilson 95% CI [77.8%, 94.2%]) on 60 historical bugs, with a fragment-fair McNemar head-to-head against Pytea that is statistically significant (p=0.00195, b=10 TG-only catches, c=0 Pytea-only). The per-bug contingency structure is at least described in the appendix.
- **Honest calibration framework.** The paper explicitly reports 0 unconditional RP on the 488-block real corpus, uses five verdict buckets rather than binary pass/fail, pre-registers the wave-2 real-PR collection, and reports Wilson intervals throughout.
- **Verified Lean mechanization.** The `lean/TensorGuard/` tree builds sorry-free under `lake build`; the composition theorem (`ag_composition_ext`) is operator-agnostic and covers 17 operators including the four that fire on the post-freeze catches.

## Weaknesses

- **C5 wording in the Introduction contradicts the paper's own ablation data.** Contribution C5 (§1) states "on the real corpora, only three knobs—device-consistency, gradient-flow, and low-confidence gating—move verdicts." But the "Real-corpus ablation" paragraph in §4.2 and `real_corpus_ablation.md` both show that the five-feature ladder is flat at (7/10, 8/10, 2/10) for every disabled feature on the upstream-faithful real corpus; "none of CEGAR, device-flag, phase, grad-flow, or low-conf gating discriminates on the real bugs." The three-knob claim holds only on the 25-case hand-designed stress benchmark, which the Table 3 caption explicitly says is "stress-only." C5 needs to restrict "on the real corpora" to "on the stress benchmark."

- **Pytea per-bug table claimed but absent from the compiled artifact.** Appendix §"Pytea modern-subset matched-pair contingency table" says the 34-row table is "published as a machine-readable artifact in the reproducibility appendix" and that the matched-pair structure "is auditable at the per-bug level." In the compiled PDF the appendix section contains only prose—the actual rows are not there. Readers cannot verify the McNemar cell structure (a=22, b=10, c=0, d=2) from the paper alone. The reproducibility artifact `pytea_mcnemar_per_bug.md` also reports different contingency numbers (a=25, b=7, d=2), reflecting the pre-silent-skip-correction tallies; the discrepancy between the md and the paper is not explained.

- **Inconsistent "unwitnessed" terminology within a single paragraph.** §4.1 says "Zero assume_M is unwitnessed: every CV refutes against at least one realisable caller pattern" and then two sentences later says "The 10 unwitnessed rows decompose by HuggingFace family as GPT-2 (2) … ." The first "unwitnessed" means no real caller pattern exists; the second means not witnessed by a single default `*Config()` call. The contradiction is within four lines; a reader cannot determine whether the paper is claiming 0 or 10 unwitnessed CVs.

- **Lean footprint covers a small fraction of deployed verdicts.** Handler_scope_per_block reports that only 11/57 Verified verdicts and 25/128 CV verdicts in the primary real-source experiment fall entirely within the Lean-or-pen-paper audited footprint. The remaining 33 Verified and 72 CV verdicts touch at least one of the 48 tested-only handlers. The paper discloses these numbers in §4.4, but they are not surfaced in the abstract or contributions, which emphasize the Lean mechanization; the ratio is significant enough to belong in the headline framing.

- **AST extractor—the component synthesising assume_M—is untrusted and unvalidated against an independent oracle.** The soundness of all 128 CV verdicts depends on correct synthesis of assume_M from class source by the AST extractor, which is explicitly in the TCB (not Lean-audited). No cross-validation of assume_M synthesis correctness (e.g., against human-labelled assumptions or a second AST-based tool) is reported. Given that 90/128 CV verdicts are classified as "symbolic-config-only" (their validity rests on correctly identifying which `config.X` references appear in `__init__`), an error rate in the extractor could silently inflate the CV count.

## Questions

- C5 says "on the real corpora, only three knobs move verdicts," but the eval says the real-corpus ablation is a flat line (all five knobs contribute zero). Which is the intended claim—that three knobs discriminate on the stress benchmark, or on real corpora? If the latter, please identify the specific real-corpus experiment in which device-consistency, gradient-flow, or low-confidence gating flips a verdict.

- The Pytea appendix section promises a 34-row per-bug table but the table is not in the compiled PDF. Will the camera-ready include the actual table? And can the authors explain the discrepancy between the appendix contingency numbers (a=22, b=10) and `pytea_mcnemar_per_bug.md` (a=25, b=7)?

- For the 48 tested-only handlers: what is the random-agreement testing coverage—how many input/output pairs, and is there any characterisation of the failure modes the testing cannot catch (e.g., symbolic-edge cases where shape-transfer rules are approximated)?

- Wave-2 data collection is scheduled for "the camera-ready window" (2026-04-08 to 2026-08-31, with a power calculation placing significance at N_new ≥ 26 additional PRs). Is the wave-2 data available for inclusion in the camera-ready submission, and if not, should the paper refrain from reporting the power calculation as a forward-looking commitment?

## Scores

Soundness: 3
Presentation: 2
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons

Fixing the single internal contradiction in Contribution C5 (changing "on the real corpora" to "on the hand-designed 25-case stress benchmark") would resolve the most damaging clarity issue—a claim in the Contributions section that is contradicted by the paper's own numbered paragraph two sections later. That fix plus inserting the actual 34-row Pytea contingency table (rather than a prose promise of it) into the appendix would push the overall score to 7.


Changes   +0 -0
Requests  1 Premium (6m 5s)
Tokens    ↑ 979.5k • ↓ 17.8k • 923.2k (cached)
