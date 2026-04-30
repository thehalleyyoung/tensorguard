● Read round02_reviewer.md
  │ .comet_neurips/round02_reviewer.md
  └ 72 lines read

● List repo and tex size (shell)
  │ ls -la
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/ &&
  │ wc -l /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/neurips.tex
  └ 46 lines...

● Read neurips.tex
  └ 93 lines read

● list paper dir (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/
  └ 34 lines...

● list sections (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5/ && wc -l /Users/halleyyoung/Documents/div/mathdiverge
  │ nce/halley-labs/tensorguard/docs/paper/sections_v5/*.tex
  └ 42 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ 1 line read

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ 160 lines read

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ 108 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:250 (250 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ 141 lines read

● Read impl_v6.tex
  │ docs/paper/sections_v5/impl_v6.tex
  └ 86 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L250:600 (351 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L600:845 (246 lines read)

● Read handler_soundness_table.tex
  │ docs/paper/sections_v5/handler_soundness_table.tex
  └ 85 lines read

● Read related_v6.tex
  │ docs/paper/sections_v5/related_v6.tex
  └ 53 lines read

● list lean (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/l
  │ ean/ 2>/dev/null; echo "---"; find
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ -name "*.lean" | head -20
  └ 13 lines...

● check sorries (shell)
  │ grep -c "sorry" /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/
  │ tensorguard/lean/TensorGuard/*.lean; grep -nE "axiom|sorry"
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ /TensorGuard/*.lean | head -40
  └ 12 lines...

## Summary
TensorGuard is a sound, no-execution refinement-type checker for PyTorch `nn.Module` forward methods. Tensor types carry a symbolic shape, a static gradient-required flag, and a Z3-decidable refinement; the calculus is composed at the class boundary by an assume/guarantee discipline; a backward verifier flags three silent-zero-grad bug classes; and TorchDynamo's runtime guards are claimed to be a one-directional shadow of the refinements. Empirically, TG returns Refuted-Proof on 53/60 historical bugs and beats Pytea 32/34 vs 22/34 on a 2022-catalogue modern subset (McNemar p=0.00195), while producing zero unconditional Refuted-Proof verdicts on the 488-block real-source corpus. A Lean 4 audit covers 28 of 79 shape-transfer handlers sorry-free and bit-mirrors them against torch 2.9.1 on 28k samples. The headline contribution is thus a refinement-typed shape+grad calculus with a partial Lean audit and a curated bug-corpus evaluation.

## Strengths
- The verdict taxonomy (RP / CV / LW / Abstain / V) is genuinely well-calibrated: the paper draws a clear line between which verdicts the soundness theorem covers (RP+CV) and which are conservative warnings (LW), and reports the headline `0 unconditional RP` on the 488-block corpus instead of hiding it (Sec. 5.1, Table 1). This level of self-disclosure is unusual and welcome.
- The Lean operator-rule audit is non-trivial: 28 handlers are formalised as total `List Nat` functions, the previously-axiomatic lemmas are closed sorry-free under `lake build`, and the rules are bit-mirrored against torch 2.9.1 on 28k in-fragment samples plus ~2.4k off-envelope boundary samples (Sec. 5.5).
- The fairness instrumentation around the Pytea comparison is careful: AST-screened fragment fairness applied at verification time on both sides, silent-skip correction documented, modern-subset N=34 with McNemar's exact p=0.00195 and a paired bootstrap CI (Sec. 5.1).
- The hybrid-falsification table (Table 4, N=25) genuinely demonstrates that TG and FakeTensor are complementary on importable inputs (TG-only 20, FT-only 5), rather than letting the 488-block coverage win be conflated with a tool-capability win.

## Weaknesses
- **The mechanised soundness story is much narrower than the paper title suggests.** Theorem 1 (`thm:soundness`) is whole-program, but its proof "reduces … to per-operator preservation lemmas already covered by the Lean rule audit", and that audit covers 28/79 handlers (Table 6 in `handler_soundness_table.tex`); the assume/guarantee composition Theorem (Thm 3) is explicitly mechanised "on the same 3-operator DSL (matmul/view/add)" (Sec. 3.1, lines 137–141). On the 488-block corpus the paper itself reports that only 36/185 in-soundness verdicts touch handlers in the Lean+pen-and-paper set, while 105/185 touch at least one tested-only handler (Sec. 5.5, lines 819–822). The headline phrase "Sound Static Verification … with a … Lean-Audited Operator-Rule Table" is therefore technically supported only on a thin slice of the verdicts the paper actually reports.
- **The 60-bug corpus is selected in a way that pre-removes the hardest cases.** The exclusion funnel (Sec. 5.1, "Bug-corpus exclusion denominator", lines 423–444) drops 124 distributed-shape bugs (auto-Abstain) and 113 "config-attribute" bugs that reduce to constructor sentinel-resolution — but config-attribute resolution is exactly what the paper sells as TG's central advantage over execution-based baselines (Sec. 1, lines 21–26 and "symbolic config" front-end in Sec. 4.1). Excluding them by protocol from the bug corpus while keeping them in the marketing argument is a form of double-booking. Please report the RP rate on the 113-bug config-attribute slice rather than discarding it.
- **The strongest external-validity test reverses the headline.** The "unfiltered pre-registered post-freeze" sample (Table 3, N=15) gives TG 5/15 vs FakeTensorMode 2/15 vs Pytea 3/15, with two-sided Fisher-exact p=0.39 (vs FT) and p=0.68 (vs Pytea). On the only sample drawn without filtering for fragment fit, TG is not statistically separable from either baseline (Sec. 5.1, lines 354–393). The 88.3% / 32-vs-22 numbers in the abstract should at minimum be contextualised by this; right now the abstract advertises only the curated-corpus numbers.
- **The Pytea comparison is on a frozen 2022 catalogue.** The fair-head-to-head subset (`N=34`) is "operators in Pytea's 2022 catalogue" with Pytea pinned at commit `cb02a8a` (2022-04-26) (Sec. 5.1, lines 273–294). Pytea has not been updated for SDPA, RMSNorm, modern attention, etc.; the McNemar p=0.00195 result mostly measures the four-year catalogue gap, not a methodological advance. A fairer comparison would either re-implement the missing handlers in Pytea's framework or down-scope TG to the same modern-operator surface and re-run *both* tools with all current handlers; the 2022 freeze is the wrong way around.
- **CEGAR and phase-check are admitted no-ops yet remain in the architectural pitch.** Section 5.4 / Table 5 explicitly state that L1 (CEGAR) and L3 (phase) "did not discriminate any case in the current implementation" and that on the real-corpus ablation (Sec. 5.4, lines 608–619) "none of CEGAR, device-flag, phase, grad-flow, or low-conf gating discriminates on the real bugs." The contributions list (C5, Sec. 1) acknowledges this and scopes the empirical contribution to three features, but the Implementation section still describes CEGAR and phase-check as live machinery. Either remove them from the system description or report a measurement on which they fire on real code.
- **The localisation 33/33 within ±5 lines is essentially uncalibrated.** Sec. 5.4 ("Localisation", lines 621–636) admits the AST-walk strategy and the heuristic ground truth share information for the patterns where the line marker was unavailable; only 3/3 marker-only items are independently scored, on which no localisation claim survives a sample size that small. The 33/33 figure should be removed from headline summaries until a ≥30-item marker-only audit is run; right now the conclusion paragraph leaves the impression that localisation is a solved problem.
- **The Dynamo-correspondence theorem is effectively never instantiated end-to-end.** Sec. 5.4 ("Empirical audit", lines 693–711) admits that "on 16 of the 17 modules the audit takes the documented `forward` signature as the contract because the full instantiated module exceeds end-to-end constraint solving; we therefore treat the 8.8%/97.9% row as a calibration of Dynamo against hand-written contracts rather than as an in-theorem instantiation of `thm:dynamo-corr`." The "End-to-end TG–Dynamo correspondence on constituent blocks" paragraph then exhibits 5 (five) modules. A theorem whose empirical instantiation is on 5 of the simplest torchvision/timm blocks is worth marking as preliminary, not as the C4 contribution it currently is.
- **The 0/RP-on-real-source result undermines the "no-execution alternative for modern ML code" framing.** On 488 real `nn.Module` blocks under the user-visible (free-symbolic-config) verdict regime, the paper reports `34 V / 0 RP / 206 LW / 248 A` (Sec. 5.1, line 77). I would like to see one example, in the body, of a `Refuted-Proof` verdict on a real upstream class (not on a hand-distilled or upstream-faithful re-extract), or an honest reframing of the contribution as "shape-bug verification on hand-distilled bug repros plus a coverage map on real code".

## Questions
- What is TG's RP rate on the 113 "config-attribute" bugs you excluded under exclusion rule (iv) (Sec. 5.1, line 437)? If the symbolic-config front-end is genuinely a contribution, this number is the actual evidence; please add it.
- For the unfiltered post-freeze sample (Table 3), can you run TG, FakeTensorMode, and Pytea on a substantially larger pre-registered post-freeze sample (e.g. N=60) drawn by the same query, and report the corrected confidence intervals? On N=15 the headline-advantage claim is undersupported.
- Of the 105/185 in-corpus verdicts that touch at least one tested-only handler (Sec. 5.5), how many would be downgraded from V/CV to "outside-soundness-scope" if the verdict were restricted to handler chains that are entirely Lean-audited or pen-and-paper?
- Please re-run the Pytea modern-subset comparison with TG also restricted to its handler intersection with Pytea's *current* HEAD (not its 2022-04-26 commit), or down-scope both tools to a contemporaneous catalogue. What is the resulting (TG, Pytea) refutation pair?
- For the post-freeze unfiltered RP-fire-rate of 6/15 with one false positive on `rb_uf_010`, what is the false-positive rate on the same 488-block corpus when low-confidence violations (the L5 feature in Table 5) are enabled? L5 contributes +8 verdicts on the stress benchmark, but its precision on real source is not reported.
- Is the assume/guarantee composition Theorem (Thm 3) extensible from the 3-operator DSL to the full handler set in principle, or does the proof rely on properties special to matmul/view/add? A one-paragraph statement of the obstacle would clarify the gap.

## Scores
Soundness: 2
Presentation: 3
Contribution: 2
Confidence: 4
Overall: 4

## Borderline reasons
The single change that would push the overall score from 4 to 5 is an unbiased external-validity result: a ≥60-item pre-registered post-freeze sample (no fragment-fit filter, no exclusion rules (iii)/(iv)) on which TG's catch-rate confidence interval is strictly above both FakeTensorMode's and a current-HEAD Pytea's. With the curated 60-bug corpus and the 2022-frozen Pytea baseline carrying the headline numbers and the only unbiased sample (N=15) showing no separation, the empirical contribution is currently below the bar.


Changes   +0 -0
Requests  7.5 Premium (2m 26s)
Tokens    ↑ 495.7k • ↓ 7.3k • 429.0k (cached)
