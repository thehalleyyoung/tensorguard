● Read round16_reviewer.md
  │ .comet_neurips/round16_reviewer.md
  └ 114 lines read

● List paper files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls neurips.pdf neurips.tex 2>&1 | head -20 && ls docs/paper/ 2>&1 | head
  │ -30
  └ 33 lines...

● List sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls docs/paper/sections_v5/ 2>&1; echo "---"; wc -l neurips.tex
  └ 14 lines...

● Read neurips.tex
  └ 93 lines read

● size of sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5 && wc -l intro_v6.tex calculus_v6.tex impl_v6.tex
  │ eval_v6.tex related_v6.tex limconc_v6.tex appendix_v6.tex
  └ 9 lines...

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ 134 lines read

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ 171 lines read

● Read impl_v6.tex
  │ docs/paper/sections_v5/impl_v6.tex
  └ 86 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:200 (200 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L200:500 (301 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L500:900 (391 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L891:1300 (373 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1264:1534 (270 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ 192 lines read

● Read related_v6.tex
  │ docs/paper/sections_v5/related_v6.tex
  └ 53 lines read

● List lean and benchmarks (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/ 2>&1 | head -20 && echo "---" && ls real_benchmarks/ 2>&1 | head
  │ -20
  └ 21 lines...

● Check for sorry in Lean (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && grep -rn "sorry" --include="*.lean" 2>&1 | head -20
  └ 12 lines...

## Summary
TensorGuard is a static refinement-type checker for PyTorch `nn.Module` classes that infers symbolic shapes plus a flat grad-flag lattice (`has_grad`/`no_grad`/⊤) from class source alone, without instantiating or tracing the model. The system emits a five-way verdict (Verified / Refuted-Proof / Contract-Violation / Library-Warn / Abstain), discharges shape obligations via Z3, and adds a backward verifier for canonical silent-zero-grad bugs and an assume/guarantee discipline at the class boundary. A Lean 4 audit closes 11/11 soundness lemmas sorry-free for 28 of the 79 shape-transfer handlers and mechanises a 17-operator composition theorem. Empirically the paper reports 53/60 Refuted-Proof on a curated historical bug corpus, 32/34 vs. Pytea 25/34 on a fragment-fair modern subset (McNemar p=0.0156), 0 unconditional RP on a 488-block real-source corpus under the user-visible free-symbolic regime, 7/7 RP on transcribed HF transformers PRs, and 5/15 catches on an unfiltered post-freeze sample (vs. 2/15 FakeTensor, 3/15 Pytea, not separable). A one-directional Dynamo-guard inclusion lemma (Thm. 5) is reported as exploratory.

## Prior weakness disposition
(none — first round)

## Strengths
- The Lean 4 audit is real and matches what the paper claims: `lean/TensorGuard/V5OperatorRules.lean`, `Extended.lean`, `AssumeGuaranteeExtended.lean`, and `Parity.lean` are all sorry-free, and the in-range restatement of `permList_compose` is honestly disclosed in the appendix. The handler-soundness scope table (28 Lean / 7 pen-and-paper / 44 tested-only) is the right level of calibration for a system of this size.
- The reporting discipline is unusually disciplined: the five-way verdict taxonomy, the per-corpus separation of headline RP from CV/LW, the explicit "0 unconditional RP under the user-visible regime" admission, and the Wilson/Clopper-Pearson intervals around small-N rates avoid the usual overclaiming. The McNemar pairing structure for the Pytea head-to-head ($b{=}7$, $c{=}0$) is a much stronger evidential design than a marginal comparison.
- Operating regime (no instantiation, no example inputs, no tracer) is genuinely novel relative to FakeTensorMode / torch.export / TorchDynamo, and the 481/488 N/A column for those baselines is a real structural advantage on HuggingFace-style code that needs a `config` to instantiate.
- The transcribed-from-real-PR result (7/7 RP on naturally-occurring HF Llama/Qwen2/Mistral/Phi-3 bug PRs with citations to specific PR numbers) is the most credible bug-finding evidence in the paper.

## Weaknesses
- **The natural-distribution bug-finding result is essentially negative and the paper acknowledges it.** On the 488-block real-source corpus the user-visible (free-symbolic-config) regime returns 0 unconditional Refuted-Proof verdicts (eval, "Calibration first" paragraph, lines 67–82). The 128 CV verdicts are sound only under synthesised caller-rely contracts and the 78 LW verdicts are explicitly "not a soundness claim." The headline soundness result therefore rests on (i) a 60-bug corpus the catalogue was built against, with an over-determined refute path (eval lines 813–820: AST-only and operator-only each independently refute 53/60, suggesting heavy fitting), and (ii) hand-transcribed PR repros. The paper has no clean win on naturally drawn library code without contract synthesis.
- **The fragment-fair head-to-head is against a frozen 2022 baseline.** Pytea's last upstream commit is `cb02a8a` (2022-04-26) per the paper's own audit (eval lines 535–541). On the same modern subset, `torch.compile(fullgraph=True)` with FakeTensor catches 34/34 (vs. TG's 32/34), as the paper concedes (lines 554–567). The McNemar p=0.0156 result is therefore against a stale tool that loses to a modern one in the same fragment. The Pytea comparison answers a 2022 question; it does not establish state-of-the-art bug-finding in 2026.
- **The pre-registered post-freeze evaluation, which is the only attempt at unbiased generalisation, fails to separate from baselines.** On the N=15 unfiltered sample TG catches 5/15 vs. FakeTensor 2/15 vs. Pytea 3/15; pairwise Fisher exact p=0.39 and 0.68, all BH-corrected to 1.00 (eval lines 686–692). The accompanying power calculation says N≥26 (vs. FakeTensor) and N≥77 (vs. Pytea) would be needed for α=0.05. A trend on N=15 is not a result; the paper would be more compelling with the second wave actually run rather than scoped.
- **Soundness coverage on the deployed system is much narrower than the headline suggests.** Of the 185 in-soundness verdicts on the 488-block corpus, only 36 (11 V + 25 CV) touch *only* Lean-or-pen-paper audited handlers; 105 touch at least one of the 44 tested-only handlers (eval lines 1346–1355). Theorem 1 therefore certifies <20% of the verdicts the paper reports on the real corpus. This gap should be in the abstract, not the appendix.
- **The grad-flag claim has a 25% worst-case runtime false-verified rate on the construct family that matters.** The held-out runtime harness on parameter-sharing / `torch.utils.checkpoint` subjects gives 6/8 RP and 2/8 silently Verified — a 25.0% false-verified rate on tied / renamed-attribute parameter sharing (eval lines 1487–1503; limconc lines 124–131). The 12% prevalence ceiling is a regex-detectable bound only and the authors say so. C3's "8/8 canonical bugs caught, 0/50 false positives" headline buries this caveat; the false-*negative* rate on the construct family the user actually cares about is the operative number.
- **The `500/500` static↔runtime backward-verifier agreement is on grammar-generated tiny modules**, not on a meaningful distribution (impl §3.2). The 10-module real-world sweep (resnet18, vit_b_16, bert-base, gpt2, …) is the relevant evidence and it explicitly excludes `torch.utils.checkpoint` and parameter sharing, which is exactly where the lattice is unsound. The sweep therefore measures the verifier on inputs where it is by construction expected to be correct.
- **Mutation-kill rates are weak for a soundness-oriented paper.** The triple-corpus union kill rate is 7/50 = 14%; even after the targeted handler-extension corpus is added, the conv2d kill rate is 53% and the union 60% on comparison/arithmetic mutations only (eval lines 1242–1272). 40–47% of single-edit mutants on the load-bearing handlers survive on three corpora combined. For a paper that leans on Lean for soundness messaging, this is a meaningful gap on the Python implementation, which is *not* mechanised.
- **Theorem 5 (Dynamo-guard correspondence) carries little theoretical weight.** It is a necessary-direction inclusion proved against a frozen torch 2.9.1 commit, audited on 17 modules of which 16 use a "documented forward-signature surrogate" rather than the actual TG-emitted contract, and the 8.8% in-contract recompile rate openly contradicts the equivalence reading (eval lines 1046–1064). C4 is reported as "exploratory" — fine — but it then occupies a contribution slot and a theorem environment as if it were load-bearing.

## Questions
- Run the second wave on the unfiltered post-freeze sample to N≈26 (the smaller of your own two power thresholds) and report whether the TG vs. FakeTensor gap survives Fisher-exact at α=0.05; this is the single experiment that would convert the directional trend in §4.1 into a separation.
- What is the actual TG vs. `torch.compile(fullgraph=True, dynamic=True)` head-to-head on the full 60-bug corpus (not just the modern 34 subset), enforcing the same fragment-fairness on TG, and what is the sample-paired McNemar?
- Quantify how many of the 53/60 historical bug catches survive when the operator catalogue is restricted to the 28 Lean-audited handlers plus the 7 pen-and-paper handlers (i.e., the in-soundness footprint only). The current LOO audits do not isolate this.
- For the parameter-sharing silent-error class, can the analyser be made to *Abstain* (rather than silently Verify) on detection of any attribute-aliasing pattern, even at the cost of additional abstentions on the 488-block corpus? Report the abstention-rate cost.
- The 60-bug corpus is "over-determined" (operator-only and AST-pattern-only each refute 53/60). On a held-out corpus that the catalogue and AST patterns have *not* been tuned against, what is the AST-pattern-only refute rate? If it remains high, the operator catalogue's claimed contribution to the headline is unsupported.
- For the 6 RP fires in the post-freeze soundness-footprint table (eval line 706), please report the per-fire result of replacing the in-soundness handler on the bug path with its tested-only counterpart, to confirm the in-soundness handler is the one actually emitting the witness.

## Scores
Soundness: 3
Presentation: 2
Contribution: 3
Confidence: 3
Overall: 5

## Borderline reasons
The single change that would push this to a clear accept is replacing the 53/60 headline (and its over-determined LOO audit on a corpus the catalogue was built against) with a comparable result on a held-out distribution where TG separates from `torch.compile`+FakeTensor under a paired test — the unfiltered post-freeze experiment is the right design but is run at N=15 and fails to separate, so the paper's strongest empirical claim is currently a directional trend against a 2022 baseline. Either running the pre-registered second wave or restricting the headline to the in-soundness handler footprint would do it; the second-most-impactful change would be tightening the eval section (currently 1533 lines, with extensive defensive bookkeeping) into a presentation a reviewer can read in one pass.


Changes   +0 -0
Requests  7.5 Premium (2m 52s)
Tokens    ↑ 1.0m • ↓ 7.2k • 966.4k (cached)
