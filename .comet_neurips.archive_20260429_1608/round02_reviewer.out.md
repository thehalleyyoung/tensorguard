● Read round02_reviewer.md
  │ .comet_neurips/round02_reviewer.md
  └ 139 lines read

● List relevant files (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/.
  │ comet_neurips/ 2>/dev/null; ls *.tex *.pdf 2>/dev/null | head
  └ 5 lines...

● Check sizes and prior round (shell)
  │ wc -l neurips.tex && cat .comet_neurips/round01_reviewer.out.md | head -200
  └ 83 lines...

● Read improver response (shell)
  │ cat .comet_neurips/round01_improver.out.md | head -150
  └ 150 lines...

● Check timestamps (shell)
  │ stat -f "%Sm" neurips.pdf neurips.tex; wc -l docs/paper/sections_v5/*.tex
  │ 2>/dev/null
  └ 33 lines...

● Read neurips.tex
  └ 90 lines read

● Check eval section (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "checkpoint\|published\|jointly\|holdout\|catalogue\|recompile\|C
  │ EGAR\|phase\|N=15\|N=60\|N=30\|marker\|localis\|shared\|grad-flag\|round-"
  │ docs/paper/sections_v5/eval_v6.tex | head -80
  └ 58 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L60:220 (161 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L260:470 (211 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L470:700 (231 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L700:782 (83 lines read)

● Check rebuttal language (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "round\|Q[0-9]\|rebuttal\|W[0-9]\|honest\|admit"
  │ docs/paper/sections_v5/*.tex | grep -v "_v5\|_v6\|round-\|round " | head -30
  └ 1 line...

● Check rebuttal language (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "round\|rebuttal\|honest\|admit"
  │ docs/paper/sections_v5/eval_v6.tex docs/paper/sections_v5/intro_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex | head -40
  └ 8 lines...

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ L1:101 (101 lines read)

## Summary
TensorGuard is a static, no-execution refinement-type checker for PyTorch `nn.Module` forward methods that reasons about symbolic shapes and a flat first-order grad-flag lattice (`{has_grad, no_grad, ⊤}`), discharging side conditions to Z3, and reporting a five-way verdict taxonomy (`Verified / Refuted-Proof / Contract-Violation / Library-Warn / Abstain`). The empirical evaluation reports 53/60 RP on a curated historical bug corpus, 32/34 vs. Pytea 22/34 on a fragment-fair modern subset (McNemar p=0.00195), 5/15 catches on the unfiltered pre-registered post-freeze N=15 PR sample (vs. FakeTensorMode 2/15 and Pytea 3/15, point-above but Fisher-not-separable), 0 unconditional RP on the 488-block real-source corpus under the free-symbolic regime, and a Lean 4 audit covering 28/79 shape-transfer handlers with 11/11 axiomatic soundness lemmas now closed sorry-free. The Dynamo-guard correspondence (Thm. 5) is necessary-direction only and now empirically audited on an extended 14-module corpus (in addition to the 17-module audit), with $48/544$ ($8.8\%$) in-contract recompile rate and zero out-of-catalogue SHAPE/DTYPE/RANK guards observed.

## Prior weakness disposition
- [UNRESOLVED] The unconditional-RP headline rests almost entirely on a curated 60-bug historical corpus (53/60). On the 488-block real-source corpus the user-visible regime returns zero unconditional RP -- The unfiltered post-freeze sample is still N=15 with two-sided Fisher p=0.39 vs. FakeTensorMode and p=0.68 vs. Pytea; a Bayesian BF supplement was added but the empirical-superiority headline is still not statistically separable on the only unfiltered, pre-registered evaluation.
- [RESOLVED] The title advertises "Sound Static Verification … with a 28/79-Handler Lean-Audited … Calculus", but only 28/79 ≈ 35% of handlers are Lean-audited -- The title now reads "Refinement-Type Verification of nn.Module Shapes and Gradient Flow with a Lean-Audited Operator-Rule Table"; "Sound Static Verification" and "28/79" are removed and the abstract explicitly carves out the analyser/AST extractor/backward verifier/Z3 dispatch as TCB.
- [PARTIAL] The 488-block "0 unconditional RP / 57 Verified" headline depends critically on the synthesised caller-rely `assume_M`. CV-witness audit cites 26/128 empty assumes, 90/128 documented config attributes -- 12 randomly-sampled CV verdicts are now each paired with a documented `transformers` `*Config`-default instantiation and a named published checkpoint that witnesses the joint $\mathit{assume}_M$, but the all-128-or-uniform-subsample-with-CI joint-realisability ratio asked for last round is still not reported.
- [PARTIAL] Theorem 5 is necessary-direction-only and the empirical audit reports an 8.8% in-contract recompile rate; falsification predicate exercised on only 17 modules and 48 in-contract recompiles all in INT bucket -- Audit is now extended to a further 14 importable blocks (9 CNN end-to-end, 4 transformer + 1 ResNet50 layer surrogate) with all 19 recompile events classified `{shape:19, dtype:0, rank:0, int:0}` and zero out-of-catalogue guards; total ≈31 modules still falls well short of the ≥100 threshold the prior round named.
- [RESOLVED] The per-feature ablation on the real corpus is "a flat line": CEGAR contract discovery and the train/eval phase check are honestly reported as no-ops -- C5 is now explicitly restricted to the three discriminative knobs (device-consistency, gradient-flow, low-confidence gating) and the contributions paragraph states "the unused CEGAR loop and the always-satisfiable phase encoder ship with the analyser but are not claimed as contributions."
- [PARTIAL] The first-order grad-flag lattice `{has_grad, no_grad, ⊤}` is silently incorrect on parameter-sharing-under-renamed-attribute; prevalence ≤12% by self-conducted GitHub sweep with no independent corroboration -- A grad-flag silent-error audit on the 16 importable Track-E modules reports 0/16 `torch.utils.checkpoint` and 0/16 renamed-attribute parameter sharing, but this is a same-author pattern check on the same 17-module fixture, not the held-out HF training-script false-verified-rate measurement requested.
- [RESOLVED] The 33/33 within-±5-line localisation result is, by the paper's own admission, a "consistency check, not a precision claim" -- A 30-item marker-only audit (independent `# BUG`-comment ground truth) is now reported: 14/17 within ±5 lines and 11/17 within ±1 of the marker on the 17 cases TG refutes, with the 13 non-computable cases explicitly named as the relevant coverage gap and the tracer relegated to engineering rather than a research contribution.
- [RESOLVED] Presentation: paper is exceptionally dense with rebuttal-style apparatus (round-2 Q4, round-3 Q6, etc.); abstract ~22 sentences -- Round-N markers and rebuttal-style narration are removed from `intro_v6`, `eval_v6`, and `limconc_v6`; the abstract is now ~10 sentences and reads as a calibrated headline (53/60 on the historical corpus, $32/34$ vs. Pytea, $5/15$ on the unfiltered post-freeze sample with explicit non-separability, $0$ unconditional RP on the 488-block surface as a fragment-coverage measurement); §4 is structured as paragraphs rather than a per-round patch stack.

## Strengths
- The title/abstract cleanup is substantive: "Sound Static Verification" is gone, the Lean audit is correctly described as an operator-rule table audit (28/79 with $11/11$ axiomatic lemmas closed sorry-free, $28{,}000/28{,}000$ byte-mirror against `torch 2.9.1`), and the abstract explicitly carves out the analyser implementation, AST extractor, backward verifier, and Z3 dispatch as TCB. This brings the title in line with what the artefact actually certifies.
- Calibration discipline remains unusually high: the $0$ unconditional RP on the $488$-block free-symbolic surface is still reported as the headline; the $5/15$ post-freeze catch is reported with Wilson CI $[15.2\%, 58.3\%]$ and explicit Fisher non-separability; the off-axis fire on `rb_uf_010` is recorded as a false positive against ground truth; CEGAR and the always-satisfiable phase encoder are dropped from the contributions list.
- The CV-witness audit now exhibits 12 randomly-sampled CV verdicts each paired with a published-checkpoint config that witnesses $\mathit{assume}_M$ — a real step beyond "the symbols exist in some config".
- The 30-item marker-only localisation audit (independent `# BUG` ground truth, no information shared with the AST-walk strategy) replaces the AST-coupled $33/33$ "consistency check" with a properly held-out measurement (14/17 ±5, 11/17 ±1 on the 17 refuted cases, 13/30 explicitly named as coverage gap).

## Weaknesses
- W1 (UNRESOLVED): the unfiltered pre-registered post-freeze evaluation is still $N{=}15$. The Bayesian supplement ($\mathrm{BF}_{10}{=}8.1$ vs. FT, $3.6$ vs. Pytea) does not exceed the conventional "strong evidence" threshold $\mathrm{BF}{=}10$, and the frequentist Fisher gap remains $p{=}0.39$ / $0.68$. The headline claim that TG catches real PyTorch shape bugs that execution-based tools cannot is therefore still not statistically separable from the baselines on the only unfiltered, pre-registered evaluation. Either expand the pre-registered post-freeze sample to $N\ge 60$ on the same query and report the resulting Fisher comparisons, or restate the contribution as a calibrated-coverage result.
- W3 (PARTIAL): the 12-CV joint-realisability audit is a sample of $12/128$ ($\sim 9.4\%$) selected as "12 randomly-sampled CV verdicts". The prior round asked for either the full 128-set or a uniformly random subsample with a stated CI on the joint-realisability ratio. With $12/12$ witnessed and no explicit Wilson/Clopper-Pearson CI on $128$, the published one-sided lower bound on the ratio is $\sim 80.4\%$ Clopper-Pearson at $12/12$, which is a meaningfully weaker statement than the headline. Report the audit on either the full $128$ or a power-justified random subsample with the resulting witnessed-ratio CI.
- W4 (PARTIAL): the Dynamo-falsification corpus is now $\sim 31$ modules ($17$ original + $14$ extended), still well short of the $\ge 100$ timm/HF blocks the prior round asked for, and 4 of the 14 extended modules use the documented forward-signature surrogate rather than full instantiation. The "zero out-of-catalogue SHAPE/DTYPE/RANK guards" result on $19$ aggregate recompile events is therefore based on a small denominator, and the surrogate split (9 CNN end-to-end vs. 4 transformer surrogate) is not separated in the headline. Either run the falsifier on $\ge 100$ end-to-end modules or restrict Theorem 5's empirical claim to the CNN-block subset.
- W6 (PARTIAL): the grad-flag silent-error audit ("$0/16$ `torch.utils.checkpoint`, $0/16$ renamed-attribute parameter sharing") is a same-author pattern check on the same $17$-module Theorem 5 fixture, with the patterns mechanically grep-detected. The prior round asked for the backward verifier to be run on a held-out set of HF training scripts containing this construct and the false-verified rate reported. Either provide that measurement (a positive sample from a HF-training-scripts sweep, with the verifier's verdict per script and the resulting false-verified rate), or restrict C3 to non-shared-parameter modules in the contributions list rather than only in the limitations section.
- The catalogue-coverage residual $12/78$ "could-in-principle convert to RP" upper bound on the LW→RP gap is asserted in §4.1 but not exhibited at the per-block level inside the body. Provide a per-block list of the $12$ "fragment-only forward bodies" so the upper bound can be independently checked, or reduce the claim to "≤12/78" without the structural decomposition.

## Questions
- For the 12-CV joint-realisability audit, what is the Clopper-Pearson 95% CI on the witnessed-ratio over 128, and was the random-sampling seed pre-registered? Please report the full $128$-set witnessed-ratio.
- The post-freeze $N{=}15$ table reports `rb_pf_002`, `rb_pf_005`, `rb_pf_006` as silent verifieds in the constructor-bound integer-attribute envelope class. On the round-6 envelope synthesiser specifically (init-time local-scalar fold; single-dim shape-alias recogniser; shape-tuple propagator), how many of the $5/15$ catches are *only* enabled by the round-6 additions, and how many would also be caught by the v4 path?
- For the extended 14-module Dynamo audit, does the partition `{shape:19, dtype:0, rank:0, int:0}` survive when restricted to the 9 fully end-to-end CNN blocks (i.e., excluding the 4 surrogate transformer blocks and the ResNet50 layer)? If yes, the falsification claim for Theorem 5 is much stronger.
- The handler-soundness scope table (Table) maps $48$ of $79$ handlers as "tested-only". Of the $5/15$ catches in the post-freeze unfiltered sample, how many fire through a tested-only handler vs. a Lean-audited or pen-and-paper handler? This bears directly on the calibrated soundness scope of the empirical headline.
- The marker-only audit reports $17/30$ refuted with $\pm 5$-line accuracy on $14/17$. What is the verdict on the $13/30$ non-computable cases — is the breakdown silent-verified vs. explicit-abstain, and how many fall into the constructor-bound integer-attribute envelope class?
- For the grad-flag silent-misclassification audit, the $\le 12\%$-of-training-scripts prevalence is cited as a self-conducted GitHub sweep. What query, denominator, and date were used, and is there a held-out positive sample on which the backward verifier's false-verified rate can be measured directly?

## Scores

Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 4
Overall: 6

## Borderline reasons
The single change that would push my overall from 6 to 7 is a substantively larger pre-registered post-freeze sample ($N \ge 60$) on the same frozen GitHub-search query, with the resulting Fisher-exact two-sided $p$-value vs. FakeTensorMode and Pytea reported; a $p<0.05$ on either head-to-head would convert the calibrated-confidence framing of Table 4 into a statistically separated empirical-superiority claim on the only unfiltered surface. Alternatively, completing the 128-CV joint-realisability audit with a stated witnessed-ratio CI would solidify the $488$-block calibration story and similarly justify a 7.


Changes   +0 -0
Requests  7.5 Premium (2m 52s)
Tokens    ↑ 745.3k • ↓ 8.7k • 695.9k (cached)
