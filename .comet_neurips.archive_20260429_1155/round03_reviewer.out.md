● Read round03_reviewer.md
  │ .comet_neurips/round03_reviewer.md
  └ 72 lines read

● List comet_neurips dir (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/.
  │ comet_neurips/ 2>&1 | head -30
  └ 23 lines...

● Inspect key files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l neurips.tex neurips.txt README.md API.md 2>&1; ls src/ benchmarks/
  │ tests/ 2>&1 | head -60
  └ 66 lines...

● Read neurips.tex
  └ 93 lines read

● List sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls sections_v5/ && wc -l sections_v5/*.tex 2>&1 | tail -20
  └ 2 lines...

● Find tex sources (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && find . -path ./node_modules -prune -o -name "sections_v5" -print
  │ 2>/dev/null; find . -name "*.tex" 2>/dev/null | head -30
  └ 32 lines...

● Section sizes (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l docs/paper/sections_v5/{intro_v6,calculus_v6,impl_v6,eval_v6,relate
  │ d_v6,limconc_v6,appendix_v6}.tex
  └ 9 lines...

● Look at round 2 review (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat .comet_neurips/round02_reviewer.out.md | head -100
  └ 101 lines...

● Round 2 cont (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat .comet_neurips/round02_reviewer.out.md | sed -n '100,200p'
  └ 22 lines...

● Look at round 3 improver output (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && tail -200 .comet_neurips/round03_improver.out.md
  └ 201 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:100 (100 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L100:260 (160 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L260:500 (241 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L500:700 (201 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L700:855 (146 lines read)

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ L1:141 (141 lines read)

## Summary
TensorGuard is a sound, no-execution refinement-type checker for PyTorch `nn.Module` forward methods. Tensor types carry a symbolic shape, a static `requires_grad` flag, and a Z3-decidable refinement; an assume/guarantee discipline composes contracts at the class boundary; a backward verifier targets three silent-zero-grad classes; and TorchDynamo's runtime guards are placed in a one-directional (necessary) correspondence with the static refinements. Empirically TG returns Refuted-Proof on 53/60 historical bugs (Wilson [77.8%, 94.2%]) and beats Pytea 32/34 vs 22/34 on a 2022-catalogue modern subset (McNemar exact p=0.00195), while producing zero unconditional Refuted-Proof verdicts on the 488-block real-source corpus. A Lean 4 audit closes 11/11 previously-axiomatic soundness lemmas sorry-free over 28 of 79 shape-transfer handlers and bit-mirrors them against torch 2.9.1 on 28k in-fragment samples. The headline contribution is a refinement-typed shape+grad calculus with a partial Lean audit, a Dynamo-guard correspondence on the catalogue intersection, and a curated bug-corpus evaluation supplemented by a transparent block-corpus coverage measurement.

## Strengths
- The verdict taxonomy (RP / CV / LW / Abstain / V) and the explicit scope of `thm:soundness` (RP+CV only; LW excluded) is well-engineered and consistently applied. The `0` unconditional RP on the 488-block corpus is reported as the headline rather than hidden (Sec. 5.1, Tab. 1, lines 76–82).
- Wilson / Clopper-Pearson intervals and Fisher-exact p-values were added at every headline this round (Sec. 5.1: 5/15 → [15.2%, 58.3%]; 7/10 → [34.8%, 93.3%]; 53/60 → [77.8%, 94.2%]; 32/34 → [80.9%, 98.4%]). The pre-registered unfiltered post-freeze sample now states honestly that TG is not statistically separable from FakeTensor (p=0.39) or Pytea (p=0.68) at α=0.05 (lines 361–365).
- The Lean audit is non-trivial and now reconciled to the verdict surface: per-block scope mapping yields V 11/57 + CV 25/128 = 36/185 in-soundness verdicts whose handler chain is *entirely* Lean-audited or pen-and-paper (Sec. 5.5, lines 816–822); the 105/185 majority that touches a tested-only handler is named.
- The CV witness audit ("zero `assume_M` is unwitnessed", with 12 randomly-sampled CV/`*Config` pairings) addresses the round-2 question about whether CV verdicts are vacuous (Sec. 5.1, lines 83–101).

## Weaknesses
- **The joint-LOO result is so flat it is suspect rather than reassuring.** The new joint test in `bug_corpus_loo_joint.py` reports that disabling *all* per-category operator handlers *and* the AST-pattern intent-bug analyser simultaneously still leaves the bug-corpus refute count at 53/60 (Sec. 5.1, lines 501–511). The paper attributes this to "a constraint-based shape back-end [that] harvests shape predicates from explicit asserts, control-flow guards, and the symbolic interpreter". If true, this means the curated 60-bug corpus is catchable by predicates the bug repros themselves carry (assert lines, explicit shape literals), and the operator catalogue is doing very little load-bearing work on this benchmark. Either the per-handler "load-bearing for 7 RPs / 6 RPs / …" attribution (lines 489–496) or the joint-LOO 53/60 result must be wrong, because handler-dependency cannot be both 7+7+6+5+4·4+3 = 49 and 0. Please report the *handler-attributable refute count under joint-LOO* (i.e., bugs that joint-LOO converts from RP to non-RP), not just the headcount.
- **The unbiased external-validity sample is still flat and still N=15.** The pre-registered post-freeze unfiltered sample (`tab:unfiltered-postfreeze`) is unchanged in size since round 2: TG 5/15 vs FakeTensor 2/15 vs Pytea 3/15, Fisher p=0.39 and p=0.68 (lines 354–367). The round-2 review explicitly asked for this sample to be expanded to N≥60 with corrected CIs; it has not been. Adding Wilson intervals to the same N=15 cell (lines 355–360) is a formatting improvement, not new evidence. The paper then says TG is "strictly above the two execution-based baselines on the same unfiltered surface, although on N=15 the head-to-head Fisher-exact comparisons are not statistically separable" (lines 383–388) — but "strictly above on point estimate, not separable" is a circular description of insufficient data, not a result.
- **The 113 config-attribute exclusion (round-2 W2 / Q1) was not measured.** The bug-corpus exclusion funnel still drops `~113` config-attribute bugs (Sec. 5.1, lines 436–439, exclusion rule (iv)) — exactly the class the paper sells as TG's central advantage over execution-based baselines (Sec. 1, intro framing, and the "symbolic config" front-end in Sec. 4). Round 2 asked for the RP rate on this 113-bug slice; round 3 still discards them by protocol. Without that number the headline 53/60 / 88.3% rate measures TG against the corpus from which its hardest target class has been removed.
- **The Pytea baseline is still pinned at 2022-04-26.** Round 2 asked for either a current-HEAD Pytea run *or* a TG/Pytea down-scope to a contemporaneous catalogue; this round restricts TG to the 2022 catalogue intersection (lines 282–286), which is the right adjustment on the *TG* side but does not address that Pytea itself has not been retargeted to 2025/2026 PyTorch. The 32/34 vs 22/34 McNemar p=0.00195 therefore still substantially measures the 4-year operator-catalogue freeze on the *baseline*, not a methodological advance.
- **The CV "witness" audit is qualitative.** The claim "zero `assume_M` is unwitnessed" (lines 94–95) is supported by 12 hand-paired CV verdicts out of 128. The remaining 116 are classified by *type* of `assume_M` reference (empty / `config.*` / `*PreTrainedModel` stub) but no real upstream call site is exhibited. "Every published checkpoint config exposes `config.hidden_size`" (line 91) shows a config has the *attribute*, not that any real caller's value satisfies the synthesised inequality. Either (i) check the synthesised `assume_M` against the actual config defaults of N≥30 published checkpoints, or (ii) report the witness coverage as 12/128 rather than as a categorical "zero unwitnessed".
- **CEGAR and phase-check remain in the architectural pitch but are admitted no-ops on every measured corpus.** Sec. 5.4 reports that L1 and L3 "did not discriminate any case in the current implementation" on the synthetic stress benchmark (lines 573–582) and also that "none of CEGAR, device-flag, phase, grad-flow, or low-conf gating discriminates" on the real 10-bug ablation (lines 614–619). Yet they remain in the contributions/system pitch. Per round-2 W5: either remove them from the system description or report a measurement on which they fire.
- **The Lean reconciliation makes the soundness footprint smaller than the contributions list implies.** Of TG's 185 in-soundness verdicts on the 488-block corpus, 36 (≈19%) touch only handlers that are Lean-audited or pen-and-paper; 105 (≈57%) touch at least one tested-only handler (lines 816–822). The paper title still advertises "Sound Static Verification … with a … Lean-Audited Operator-Rule Table"; the calibrated reading is that on real source roughly four-fifths of the soundness-bearing verdicts depend on at least one un-mechanised handler.
- **The Dynamo correspondence's empirical instantiation is still 5 modules.** The end-to-end TG-verifies-class-body + `torch.compile(dynamic=True)` audit (lines 713–733) covers BasicBlock, Bottleneck, InvertedResidual, Fire, and ViT Block — i.e., five small constituent blocks. The 17-module `8.8% / 97.9%` audit is now explicitly downgraded to "calibration of Dynamo against hand-written contracts rather than … in-theorem instantiation of `thm:dynamo-corr`" (lines 705–711). With a 5-module in-theorem witness set, `thm:dynamo-corr` should be presented as preliminary rather than a headline contribution (C4).
- **Localisation 33/33 ±5 lines is still in the body of Sec. 5.4 (lines 626–636).** It is correctly caveated this round, but the figure is not removed and a marker-only audit at N≥30 has not been run; the headline number a casual reader will retain is still 33/33.

## Questions
- Reconcile the joint-LOO 53/60 result with the per-handler attribution table (7/7/6/5/4×4/3 = 49 RPs attributed to handler classes). Please report the *handler-attributable refute count under joint-LOO* — i.e., the count of bugs that joint-LOO converts from RP to non-RP, broken out by category — so a reader can distinguish "the catalogue is over-determined" from "the corpus is catchable from predicate harvesting alone".
- What is TG's RP rate on the ~113 "config-attribute" bugs excluded by exclusion rule (iv)? This was Q1 in round 2 and is the actual evidence for the symbolic-config contribution.
- Can the unfiltered post-freeze sample be enlarged to N≥60 (same pre-registered query, same exclusion rules), and the Fisher-exact comparisons re-run? The current N=15 is too small to resolve any of TG/FakeTensor/Pytea against each other.
- Run the head-to-head against current-HEAD Pytea (or, equivalently, a Pytea fork retargeted to a contemporaneous PyTorch). What are the resulting (TG, Pytea) refutation pairs on the modern-subset N=34?
- For the 116 CV verdicts not in the 12-witness sample, can the synthesised `assume_M` be evaluated against published `*Config()` defaults from the matching HuggingFace checkpoint repository, and the share that *concretely* satisfies `assume_M` reported as a fraction?
- The first-order grad-flag lattice silently misclassifies parameter-sharing/checkpoint cases (Sec. limitations, lines 97–113). What is the frequency of such patterns in the bug corpus and the 488-block real corpus, and is any such block currently in the V or RP column?

## Scores

Soundness: 2
Presentation: 3
Contribution: 2
Confidence: 4
Overall: 4

## Borderline reasons
The single change that would push the overall score from 4 to 5 is the round-2-asked-for unbiased external-validity result that round 3 still has not delivered: a ≥60-item pre-registered post-freeze sample (no fragment-fit filter, no exclusion rules (iii)/(iv), Pytea at current HEAD or both tools at a contemporaneous catalogue) on which TG's catch-rate Wilson interval is strictly above both FakeTensorMode's and Pytea's. With the headline numbers still carried by the curated 60-bug corpus and a 2022-frozen Pytea, and with the only unbiased sample (N=15) still showing no separation, the empirical contribution remains below the bar despite the genuinely improved calibration around it.


Changes   +0 -0
Requests  7.5 Premium (3m 1s)
Tokens    ↑ 744.2k • ↓ 7.9k • 692.6k (cached)
