● Read round01_reviewer.md
  │ .comet_neurips/round01_reviewer.md
  └ 114 lines read

● List repo (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  └ 14 lines...

● Inspect paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l neurips.txt neurips.tex README.md && head -100 neurips.txt
  └ 105 lines...

● Read more paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '100,300p' neurips.txt
  └ 202 lines...

● More paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '300,600p' neurips.txt
  └ 302 lines...

● More paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '600,900p' neurips.txt
  └ 302 lines...

● More paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '900,1300p' neurips.txt
  └ 402 lines...

● Continue paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '1300,1700p' neurips.txt
  └ 402 lines...

● Continue paper (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && sed -n '1700,2100p' neurips.txt
  └ 401 lines...

● Inspect repo structure (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/ src/ tests/ 2>/dev/null | head -100 && echo "---" && ls
  │ benchmarks/ real_benchmarks/ 2>/dev/null | head -50
  └ 152 lines...

● Lean structure (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/TensorGuard/ | head -50 && echo "---" && grep -c "sorry"
  │ lean/TensorGuard/*.lean 2>/dev/null | head -30
  └ 14 lines...

● Look for sorrys (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -n "sorry" lean/TensorGuard/*.lean
  └ 8 lines...

● Headline data (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && cat benchmarks/headline_results.json 2>/dev/null | head -80 && echo "---"
  │ && cat benchmarks/real_bug_corpus.json 2>/dev/null | head -30
  └ 59 lines...

● find rb references (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls benchmarks/realcode_corpus 2>/dev/null && grep -l
  │ "rb_001\|rb_pf_001\|rb_uf" benchmarks/ src/ -r 2>/dev/null | head -10
  └ 12 lines...

● Stop shell
  └ 1 line...

● Find rb references (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rln "rb_001\|rb_pf\|rb_uf" benchmarks/ experiments_v5/
  │ reproducibility/ 2>/dev/null | head -10
  └ 1 line...

● Stop shell
  └ 1 line...

● Search (grep)
  │ "rb_pf_001|rb_uf_008|rb_001"
  └ 59 files found

● Read real_bugs_unfiltered.json
  │ reproducibility/real_bugs_unfiltered.json
  └ L1:80 (80 lines read)

## Summary
TensorGuard is a no-execution refinement-type checker for `nn.Module` forward methods that statically verifies tensor shapes and a coarse `requires_grad` flag from class source, without instantiating or tracing the module. It introduces a Tensor{shape, grad | φ} calculus discharged by Z3, an assume/guarantee discipline at the module boundary (mechanised in Lean 4 on a 13-operator DSL), a backward-pass verifier for three silent-zero-grad classes, and a one-directional inclusion lemma against TorchDynamo's guards. The shape-transfer rule table is mechanised in Lean for 28 of 79 handlers (sorry-free, with 28k/28k torch-parity samples). Empirically the paper reports 53/60 RP on a curated historical bug corpus, 32/34 vs Pytea 22/34 on a fragment-fair subset (McNemar p=0.00195), 5/15 catches vs FakeTensorMode 2/15 / Pytea 3/15 on a pre-registered post-freeze unfiltered sample (not statistically separable), and 0 unconditional RP on the 488-block real-source corpus under the user-visible free-symbolic regime.

## Prior weakness disposition
(none — first round)

## Strengths
- Genuinely calibrated reporting: a five-way verdict taxonomy (V/RP/CV/LW/A), an explicit pre-registered post-freeze unfiltered N=15 sample with confidence intervals and Fisher-exact p-values that the paper does *not* spin into a significance claim, and an upfront acknowledgment that the user-visible 488-block headline produces 0 unconditional RP.
- Lean 4 mechanisation is real and auditable: `lean/TensorGuard/{Extended,Soundness,Parity,V5OperatorRules}.lean` build sorry-free in the tree (verified by grep), 28 shape-transfer rules with 11/11 previously-axiomatic lemmas closed, and a JSON registry export gating Python/Lean drift.
- The fragment-fair Pytea head-to-head is rigorously isolated from the catalogue confound (Pytea repo has no commits after 2022-04-26, mechanically restricted to common ops at verification time, silent-skip-corrected), giving a genuine 32/34 vs 22/34.
- The hybrid-mode N=25 stress contingency table (Table 4) is a clean falsification of the "TG and FakeTensor refute the same things on importable code" reading: 20 TG-only / 5 FT-only / 0 overlap.
- Useful negative result on `nn.Module` execution baselines (FakeTensorMode/torch.fx/torch.export N/A on 481/488 blocks) due to constructor-argument requirements — a structurally meaningful gap.

## Weaknesses
- The headline "soundness theorem" surface is much narrower than the paper's prose suggests. Theorem 2 covers only 28 Lean-audited + 3 pen-and-paper of 79 handlers (Section 4.4); on the 488-block corpus only 36/185 verified+CV verdicts touch *exclusively* in-soundness handlers, while 105/185 traverse at least one tested-only handler. The composition rule (Theorem 3) is mechanised on 13 operators of 79. The user is therefore being asked to trust an analyser whose Lean guarantee covers a minority of verdict paths, and the paper does not quantify how many of the 53/60 historical RPs land entirely inside the in-soundness footprint.
- The user-visible real-source result is a non-result. Under the free-symbolic-config regime — the only regime that does not assume a synthesised, possibly vacuous caller-rely — TG returns 0 unconditional RP and 34/0/206/248 on 488 blocks. The 12/78 LW→RP "ceiling" is a prediction, not a measurement; the paper should report the actual count after implementing at least the smallest-cost rule (`Tensor.unbind(dim)`) it identifies, since that conversion is described as ~30 LoC.
- The N=15 pre-registered post-freeze sample is the best generalization evidence, and it is not statistically separable from either baseline (Fisher p=0.39 vs FakeTensorMode, p=0.68 vs Pytea). 2/5 of the catches are attributed solely to the round-6 envelope additions and 1 of 6 RP fires (rb_uf_010) is an off-axis false positive against ground truth. The honest reading is "directional only on N=15"; the abstract still places "5/15 catches versus FakeTensorMode 2/15 and Pytea 3/15" as a contribution-bullet without making the non-separation salient at that altitude.
- The 60-bug corpus has substantial curator latitude: 1,087 keyword-search hits filtered to 60 by four hand-defined exclusion rules, including ~113 "config-attribute bugs" excluded under rule (iv) on which TG returns 0/113 RP. The 88.3% headline is therefore conditional on the same team's exclusion criteria; the leave-one-out audit retains 53/60 only because an independent AST-pattern path catches them, which is precisely the kind of redundancy that masks rule-table overfitting.
- Theorem 5 is empirical for the transformer case. 4/13 modules in the extended audit (Sec. 4.3) and 16/17 in the original audit are evaluated through a "documented forward-signature surrogate" because full instantiation exceeds constraint solving. The denominator audit on 55 successful modules contributes 0 SHAPE/DTYPE/RANK guards (all 72 recompiles are INT specialisations), so the falsifier is *not exercised* on the larger population — it is exercised only on 13 CNN events. This is a much narrower correspondence than the abstract's "TorchDynamo guards become the runtime shadow of these refinements."
- Mutation testing is weak: 3/50 kill rate on the 60-bug corpus, 7/50 union across three corpora (14%). For an analyser whose sales pitch is soundness, a 86% mutant-survival rate is a meaningful negative signal about test sensitivity, and the paper presents this as a complement to the four hand-picked TCB faults rather than as the limitation it is.
- The grad-flag contribution is bounded by a 12% prevalence ceiling for parameter-sharing-under-renamed-attribute and `torch.utils.checkpoint`, but the 500/500 random-module agreement is on a generated grammar and the 10-model real-world sweep is on models the paper itself notes do not exercise either failure mode. The 8/8 trainer harness with `gradient_checkpointing_enable()` returns RP because the construct flips out of the lattice — this measures detection of the construct, not soundness of the lattice on it.
- Per-feature ablation on the real corpora is a flat line for all five knobs (Sec. 4.2), and CEGAR/phase are explicitly "shipped, did not discriminate." Combined with the 0-RP free-symbolic regime, this means three of the six contributions (C5's CEGAR, the phase encoder, the localisation tracer) make no difference to the headline numbers; the contribution list overstates what is actually load-bearing.

## Questions
- Of the 53/60 RP on the historical corpus and the 7+1/10 RP on the upstream-faithful corpus, how many bugs are caught entirely along handler paths that are Lean-audited or pen-and-paper-audited (i.e. inside Theorem 2's footprint)? A per-bug scope column would tell the reader how much of the "88.3%" is actually backed by the mechanised guarantee.
- After implementing the `Tensor.unbind(dim)` rule that the paper itself identifies as ~30 LoC, what is the actual converted RP count on the 12 residual LW blocks? The paper currently reports a falsifiable prediction (12 RPs) but not the measurement.
- For the N=15 post-freeze sample, what is the per-PR ground-truth label (shape vs. dtype vs. distributed vs. autograd) and the per-tool TP/FP/FN matrix? With one off-axis fire already documented (rb_uf_010), the relevant comparison is precision at fixed recall, not a raw catch count.
- Theorem 5's transformer surrogate replaces full instantiation with the documented forward signature. On at least one transformer block (say `timm ViT Block`), can you exhibit a single in-contract input on which the surrogate's predicted guard set and the actually-installed Dynamo guard set agree, and quantify the gap if not?
- The 1,087→60 funnel relies on four hand-curated exclusion categories; for category (iv) ("config-attribute bugs") the paper reports 0/113 RP. Could an independent third party reconstruct the inclusion list from the published query, and what fraction of the 60 would survive a held-out reviewer's re-application of the same rules without seeing the analyser's behaviour on each candidate?
- The mutation-kill rate is 14% at union of three corpora; what is the kill rate on the in-soundness-handler subset, and would adding the standard mutation operators (statement deletion, condition negation in branch guards) on `src/typing_rules.py` and `src/backward/` materially change the estimate?

## Scores
Soundness: 3
Presentation: 3
Contribution: 2
Confidence: 3
Overall: 5

## Borderline reasons
The single change that would push this to a 6 is closing the user-visible real-source headline: implement at least one of the named LW→RP rules (`Tensor.unbind(dim)` is described as ~30 LoC) and report the resulting unconditional-RP count on the 488-block corpus under the free-symbolic-config regime, so that the paper has at least one positive number on the surface its abstract is being read against rather than only on curated/historical corpora.


Changes   +0 -0
Requests  7.5 Premium (3m 49s)
Tokens    ↑ 1.1m • ↓ 7.0k • 1.1m (cached)
