● Read round02_reviewer.md
  │ .comet_neurips/round02_reviewer.md
  └ 121 lines read

● Check repo files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls -la && wc -l neurips.tex neurips.txt 2>/dev/null
  └ 58 lines...

● Check actual paper source (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l docs/paper/neurips.tex && head -50 neurips.txt
  └ 52 lines...

● Read neurips.tex
  │ docs/paper/neurips.tex
  └ 90 lines read

● List paper sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper && ls sections_v5/ && wc -l sections_v5/*.tex
  └ 33 lines...

● Read shell output Waiting up to 5 seconds for command output
  └ 39 lines...

● Stop shell
  └ 1 line...

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ 123 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:200 (200 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L200:500 (301 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L500:800 (301 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L800:1100 (301 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1100:1209 (110 lines read)

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ 166 lines read

● Check Lean sorry usage (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls lean/ && echo --- && grep -rn "sorry" lean/ 2>/dev/null | grep -v
  │ ".lake" | head -30
  └ 14 lines...

● Look for sorry/axiom in Lean (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && grep -rn "^[^-]*\bsorry\b" --include="*.lean" 2>/dev/null | grep -v
  │ "comment\|--\|/-" | head -30 ; echo ---axiom---; grep -rn
  │ "^axiom\|^[[:space:]]*axiom " --include="*.lean" 2>/dev/null | head -30
  └ 11 lines...

● Lean file sizes (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && wc -l TensorGuard/*.lean *.lean 2>/dev/null && echo --- && ls
  │ TensorGuard/
  └ 15 lines...

● Count operator soundness lemmas (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && grep -c "applyOp_sound\|applyOpExt_sound"
  │ TensorGuard/AssumeGuaranteeExtended.lean TensorGuard/Extended.lean
  │ TensorGuard/V5OperatorRules.lean
  └ 4 lines...

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ 156 lines read

## Summary
TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module.forward` that statically verifies symbolic shapes and a flat gradient-flag lattice from class source. Its core artefact is a refinement-typed calculus `Tensor[τ, φ_shape, φ_grad]` with an assume/guarantee discipline at the class boundary, paired with a Lean 4 audit of 28 of 79 shape-transfer rules (11/11 previously-axiomatic lemmas closed sorry-free) and a five-verdict calibrated reporting taxonomy (Verified, Refuted-Proof, Contract-Violation, Library-Warn, Abstain). Empirically, on a curated 60-bug historical corpus it reports 53/60 (88.3%) Refuted-Proof; on a 34-bug fragment-fair head-to-head it reports 32/34 vs Pytea 22/34 (McNemar p=0.00195); on an N=15 pre-registered post-freeze unfiltered real-PR sample it reports 5/15 catches vs FakeTensorMode 2/15 and Pytea 3/15 (Fisher p=0.39, 0.68 — not separable at α=0.05); and on the 488-block real-source corpus the user-visible free-symbolic regime returns 0 unconditional Refuted-Proof verdicts. The paper additionally claims a necessary-direction Dynamo-guard inclusion lemma (CNN-only-headline 13 SHAPE recompiles, all in-catalogue), an 88.3% bug-corpus rate, and a hybrid-mode complementarity result with FakeTensorMode.

## Prior weakness disposition
- [PARTIAL] The headline "soundness theorem" surface is much narrower than the paper's prose suggests. Theorem 2 covers only 28 Lean-audited + 3 pen-and-paper of 79 handlers... -- The in-soundness footprint is now 28 Lean + 7 pen-and-paper = 35/79, and the new per-block scope table reports only 38/185 V+CV verdicts touch only in-soundness handlers; 44 handlers remain tested-only and 103/185 verdicts still touch one of them.
- [PARTIAL] The user-visible real-source result is a non-result. Under the free-symbolic-config regime... TG returns 0 unconditional RP and 34/0/206/248 on 488 blocks. -- The 0/488 unconditional-RP count is unchanged and remains the headline triple; the new 12-row LW→RP candidate table flips only 2 rows (`ChannelAttention(V2)`) to measured RP and the rest stay predictions, so the user-visible non-result on real source persists.
- [UNRESOLVED] The N=15 pre-registered post-freeze sample is the best generalization evidence, and it is not statistically separable from either baseline (Fisher p=0.39 vs FakeTensorMode, p=0.68 vs Pytea). -- Same N=15, same 5/15 vs 2/15 vs 3/15, same Fisher p-values; the paper now states the non-separation explicitly and adds a power calculation, but no new evidence has moved the headline.
- [PARTIAL] The 60-bug corpus has substantial curator latitude: 1,087 keyword-search hits filtered to 60 by four hand-defined exclusion rules, including ~113 "config-attribute bugs" excluded under rule (iv) on which TG returns 0/113 RP. -- The exclusion rationale and the 0/113 RP measurement on the excluded slice are now disclosed openly with 16/113 silent-verified counts; the structural curator latitude (1,087→60 with rule (iii)+(iv) removing ~22%) and the 0/113 scope bound itself remain.
- [PARTIAL] Theorem 5 is empirical for the transformer case. 4/13 modules in the extended audit (Sec. 4.3) and 16/17 in the original audit are evaluated through a "documented forward-signature surrogate"... -- The CNN-only restriction (10 modules, 13 SHAPE recompiles, all in-catalogue) is now promoted to the headline for the necessary direction, with the 4 transformer blocks explicitly scoped as surrogate; the 16/17 original-audit surrogate fraction is unchanged.
- [PARTIAL] Mutation testing is weak: 3/50 kill rate on the 60-bug corpus, 7/50 union across three corpora (14%). -- The 14% best-of (7/50 union) figure is now reported with a per-corpus decomposition and a four-fault hand-picked TCB pair, but 43/50 mutants still survive and the headline kill rate remains an order of magnitude below standard mutation-testing acceptance thresholds.
- [PARTIAL] The grad-flag contribution is bounded by a 12% prevalence ceiling for parameter-sharing-under-renamed-attribute and `torch.utils.checkpoint`, but the 500/500 random-module agreement is on a generated grammar and the 10-model real-world sample... -- A held-out 1/42 (2.4%) AST-grep audit on `examples/pytorch/` and a runtime 0/8 trainer-with-checkpointing harness now sit inside the 12% ceiling and tighten the silent-error envelope; the lattice flatness and ceiling itself are unchanged.
- [UNRESOLVED] Per-feature ablation on the real corpora is a flat line for all five knobs (Sec. 4.2), and CEGAR/phase are explicitly "shipped, did not discriminate." -- The new 10-bug real-corpus ladder is reported and is again a flat line; CEGAR and phase remain "shipped, did not discriminate" no-ops with zero verdict-touching call sites, and the only change is that the paper now declines to claim them as contributions.

## Strengths
- Calibrated reporting is exceptionally honest by NeurIPS standards: the five-way verdict taxonomy with explicit RP/CV/LW split, the up-front statement that the user-visible free-symbolic regime returns 0 unconditional RP on 488 blocks, and the Wilson/Fisher/McNemar intervals throughout (e.g. lines 67–82, 246–253, 435–446 of `eval_v6.tex`) are unusually disciplined for a tool paper.
- The refinement-typed calculus `Tensor[τ, φ_shape, φ_grad]` over LIA + Div + bounded BMul, with a contravariant/covariant assume/guarantee subclassing rule (`calculus_v6.tex` lines 116–129) and the `ag_composition_ext` operator-agnostic composition theorem mechanised in Lean 4, is a genuinely principled core, not a rebadged execution-based analyser.
- The Lean 4 rule audit is real: `lean/TensorGuard/V5OperatorRules.lean`, `Extended.lean`, and `AssumeGuaranteeExtended.lean` build sorry-free under `lake build`, the per-rule Python byte-mirror agrees on 28,000/28,000 in-fragment samples against torch 2.9.1, and the off-envelope check on ~2,400 mutators reports zero silent-through cases.
- The 32/34 vs 22/34 fragment-fair head-to-head against Pytea (McNemar exact p=0.00195, paired-bootstrap CI [+14.7, +44.1] pp, with a verification-time AST screen forcing fragment fairness on both sides) is the one place the paper has a clean significant separation, and it is constructed defensibly.
- The hybrid falsification corpus (Table `tab:hybrid-falsify`, 25 importable no-arg modules) gives a genuinely informative complementarity result with FakeTensorMode (TG-only 20, FT-only 5), turning what could have been a vacuous "non-importable corpus → mechanical N/A" critique into a falsifiable comparison.

## Weaknesses
- The user-visible headline on real source is still a non-result: under the free-symbolic-config regime the 488-block triple is 34 V / 0 RP / 206 LW / 248 A (`eval_v6.tex` line 77), and the 12-row LW→RP candidate table only flips 2 rows (`ChannelAttention`/`ChannelAttentionV2`) to measured RP this round, with the other 10 still labelled "predicted." The unconditional-RP claim is therefore still entirely carried by curated bug corpora; the "audited shape-transfer rule table" front-end produces zero unconditional refutations on a representative slice of real torchvision/timm/transformers source code.
- The N=15 pre-registered post-freeze sample is the only post-hoc-curation-free generalization evidence, and on it TG is not statistically separable from FakeTensorMode (p=0.39) or Pytea (p=0.68); 2/5 of the catches are attributed to the constructor-bound integer-attribute envelope synthesiser that was added after the freeze (paper acknowledges this in lines 462–470), so the load-bearing differentiator is a recent engineering addition rather than the calculus itself, and the fragment-honest catch rate of the published v4 calculus on this sample is 3/15. The pre-registered evidence base for the headline contribution is thinner than the abstract's framing implies.
- The conclusion (`limconc_v6.tex` line 142) states the assume/guarantee composition theorem "is mechanised on a 3-operator DSL only," whereas `intro_v6.tex` lines 52–69 and `eval_v6.tex` lines 997–1004 say 17 operators, and `calculus_v6.tex` line 138 says 13. This is a self-contradiction inside the paper about the size of the mechanised composition fragment, not a presentation nit; the actual sorry-free file is `lean/TensorGuard/AssumeGuaranteeExtended.lean` with 15 `applyOp_sound_*`/`applyOpExt_sound_*` lemmas. Pick one number and make it consistent across abstract, intro, calculus, eval, and conclusion.
- Mutation testing kill rate is 7/50 (14%) at union over three corpora, with 43/50 mutants surviving (`eval_v6.tex` lines 932–957). For an analyser whose central marketing claim is soundness, an 86% mutant survival rate on AST-edit mutations of the analyser core is a meaningful negative signal about test sensitivity and indirectly about the implementation TCB that sits between Theorem `thm:soundness` and the headline 53/60 figure.
- 44 of 79 shape handlers remain tested-only (Sec. 4.4 / `tab:handler-soundness`), and the paper's own per-block scope analysis reports that 103/185 V+CV verdicts on the 488-block corpus touch at least one tested-only handler. The "Lean-audited operator-rule table" framing in the abstract therefore covers only 38/185 in-soundness verdicts; this proportion should be at least as prominent as the 28/79 handler count in the contribution claims.
- The CEGAR loop and the train/eval phase-check ship in the analyser, never discriminate on any corpus, and have zero verdict-touching call sites (`eval_v6.tex` lines 681–689, 713–717). The paper now correctly declines to claim them as contributions, but they remain in the codebase and are still part of the surface area Theorem `thm:soundness`'s implementation depends on; the per-feature ablation as a result demonstrates only 3 of the 5 advertised analyser knobs do work on any corpus.
- The 60-bug corpus filtering pipeline (1,087 keyword hits → 612 not-shape-bug → 178 user-error → 124 distributed → 113 config-attribute → 60) plus the 0/113 RP measurement on the rule-(iv) excluded slice (`eval_v6.tex` lines 568–582) means the headline 88.3% applies to a sample that was selected away from the 113-bug class on which TG is known to score 0/113. The rule-(iv) exclusion is principled but the headline rate is conditional in a way the abstract does not communicate.
- The Dynamo-guard correspondence theorem is now headlined on a 10-CNN-block end-to-end audit yielding 13 SHAPE recompile events; this is a small denominator for a "necessary-direction inclusion lemma" framed as a theorem about the catalogue intersection, and the 4 transformer blocks in the extended audit and 16/17 in the original audit still go through the documented forward-signature surrogate. The theorem-versus-empirical-audit distinction would be cleaner if the surrogate-audited cases were not counted under the same theorem statement.

## Questions
- On the 488-block free-symbolic-config regime, can you implement the `Tensor.unbind(dim)` rule plus one further smallest-cost LW→RP candidate from the 12-row table and report the resulting unconditional-RP count on the same 488 blocks under the same harness? The paper currently labels 10 of the 12 as "predicted RP"; converting two more would falsify or substantiate the 12/78 LW→RP ceiling and convert the headline triple from `34/0/206/248` to a non-zero RP figure.
- For the N=15 unfiltered post-freeze sample, can you extend the sample to N=41 (TG-vs-FakeTensorMode passes p<0.05 at N_new=26 by your own power calculation) using the same pre-registered query and report the new pairwise Fisher-exact p-values? Without this, the strongest generalization evidence remains a 5/15 point estimate not separable from either baseline.
- Reconcile the size of the mechanised assume/guarantee composition DSL: is it 3 (conclusion), 13 (Theorem `thm:ag-sound` statement), or 17 (intro and eval) operators? Please also state which of the 17 operators correspond to `applyOpExt_sound_*` lemmas and which to `applyOp_sound_*`, with a single citation back to the file in `lean/TensorGuard/`.
- The mutation-testing union is 7/50 and the surviving 43 mutants are described as sitting on "handler paths none of the three corpora exercise." Can you report the kill rate restricted to mutations of the four handlers most load-bearing on the headline 53/60 (`view_reshape_total_size`, `broadcasting`, `conv_channel_mismatch`, `einsum_dim`)? A targeted kill rate would be more informative about the soundness-relevant TCB than the union number.
- For the per-block scope table, what fraction of the 6/15 RP fires (including the off-axis `rb_uf_010` device-mismatch fire counted as a false positive against ground truth) traverse only the 35-handler in-soundness footprint, and what fraction traverse at least one tested-only handler? This would let a reader read the unfiltered post-freeze headline against the soundness scope of Theorem `thm:soundness` directly.
- The `rb_uf_010` off-axis RP fire is reported as a false positive against ground truth (device-mismatch where the upstream PR fixes a dtype bug). Is this the only false positive observed across the union of the 60-bug, 10-real-bug, 6-post-freeze, and 15-unfiltered corpora, and if so what is the corresponding precision interval?

## Scores
Soundness: 3
Presentation: 3
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons
The single change that would push the score from 5 to 6 is converting the user-visible 488-block free-symbolic-config triple from `0` unconditional Refuted-Proof to a non-trivial RP count by implementing two of the smallest-cost LW→RP candidates (`unbind(dim)` plus one more from the 12-row table) and reporting the measured RP under the same harness; alternatively, extending the unfiltered post-freeze sample from N=15 to ~N=41 and reporting a Fisher-exact separation from FakeTensorMode at α=0.05 would have the same effect, since the headline contribution would then be supported by either (a) a real-source bug-finding result or (b) statistically-separable generalization evidence rather than only by a curated 60-bug corpus.


Changes   +0 -0
Requests  7.5 Premium (5m 40s)
Tokens    ↑ 1.1m • ↓ 9.8k • 1.0m (cached)
