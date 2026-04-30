● Read round07_reviewer.md
  │ .comet_neurips/round07_reviewer.md
  └ 118 lines read

● List repo (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/
  │ && wc -l neurips.tex neurips.txt 2>/dev/null
  └ 18 lines...

● Read neurips.tex
  └ 94 lines read

● List sections (shell)
  │ ls sections_v5/ 2>/dev/null && wc -l sections_v5/*.tex 2>/dev/null
  └ 1 line...

● Find section files (shell)
  │ find . -name "intro_v6*" -o -name "eval_v6*" -o -name "calculus_v6*" -o
  │ -name "impl_v6*" -o -name "limconc_v6*" -o -name "appendix_v6*" 2>/dev/null
  │ | head -30
  └ 13 lines...

● Section line counts (shell)
  │ wc -l docs/paper/sections_v5/*.tex
  └ 31 lines...

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ 136 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:200 (200 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L200:500 (301 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L500:900 (394 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L894:1346 (368 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1262:1346 (85 lines read)

● Read calculus_v6.tex
  │ docs/paper/sections_v5/calculus_v6.tex
  └ 171 lines read

● Read limconc_v6.tex
  │ docs/paper/sections_v5/limconc_v6.tex
  └ 171 lines read

● Read impl_v6.tex
  │ docs/paper/sections_v5/impl_v6.tex
  └ 86 lines read

● Read related_v6.tex
  │ docs/paper/sections_v5/related_v6.tex
  └ 53 lines read

## Summary

The paper presents \textsc{TensorGuard}, a no-execution refinement-type checker for PyTorch \texttt{nn.Module} classes that statically infers shape and \texttt{requires\_grad} refinements (\(\mathsf{Tensor}\{s,g\mid\varphi\}\)) from class source via Z3. It introduces an assume/guarantee discipline at the class boundary, an autograd-aware backward verifier, and a one-directional Dynamo-guard correspondence lemma (Thm.~5). Empirical results report \(53/60\) RP on a historical bug corpus, a fragment-fair head-to-head with Pytea (\(32/34\) vs.\ \(25/34\), McNemar exact \(p{=}0.0156\)), \(5/15\) catches on a pre-registered post-freeze unfiltered real-PR sample (vs.\ FakeTensor \(2/15\), Pytea \(3/15\); not separable at \(\alpha{=}0.05\)), and \(0\) unconditional RP on the \(488\)-block real-source corpus under the user-visible free-symbolic regime. The Lean side mechanises \(28/79\) shape-transfer rules sorry-free plus a \(17\)-operator assume/guarantee composition theorem (\texttt{ag\_composition\_ext}); the analyser implementation, AST extractor, backward verifier, and Z3 dispatch remain in the TCB.

## Prior weakness disposition

- [RESOLVED] Abstract/body numerical inconsistency on the Pytea head-to-head (32/34 vs 25/34 p=0.0156 vs body 32/34 vs 22/34 p=0.00195) -- abstract and body now both report \(32/34\) vs \(25/34\), \(p{=}0.0156\) (eval\_v6.tex L352-369, neurips.tex L44-45); the alternate \(p{=}0.00195\) is explicitly an auditable reclassification convention, not a contradicting count.
- [PARTIAL] The symbolic calculus contributes 0/60 detections on the headline benchmark -- the paper now shows operator-dispatch-only and AST-pattern-only \emph{each} refute \(53/60\) and frames the corpus as "over-determined" (eval\_v6.tex L700-718), but the underlying point that the rule-driven Z3 path is not independently load-bearing on the headline corpus (and that the \(488\)-block free-symbolic surface is \(0\) RP) is acknowledged rather than fixed.
- [RESOLVED] Dynamo ABSTAIN bidirectional intro vs unidirectional Theorem 5 -- the intro now explicitly scopes Thm.~5 as "one-directional inclusion," "necessary-direction only" (intro\_v6.tex L38-44, L86-100); abstract and theorem statement match.
- [PARTIAL] The 55-module Dynamo population fails to exercise SHAPE-correspondence falsification predicate (all 72 in-contract recompiles are INT) -- the paper now adds a curated \(N{=}5\) adversarial custom-op corpus on which the predicate is non-vacuously evaluable (5/5 hold) (eval\_v6.tex L999-1013), but the larger 55/107-module population is still a denominator audit, not a falsifier evaluation; the empirical falsifier still rests on a small CNN-only \(13\)-event count plus 5 hand-built fixtures.
- [UNRESOLVED] Mutation kill rate 3/50 = 6.0% on the 60-bug corpus (union 14%) -- the paper retains the same numbers (eval\_v6.tex L1052-1058) and adds a load-bearing-handler breakdown which actually \emph{strengthens} the concern: \texttt{conv2d} \(0/10\) and \texttt{einsum} \(0/10\) (eval\_v6.tex L1079-1090). 43/50 mutants still survive at union; the analyser's empirical robustness on its core handlers is essentially unchanged.

## Strengths

- The verdict taxonomy (RP/CV/LW/Abstain) is genuinely calibrated: the paper does not silently report Verified on out-of-fragment code, and the soundness theorem is restricted exactly to RP+CV. The \(0\)-unconditional-RP headline on \(488\) blocks is reported up-front in the abstract, not buried.
- The Pytea head-to-head is now a clean matched-pair design with documented fragment fairness on both sides (membership predicate + AST scan), and the contingency \((a{=}25, b{=}7, c{=}0, d{=}2)\) is reported per-bug. McNemar \(p{=}0.0156\) on the conservative convention is genuine evidence on the modern subset.
- The Lean audit, while limited in footprint (\(28/79\) handlers), is concrete: \(11/11\) previously-axiomatic lemmas closed sorry-free, full operator-rule tree builds under \texttt{lake build}, and \(28{,}000/28{,}000\) in-envelope agreement with torch~2.9.1. The per-handler scope table and the \(36/185\) "in-soundness footprint" accounting on the block corpus are honest.
- The pre-registered \(N{=}15\) post-freeze sample with explicit non-separation at \(\alpha{=}0.05\), Bonferroni/BH correction, and a power calculation locating \(N_{\text{new}}{=}26\)/\(77\) for separation is the right way to report a small-sample directional result.

## Weaknesses

- \textbf{Mutation testing on load-bearing handlers shows \(0/10\) kill on \texttt{conv2d} and \(0/10\) on \texttt{einsum}} (eval\_v6.tex L1079-1090). The paper attributes this to "corpus-coverage gap, not a handler correctness concern," but for a system whose soundness story rests on these handlers being mechanised in Lean (\texttt{applyOp\_sound\_conv2d}, \texttt{applyOp\_sound\_einsum}) the inability of the historical corpus to distinguish a mutated handler from the correct one means the empirical-agreement story for these two operators is carried entirely by the in-envelope sampling test, not by any bug the corpus actually catches. Either supply a small targeted bug-corpus extension that \emph{does} kill comparison-flip and arithmetic-swap mutants on these two handlers, or report the kill rate as a residual robustness gap on the load-bearing path.
- \textbf{Of the \(185\) in-soundness verdicts on the \(488\)-block corpus, only \(36\) (\(11+25\)) touch \emph{only} the Lean-or-pen-paper audited footprint; \(105/185\) touch at least one of the \(44\) tested-only handlers} (eval\_v6.tex L1160-1168). The paper is forthright about this, but the practical implication is that a majority of in-soundness verdicts on real source rest on random in-fragment sampling against torch, not on a mechanised theorem. The paper would be substantially strengthened by reporting which 5--10 specific tested-only handlers, if promoted to Lean, would convert the largest fraction of these \(105\) verdicts into the audited footprint, with the resulting projected ratio.
- \textbf{The \(N{=}15\) unfiltered post-freeze headline does not separate from baselines under any standard test} (Fisher \(p{=}0.39\) vs.\ FakeTensor, \(p{=}0.68\) vs.\ Pytea; BH-adjusted \(p{=}1.00\) on every pair). The paper reports this honestly, but the abstract's framing ("\(5/15\) catches versus \texttt{FakeTensorMode}~\(2/15\) and Pytea~\(3/15\)") still uses a separation-suggesting comma list while the body retracts the separation. The headline contribution on the post-freeze surface is therefore a directional point estimate on a small \(N\); the second-wave power calculation (\(N_{\text{new}}{=}26\)/\(77\)) is the real ask.
- \textbf{The hybrid-mode "complementarity" claim (Tab.~\ref{tab:hybrid-falsify}: TG-only \(20\), FT-only \(5\), zero overlap) is on a \(25\)-block stress set the authors hand-designed} to exercise three knobs (eval\_v6.tex L737-746). This is structurally identical to the L2/L4/L5 stress benchmark caveat: the per-feature stress set is constructed so that exactly one feature fires, and the corresponding real-corpus ablation is a flat line. The \(20/0/0/5\) split should carry the same "stress-only" caveat as Table~\ref{tab:ablation}; on real importable code we have no measurement of complementarity, only of complementarity on a benchmark that was built to show complementarity.
- \textbf{The grad-flag silent-error rate on the \emph{worst-case} construct family is \(2/8 = 25\%\)} on the runtime trainer harness (eval\_v6.tex L1290-1316), not the \(\le 12\%\) "regex-detectable prevalence" ceiling that the paper foregrounds. The paper notes the runtime probe is "deliberately oversampled on positives," but the claim "\(0/50\) false positives" on the backward verifier (intro\_v6.tex L79) is for shape misclassification on clean models, not for silent verifies on tied-weight gradient bugs. The intro's \(8/8\) canonical-bug headline should be paired with the \(2/8\) silent-verify on tied/renamed-attribute parameter sharing in the same bullet, since both are first-order grad-lattice claims.
- \textbf{Theorem 5 (Dynamo-guard correspondence)'s end-to-end empirical anchor is \(13\) SHAPE recompile events on \(10\) CNN-type modules} (eval\_v6.tex L966-976), with the \(4\) transformer blocks audited via the documented forward-signature surrogate and the \(55\)/\(107\) larger module population producing \(0\) SHAPE recompiles to test the predicate against. Calling this an "exploratory" result is correct, but the C4 contribution bullet still claims the lemma is "shipped as data (\(28\) rows; mechanically rechecked against the imported torch --- \(28/28\) pass)"; the \(28/28\) row check is variable-mapping, not the falsification predicate, and the conflation should be tightened.

## Questions

- For the \(0/10\) and \(0/10\) mutation kills on \texttt{conv2d} and \texttt{einsum} handlers, can you exhibit a minimum-size bug corpus extension (one bug per surviving mutant class on each handler) that flips the kill rate on these two load-bearing handlers above \(50\%\)? If not, what does the empirical agreement test on \(28{,}000/28{,}000\) in-envelope samples actually rule out for these two handlers?
- Of the \(105/185\) in-soundness verdicts that touch at least one of the \(44\) tested-only handlers, which \(k\) handlers (smallest \(k\)) would, if promoted to Lean, push the audited-only fraction above \(50\%\) of in-soundness verdicts on the \(488\)-block corpus?
- On the \(2/8\) silent-verify rate for tied / renamed-attribute parameter sharing in the runtime trainer harness: does the \(371\)-Verified V/RP/Abstain sweep over the \(1{,}957\) tied-weight \texttt{nn.Module} subclasses (eval\_v6.tex L1266-1274) include any subclass whose \texttt{forward} actually exercises a renamed-attribute alias on a parameter that receives gradient? If yes, what is the silent-verify rate on that subset specifically?
- For the unfiltered post-freeze \(N{=}15\) sample, what would the catch-rate point estimate need to be on a same-sized \(N{=}15\) second wave to give a Fisher one-sided \(p<0.05\) against FakeTensor's currently-observed \(2/15\), holding TG's denominator fixed? (The paper gives \(N_{\text{new}}{=}26\), but it would help to fix the actual minimum margin on a fixed \(N\).)
- Theorem~5 is scoped to "the catalogue intersection." Is there a \texttt{torch.compile} configuration on any of the \(4\) transformer blocks (Swin, ViT, MLP-Mixer, EncoderBlock) under which a SHAPE/DTYPE/RANK guard can actually be elicited (rather than the surrogate forward-signature audit)? The audit on the \(13\) CNN events is the only direct empirical test of the falsification predicate in the body.
- The Pytea baseline is frozen at commit \texttt{cb02a8a} (2022-04-26) and the paper notes Pytea has "zero commits after" that. Can you confirm Pytea's catalogue does not silently extend through any monkey-patch / plugin layer, so the modern-subset filter is exhaustive rather than an inclusion approximation?

## Scores

Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 6

## Borderline reasons

The single change that would push the score up is closing the load-bearing-handler mutation gap: a small targeted bug-corpus extension that flips the \texttt{conv2d} and \texttt{einsum} kill rates from \(0/10\) above \(50\%\), reported alongside the existing \(7/50\) union number. Currently the analyser's empirical robustness on its two most central operator handlers is \emph{measured at zero kills on the historical corpus}, which is the largest residual weakness in an otherwise carefully-calibrated paper.


Changes   +0 -0
Requests  7.5 Premium (2m 48s)
Tokens    ↑ 930.5k • ↓ 8.2k • 860.4k (cached)
