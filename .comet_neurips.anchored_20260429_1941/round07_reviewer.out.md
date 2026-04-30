● Read round07_reviewer.md
  │ .comet_neurips/round07_reviewer.md
  └ 118 lines read

● List repo contents (shell)
  │ ls -la
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/ |
  │ head -50
  └ 51 lines...

● Check sizes (shell)
  │ wc -l /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/neurips.txt /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/te
  │ nsorguard/docs/paper/neurips.tex
  └ 4 lines...

● Read neurips.tex
  │ docs/paper/neurips.tex
  └ 90 lines read

● List sections (shell)
  │ ls /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/d
  │ ocs/paper/sections_v5/
  └ 12 lines...

● Sizes of sections (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && wc -l docs/paper/sections_v5/*.tex
  └ 31 lines...

● Read intro_v6.tex
  │ docs/paper/sections_v5/intro_v6.tex
  └ 123 lines read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ 1 line read

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1:200 (200 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L200:450 (251 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L450:700 (251 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L700:950 (251 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L950:1209 (260 lines read)

● List lean dir (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && ls 2>&1 | head -20
  └ 9 lines...

● Check Lean files (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && grep -l "applyOpExt_sound\|ag_composition_ext\|unbind\|reshape\|conv2d"
  │ TensorGuard/*.lean 2>&1 | head -20 && echo "---" && grep -c "sorry"
  │ TensorGuard/*.lean 2>&1 | head -30
  └ 11 lines...

● find sorry usages (shell)
  │ cd
  │ /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/lean
  │ && grep -n "sorry" TensorGuard/AssumeGuaranteeExtended.lean
  │ TensorGuard/Extended.lean TensorGuard/Parity.lean
  │ TensorGuard/V5OperatorRules.lean
  └ 8 lines...

● Check unbind impl and lean lemmas (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rn "unbind" src/*.py 2>&1 | head -10 && echo "---" && grep -n
  │ "applyOpExt_sound_unbind\|applyOpExt_sound_view\|applyOpExt_sound_reshape\|a
  │ pplyOpExt_sound_conv2d\|applyOpExt_sound_einsum" lean/TensorGuard/*.lean
  └ 18 lines...

## Summary
The paper presents \textsc{TensorGuard}, a static refinement-type checker for PyTorch \texttt{nn.Module} forward methods that infers $\mathsf{Tensor}\{s,g\mid\varphi\}$ signatures from class source without instantiating the module, dispatches refinements to Z3, and ships an assume/guarantee discipline at the class boundary. On the curated 60-bug historical corpus it returns Refuted-Proof on 53/60 (88.3%, Wilson [77.8%, 94.2%]); on a 34-row Pytea-fragment-fair head-to-head it claims 32/34 vs 22/34 (McNemar exact $p=0.00195$); on a pre-registered post-freeze unfiltered N=15 sample it catches 5/15 vs FakeTensorMode 2/15 and Pytea 3/15, reported as a directional rather than statistically separable result. The shape-transfer rule table is mechanised in Lean 4 (28/79 handlers, 11/11 previously-axiomatic soundness lemmas closed sorry-free, 28k/28k torch-2.9.1 in-envelope agreement), and \texttt{ag\_composition\_ext} is now mechanised over a 17-operator DSL that includes the operators firing on the post-freeze catches (\texttt{view}, \texttt{reshape}, \texttt{conv2d}, \texttt{einsum}, \texttt{unbind}). Headline 488-block result is reported as 0 unconditional Refuted-Proof under the user-visible free-symbolic regime, with a 14/488 contracted rerun reported as fragment-coverage rather than as the bug-finding headline.

## Prior weakness disposition
- [PARTIAL] The 488-block headline regressed on the load-bearing measured-RP claim from round 5 -- The abstract and §4.1 now coherently report 0 unconditional RP under the free-symbolic regime as the headline, with the 14/488 (and 14/289 over the analysable subset) input-shape-contract+unbind rerun preserved as a separately labelled measurement and as the falsifiability anchor for the LW$\to$RP table; the framing collision with round 5 is gone, but the load-bearing measured-RP number itself is still the smaller "fragment-coverage" rate, not a strengthened headline.
- [RESOLVED] The \texttt{ag\_composition\_ext} mechanisation footprint is unchanged and still does not cover the operators that fire on the empirical headline -- The 17-operator DSL now explicitly includes \texttt{view}/\texttt{reshape}/\texttt{conv2d}/\texttt{einsum}/\texttt{unbind} with named, sorry-free \texttt{applyOpExt\_sound\_*} witnesses (\texttt{lean/TensorGuard/AssumeGuaranteeExtended.lean:436--503}); all five post-freeze catches in §4.1 thus traverse a Lean-audited handler.
- [RESOLVED] The §4.4 105/185 split was promoted to 103/185 in round 5 and is now back to 105/185 in round 6 -- The current §6 reinstates the round-5 view-reducible promotions (\texttt{flatten}/\texttt{squeeze}/\texttt{unsqueeze}/\texttt{softmax}), giving a 28-Lean / 7-pen-and-paper / 44-tested-only split and an in-soundness verdict count of 38/185 vs 103/185 tested-only; the regression is undone and the four promotions are back inside the soundness footprint.
- [PARTIAL] The 488-block denominator question is still unaddressed -- §4.1 ("Two denominators on the unbind rerun") now reports both 14/488=2.87% over the full corpus and 14/289=4.84% over the analysable subset, with the 199-row not-analysable bucket attributed to the block-extractor stripping class context; the disclosure is honest but the extractor itself was not fixed, so the analysable denominator is still bounded by an extractor artefact.
- [PARTIAL] Theorem 5's empirical surface still has no measured cardinality on non-INT falsifier events -- The 55-module larger-population audit still reports 0 SHAPE/DTYPE/RANK guards, and the gap is now backstopped by a curated N=5 \texttt{torch.library.custom\_op} fixture set on which the falsifier is "non-vacuously evaluable on 5/5"; this is a synthetic, hand-built falsifier corpus rather than an in-the-wild one, so the falsification predicate has still never been exercised on a non-curated module.
- [RESOLVED] The 365-Verified tied-weight subset is still bounded only by abstention rather than by direct measurement -- The new "Stub-mocked runtime sample on the 371-Verified subset" reports 0/25 silently-incorrect Verifieds (Wilson [0.00%, 13.32%]) drawn from exactly the V-population in question, complementing the 8-HF runtime-trainer harness; the 371-Verified envelope is now bounded by a measured Wilson interval rather than only by abstention.

## Strengths
- The Lean side is now genuinely tight against the empirical headline: every operator firing on the 5/5 unfiltered post-freeze catches has a sorry-free \texttt{applyOpExt\_sound\_*} lemma in \texttt{AssumeGuaranteeExtended.lean} and lives inside \texttt{ag\_composition\_ext}, rather than only sitting in the 28-handler shape-transfer table; the soundness story for the headline is no longer "Lean covers a different fragment than what fires."
- Calibrated reporting is consistent across regimes (free-symbolic 0/488, contract-rerun 14/488, analysable-subset 14/289 with explicit Wilson intervals; modern-subset 32/34 vs Pytea 22/34 with McNemar and bootstrap CI; unfiltered post-freeze 5/15 with explicit Fisher-exact $p=0.39$ and $p=0.68$ vs the two baselines and explicit non-separation language). The paper does not hide behind a single number.
- The TCB exposure $+$ measured-flip pair on F1--F4 (exposure 0/0/2/7 on the 60-bug corpus, measured flips 0/0/0/0) is a methodologically clean way to bound how badly a single AST/Z3/handler fault could degrade the headline; combined with the 7/50 best-of mutation-testing kill rate it is a more disciplined robustness argument than typical for this paper genre.
- The new stub-mocked runtime sample (0/25 silent Verifieds on the 371-V subset) and the 8-HF \texttt{gradient\_checkpointing\_enable} runtime-trainer harness (8/8 Refuted-Proof, 0/8 false-Verified) jointly close the "abstention-bounded silent error" loophole that this reviewer flagged in earlier rounds.

## Weaknesses
- The user-visible default headline on the 488-block real-source corpus is still 0 unconditional Refuted-Proof, and the only path that produces unconditional refutations on this corpus (the unbind+input-shape-contract rerun, 14/488) is itself reported as "fragment-coverage" rather than a bug-finding result. The paper's load-bearing real-source bug-finding evidence therefore still rests on N=10 (\Cref{tab:real-bugs}, 7/10 at $\ge 0.99$) and N=15 (unfiltered post-freeze, 5/15), neither of which is statistically separable from the FakeTensorMode/Pytea baselines on the latter (Fisher exact $p=0.39$, $p=0.68$). The headline empirical claim is still under-powered for what the paper asks the reader to conclude.
- The 199/488 "not-analysable" Abstain bucket (§4.1, "Two denominators on the unbind rerun") is attributed to an extractor that strips surrounding class context from those rows. Either fix the extractor and report the rate on a single denominator, or report a per-class-context re-extraction rate over (some honest sample of) those 199 rows so the reader can verify that the 14/289=4.84% rate is not concentrated in a particular family of strippable blocks.
- The Theorem-5 falsifier predicate is still vacuously satisfied on every non-curated population (0 SHAPE/DTYPE/RANK guards on the 55-module audit; N=5 hand-built \texttt{torch.library.custom\_op} fixtures are the only non-vacuous evaluation). Either run the audit on a population large enough that some real (non-custom-op) module installs a SHAPE guard outside \texttt{catalogue}$(M)$, or weaken the theorem's empirical-instantiation language; the current setup means the falsification predicate has only ever been exercised on a corpus the authors themselves wrote.
- The C3 backward-verifier headline is "8/8 canonical bugs, 0/50 false positives, 500/500 static$\leftrightarrow$runtime agreement" (intro, C3) on randomly-generated small modules, supplemented by 10/10 real-model agreement and the new 8/8 trainer-harness result. The held-out "1/42 silent-error-positive" rate from the HuggingFace \texttt{examples/pytorch/} sweep is consistent with a real-world silent-error prevalence near the bottom of the $\le 12\%$ ceiling, but the analyser has not actually been run \emph{on those 42 scripts} and shown to refute the 1 silent-error-positive case --- the 1/42 is a static-grep ground-truth count, not a verifier outcome. Report the verifier verdict on the 1 positive script.
- The "AST-pattern path alone refutes 53/60, operator-dispatch alone refutes 53/60" over-determination claim (§4.1, "Rule-development holdout") is a strong assertion that the 60-bug corpus does not separate the two reasoning paths. If the two paths are truly independently saturating, then the 53/60 number is being driven by corpus-design properties (each bug is recognisable by either a simple AST pattern or a single operator rule), which weakens its evidentiary value for the operator-rule contribution. A per-bug attribution counting how many bugs each path uniquely catches in a bug-difficulty-stratified random subsample would substantively address this.
- The Pytea modern-subset comparison is still an in-2022-catalogue intersection on N=34 historical bugs; the post-freeze unfiltered N=15 (where the gap is 5/15 vs 3/15, $p=0.68$) is the only out-of-corpus head-to-head and is not separable from Pytea at $\alpha=0.05$. The McNemar $p=0.00195$ from §4.1 should not be cited in the abstract as the load-bearing significance test if the post-freeze unfiltered rate against the same baseline is not separable.

## Questions
- For the 199 "not-analysable" rows in the 14/488 unbind rerun: what is the per-class-context re-extraction rate on a uniform random sample of $\sim 30$--$50$ of them, so the reader can verify the 14/289=4.84% rate is not biased by which blocks the extractor happens to strip?
- On the 1/42 silent-error-positive HuggingFace \texttt{examples/pytorch/} script, what verdict does the analyser actually return? If it is silent-Verified, the held-out worst-case rate is $1/42$; if it is Refuted-Proof or Abstain, that is materially different.
- For \Cref{thm:dynamo-corr}: on a long-tail population of importable real modules large enough to observe at least one in-contract SHAPE/DTYPE/RANK recompile (i.e. extending the 55-module audit further until the predicate is non-vacuously evaluable on real code), does the predicate still hold? What is the smallest such non-vacuous denominator the paper can report?
- The two \texttt{ChannelAttention}/\texttt{ChannelAttentionV2} rows in the 12-row LW$\to$RP candidate table (§4.1) are now flipped from "predicted" to "measured RP" via the unbind handler. What is the implementation cost (in source lines, by analogy to the unbind addition) of the smallest of the remaining 10 candidates, and does the authors' own roadmap commit to flipping any specific further row before submission?
- For the rule-development holdout (53/60 with operator-dispatch alone, 53/60 with AST-pattern alone, parser-artefact 53/60 quad-disabled): on a strictly held-out per-bug random partition, does each path independently catch the same per-bug subset, or do they catch different bugs that just sum to 53? The current text reports only the aggregate.

## Scores
Soundness: 3
Presentation: 3
Contribution: 3
Confidence: 3
Overall: 7

## Borderline reasons
The single change that would push my score up by one point is a non-vacuous evaluation of the Theorem-5 falsification predicate on a real (non-curated) module population --- i.e., extending the 55-module audit until it observes at least one in-contract SHAPE/DTYPE/RANK recompile and reporting the catalogue-membership rate on those events. Currently the necessary direction of \Cref{thm:dynamo-corr} is empirically supported only by curated custom-op fixtures and by audits whose denominator on the falsifiable kind is zero, which is the load-bearing remaining gap.


Changes   +0 -0
Requests  7.5 Premium (2m 56s)
Tokens    ↑ 924.5k • ↓ 8.8k • 866.5k (cached)
