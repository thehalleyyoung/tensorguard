# Role: paper authors writing a brief rebuttal

A NeurIPS reviewer just posted the review below on your paper. Before
you start any code or paper changes this round, you have ONE chance to
push back on weaknesses you believe are misweighted, factually wrong,
or already-resolved-in-the-current-repo. The next round's reviewer
WILL read this rebuttal and must either accept it (drop the weakness)
or sharpen it (restate with a concrete counter-example).

## The review you are rebutting
## Summary

TensorGuard is a no-execution refinement-type checker for PyTorch `nn.Module` source that statically verifies tensor shapes and gradient flow without instantiating or tracing the module. The system computes refinement types `Tensor{s, g | φ}` (LIA + divisibility + bounded multiplication for shape, plus a flat `{has_grad, no_grad, ⊤}` lattice for grad), discharges obligations with Z3, and emits a five-way verdict taxonomy (VERIFIED, REFUTED-PROOF, CONTRACT-VIOLATION, LIBRARY-WARN, ABSTAIN). Headline empirics: 53/60 RP on a curated historical bug corpus (free-symbolic regime) / 56/60 with concrete input shapes; 32/34 vs Pytea's 22/34 on a fragment-fair subset (McNemar p=0.00195); 5/15 on a pre-registered post-freeze unfiltered sample, not statistically separable from FakeTensorMode (2/15) or Pytea (3/15) at α=0.05; 0/488 unconditional RP on the real-source block corpus, reported up-front as a fragment-coverage limitation. 28 of 79 shape-transfer handlers and 11/11 lemmas are mechanised sorry-free in Lean 4; the analyser, AST extractor, backward verifier, and the remaining 51 handlers stay in the TCB.

## Prior weakness disposition

- [RESOLVED] 53/60 vs. 56/60 internal inconsistency in headline RP count -- `reproducibility/reproduce_headline_60bug.py` runs in ~1.6s and prints both numbers under named regimes; `eval_v6.tex` lines 339–351 explicitly reconcile them as zero-shape vs. input-shape, with the three differential bugs (003/004/005) named and re-derivable per item.
- [RESOLVED] CEGAR and phase-check ship but are architecturally non-functional as described -- the paper now explicitly excludes them from the contribution claim ("ship with the analyser but are not claimed as contributions"; Table 5 caption + L1/L3 zero-delta rows + dead-code TCB audit `cegar_phase_deletion_tcb.py`); rebuttal accepted: the load-bearing knobs are now scoped to the three discriminative ones.
- [PARTIAL] Mutation-testing kill rate on load-bearing handlers is low without corpus extension -- rebuttal partially accepted: `mutation_kill_rate_loadbearing_v2.json` shows einsum=0.73, but conv2d_channel_mismatch is only 0.42 (still below the paper's "above 50%" framing in the rebuttal); the methodological argument that the regression corpus is for verdict-taxonomy coverage rather than handler arithmetic is fair, but the headline mutation number for `conv2d` remains under the 50% threshold the authors invoke and that gap should be acknowledged in the paper rather than only in the rebuttal.
- [RESOLVED] Theorem 5 (Dynamo) falsification predicate is vacuously satisfied on the large-corpus audits -- rebuttal accepted: `dynamo_e2e_15modules.json` reports 19/19 in-catalogue SHAPE recompiles on the end-to-end CNN+transformer base, so the inclusion-test denominator is non-zero independent of the 55/67-module null findings; Theorem 5 is now stated explicitly as the necessary direction only with the 48/544 (8.8%) in-contract rate quantifying the gap to a two-directional reading.
- [RESOLVED] No single command reproduces the headline 53/60 RP figure -- `reproducibility/reproduce_headline_60bug.py` (cited in README L284, L321 and `eval_v6.tex` L352) is a single-command, ~1.6s reproducer that prints `53/60` (paper-headline regime) and `56/60` (ablation regime) with per-item JSON; verified by direct execution.

## Strengths

- Unusually disciplined verdict-taxonomy reporting: the V/RP/CV/LW/A split and the per-cell N/A bookkeeping in Table 1 make the soundness footprint of every claim explicit; `Theorem 2` cleanly restricts coverage to RP+CV and the paper does not silently bundle LW into "refuted."
- The scope of the Lean 4 audit (28/79 handlers, 11/11 lemmas, sorry-free) is honestly disclosed in the contribution list and matches the artifact (`grep \bsorry\b` over `lean/TensorGuard/` returns only sorry-free comments), and the analyser/AST extractor/backward verifier are explicitly carved out of the trusted base.
- The fragment-fair Pytea head-to-head (32/34 vs 22/34, McNemar exact p=0.00195) is set up to neutralise the obvious catalogue-confound objection and the inclusion rule (post-2022-04-26 commits = none) makes the comparator effectively a fixed target.
- The post-freeze N=15 result is reported with calibrated CIs and *not* upgraded to a significance claim — the paper resists a familiar overclaim and explicitly states the second-wave N needed to reach p<0.05.
- The 0/488 unconditional RP on real source is reported up front as a limitation rather than buried, and the LW→RP "ceiling" of 12/78 is exhibited per-block with the named missing rule for each, making the upper bound falsifiable from the paper alone.

## Weaknesses

- **Conceptual novelty over Pytea is thinner than the contribution list suggests.** C1's refinement-typed calculus is a presentation reorganisation of constraint-based shape analysis; the genuinely new ingredient is the joint shape+grad refinement, but the grad lattice is acknowledged to be flat (`{has_grad, no_grad, ⊤}`) and silently incorrect under parameter-sharing-under-renamed-attribute. The paper's own framing ("we are the first to combine shape and grad in one Z3 procedure") is exactly the kind of "first to apply X to Y" claim that needs a sharper articulation of why a flat-lattice, single-parameter-name grad analysis qualifies as a new abstract domain rather than a Pytea+grad-flag engineering composition. A positioning paragraph that names the closest prior shape-and-effect type system (e.g., refinement types for tensor programs in Hasktorch/JAX-typing work, or the Dex/Dependently-typed-tensor literature) and says concretely what TG can prove that they cannot would strengthen Contribution.
- **Headline 5/15 on the post-freeze unfiltered sample is statistically indistinguishable from the execution-based baselines** (Fisher exact two-sided p=0.39 vs. FakeTensorMode, p=0.68 vs. Pytea). The paper acknowledges this but then continues to use the 5/15 vs 2/15 vs 3/15 triple as a "directional" headline. On N=15 with these p-values, the directional reading is not supported; the only honest headline on the unfiltered post-freeze sample is "no separation." The paper should either drop the 5-vs-2-vs-3 framing from the abstract-adjacent positioning or run the pre-computed N_new=26/77 second wave whose required size the authors themselves derived.
- **The headline 53/60 RP on the historical bug corpus is on a corpus assembled by keyword search and curated by the authors with knowledge of the operator catalogue.** The leave-one-out audits (`bug_corpus_loo.py`) "leave the aggregate RP rate at 53/60" because of an "independent AST-pattern verification path" that runs in parallel with operator dispatch. Concretely: this means the LOO is ablating a path that the AST-pattern path can recover, so the LOO does not actually probe handler-removal sensitivity. The genuine handler-LOO would disable both paths; without that, the 53/60 number is partially attributable to category-keyword AST pattern matches and the paper's own "rule-development holdout" claim is weakened.
- **The 0/488 unconditional RP on real source is much more damaging to the contribution than the paper allows.** The narrative treats this as "principled abstention" by exhibiting 12/78 LW→RP candidates, but those 12 candidates require six fragment-extension rules that are exactly the kind of operator (`unbind`, `super().forward()→Embedding`, transposed-Parameter matmul, `cumsum`) that any modern transformer block uses. The paper would benefit from running the smallest-cost LW→RP candidate (the unbind rule, claimed to be ~30 LoC) and reporting the resulting unconditional RP count on the 488-block corpus, rather than keeping the count at zero and asserting the ceiling is 12.
- **Mutation kill rate on `conv2d` is 0.42 on the load-bearing extension corpus** (`mutation_kill_rate_loadbearing_v2.json`), below the "above 50%" threshold the rebuttal invokes. The methodological point that the regression corpus is for verdict taxonomy not handler arithmetic is reasonable, but the paper should report the 0.42 figure directly with the einsum 0.73 in the empirical evaluation rather than only in supplementary mutation logs.
- **Theorem 5 is now explicitly the necessary direction only, with an 8.8% in-contract recompile rate quantifying the converse gap.** This is a calibration win, but the residual claim ("TG-abstention is necessary for guard-stability") is so weak that calling it a contribution (C4) is borderline; the empirical instantiation covers 9 CNN blocks fully + 4 transformer blocks via "documented surrogate." The transformer evidence is essentially absent at the full-instantiation level. C4 should be either downgraded to a remark or backed by at least one full transformer-block instantiation (no surrogate).
- **Soundness scope is fragile in a way the abstract does not surface.** Theorem 2 covers RP and CV; CV soundness is conditional on `assume_M` holding at the call site; 92.2% of CV rows are joint-realisable but 10/128 are not (CV verdicts whose default-Config caller doesn't satisfy the synthesised rely). Those 10 unwitnessed CV rows are reported as "uniform across HF families" but they are still 10 verdicts whose soundness statement is empty in practice; the abstract's "53/60 (88.3%) RP" headline is clean only if the paper consistently treats unwitnessed CVs as the same kind of soundness gap that LW is — it doesn't currently do that.

## Questions

- For the 12/78 LW→RP candidates, what is the unconditional RP count on the 488-block corpus *after* implementing the unbind rule the paper identifies as the smallest-cost extension? The paper claims ~30 LoC; please run it and report the resulting (V, RP, CV, LW, A) triple.
- Run a true handler-LOO that disables both the operator-dispatch path *and* the parallel AST-pattern verification path on the same handler. What is the 60-bug RP rate under this combined LOO for the top three load-bearing categories (view/reshape, broadcasting, conv-channel)?
- For C4, please give one full-instantiation in-contract recompile audit on a transformer block (BERT, GPT-2, or T5 sublayer) without the documented surrogate. Even a single block's per-fire guard table would substantiate the claim across architectures rather than only on the CNN side.
- For the 10 unwitnessed CV rows: in the paper's own taxonomy, is the soundness statement of Theorem 2 vacuous on those modules (because no caller satisfies `assume_M`)? If yes, please subtract them from the 128-CV soundness footprint or argue why they should be retained.
- For the post-freeze unfiltered N=15: are you committing to running the second wave (N_new=26 vs FakeTensorMode, N_new=77 vs Pytea) the paper itself pre-computes? If not, the 5/15 directional framing in the introduction should be replaced with the explicit "no separation" reading.
- The grad lattice is flat and acknowledged silently incorrect on parameter-sharing-under-renamed-attribute. What fraction of the 488 blocks tie weights via aliasing (e.g., `output_embedding.weight = input_embedding.weight`) or assign a single `nn.Parameter` to multiple module attributes? A concrete prevalence number on the paper's own corpus would replace the asserted ≤12% figure with a measured one.

## Scores

Soundness: 3
Presentation: 3
Contribution: 2
Confidence: 4
Overall: 5

## Borderline reasons

The single change that would push my score from 5 to 6 is a non-zero unconditional-RP count on the 488-block real-source corpus — implementing one of the 12 LW→RP candidates the paper itself identifies as low-cost (the `unbind` rule is the explicit smallest-cost candidate at ~30 LoC) and reporting the new RP/LW split on the same 488 blocks would convert the paper's "0-RP gap is principled abstention" defense into an empirically falsified upper bound and give the contribution a concrete bug-finding result on real library source rather than only on a curated historical corpus.


Changes   +0 -0
Requests  7.5 Premium (4m 13s)
Tokens    ↑ 2.2m • ↓ 13.1k • 2.1m (cached)

## Output requirements

Pick **at most 3** of the listed weaknesses. For each, write a
paragraph of strict format:

  ### Rebuttal of weakness: <verbatim wording, truncated to ~100 chars>
  Concise argument (4-8 sentences) for why this weakness is
  overweighted, factually wrong, or already addressed. Cite specific
  artifacts in the repo (concept names, theorem names, table numbers
  — NOT file paths) that prove your point. Do NOT add caveats. Do
  NOT use the word "honest" or any rebuttal-style narration that
  mentions the reviewer.

If you have nothing strong enough to rebut, write only the line:
`(no rebuttal this round — addressing all weaknesses in the improver pass)`

Do not preface with anything; the first non-blank line of your output
must be either the first `### Rebuttal of weakness:` header or the
`(no rebuttal this round...)` sentinel. Do not write to a file.

Round: 4
