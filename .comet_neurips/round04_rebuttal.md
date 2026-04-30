● Read round04_rebuttal.prompt.md
  │ .comet_neurips/round04_rebuttal.prompt.md
  └ 79 lines read

● List paper sections (shell)
  │ ls docs/paper/sections_v5/ 2>/dev/null && ls docs/paper/ 2>/dev/null | head
  │ -30
  └ 42 lines...

● Search abstract claims (shell)
  │ grep -n "26 unconditional\|15/488\|26/356\|fragment-fair\|34"
  │ docs/paper/sections_v5/intro_v6.tex docs/paper/sections_v5/eval_v6.tex
  │ 2>/dev/null | head -50
  └ 29 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L280:350 (71 lines read)

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L60:110 (51 lines read)

### Rebuttal of weakness: "headline numerical claim in the abstract is in tension with Table `tab:headline`"
The abstract's $15/488$ and $26/356$ are not in tension with the $0$-RP row of `tab:headline`; they refer to a named, in-paper-defined refinement of the same regime, and the eval section labels it explicitly. `eval_v6.tex` ("Calibration first") leads with the $0$-unconditional-RP figure under the high-confidence Z3 regime as the headline, then in the same paragraph announces that "the unconditional-RP claim is carried by the bug corpora, not by the block corpus." The $26/356$ figure is then defined as "the no-synthesised-assume subset of the corpus: $26$ Refuted-Proof verdicts on the $356$-block subset where the Contract-Violation classifier returns an empty caller-rely obligation," which is precisely the $26/128$ empty-$\mathit{assume}_M$ partition disclosed in the caller-rely satisfiability paragraph. The $15/488$ figure is the rule-extension rerun with `Tensor.unbind`, transposed-Parameter matmul, and the `nn.Embedding`/`nn.Linear`/`nn.LayerNorm` subclass recogniser enabled, with both the $15/488$ and $15/295$ analysable-subset Wilson CIs reported. Both refinements are derivations from `tab:headline`, not contradictions of it, and both have a falsifiability anchor in the $12$-row LW$\to$RP candidate table. The reconciliation is in the same eval section that hosts `tab:headline` and was not exfiltrated to the abstract.

### Rebuttal of weakness: "fragment-fair head-to-head … paper does not make it possible to audit how 60 bugs were filtered to 34"
The deterministic filter and the per-bug audit table both exist as paper artefacts, not as un-released claims. `eval_v6.tex` states that "the full $34$-row matched-pair contingency table (each row: bug-id, TG verdict, fragment-fairness enforcement, and Pytea verdict) is included in the reproducibility appendix," and that the fragment-fair filter is the operator-surface intersection with Pytea's $2022$ catalogue, with the $32/34$ figure reproduced at verification time at a recorded commit SHA (`cc75834b...`) in the reproducibility appendix. The filter is not a curated selection from $60$ but the deterministic projection onto Pytea's catalogued operators; the paper further defends this by noting that even on the $2024{-}26$ catalogue the headline is the modern-mirror figure, and discloses the $22/34$ alternative count under a stricter "Pytea-catches" denominator. A McNemar audit therefore requires the appendix table, which is shipped, not a re-derivation from the prose.

### Rebuttal of weakness: "four different `{V, R, A}` triples for the same 488-block corpus across regimes and re-runs"
The four triples are not numerical drift over a single regime; they are the $2\times 2$ product of two named, structural axes that the bookkeeping note enumerates. The first axis is high-confidence (Z3-only) vs.\ public default (adds the lower-confidence heuristic post-pass), and the eval section pins down that this axis moves exactly $7$ rows from Verified to Refuted with the $225$ Abstain count invariant. The second axis is original-capture vs.\ re-execution against current code, with the eval section certifying that the $5$-block absolute drift is bookkeeping-clean (no Abstain transitions, no refutation flips), so the same $7$-row Verified$\to$Refuted diff structure carries across re-execution. The high-confidence regime is identified as canonical for the abstract and `tab:headline` ("the high-confidence regime is the one cited throughout this paper"), and the per-id audit accompanies the open-source implementation. This is calibration of two orthogonal knobs with a documented diff structure, not unexplained re-run variance on a single configuration.


Changes   +0 -0
Requests  7.5 Premium (54s)
Tokens    ↑ 231.3k • ↓ 2.5k • 188.0k (cached)
