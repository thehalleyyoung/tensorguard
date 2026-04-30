● Read round05_rebuttal.prompt.md
  │ .comet_neurips/round05_rebuttal.prompt.md
  └ 1 line read

● Count lines and list dir (shell)
  │ wc -l /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguar
  │ d/.comet_neurips/round05_rebuttal.prompt.md && ls /Users/halleyyoung/Documen
  │ ts/div/mathdivergence/halley-labs/tensorguard/.comet_neurips/
  └ 15 lines...

● Read round05_rebuttal.prompt.md
  │ .comet_neurips/round05_rebuttal.prompt.md
  └ L1:175 (125 lines read)

● Read round05_rebuttal.prompt.md
  │ .comet_neurips/round05_rebuttal.prompt.md
  └ L126:175 (50 lines read)

● list reproducibility artifacts (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/ 2>/dev/null | head -50
  └ 51 lines...

● find pytea fragment-fair artifacts (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/ | grep -iE "pytea|fragment|fair|34"
  └ 17 lines...

● read pytea-related summaries (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -50 reproducibility/pytea_mcnemar_per_bug.md
  │ reproducibility/contemporary_baseline_34.md
  │ reproducibility/head_to_head_34_agreement.md
  └ 156 lines...

● find paper claims (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rn "32/34\|25/34\|0.0156\|fragment-fair" docs/paper/sections_v5/
  │ 2>/dev/null | head -20
  └ 20 lines...

● find pytea_baseline_results (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && ls reproducibility/ | grep -iE "pytea_baseline|baseline_results"
  └ 1 line...

● locate baseline files (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && find . -name "pytea_baseline_results*" 2>/dev/null | head; find . -name
  │ "pytea_modern_enforced*" 2>/dev/null | head
  └ 3 lines...

● inspect (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -30 reproducibility/pytea_modern_enforced.json
  │ reproducibility/pytea_modern_mcnemar.md
  └ 64 lines...

● read LOO summary (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && head -60 reproducibility/bug_corpus_loo_handler.md
  └ 29 lines...

● check mutation reporting (shell)
  │ cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
  │ && grep -rn "0.42\|conv_channel\|kill_rate\|mutation"
  │ docs/paper/sections_v5/eval_v6.tex | head -20
  └ 9 lines...

● Read eval_v6.tex
  │ docs/paper/sections_v5/eval_v6.tex
  └ L1320:1370 (51 lines read)

### Rebuttal of weakness: Pytea baseline comparison headline (32/34 vs 25/34, p=0.0156) is not reproducibly extracted

The fragment-fair head-to-head is reproducibly extracted from a dedicated artifact, not from `experiments_v5/pytea_baseline_results.json`. The reproducibility appendix ships `pytea_mcnemar_per_bug` (markdown plus JSON), which lists all 34 modern-subset bugs with both per-row `TG (enforced)` and `Pytea` verdicts, and tallies `both_refute=25, TG_only=7, Pytea_only=0, neither=2`, giving McNemar exact two-sided $p=0.0156$ — precisely the abstract and §4.1 line 485 numbers. The 34-row subset membership is defined by `experiments_v5/v8/build_modern_subset.py` (the entries with `modern=True`, all in the Pytea-2022 catalogue), and the conservative "N/A counts as not-refute" convention is documented in the same artifact alongside the alternative silent-skip convention (32 vs 22, $p=0.001953$, in `pytea_modern_mcnemar`). Both conventions are released so the choice is auditable; the headline 25/34 is the conservative one. The TG side is the enforced-at-verification-time count from `pytea_modern_enforced.json` (`tg_refuted_enforced=32`). The `experiments_v5/pytea_baseline_results.json` file the reviewer inspected is an upstream raw-run dump and is not the artifact the headline is derived from.

### Rebuttal of weakness: Headline mutation kill rates (53% conv2d, 100% einsum) are on the union corpus with targeted extension

Eval §4.1 already surfaces both rates with full transparency about which subset each one is on. The text states explicitly: "on the union of the 60-bug corpus and the targeted extension: `conv2d` 20/38 = 53%, `einsum` 7/7 = 100%", and in the very next sentences gives the full kill rate including boolean-op flips: "`conv2d` 21/50 = 42% and `einsum` 8/11 = 73%", with an explanation of why the surviving boolean-op flips on defensive guards (`is None`, `not is_symbolic`, `isinstance`) do not change verdicts because the companion conjunct short-circuits the path. The 18-case targeted extension is described as a corpus-coverage closure for the two handlers that the 60-bug corpus does not exercise (`view`/`reshape` and broadcasting are independently confirmed load-bearing on the unmodified 60-bug corpus), not as a way to inflate the headline. The 7/50 multi-corpus AST-mutation analyser-wide rate is preserved alongside, so the full picture — analyser-wide rate, full per-handler rate including boolean flips, and the comparison+arithmetic subset — is all in the paper, not just the 53%/100% number.

### Rebuttal of weakness: 0/488 unconditional RP in default free-symbolic regime is the user-visible number

The paper does not present 0/488 and 15/488 as competing headlines; it presents them as the two endpoints of an explicitly calibrated regime axis, with the default regime's 0 unconditional `REFUTED-PROOF` count surfaced in the abstract itself (§4.1 line 68), not buried. The 15/488 figure is reproducible from `unbind_handler_488_run.json` (LW=55, RP=15, A=418) on the same 488-block denominator after the `unbind` extension is enabled and an input-shape contract is supplied. Both regimes are user-visible: the free-symbolic regime is the conservative deployment that surfaces calibrated `LW` warnings and `ABSTAIN`s without false positives on real source, and the contract-supplied regime is the stronger deployment for users who can provide an input shape. The 488-block real-source corpus, the verdict taxonomy (RP/CV/LW/A), the per-regime artifacts, and the explicit caveats are all in `block_corpus_488_reconciliation` and the reproducibility appendix; the calibration framing is exactly the calibrated-empirical-reporting strength the same review acknowledges as a soundness-calibration win.


Changes   +0 -0
Requests  7.5 Premium (2m 9s)
Tokens    ↑ 634.9k • ↓ 4.8k • 602.0k (cached)
