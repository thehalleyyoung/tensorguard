# TensorGuard NeurIPS v7 Rigor Pass — Summary

Backup of pre-revision PDF/source: `docs/paper/neurips.tex.before_v7`,
prior PDF in git history. Body length: **9 pages** (refs start on p.10);
total 21 pages (body 9 + refs 1 + appendix 4 + checklist 7). Line numbers
visible: **32 on p.1** (≥30 required); `\renewcommand{\sfdefault}{ptm}`
preserved.

## Per-ask reconciliation

| # | Reviewer ask | Status | Where |
|---|---|---|---|
| 1 | Verdict-taxonomy refactor (`REFUTED-PROOF` / `CONTRACT-VIOLATION` / `LIBRARY-WARN`); rewrite Theorem 1; reclassify the 206 block-corpus refutations | **Done** | `src/v5/verdict_taxonomy.py`; `tests/v5/test_verdict_taxonomy.py` (13/13 pass); `experiments_v5/run_verdict_reclassification.py` → `experiments_v5/verdict_reclassification.json`. Theorem 1 (`thm:soundness`) rewritten in `sections_v5/calculus_v6.tex` to cover only V/RP/CV. Headline table in `sections_v5/eval_v6.tex` (Tab. 1) splits 206 → 0 RP / 128 CV / 78 LW. |
| 2 | Pytea baseline | **Done** (install succeeded) | `experiments_v5/_pytea_src/` (cloned ropas/pytea, npm-built); `experiments_v5/pytea_baseline_results.json`; row added to `sections_v5/eval_v6.tex` Tab. 1 and `sections_v5/related_v6.tex`. Pytea: 27/60 bug refute, 31/60 silent verified, 2 N/A; 53/488 verified, 435/488 N/A. |
| 3 | Reduce Lean language → "operator-rule audit"; per-handler scope table | **Done** | `sections_v5/eval_v6.tex` (§ Lean-audited operator-rule table); `sections_v5/handler_soundness_table.tex` (Tab. 5: 28 Lean-audited / 3 pen-and-paper / 48 tested-only); `sections_v5/H_contribution_table.tex` ("Lean-audited rules"); `sections_v5/intro_v6.tex` (C6); `sections_v5/limconc_v6.tex`; abstract; title. |
| 4 | Auditable benchmark manifest | **Done** | `experiments_v5/build_bug_corpus_manifest.py` → `experiments_v5/bug_corpus_manifest.json` (60 items, schema_version=1) and `experiments_v5/bug_corpus_manifest.md`. Cited in body via "released benchmark manifest" (no script-file name in body). 60/60 in-fragment; 56 RP / 4 silent V / 0 A. |
| 5 | Checklist consistency on Lean | **Done** | `docs/paper/neurips_2026_checklist.tex` lines 37, 49, 65: removed "17 theorems, 0 sorry" claim; explicitly notes the operator-rule audit scope and excludes parser/analyser/AG/backward from mechanisation; soundness verdict scope (V / RP / CV only) restated. |
| 6 | Dynamo correspondence weakened to "necessary, not sufficient" | **Done** | `sections_v5/eval_v6.tex` (`thm:dynamo-corr` rewritten as necessary direction only; converse explicitly disclaimed); calibrated paragraph below the table notes the 48/544 in-contract recompiles as a one-directionality reminder; `sections_v5/limconc_v6.tex` reference softened; appendix proof in `sections_v5/appendix_v6.tex` updated. |

## Outputs touched

* `docs/paper/neurips.tex` (title, abstract)
* `docs/paper/sections_v5/{intro,calculus,impl,eval,limconc,appendix,related,handler_soundness_table,H_contribution_table,F_benchmark}_v6.tex` (touched all relevant)
* `docs/paper/neurips_2026_checklist.tex`
* `src/v5/verdict_taxonomy.py` (new)
* `tests/v5/test_verdict_taxonomy.py` (new, 13/13 pass)
* `experiments_v5/run_verdict_reclassification.py` (new)
* `experiments_v5/verdict_reclassification.json` (new)
* `experiments_v5/build_bug_corpus_manifest.py` (new)
* `experiments_v5/bug_corpus_manifest.json` / `.md` (new)
* `experiments_v5/pytea_baseline_results.json` (new)
* `experiments_v5/_pytea_src/` (vendored Pytea checkout)

## Self-estimate

Target: **6+ → solid weak accept.** The single reviewer-blocking
soundness/evaluation tension is resolved by a verdict refactor that
is mechanically reflected (a) in code (`verdict_taxonomy.py` + tests),
(b) in the headline evaluation table (RP/CV/LW columns with
0/128/78 counts), and (c) in Theorem 1's statement (V/RP/CV only).
The Dynamo theorem is honest about being necessary, not sufficient,
with the 48/544 in-contract recompiles cited as a one-directionality
reminder. Pytea was actually built and run, contributing a real
no-execution baseline row (not a citation). The Lean language is
consistently downgraded to "operator-rule audit" across body,
checklist, contribution table, and per-handler scope table. Bug
corpus has full provenance manifest. Body is at the 9pp limit;
≥30 line numbers on p.1; ordering paper→refs→appendix→checklist.

Residual risk: the 4 silent misses on the bug corpus (V-but-buggy)
are still a soft target; we list them in the released JSON as
calibration notes. The honest 8.8% Dynamo in-contract recompile
rate is now framed as a known limitation rather than an edge case.
