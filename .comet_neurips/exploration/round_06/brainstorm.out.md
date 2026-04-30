● Read brainstorm.prompt.md
  │ .comet_neurips/exploration/round_06/brainstorm.prompt.md
  └ 110 lines read

### Candidate 1: Mechanised classifier for the 13 pen-and-paper handlers
goal: Replace the manual T-Identity/T-Broadcast classification of the 13 pen-and-paper handlers with an executable checker that emits a per-handler JSON certificate, eliminating the silent over-count risk in the 128/185 audited footprint. Lifts Soundness by ~0.5 (closes obligation on "no formal check that all 13 pen-and-paper handlers are correctly classified") by replacing prose with a reproducible artifact the reviewer can rerun.
plan:
- Locate the 13 pen-and-paper handlers in `src/model_checker.py` (or wherever the audited-footprint list lives) and the `applyOp_sound_*` Lean theorem set under `lean/`.
- Write `reproducibility/classify_pen_and_paper_handlers.py` that, for each of the 13 handler names, inspects the handler's Python implementation and asserts via AST pattern that it either (a) returns input shape unchanged (T-Identity) or (b) applies `torch.broadcast_shapes`-equivalent logic on inputs (T-Broadcast); emit `reproducibility/pen_and_paper_classification.json` with `{handler, class, evidence_lines, sha}` per row.
- Add `tests/test_pen_and_paper_classification.py` that loads the JSON, asserts all 13 handlers are classified, asserts no `class == "unknown"`, and asserts each `evidence_lines` non-empty.
- Wire the script into a `make pen-paper-audit` target or top-level shell call documented in `reproducibility/README.md`.
success_criterion: `python reproducibility/classify_pen_and_paper_handlers.py && pytest tests/test_pen_and_paper_classification.py -x` exits 0, AND `python -c "import json;d=json.load(open('reproducibility/pen_and_paper_classification.json'));assert len(d)==13 and all(r['class'] in ('T-Identity','T-Broadcast') for r in d)"` exits 0.
fallback_message: If 10 minutes is insufficient to enumerate the 13 handlers and emit certificates, the subagent should print `INFEASIBLE: pen-and-paper handler enumeration requires manual cross-reference with Lean theorem list; reverting` and exit non-zero so the harness reverts.

### Candidate 2: AST-based independent audit of the ≤12% tied/renamed-attribute prevalence
goal: Replace the unaudited regex-screened ≤12% prevalence figure with an AST-level detector that scans the 488-block real-source corpus for tied/renamed-attribute parameter sharing and emits an independent prevalence point estimate with Wilson CI, then recomputes the deployment-side false-Verified bound from the new prevalence × 25% worst-case rate. Lifts Soundness by ~0.5 by directly addressing the round-6 obligation on the unaudited prevalence denominator.
plan:
- Add `reproducibility/ast_tied_param_audit.py` that walks every block in the 488-block real-source corpus (path discoverable from `experiments_v5/` / `real_benchmarks/`), parses each `nn.Module` via `ast`, and flags blocks where two attribute assignments bind the same `nn.Parameter`/`nn.Linear.weight` object or where `setattr` aliases an existing parameter — i.e., true tied/renamed-attribute sharing, not regex hits.
- Emit `experiments_v5/ast_tied_param_prevalence.json` with `{n_blocks, n_flagged, prevalence, wilson_low, wilson_high, recomputed_bound = wilson_high * 0.25}`; use `statsmodels.stats.proportion.proportion_confint` (or hand-rolled Wilson) and seed-free deterministic enumeration.
- Add `tests/test_ast_tied_param_audit.py` with at least 3 synthetic positive fixtures (tied weights via shared module, via `setattr`, via parameter aliasing) and 2 negatives (independent linears, fresh parameters); assert detector flags positives and not negatives, and assert the JSON file exists with all required keys and `0 <= prevalence <= 1`.
- Print a single summary line `PREVALENCE_AUDIT prevalence=<x> wilson=[<lo>,<hi>] bound=<b>` to stdout for harness parsing.
success_criterion: `python reproducibility/ast_tied_param_audit.py && pytest tests/test_ast_tied_param_audit.py -x` exits 0, AND the JSON file contains a numeric `prevalence` field, AND stdout contains a line matching regex `^PREVALENCE_AUDIT prevalence=0\.\d+ wilson=\[0\.\d+,0\.\d+\] bound=0\.\d+$`.
fallback_message: If the 488-block corpus path is not resolvable or `ast` parsing of the source files fails system-wide within 10 minutes, the subagent should print `INFEASIBLE: real-source corpus path unresolved or unparseable; reverting` and exit non-zero so the harness reverts.


Changes   +0 -0
Requests  7.5 Premium (43s)
Tokens    ↑ 72.6k • ↓ 2.5k • 52.4k (cached)
