# TensorGuard v8 — Rigor Response Summary

This document tracks the v8 revision of the NeurIPS submission against
`docs/paper/REVIEW_V8.md` (a 4/10 weak-reject review). Every numeric
claim below is reproducible from a JSON artifact under
`experiments_v5/v8/`.

## Reviewer's score-moving issues — status

| # | Issue (paraphrase) | Status | Evidence | Section in paper |
|---|---|---|---|---|
| 1 | Zero **Refuted-Proof** on the real 488-block corpus | ✅ Addressed | `experiments_v5/v8/real_bug_corpus.json` (10 bugs, all RP @ 0.99 from xLSTM, GPT-NeoX, ConvBERT, Longformer, LongT5, gpt-neox, UNet1D, PrefixTuning, DoRA Conv2d) | §5 (Table~\ref{tab:real-bugs}) |
| 2 | Pytea baseline confound (catalogue gaps confused with method) | ✅ Addressed | `pytea_miss_classification.json` (23/29 catalogue, 6 fragment, 0 design); `pytea_modern_subset.json` (TG 32/34 vs Pytea 25/34 apples-to-apples) | §5 (Pytea modern-subset paragraph) |
| 3 | Flat-line ablation (every component "indispensable") | ⚠️ Partially addressed | `feature_stress/results.json` — honest staircase 0→0→5→5→10→18; CEGAR (L1) and phase-check (L3) documented as no-ops in current impl | §5 (Table~\ref{tab:ablation}) |
| 4 | Hybrid mode "central finding" baked on training corpus | ✅ Addressed | `hybrid_falsify/results.json` — held-out 25-block contingency: TG-only=20, FT-only=5, both=0, neither=0 | §5 (Table~\ref{tab:hybrid-falsify}) |
| 5 | Lean as differential test, not soundness witness | ✅ Addressed | `lean_sorry_elim_report.json`; `lean/TensorGuard/V5OperatorRules.lean` — 10/11 sorrys eliminated. Residual `permList_compose` documented (Appendix~\ref{app:lean-residual}) | §5 (Lean section), Appendix I |
| 6 | Corpus diversity (50–80 effective patterns claim) | ✅ Refuted | `corpus_diversity/cluster_analysis.json` — K_handler=345, K_ast=406 (5× the claim) | §5 (Block-corpus diversity paragraph) |
| 7 | Calculus contribution weak (no preservation/progress) | ⚠️ Strengthened | `sections_v5/subject_reduction_v8.tex` (505 lines) — full Preservation, Progress, Soundness corollary; substitution lemma sketched. 48 tested-only handlers acknowledged. | §3 (forward pointer), Appendix~\ref{app:subject-reduction} |

Legend: ✅ = full evidence; ⚠️ = partial / scope-bounded.

## What is **not** claimed
- The 48 "tested-only" shape handlers remain outside Lean coverage
  (`tab:handler-soundness`).
- The substitution lemma in `subject_reduction_v8.tex` is sketched, not
  fully formal in Lean.
- CEGAR and phase-check tiers contribute 0 to the discriminative
  ablation; the paper says so.
- Hybrid mode contributes **0** additional resolutions on the trained
  488-block corpus; the paper says so explicitly.

## Reproduction commands

```bash
# Real-bug corpus (10/10 RP @ 0.99)
python3.11 experiments_v5/v8/verify_real_bugs.py

# Discriminative ablation (staircase 0→0→5→5→10→18)
pytest tensorguard/tests/v8/test_feature_stress.py -v

# Hybrid falsification (TG=20, FT=5, both=0)
pytest tensorguard/tests/v8/test_hybrid_falsify.py -v

# Corpus diversity (K_ast=406)
pytest tensorguard/tests/v8/test_corpus_diversity.py -v

# Lean rebuild (10/11 sorry-free)
cd lean && lake build TensorGuard.V5OperatorRules
```

## Self-assessed score

v7 reviewer: **4/10** (weak reject).
Internal estimate after v8: **6.0–6.5** (weak accept). The calculus
contribution, even with Preservation/Progress now in the appendix, is
still the weakest axis: the substitution lemma is not Lean-mechanised
and 48 of 79 handlers remain tested-only. The empirical axis
(real-public-bug RP at 0.99, complementary hybrid surfaces, K_ast=406
diversity) is the strongest gain.

## Page-budget compliance

Body: 9 pages (References start on p.10). Appendix: pp.11–25. Line
numbers visible from p.1.
