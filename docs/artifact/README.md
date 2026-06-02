# TensorGuard — Artifact Appendix

This artifact accompanies the TensorGuard tool paper. TensorGuard is a **sound,
static** verifier for PyTorch `nn.Module`s: it proves the absence of (or finds)
shape, device, dtype, training-phase, and gradient bugs using refinement types
discharged by an SMT solver plus a reduced-product abstract domain — with **no
model execution, no concrete inputs, and no GPU**.

- **Getting started / install:** `docs/artifact/INSTALL.md`
- **Hardware/software requirements:** `docs/artifact/REQUIREMENTS.md`
- **Badges sought & justification:** `docs/artifact/STATUS.md`

## One command reproduces every quantitative claim

```bash
python reproducibility/reproduce_all.py --check
```

This regenerates, in dependency order, every CI-reproducible artifact and then
runs the **numeric-claim audit**, which recomputes every `x/y` ratio and percent
token in `README.md` from the freshly regenerated files. `--check` additionally
asserts byte-level determinism. The harness exits non-zero on any mismatch, so a
green run *is* the reproduction.

## Claim → command map

Each paper claim is backed by a committed artifact and a regeneration command.
Claims marked *(env-qualified)* need an extra toolchain (Lean / CUDA / network)
and are validated against their committed copies in a standard CI box.

| # | Paper claim | Regenerate with | Committed artifact |
| - | --- | --- | --- |
| 1 | Headline Refuted-Proof bug figure (60-bug corpus) | `python reproducibility/reproduce_headline_60bug.py` | `reproducibility/reproduce_headline_60bug.json` |
| 2 | Precision/recall/F1 vs. PyTea + runtime + no-op baselines | `PYTHONPATH=. python evaluation/precision_recall.py` | `evaluation/confusion_matrices.json` |
| 3 | **Statistical significance** of the comparison (paired McNemar exact + Holm-Bonferroni + paired bootstrap CI) | `PYTHONPATH=. python evaluation/significance.py` | `evaluation/significance.json` |
| 4 | Sound-mode false-positive hunt (zero false alarms) | `PYTHONPATH=. python evaluation/sound_mode_fp.py` | `evaluation/sound_mode_fp.json` |
| 5 | Formalization: lattice laws, widening termination, reduced-product soundness reconciled with the code | `python -m pytest tests/test_formalization.py -q` | `docs/formalization/type_system.md` |
| 6 | Lean soundness proofs build sorry-free; core transfer functions are axiom-clean *(env-qualified: Lean 4)* | `cd lean && lake build TensorGuard` then `python -m pytest tests/test_lean_soundness.py -q` | `lean/TensorGuard/AxiomAudit.lean` |
| 7 | Labeled real-world bug benchmark with integrity manifest | `python -m real_benchmarks.build_manifest` | `real_benchmarks/manifest.json` |
| 8 | Every README number recomputed from artifacts | `python reproducibility/audit_numeric_claims.py` | `reproducibility/numeric_claims_audit.json` |

## Reproducing the significance result (claim 3)

`evaluation/significance.py` consumes the per-item predictions in
`evaluation/confusion_matrices.json` and, for each `tensorguard` vs. baseline
pair, runs an exact two-sided **McNemar** test (conditioning on the discordant
items), corrects the family with **Holm-Bonferroni**, and attaches a paired
percentile **bootstrap** confidence interval for the accuracy gap. The
statistics primitives live in `src/statistical_rigor.py` and are validated
against closed-form binomial tails in `tests/test_significance.py`.

The result is deliberately honest about the corpus size: TensorGuard reaches
1.000 accuracy and never loses a discordant pair to any baseline, beats the
trivial no-op floor significantly after correction, yet its gap over PyTea is
not yet significant at this sample size — exactly the kind of result a reviewer
should be able to trust.

## Notes for evaluators

- Nothing here downloads models or executes target code during the functional
  path; analysing an untrusted file never runs it (`SECURITY.md`).
- Env-qualified artifacts (claim 6 and the CUDA/HuggingFace studies) are
  rebuilt by `make reproduce-full` in an environment that has the toolchains;
  the audit reports their claims as `QUALIFIED_ENV` with the exact command.
