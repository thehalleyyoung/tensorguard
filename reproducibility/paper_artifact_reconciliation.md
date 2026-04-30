# Paper / artifact reconciliation (round 8)

## Command

```
python3 reproducibility/paper_artifact_reconciliation.py
```

## What it does

For every load-bearing numerical claim that the round-8 reviewer
flagged as inconsistent between the paper and the shipped JSON
artifacts in this directory, the script asserts that the two agree.
It exits non-zero on the first mismatch.

## Checks performed

| Check | Artifact(s) | Paper section | Reviewer item |
|---|---|---|---|
| grad-lattice runtime false-verified rate | `grad_lattice_runtime_holdout.json` | eval (§4.4 grad-lattice runtime) and limitations | R8-W1 / R8-Q1 |
| Theorem 5 larger-population audit | `dynamo_theorem5_n100.json`, `dynamo_theorem5_n200.json` | eval §4.3 audit-on-strictly-larger-module-population | R8-W2 / R8-Q2 |
| hybrid-mode complementarity scope | (no JSON; tex-only) | eval §4.2 hybrid-mode + §4.2 stress-set table | R8-W3 / R8-Q4 |
| post-freeze 5/15 headline still present | (covered by `real_bugs_postfreeze.json`) | eval post-freeze paragraph | prior W3 (resolved) |
| abstract structure / no repo paths | (`neurips.tex` itself) | abstract block | improver hard constraint 1 + 4 |

## Latest run

5/5 checks passing.

## Paper claim mapping

* The 2/8 = 25.0% false-verified rate is now stated in **both** the
  evaluation section (where the harness is described) and the
  limitations paragraph (where the reviewer expected it).  The
  separate 0/8 figure is retained but explicitly relabelled as the
  data\_ptr aliasing prevalence on input parameters, not as a
  false-verification metric.

* The Theorem 5 paragraph now explicitly references both the
  n=107/55/72-INT audit and the n=146/67/0 extended audit and
  notes that both agree on the only quantity the theorem requires
  (zero out-of-catalogue SHAPE/DTYPE/RANK guards, zero falsifier
  events).  The paragraph attributes the in-contract recompile gap
  to the additional torchvision/timm vision backbones in the
  larger pool, which do not enter the dynamic-input
  specialisation regime that produced the 72 INT recompiles on
  the smaller HuggingFace-heavy pool.

* The hybrid-mode complementarity claim is now restated as an
  existence demonstration on a hand-designed importable stress
  set.  The zero-gain result on the 488-block real-source corpus
  is reported in the same paragraph and labelled as the
  natural-distribution reading.

## Reproducibility

The script is pure-Python and reads only files already in this
repository.  It does not invoke `torch`, the analyser, or any
external service.  Running time on a 2024 MacBook is well under a
second.
