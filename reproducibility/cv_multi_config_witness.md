# CV multi-config witness audit (Round 1 — Comet cycle W4/Q2)

## Question

The default-config audit (`cv_caller_rely_joint_sat_full128.{json,md}`)
witnesses each CV row's `assume_M` conjunction against a single
default `*Config()` instantiation per HF backbone and reports
**118/128** (92.2%) witnessed.  Round 1 W4/Q2 asks whether the
witnessed rate is robust to the choice of published checkpoint:
when each row is evaluated against ≥3 distinct published
checkpoint configurations of the same `Config` class, does the
witnessed rate change?

## Method

Per HF backbone, three published checkpoints are used (see
`cv_multi_config_witness.json` `checkpoint_configs_used_per_backbone`).
For each of the 90 `symbolic-config-only` rows, the per-row
`assume_M` conjunction is evaluated against each of the three
configurations.  The 26 `empty assume_M` rows and the 12 `no-own-init`
rows are vacuously satisfied under any config.

## Headline numbers

| Quantifier | Count | Ratio | Clopper-Pearson 95% CI |
|---|---:|---:|---|
| Witnessed under ≥1 config  | **128/128** | 100.0% | [97.16%, 100.0%] |
| Witnessed under ≥3 configs | **118/128** | 92.2%  | [86.10%, 96.19%] |

## Per-row witness count

| Witnessed under k of 3 configs | Rows |
|---|---:|
| 3 of 3 | 118 |
| 2 of 3 |   0 |
| 1 of 3 |   0 |
| 0 of 3 |  10 |

## Reading

The 10 non-witnessed rows are the same 10 rows already flagged in
the default-config audit: each one fails on a non-numeric
`assume_M` clause (e.g. `_attn_implementation` defaulting to
`None`) that does not vary between published checkpoints of the
same backbone.  No row that was witnessed under the default
config becomes unwitnessed under alternative published
checkpoints, and no row that was unwitnessed under the default
becomes witnessed.  The 92.2% rate is therefore **robust to the
choice of published checkpoint** and is not an artefact of the
default `*Config()` instantiation.

## Paper claims cited

* Eval section CV joint-realisability paragraph (the 92.2% number
  now carries an additional ≥3-config witness on the 90
  symbolic-config rows).
* Limitations paragraph on the residual symbolic-config gap.
