# Step 15 -- false UNKNOWN rate in sound mode

Strict `sound` mode was run over **86 executable, ground-truthed** models: 8 frozen real clean benchmarks, 72 generated clean models admitted only after eager PyTorch execution, and 6 real latent phase/path bugs confirmed by the hard-recall validators.

## Result

| Metric | Value |
|---|---|
| Eligible models | 86 |
| Decided by sound mode | 86 |
| **False UNKNOWNs** | **0** |
| False UNKNOWN rate | 0.00% |
| Decision rate | 100.00% |
| Misclassifications | 0 |

A false UNKNOWN is an abstention on code that is already executable and ground-truthed. The measured rate is **0.00%**, so sound mode is not buying its zero-false-positive guarantee by refusing to decide this in-fragment benchmark.

## By kind

| Kind | Total | SAFE | UNSAFE | UNKNOWN | Decided |
|---|---|---|---|---|---|
| `buggy` | 6 | 0 | 6 | 0 | 6 |
| `clean` | 80 | 80 | 0 | 0 | 80 |

## By family

| Family | Total | SAFE | UNSAFE | UNKNOWN |
|---|---|---|---|---|
| `generated_attention` | 12 | 12 | 0 | 0 |
| `generated_cnn` | 12 | 12 | 0 | 0 |
| `generated_groupnorm_conv` | 12 | 12 | 0 | 0 |
| `generated_layernorm_mlp` | 12 | 12 | 0 | 0 |
| `generated_mlp` | 12 | 12 | 0 | 0 |
| `generated_residual_mlp` | 12 | 12 | 0 | 0 |
| `path_flag` | 3 | 0 | 3 | 0 |
| `phase_eval` | 3 | 0 | 3 | 0 |
| `real_clean` | 8 | 8 | 0 | 0 |
