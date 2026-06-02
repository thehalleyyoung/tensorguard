# TensorGuard on the extended benchmark corpus

Scored over **227** content-addressed cases (**153** buggy / **74** clean) across **9** shape-error families. Every case is runtime-validated against real PyTorch at build time, so the labels are ground truth and nothing is cherry-picked. Rates are reported with **Wilson score 95 percent confidence intervals**.

## `balanced` mode

| metric | value |
| --- | --- |
| confusion (tp / fp / tn / fn) | 153 / 0 / 74 / 0 |
| abstained (buggy / clean) | 0 / 0 |
| recall on decided | 1.0000 [0.9755, 1.0000] (n=153) |
| recall on all buggy | 1.0000 [0.9755, 1.0000] (n=153) |
| specificity on decided | 1.0000 [0.9507, 1.0000] (n=74) |
| false-positive rate on decided | 0.0000 [0.0000, 0.0493] (n=74) |
| precision | 1.0000 [0.9755, 1.0000] (n=153) |
| no false positive on clean | True |

Per-family recall (on decided buggy cases):

| family | caught | total | abstained | recall (95% CI) |
| --- | --- | --- | --- | --- |
| add_broadcast_mismatch | 27 | 27 | 0 | 1.0000 [0.8754, 1.0000] (n=27) |
| cat_dim_mismatch | 27 | 27 | 0 | 1.0000 [0.8754, 1.0000] (n=27) |
| conv_channel_mismatch | 24 | 24 | 0 | 1.0000 [0.8620, 1.0000] (n=24) |
| flatten_fc_mismatch | 12 | 12 | 0 | 1.0000 [0.7575, 1.0000] (n=12) |
| linear_inout_mismatch | 36 | 36 | 0 | 1.0000 [0.9036, 1.0000] (n=36) |
| matmul_inner_mismatch | 27 | 27 | 0 | 1.0000 [0.8754, 1.0000] (n=27) |

## `sound` mode

| metric | value |
| --- | --- |
| confusion (tp / fp / tn / fn) | 153 / 0 / 74 / 0 |
| abstained (buggy / clean) | 0 / 0 |
| recall on decided | 1.0000 [0.9755, 1.0000] (n=153) |
| recall on all buggy | 1.0000 [0.9755, 1.0000] (n=153) |
| specificity on decided | 1.0000 [0.9507, 1.0000] (n=74) |
| false-positive rate on decided | 0.0000 [0.0000, 0.0493] (n=74) |
| precision | 1.0000 [0.9755, 1.0000] (n=153) |
| no false positive on clean | True |

Per-family recall (on decided buggy cases):

| family | caught | total | abstained | recall (95% CI) |
| --- | --- | --- | --- | --- |
| add_broadcast_mismatch | 27 | 27 | 0 | 1.0000 [0.8754, 1.0000] (n=27) |
| cat_dim_mismatch | 27 | 27 | 0 | 1.0000 [0.8754, 1.0000] (n=27) |
| conv_channel_mismatch | 24 | 24 | 0 | 1.0000 [0.8620, 1.0000] (n=24) |
| flatten_fc_mismatch | 12 | 12 | 0 | 1.0000 [0.7575, 1.0000] (n=12) |
| linear_inout_mismatch | 36 | 36 | 0 | 1.0000 [0.9036, 1.0000] (n=36) |
| matmul_inner_mismatch | 27 | 27 | 0 | 1.0000 [0.8754, 1.0000] (n=27) |

**Sound mode has zero false positives on clean code: True.** This is the core soundness promise: no clean module is ever flagged as buggy.
