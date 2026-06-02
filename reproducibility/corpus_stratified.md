# Stratified per-class metrics (Wilson 95% CIs)

Every one of the **227** cases is scored and stratified by class, so no weak bug class can hide behind a strong average. Macro averages weight each class equally.

## `balanced` mode

- macro recall: **1.0**, macro specificity: **1.0**
- worst buggy class by recall lower bound: `flatten_fc_mismatch` (1.0000 [0.7575, 1.0000] (n=12))
- every buggy class fully caught: True; every clean class false-positive-free: True

Per buggy class (recall on decided cases):

| bug class | caught | decided | total | recall (95% CI) |
| --- | --- | --- | --- | --- |
| add_broadcast_mismatch | 27 | 27 | 27 | 1.0000 [0.8754, 1.0000] (n=27) |
| cat_dim_mismatch | 27 | 27 | 27 | 1.0000 [0.8754, 1.0000] (n=27) |
| conv_channel_mismatch | 24 | 24 | 24 | 1.0000 [0.8620, 1.0000] (n=24) |
| flatten_fc_mismatch | 12 | 12 | 12 | 1.0000 [0.7575, 1.0000] (n=12) |
| linear_inout_mismatch | 36 | 36 | 36 | 1.0000 [0.9036, 1.0000] (n=36) |
| matmul_inner_mismatch | 27 | 27 | 27 | 1.0000 [0.8754, 1.0000] (n=27) |

Per clean class (specificity on decided cases):

| clean class | true neg | decided | total | specificity (95% CI) |
| --- | --- | --- | --- | --- |
| clean_conv | 18 | 18 | 18 | 1.0000 [0.8241, 1.0000] (n=18) |
| clean_mlp | 32 | 32 | 32 | 1.0000 [0.8928, 1.0000] (n=32) |
| clean_norm_mlp | 24 | 24 | 24 | 1.0000 [0.8620, 1.0000] (n=24) |

## `sound` mode

- macro recall: **1.0**, macro specificity: **1.0**
- worst buggy class by recall lower bound: `flatten_fc_mismatch` (1.0000 [0.7575, 1.0000] (n=12))
- every buggy class fully caught: True; every clean class false-positive-free: True

Per buggy class (recall on decided cases):

| bug class | caught | decided | total | recall (95% CI) |
| --- | --- | --- | --- | --- |
| add_broadcast_mismatch | 27 | 27 | 27 | 1.0000 [0.8754, 1.0000] (n=27) |
| cat_dim_mismatch | 27 | 27 | 27 | 1.0000 [0.8754, 1.0000] (n=27) |
| conv_channel_mismatch | 24 | 24 | 24 | 1.0000 [0.8620, 1.0000] (n=24) |
| flatten_fc_mismatch | 12 | 12 | 12 | 1.0000 [0.7575, 1.0000] (n=12) |
| linear_inout_mismatch | 36 | 36 | 36 | 1.0000 [0.9036, 1.0000] (n=36) |
| matmul_inner_mismatch | 27 | 27 | 27 | 1.0000 [0.8754, 1.0000] (n=27) |

Per clean class (specificity on decided cases):

| clean class | true neg | decided | total | specificity (95% CI) |
| --- | --- | --- | --- | --- |
| clean_conv | 18 | 18 | 18 | 1.0000 [0.8241, 1.0000] (n=18) |
| clean_mlp | 32 | 32 | 32 | 1.0000 [0.8928, 1.0000] (n=32) |
| clean_norm_mlp | 24 | 24 | 24 | 1.0000 [0.8620, 1.0000] (n=24) |
