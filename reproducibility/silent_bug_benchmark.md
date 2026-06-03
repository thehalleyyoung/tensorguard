# Silent-bug benchmark (Step 261)

Curated CPU-only benchmark of **15** runtime-silent PyTorch bugs: **15/15** execute without raising, **15/15** are positive under an independent semantic oracle, and TensorGuard object-level gates catch **15/15**.

Curated CPU-only benchmark over author-constructed silent-failure families; it demonstrates coverage and regression protection, not a field prevalence estimate.

| family | cases | non-raising | oracle-positive | gate-caught |
| --- | ---: | ---: | ---: | ---: |
| gradient_freeze | 3 | 3 | 3 | 3 |
| optimizer_state_drift | 3 | 3 | 3 | 3 |
| quantization_wrong_output | 3 | 3 | 3 | 3 |
| stale_buffer | 3 | 3 | 3 | 3 |
| train_eval_mode_leakage | 3 | 3 | 3 | 3 |

## Cases

| id | family | delta | issue kinds |
| --- | --- | ---: | --- |
| gradient_freeze_mlp | gradient_freeze | 0.0 | gradient_freeze |
| gradient_freeze_conv | gradient_freeze | 0.0 | gradient_freeze |
| gradient_freeze_embedding | gradient_freeze | 0.0 | gradient_freeze |
| stale_buffer_scale | stale_buffer | 0.5 | stale_buffer |
| stale_buffer_bias | stale_buffer | 2.0 | stale_buffer |
| stale_buffer_position | stale_buffer | 10.0 | stale_buffer |
| optimizer_drift_zero_exp_avg | optimizer_state_drift | 0.001759648 | optimizer_state_drift |
| optimizer_drift_scaled_exp_avg_sq | optimizer_state_drift | 0.000739276 | optimizer_state_drift |
| optimizer_drift_stale_step | optimizer_state_drift | 0.002251178 | optimizer_state_drift |
| mode_leak_dropout_train | train_eval_mode_leakage | 1.0 | train_eval_mode_leakage |
| mode_leak_batchnorm_eval | train_eval_mode_leakage | 5.666917 | train_eval_mode_leakage |
| mode_leak_nested_dropout | train_eval_mode_leakage | 0.899001 | train_eval_mode_leakage |
| quant_wrong_scale_coarse | quantization_wrong_output | 0.2 | quantization_wrong_output |
| quant_wrong_zero_point | quantization_wrong_output | 0.2 | quantization_wrong_output |
| quant_wrong_scale_saturation | quantization_wrong_output | 0.21 | quantization_wrong_output |
