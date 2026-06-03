# Step 266 -- negative controls: where runtime value checks should win

This suite deliberately uses value-dependent failures outside TensorGuard's declared tensor-structure contract.  The inputs are crafted concrete tensors, not random smoke-test draws, because the point is to show the boundary honestly.

## Recall on value-domain controls

| Detector | Caught | Recall | Interpretation |
| --- | ---: | ---: | --- |
| TensorGuard (`sound` mode) | 0 / 6 | 0.000 | expected loss: value semantics are out of contract |
| Runtime smoke test | 2 / 6 | 0.333 | catches assertions, misses silent NaN/Inf outputs |
| Runtime finite-output check | 6 / 6 | 1.000 | explicit value monitor on crafted inputs |

## By family

| Family | Cases | TensorGuard | Smoke test | Finite-output check |
| --- | ---: | ---: | ---: | ---: |
| `nonfinite_value` | 4 | 0 | 0 | 4 |
| `value_assertion` | 2 | 0 | 2 | 2 |

## Per-case signals

| Case | Family | Runtime signal | TG verdict |
| --- | --- | --- | --- |
| `log_negative_nan` | `nonfinite_value` | `nonfinite_output` | `SAFE`, bugs=0 |
| `sqrt_negative_nan` | `nonfinite_value` | `nonfinite_output` | `SAFE`, bugs=0 |
| `divide_by_zero_inf` | `nonfinite_value` | `nonfinite_output` | `SAFE`, bugs=0 |
| `reciprocal_zero_inf` | `nonfinite_value` | `nonfinite_output` | `SAFE`, bugs=0 |
| `assert_positive_min` | `value_assertion` | `exception:AssertionError` | `UNKNOWN`, bugs=0 |
| `assert_probability_range` | `value_assertion` | `exception:AssertionError` | `UNKNOWN`, bugs=0 |
