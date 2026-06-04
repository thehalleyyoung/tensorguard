# Symbolic shape benchmark

Total cases: **100 / 100** passed.

| Family | Passed | Total |
| --- | ---: | ---: |
| `annotated_linear_bug` | 20 | 20 |
| `conv2d_bug` | 30 | 30 |
| `conv2d_safe` | 30 | 30 |
| `docstring_linear_bug` | 10 | 10 |
| `linear_abstain` | 10 | 10 |

This benchmark verifies modules without concrete `input_shapes`: Conv2d
front-ends infer symbolic `(batch, channels, height, width)` contracts,
shape-annotated and docstring-documented Linear models use symbolic API
contracts, and ambiguous Linear-first modules prove TensorGuard abstains
instead of guessing rank.
