# TensorGuard Code Scanning Trend Dashboard

| releases | current open | resolved events | recurrence events | net open delta |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | 1 | 1 | 1 |

## Per-release deltas

| release | open | opened | closed | carried | recurred | net delta | top rules |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v0.1.0 | 1 | 1 | 0 | 0 | 0 | 1 | shape-incompatible:1 |
| v0.1.1 | 0 | 0 | 1 | 0 | 0 | -1 | - |
| v0.1.2 | 2 | 2 | 0 | 0 | 1 | 2 | shape-incompatible:2 |

## Currently open alerts

| count | rule | location | message |
| ---: | --- | --- | --- |
| 1 | shape-incompatible | models/conv.py:8 | Layer conv2 (line 8) expects input dimension 9, but receives (batch, 8, 6, 6) from __inner_2 |
| 1 | shape-incompatible | models/linear.py:8 | Layer fc2 (line 8) expects input dimension 30, but receives (batch, 20) from __inner_2 |
