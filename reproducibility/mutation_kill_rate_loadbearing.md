# Targeted mutation kill rate: 4 load-bearing handlers

## Command

```bash
python3 reproducibility/mutation_kill_rate_loadbearing.py
```

## Handler ranges (model_checker.py)

| Handler | Lines |
|---|---|
| view_reshape_total_size | 8530–8640 |
| broadcasting            | 8444–8530 |
| conv_channel_mismatch   | 4874–4983 |
| einsum_dim              | 8222–8270 |

## Results

| Handler | Killed | Total | Kill rate |
|---|---|---|---|
| view_reshape_total_size | 4 | 10 | 40% |
| broadcasting | 3 | 9 | 33% |
| conv_channel_mismatch | 0 | 10 | 0% |
| einsum_dim | 0 | 10 | 0% |
| **Union** | **7** | **39** | **18%** |

Baseline: 53/60 RP on the 60-bug corpus (clean run).

## Interpretation

Targeted mutations of the four handlers most load-bearing on the headline
53/60 figure show a 18% (7/39) kill rate, substantially
higher than the full-file union kill rate of 14% (7/50). The mutations that
survive sit on guard branches not exercised by the 60-bug corpus
(config-attribute and unresolvable-symbolic-dim paths).

## Paper claim (T3)

Round-2 Q4 requested the kill rate restricted to the four load-bearing
handlers. This artefact answers that question with a targeted 7/39
(18%) kill rate vs the 7/50 (14%) union figure.
