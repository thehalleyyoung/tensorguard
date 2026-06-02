# Scaling study: analysis work vs model size

Over a sweep of feed-forward depths (1..64 stacked `nn.Linear` layers, width 32) we record the verifier's deterministic structural-work metric `lines_analyzed`, the verdict, and whether the model was decided.

| depth | params | lines_analyzed | cegar | verdict | decided |
| --- | --- | --- | --- | --- | --- |
| 1 | 1056 | 8 | 1 | SAFE | True |
| 2 | 2112 | 10 | 1 | SAFE | True |
| 4 | 4224 | 14 | 1 | SAFE | True |
| 8 | 8448 | 22 | 1 | SAFE | True |
| 16 | 16896 | 38 | 1 | SAFE | True |
| 24 | 25344 | 54 | 1 | SAFE | True |
| 32 | 33792 | 70 | 1 | SAFE | True |
| 48 | 50688 | 102 | 1 | SAFE | True |
| 64 | 67584 | 134 | 1 | SAFE | True |

Ordinary least-squares fit of analysis work versus depth: slope `2.0`, intercept `6.0`, R^2 `1.0`.

- analysis work is linear in model size: **True**
- every size decided (no abstention/blow-up at scale): **True**
- CEGAR iterations bounded at scale: **True**

Wall-clock scaling (machine-dependent) is reported separately in `scaling_walltime.json` with a log-log regression exponent below three, confirming polynomial (sub-cubic) rather than exponential growth.
