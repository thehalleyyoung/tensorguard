# Frontend reconciliation: torch.fx vs torch.export

The fx and export frontends capture each model by entirely different routes, yet a sound verifier must reach the same verdict through either. Generated against torch `2.9.1`.

Across 7 models, 5 are captured by both frontends with **0 divergences** (agreement rate 1.000). Every captured verdict also matches ground truth: fx correct on 7, export correct on 7.

| Model | Expected | fx | export | Both captured | Divergent |
|-------|----------|----|--------|---------------|-----------|
| `mlp` | safe | safe | safe | yes | no |
| `deep_mlp` | safe | safe | safe | yes | no |
| `residual_mlp` | safe | safe | safe | yes | no |
| `cnn` | safe | safe | safe | yes | no |
| `conv_bn_stack` | safe | safe | safe | yes | no |
| `mlp_bad_in_features` | UNSAFE | UNSAFE | no-capture | no | no |
| `cnn_bad_flatten` | UNSAFE | UNSAFE | no-capture | no | no |
