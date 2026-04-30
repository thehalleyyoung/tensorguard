# Cross-family expansion: Gemma 2 (round-5)

TG verdict tally on 2 naturally-occurring upstream Gemma 2 shape-bug repros: RP=2.

| module | input_shapes | verdict | max_conf | first bug |
|---|---|---|---:|---|
| `Gemma2HeadDimDivisibility` | `{'hidden_states': [1, 7, 2304]}` | RP | 0.99 | [SHAPE-INCOMPATIBLE] Reshape incompatible: cannot reshape TensorShape(dims=(1, 7, 2304)) to (1, 7, 10, 256) |
| `Gemma2GQAGroupedKVRepeat` | `{'hidden_states': [1, 16, 2304]}` | RP | 0.99 | [SHAPE-INCOMPATIBLE] Reshape incompatible: cannot reshape TensorShape(dims=(1, 16, 2304)) to (1, 16, 8, 289) |

Reproduce: `python3 reproducibility/upstream_gemma2_round5.py`.
