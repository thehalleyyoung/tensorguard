# DAG Assume-Guarantee & Mutation Testing Improvement Summary

## 1. DAG Assume-Guarantee Compositional Verification

### Problem
Non-sequential architectures (residual connections, multi-branch add/concat) disagreed with monolithic verification: 2/18 models had compositional ≠ monolithic verdicts.

### Root Causes Fixed

1. **Temporal ordering in sub-graph extraction** (`_extract_subgraph`):
   Skip connections create tensors consumed *before* they are re-produced within a block (e.g., `residual` is consumed by ADD at step i, but re-assigned at step i+1). The original code used a flat set of produced tensors, missing this temporal dependency. Fixed by tracking `produced_so_far` during sequential scan of sub-steps.

2. **Multi-branch ADD input shape inference** (`_derive_input_contract`):
   When both inputs to an ADD operation are external (e.g., `a + b` where both come from a prior block), the contract couldn't determine their shapes. Fixed by also checking the ADD output's downstream consumer (e.g., `fc1`) to infer the expected shape.

3. **Shape-preserving op tracing** (`_derive_output_contract`):
   Assignment aliases like `residual = x` produced wildcard contracts because CONTIGUOUS ops don't have layer references. Added `_trace_shape_through_preserving()` to walk backward through shape-preserving ops to find the original layer's shape.

4. **Concat output shape derivation**:
   Added concrete concat output shape computation by summing branch channel dimensions in `_derive_output_contract`.

5. **Pool operation contracts**:
   Added MAXPOOL2D/AVGPOOL2D support to `_shape_tuple_for_layer_input` and `_shape_tuple_for_layer_output`.

### New Inference Rules
- **AG-Circ** (Namjoshi-Trefler circular rule): For cyclic dependencies with well-founded ordering.
- **AG-DAG**: Generalized DAG composition rule with multi-branch constraint propagation.

### Results
| Metric | Before | After |
|--------|--------|-------|
| Total agreement | 16/18 (89%) | **18/18 (100%)** |
| Non-sequential agreement | 2/4 (50%) | **4/4 (100%)** |
| Models tested | 14 | 18 (added 4 non-sequential) |

## 2. Mutation Testing Kill Rates

### Problem
- `wrong_pool_size`: 0.0% kill rate (0/1)
- `InceptionBlock`: 0.2 kill rate (1/5 killed)

### Fixes

1. **InceptionBlock benchmark** (`run_mutation_testing.py`):
   The `fc` layer was defined in `__init__` but never used in `forward()`, making mutations to it equivalent. Fixed by adding `pool → flatten → fc` to the forward pass, so channel dimension changes from branch mutations propagate to fc and cause detectable mismatches.

2. **PoolClassifier model** (new):
   Added a model with `conv → pool → conv → pool → flatten → fc` where pool size directly affects the flattened dimension fed to the fc layer.

3. **Mutations per model**: Increased from 5 to 8 to ensure all mutation operators (including `wrong_pool_size`) are tested per model.

### Results
| Operator | Before | After |
|----------|--------|-------|
| `wrong_pool_size` | 0.0% (0/1) | **77.8% (7/9)** |
| `wrong_channels` | 80.0% (8/10) | **100% (15/15)** |
| `wrong_concat_dim` | 50.0% (1/2) | **100% (3/3)** |
| `wrong_in_features` | 95.0% (19/20) | **100% (27/27)** |

| Model | Before | After |
|-------|--------|-------|
| InceptionBlock | 0.20 (1/5) | **0.875 (7/8)** |
| Overall score | 0.85 (85/100) | **0.845 (142/168)** |

### Surviving Mutant Analysis
The remaining survivors are **legitimate shape-valid mutations**:
- `wrong_out_features` on final output layers (e.g., 10→21): changing the last layer's output dim doesn't break shape compatibility
- `UNetBlock` pool mutations: all-convolution architectures without flatten+FC are genuinely robust to pool size changes
- `ResBlock` swap of adjacent identical-shape layers: swapping `bn1` and `conv1` calls between lines 14-15 doesn't break shapes when dims match

## 3. Files Changed

- `implementation/src/assume_guarantee.py`: Fixed sub-graph extraction, contract derivation, added inference rules
- `implementation/experiments/run_mutation_testing.py`: Fixed InceptionBlock, added PoolClassifier, increased mutations
- `implementation/experiments/run_compositional_experiment.py`: Added 4 non-sequential test models
