# Lean proof examples for operator contributors

Most good-first operator work should start with Python conformance tests.
When a rule graduates to proof-backed status, use the existing Lean files
below as patterns. Each row is derived from `proof_footprint_manifest.json`
and checked against the committed Lean tree.

| Operator | Lean file | Theorem | Transfer role |
| --- | --- | --- | --- |
| `F.relu` | `lean/TensorGuard/Extended.lean` | `TensorGuard.relu_shape_preserving` | relu(input) has exactly the input shape |
| `torch.bmm` | `lean/TensorGuard/MatmulSound.lean` | `TensorGuard.MatmulSound.matmul_contraction_sound` | batched matrix multiplication contracts matching inner dimensions per batch |
| `torch.broadcast_shapes` | `lean/TensorGuard/BroadcastChain.lean` | `TensorGuard.BroadcastChain.bcDim_none_iff` | broadcast each aligned dimension by equality-or-one, folding across operands |
| `torch.broadcast_tensors` | `lean/TensorGuard/BroadcastChain.lean` | `TensorGuard.BroadcastChain.bcDim_none_iff` | broadcast each tensor shape by the same equality-or-one dimension rule |
| `torch.clamp` | `lean/TensorGuard/SoundnessV5.lean` | `TensorGuard.V5.applyOp_sound_clamp` | clamp(input, ...) has exactly the input shape |
| `torch.gather` | `lean/TensorGuard/SoundnessV5.lean` | `TensorGuard.V5.applyOp_sound_gather` | gather(input, index, dim) returns the index shape |
| `torch.index_select` | `lean/TensorGuard/SoundnessV5.lean` | `TensorGuard.V5.applyOp_sound_index_select` | index_select replaces the selected axis by the static index length |
| `torch.matmul` | `lean/TensorGuard/MatmulSound.lean` | `TensorGuard.MatmulSound.applyOpExt_sound_matmul` | (..., m, k) x (..., k, n) contracts to (..., m, n) |

Contributor rule of thumb: prove the smallest local transfer lemma first,
then connect it to the Python registry through `proof_footprint_manifest.json`
only after the torch oracle/conformance tests pass.
