# TensorGuard

**Static tensor shape verification for PyTorch.** Catches dimension mismatches, broadcast bugs, and device errors in `nn.Module` subclasses at analysis time—zero annotations, zero runtime cost—using Z3 SMT solving with a formally verified 5-theory product domain.

## 30-Second Quickstart

```bash
pip install -e .   # requires Python ≥ 3.9, z3-solver
```

```python
from src.model_checker import verify_model

result = verify_model('''
import torch.nn as nn
class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(128, 10)  # BUG: 256 != 128
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
''', input_shapes={"x": ("batch", 768)})

print(result.counterexample.pretty())
```

Output:
```
CounterexampleTrace(BuggyMLP)
  Failing step: 1
  Concrete dims: batch=1
  Computation path (3 steps):
    → [0] x: TensorShape(dims=(batch, 768))
    ✗ [1] x: TensorShape(dims=(batch, 256))
  VIOLATION [1]: Linear expects last dim=128, got 256
```

## Baseline Comparison (real results)

| Tool | Precision | Recall | F1 | Avg Time | FP |
|------|-----------|--------|-----|----------|----|
| **TensorGuard** | **0.900** | **0.900** | **0.900** | 168 ms | 1 |
| TorchScript | 0.556 | 1.000 | 0.714 | 5 ms | 8 |
| mypy | 0.000 | 0.000 | 0.000 | 26,843 ms | 0 |

On 18 benchmarks (10 buggy, 8 correct). TorchScript fails on all correct models via `torch.jit.script`. mypy finds zero shape bugs. TensorGuard catches 9/10 shape bugs with 1 false positive on complex view/reshape patterns. See `experiments/run_real_baseline_comparison.py`.

## Key Capabilities

- **117 operator transfer functions** — Linear, Conv1d–3d, BatchNorm, LSTM, GRU, MultiheadAttention, Transformer layers, pooling, padding, loss functions, and more
- **5-theory SMT domain** — Shape × Device × Phase × Stride × Permutation with Tinelli-Zarba combination; Lean 4 mechanization (1,587 lines, 71 theorems, 0 sorry)
- **IC3/PDR unbounded verification** — Proves safety for *all* values of symbolic dims, not just sampled ones
- **Decidable fragment** — 94.9% of operators produce QF_LIA constraints (polynomial time); only 6 operators (reshape, flatten, PixelShuffle, repeat) enter NP-hard territory
- **Proof certificates** — Every safe verdict backed by machine-checkable Z3 inference chain

## Usage

### Verify a model

```python
from src.model_checker import verify_model
result = verify_model(source, input_shapes={"x": ("batch", 3, 224, 224)})
print("safe" if result.safe else result.counterexample.pretty())
```

### Unbounded verification (IC3/PDR)

```python
from src.ic3_pdr import ic3_verify
result = ic3_verify(source, symbolic_dims={"batch": "batch_size"})
assert result.safe  # safe for ALL batch sizes
```

### CLI

```bash
python -m src.cli.main verify model.py -s x=batch,768
python -m src.cli.main ci-check model.py -s x=batch,3,224,224  # exit 0=safe, 1=bug
```

## Architecture

```
src/
  model_checker.py          Core verify_model() — AST → graph → Z3
  ic3_pdr.py                IC3/PDR unbounded verification
  decidability.py           Decidability characterization
  smt/                      5 UserPropagator theories
  tensor_shapes.py          Shape algebra + operator registry
lean/
  TheoryCombination.lean    Lean 4 mechanization (0 sorry)
experiments/
  run_real_baseline_comparison.py    TensorGuard vs TorchScript vs mypy
  run_decidability_characterization.py   Operator classification
```

## Limitations

- **Operator coverage**: 117 of ~2000 PyTorch operators modeled. Unsupported operators treated as identity (conservative).
- **Reshape/view**: Fully symbolic reshapes produce QF_NIA constraints (NP-hard). May yield UNKNOWN on complex patterns.
- **Complex control flow**: View/reshape with dynamic shapes can cause false positives (1 FP observed on self-attention with 4D view).
- **No value-level bugs**: Does not detect NaN, gradient explosion, or numerical instability.
- **IC3/PDR completeness**: Sound but incomplete for the nonlinear (reshape) fragment; complete for the 94.9% QF_LIA fragment.

## License

MIT
