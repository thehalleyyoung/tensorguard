# TensorGuard

**Static multi-property verification for PyTorch `nn.Module` code.**
Catches shape mismatches, device inconsistencies, train/eval phase bugs, stride violations, and permutation errors — **before runtime**, with zero annotations, by jointly reasoning across all five property domains at once.

**The only tool that catches cross-cutting bugs.** Existing tools (TorchScript, mypy, PyTEA, jaxtyping) each check at most one property in isolation. TensorGuard's 5-theory product domain (`T_shape × T_device × T_phase × T_stride × T_perm`) catches bugs that live at the **intersection** of multiple concerns — such as a buffer with wrong channel count that only surfaces in eval mode, or a device mismatch triggered by a training-only code path. On a 52-model cross-cutting benchmark drawn from real HuggingFace, torchvision, detectron2, YOLOv5, and fairseq bug patterns, **every other tool scores F1 = 0.000; TensorGuard scores F1 = 0.625 with perfect precision (1.0)**.

## Most Impressive Result

```
$ python3 experiments/run_multi_theory_eval.py

  ╔═══════════════════════════════╦═══════════╦══════════╦══════════╗
  ║ Tool                          ║ Precision ║  Recall  ║    F1    ║
  ╠═══════════════════════════════╬═══════════╬══════════╬══════════╣
  ║ TensorGuard (ours)            ║  1.0000   ║  0.4545  ║  0.6250  ║
  ║ TorchScript                   ║  0.0000   ║  0.0000  ║  0.0000  ║
  ║ mypy + torch stubs            ║  0.0000   ║  0.0000  ║  0.0000  ║
  ║ PyTEA (ECOOP'23)              ║  0.0000   ║  0.0000  ║  0.0000  ║
  ║ jaxtyping                     ║  0.0000   ║  0.0000  ║  0.0000  ║
  ╚═══════════════════════════════╩═══════════╩══════════╩══════════╝

  52 benchmarks: 44 buggy (cross-cutting), 8 correct — zero false positives
```

On these cross-cutting bugs, TensorGuard is not marginally better — it is the **only tool in the category**.

## Quick Start

```bash
pip install numpy z3-solver
```

```python
from src.model_checker import verify_model

source = """
import torch, torch.nn as nn

class TransferLearningBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(512, 256)
        self.train_head = nn.Linear(256, 100)
        self.eval_head = nn.Linear(128, 10)   # BUG: expects 128, gets 256

    def forward(self, x):
        h = self.backbone(x)
        if self.training:
            return self.train_head(h)         # OK in training
        else:
            return self.eval_head(h)          # CRASH in eval: 256 != 128
"""

result = verify_model(source, input_shapes={"x": ("batch", 512)})
print(result.safe)   # False
print(result.pretty())
# ✗ Model is UNSAFE
#   [eval mode] shape_incompatible at step 4:
#     eval_head expects in_features=128 but receives tensor with 256 features
```

No other tool catches this — it requires **joint phase + shape reasoning**.

## What TensorGuard Verifies

| Property | What it catches | Example |
|----------|----------------|---------|
| **Shape** | Dimension mismatches across layers | `Linear(128, 64)` fed a `(B, 256)` tensor |
| **Device** | Cross-device operations | GPU model × CPU buffer in `register_buffer` |
| **Phase** | Train/eval behavioral differences | Auxiliary head with wrong dims, only used in training |
| **Stride** | Memory layout violations | `view()` on non-contiguous tensor |
| **Permutation** | Axis ordering errors | `transpose` followed by incompatible `matmul` |

The key insight is that real-world PyTorch bugs are often **cross-cutting**: a buffer with wrong dimensions (shape) that was registered on CPU (device) and is only accessed during eval (phase). TensorGuard is the first tool that reasons about all five properties jointly.

## Key Results

| Benchmark | TensorGuard | TorchScript | mypy | PyTEA | jaxtyping |
|-----------|:-----------:|:-----------:|:----:|:-----:|:---------:|
| Cross-cutting multi-theory (52 models) | **F1=0.625** | 0.000 | 0.000 | 0.000 | 0.000 |
| Phase-dependent bugs (14 models) | **F1=0.783** | 0.000 | 0.000 | 0.000 | 0.000 |
| Real-world architectures (56 models) | **F1=0.889** | — | — | — | — |
| Shape-only bugs (18 models) | **F1=0.900** | 0.714 | 0.000 | — | — |
| Correct models (8 multi-theory) | **Precision=1.0** | — | — | — | — |
| IC3/PDR parametric certificates | **∀ batch sizes** | ✗ | ✗ | ✗ | ✗ |
| Lean 4 mechanized soundness | **71 theorems** | ✗ | ✗ | ✗ | ✗ |

### Real-World Bug Sources

The cross-cutting benchmark includes bugs derived from:
- **HuggingFace Transformers** — position_ids buffer device mismatch (#13666)
- **torchvision** — ResNet bottleneck downsample dimension error, FPN channel mismatch
- **detectron2** — mask head phase-dependent channel mismatch
- **YOLOv5** — anchor grid shape + device + eval-only postprocessing
- **mmdetection** — anchor generator triple-theory bug
- **timm** — squeeze-excite channel scale after transfer learning
- **fairseq** — label smoothing projection dimension error (training-only)
- **wav2vec2** — feature normalization buffer size mismatch
- **PyTorch Forums / StackOverflow** — BatchNorm channel mismatch, attention bias device

## Features

- **331 operator transfer functions** — Linear, Conv2d, BatchNorm, MultiheadAttention, LSTM, GRU, einops, and more
- **5-theory product domain** — Shape × Device × Phase × Stride × Permutation, verified jointly
- **Multi-phase verification** — both train and eval branches of `if self.training:` are checked
- **Zero annotations** — analyzes raw Python `nn.Module` source code via AST
- **SMT-backed** — Z3 for constraint solving; IC3/PDR for unbounded parametric verification
- **Proof certificates** — for safe models, produces Z3-backed safety certificates valid for all input sizes
- **Conservative** — unsupported operators produce UNKNOWN, never silently passed
- **Lean 4 mechanized** — 71 theorems proving theory combination soundness (zero `sorry`)

## Usage

### Python API
```python
from src.model_checker import verify_model

result = verify_model(
    source=open("model.py").read(),
    input_shapes={"x": ("batch", 3, 224, 224)},
)

if result.safe:
    print(result.certificate.pretty())  # safety certificate
else:
    print(result.counterexample.pretty())  # concrete bug trace
```

### Command Line
```bash
tensorguard verify model.py --input-shapes '{"x": ["batch", 3, 224, 224]}'
```

### Parametric Verification (IC3/PDR)
```python
result = verify_model(
    source=model_source,
    input_shapes={"x": ("batch", 3, "H", "W")},
    verification_mode="unbounded",
    symbolic_dims={"batch": "batch_size", "H": "height", "W": "width"},
)
# Proves safety for ALL batch sizes, ALL image heights, ALL image widths
```

## How It Works

```
Source code (nn.Module)
    │
    ▼
┌──────────────────────────────────────────────┐
│  1. AST Parse — extract layers & data flow   │
│     nn.Linear, Conv2d, register_buffer, ...  │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  2. Multi-phase graph extraction             │
│     Both if self.training branches analyzed  │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  3. Forward symbolic constraint propagation  │
│     T_shape × T_device × T_phase ×          │
│     T_stride × T_perm — joint reasoning     │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  4. Z3 verification at every operation site  │
│     UserPropagator plugins for each theory   │
└──────────────┬───────────────────────────────┘
               │
          ┌────┴────┐
          │         │
          ▼         ▼
      ✓ SAFE    ✗ UNSAFE
      │             │
      ▼             ▼
  Certificate   Counterexample
  (∀ inputs)    (concrete dims)
```

## Project Structure

```
src/
├── model_checker.py          # Core verifier (8300+ lines, 331 operators)
├── ic3_pdr.py                # IC3/PDR unbounded parametric verification
├── assume_guarantee.py       # Compositional verification with AG rules
├── smt/
│   ├── solver.py             # Z3 solver orchestration
│   ├── device_theory.py      # T_device — device placement constraints
│   ├── phase_theory.py       # T_phase — train/eval mode reasoning
│   ├── stride_theory.py      # T_stride — memory layout constraints
│   ├── permutation_theory.py # T_perm — axis ordering constraints
│   ├── broadcast_theory.py   # Broadcasting rule verification
│   └── theory_combination.py # Tinelli-Zarba theory combination
├── typing_rules.py           # 331 operator transfer functions
└── cli/                      # Command-line interface
experiments/
├── multi_theory_benchmarks.py       # 52-model cross-cutting benchmark
├── run_multi_theory_eval.py         # Cross-cutting evaluation runner
├── benchmarks/
│   └── realworld_pytorch_benchmark.py  # 56 real-world architecture benchmarks
├── eval_baselines.py                # Baseline comparison framework
└── run_realworld_pytorch_eval.py    # Real-world evaluation runner
tests/                               # 6216 tests, all passing
lean/                                # 71 Lean 4 mechanized theorems
docs/paper/tool_paper.tex            # Conference paper
```

## Honest Limitations

1. **Recall on device-only bugs is low** — the current analysis catches device issues when they co-occur with shape mismatches, but pure cross-device operations without shape errors are often missed.
2. **Dynamic control flow** — `for` loops with data-dependent iteration counts, recursive modules, and `torch.where` with shape-dependent branches are not yet supported.
3. **No real model execution** — all results are from static analysis of source code; conformance with PyTorch runtime semantics is validated by property-based testing but not formally verified.
4. **Reshape overapproximation** — complex `view()` operations with 4+ symbolic dimensions can cause false positives (1 observed across 230 benchmarks).
5. **Lean conformance gap** — the 71 mechanized theorems prove the *theory combination* is sound, but the Python implementation's faithfulness to the Lean specification is not machine-checked.

## Documentation

- [API Reference](API.md)
- [Trusted Computing Base](docs/TCB.md)
- [Typing Rules](src/typing_rules.py)
- [Multi-Theory Benchmark](experiments/multi_theory_benchmarks.py)
- [Baseline Comparisons](experiments/eval_baselines.py)

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -x -q   # 6216 tests
```

## Reproducing Results

```bash
# Cross-cutting multi-theory evaluation (the SOTA result)
python3 experiments/run_multi_theory_eval.py

# Real-world architecture evaluation
python3 experiments/run_realworld_pytorch_eval.py

# Baseline comparison (TorchScript, mypy, PyTEA)
python3 experiments/run_real_baseline_comparison.py

# IC3/PDR parametric certificate verification
python3 src/ic3_pdr.py
```

## License

MIT
