# TensorGuard

Static tensor shape verification for PyTorch. Catches shape mismatches, broadcast bugs, device inconsistencies, and dimension errors **before runtime** — with zero annotations.

## Quick Start

```bash
pip install tensorguard
```

```python
# verify_my_model.py
from src.model_checker import verify_model

source = """
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)  # BUG: expects flatten first
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.fc(x)  # Shape error: (B,64,H,W) vs (B,64)
        return x
"""

result = verify_model(source=source, input_shapes={"x": ("batch", 3, 224, 224)})
print(result.safe)  # False
for error in result.errors:
    print(f"  Line {error.line}: {error.message}")
```

## Features

- **331 operator coverage** — Linear, Conv2d, BatchNorm, attention, einops, and more
- **5-theory product domain** — Shape × Device × Phase × Stride × Permutation
- **Zero annotations** — analyzes raw Python source code
- **SMT-backed** — Z3 for constraint solving, IC3/PDR for unbounded verification
- **Conservative** — unsupported operators produce UNKNOWN (not identity)
- **Lean 4 mechanized** — 71 theorems proving theory combination soundness

## Supported Architectures

Evaluated on: ResNet-50, BERT-base, GPT-2, ViT-B/16, MobileNetV2, EfficientNet-B0, DenseNet-121

## Usage

### Command Line
```bash
tensorguard verify model.py --input-shapes '{"x": ["batch", 3, 224, 224]}'
```

### Python API
```python
from src.model_checker import verify_model
result = verify_model(source=open("model.py").read(), input_shapes={"x": ("batch", 3, 224, 224)})
```

### CI Integration
See `.github/workflows/tensorguard-ci.yml` for GitHub Actions integration.

## How It Works

1. **Parse** — AST analysis extracts layer definitions and data flow
2. **Harvest** — Shape predicates from constructors, assertions, reshape calls
3. **Propagate** — Forward symbolic constraint propagation through computation DAG
4. **Verify** — Z3 checks shape compatibility at every operation site
5. **Report** — Concrete counterexample traces for any shape errors found

## Documentation

- [API Reference](API.md)
- [Trusted Computing Base](docs/TCB.md)
- [Typing Rules](src/typing_rules.py)
- [Baseline Comparisons](experiments/eval_baselines.py)

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -x -q
```

## License

MIT
