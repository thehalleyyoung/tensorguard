# TensorGuard — Static Shape/Device/Phase Verifier for PyTorch Models

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Z3 SMT Solver](https://img.shields.io/badge/Z3-SMT%20solver-orange?logo=microsoft)
![PyTorch](https://img.shields.io/badge/PyTorch-compatible-EE4C2C?logo=pytorch&logoColor=white)

**TensorGuard** statically verifies PyTorch `nn.Module` architectures for
shape mismatches, device inconsistencies, and train/eval phase errors —
**with zero annotations**. It encodes tensor shapes as refinement type
predicates (`{v: Tensor | shape(v) == (batch, C, H, W)}`) and uses Z3 to
prove compatibility at every operation site, catching the #1 class of
runtime errors in ML codebases before any code runs.

---

## Key Features

- **142 operator transfer functions** — covers `matmul`, `conv2d`, `cat`,
  `view`, `reshape`, `transpose`, `permute`, `einsum`, `bmm`, attention
  patterns, and more
- **5-theory product domain** — jointly reasons over
  **Shape × Device × Phase × Stride × Permutation** for each tensor
- **Zero annotations required** — shapes are inferred from constructors,
  `torch.randn`, `nn.Linear(in, out)`, reshapes, and data flow
- **Multi-phase train/eval analysis** — detects `BatchNorm` / `Dropout`
  misuse across phases
- **Device tracking** — catches silent CPU ↔ CUDA mismatches before they
  become runtime errors
- **CEGAR loop** — counterexample-guided abstraction refinement discovers
  shape predicates automatically; no manual specification needed
- **Z3-backed** — all shape constraints are discharged by the Z3 SMT solver
  for soundness (0% false positives in `--high-confidence` mode)
- **SARIF 2.1.0 output** — integrates with GitHub Code Scanning / Advanced Security
- **Sub-second analysis** — typical models verified in < 1 second

---

## Installation

```bash
git clone https://github.com/thehalleyyoung/tensorguard.git
cd tensorguard
pip install -e .
```

The only required dependency is `z3-solver>=4.12` (installed automatically).

---

## Quickstart

Write a model with a shape bug:

```python
# model.py
import torch
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.fc = nn.Linear(16 * 5 * 5, 10)   # wrong: should be 16 * 222 * 222

    def forward(self, x):          # x: (batch, 3, 224, 224)
        x = self.conv(x)          # → (batch, 16, 222, 222)
        x = x.view(x.size(0), -1) # → (batch, 16*222*222)
        return self.fc(x)          # ERROR: expects 16*5*5 = 400, got 788544
```

Run TensorGuard:

```
$ tensorguard verify model.py -s x=batch,3,224,224

  ✗ model.py: 1 shape error found (243ms)

  [ERROR] :15:15  Shape mismatch at nn.Linear: input has 788544 features,
                  expected 400                                  (shape-error)
```

Verify a correct model:

```
$ tensorguard verify model.py -s x=batch,3,224,224
  ✓ model.py: Architecture verified safe (768ms)
```

---

## CLI Reference

```
tensorguard verify FILE [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `FILE` | Python file containing `nn.Module` class(es) | — |
| `-s`, `--input-shape` | `name=dim1,dim2,...` (repeatable) | auto-inferred |
| `--no-device-check` | Skip device consistency checks | off |
| `--no-phase-check` | Skip train/eval phase checks | off |
| `--cegar-iterations` | Max CEGAR refinement iterations | `10` |
| `-f`, `--format` | `text`, `json`, `sarif` | `text` |
| `--high-confidence` | Only report Z3-proven bugs (0% FP) | off |
| `--timeout` | Per-file timeout (seconds) | `300.0` |

### Additional Commands

| Command | Description |
|---------|-------------|
| `tensorguard ci-check FILE --sarif-output out.sarif` | CI mode with SARIF output |
| `tensorguard watch DIR` | Watch and re-verify on changes |
| `tensorguard version` | Show version info |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Verification succeeded — no shape/device/phase errors |
| `1` | Errors found |
| `2` | Analysis error (invalid input, timeout) |

---

## What TensorGuard Catches

| Error Class | Example |
|-------------|---------|
| **Shape mismatch** | `nn.Linear` input dimension ≠ declared `in_features` |
| **Matmul incompatibility** | `torch.matmul(A, B)` where inner dimensions differ |
| **Bad reshape/view** | `x.view(batch, -1)` where total elements don't divide evenly |
| **Conv output → Linear input** | Feature map flattened size ≠ `nn.Linear` input |
| **Cat dimension mismatch** | `torch.cat([a, b], dim=1)` with different sizes on other dims |
| **Device mismatch** | `cpu_tensor + cuda_tensor` |
| **Phase error** | `model.eval()` then calling layers that behave differently in eval |
| **Stride/permutation error** | Contiguity assumptions violated after `transpose`/`permute` |

---

## How It Works

1. **AST Parse** — extract `nn.Module` class, `__init__`, and `forward` method
2. **Shape Predicate Harvesting** — infer shapes from `nn.Linear(in, out)`,
   `nn.Conv2d(...)`, `torch.randn(...)`, input shape flags, and reshape calls
3. **5-Theory Product Domain Propagation** — propagate
   **(Shape × Device × Phase × Stride × Permutation)** through every
   operation using 142 transfer functions
4. **Z3 Constraint Solving** — at each operation site, generate and discharge
   shape compatibility constraints via Z3
5. **CEGAR Refinement** — if the initial abstraction is too coarse, discover
   new predicates from counterexamples and re-check
6. **Report** — emit shape/device/phase errors with concrete dimension values
   and fix suggestions

---

## Python API

```python
from src.api import verify_module

# Verify a file
result = verify_module("model.py", input_shapes={"x": ("batch", 3, 224, 224)})
print(f"Status: {result.status}")       # "SAFE" or "UNSAFE"
print(f"Bugs: {len(result.bugs)}")
print(f"Duration: {result.duration_ms:.0f}ms")

for bug in result.bugs:
    print(f"  {bug.location.line}: {bug.message}")
```

---

## CI / CD Integration

```yaml
# .github/workflows/tensorguard.yml
name: TensorGuard Shape Check
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .
      - run: tensorguard ci-check models/ --sarif-output results.sarif
      - uses: github/codeql-action/upload-sarif@v3
        if: always()
        with: { sarif_file: results.sarif }
```

---

## Configuration

TensorGuard reads configuration from `.tensorguard.toml` or `[tool.tensorguard]`
in `pyproject.toml`:

```toml
[tensorguard]
include = ["models/**/*.py"]
exclude = ["tests/**"]
timeout = 300.0

[tensorguard.cegar]
max_iterations = 10

[tensorguard.checks]
device = true
phase = true
stride = true
```

---

## FAQ

**Q: Z3 install fails.**
A: Ensure Python 3.9+ and pip ≥ 21.0. On Apple Silicon:
`pip install --no-cache-dir z3-solver>=4.12`.

**Q: False positive on complex `view()`/`reshape()`.**
A: TensorGuard is conservative with dynamic reshapes. Use `--high-confidence`
to suppress heuristic findings, or add `# tensorguard: ignore` on the line.

**Q: How fast is it?**
A: Typical single-model verification completes in < 1 second. Large
codebases benefit from `--timeout` and parallel `--workers`.

---

## License

MIT — see [LICENSE](LICENSE) for details.
