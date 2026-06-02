---
name: Unsound result (false SAFE)
about: TensorGuard reported SAFE but the model has a real shape/device/dtype/phase/gradient bug
title: "[UNSOUND] "
labels: ["soundness", "bug"]
---

**This is the highest-severity class of bug.** TensorGuard's core promise is that
SAFE is trustworthy.

### Minimal reproducer

```python
import torch.nn as nn

class Repro(nn.Module):
    def __init__(self):
        super().__init__()
        # ...

    def forward(self, x):
        # ...
        return x
```

### How you invoked TensorGuard

```bash
# e.g. tensorguard verify repro.py
# or the input_shapes you passed to verify_architecture
```

- `input_shapes`:
- soundness mode (strict / balanced / lenient):

### What TensorGuard reported

(verdict + any output)

### What actually happens at runtime

(the real error from eager PyTorch, with the traceback)

### Environment

- TensorGuard version (`tensorguard --version`):
- Python version:
- PyTorch version:
- OS:
