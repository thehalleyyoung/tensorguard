---
name: False positive (false UNSAFE)
about: TensorGuard reported a bug for a model that is actually correct
title: "[FALSE-POSITIVE] "
labels: ["false-positive"]
---

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

- command / API call:
- `input_shapes`:
- soundness mode (strict / balanced / lenient):

### What TensorGuard reported

(the bug message + location)

### Why it is actually correct

(show the shapes line up / the model runs cleanly under eager PyTorch)

### Environment

- TensorGuard version:
- Python version:
- PyTorch version:
- OS:
