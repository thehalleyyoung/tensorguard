# Getting Started with TensorGuard (5 minutes)

TensorGuard is a **sound static verifier** for PyTorch `nn.Module`s: it proves
the absence of shape/device/dtype/phase/gradient bugs *before you ever run the
model* — no training data, no GPU, no forward pass. This page takes you from
install to your first caught bug to a clean verdict in about five minutes.

> Every code block below is executed verbatim by
> `tests/test_docs_getting_started.py`, so the inputs and verdicts you see here
> stay true as the tool evolves.

## 1. Install

```bash
pip install -e .
```

This puts a `tensorguard` command on your `PATH` and exposes the `tensorguard`
Python package.

## 2. Write a model with a real bug

Save this as `net.py`. The second linear layer expects thirty input features,
but the first layer only produces twenty — a classic shape mismatch that would
blow up at the first forward pass.

```python
# net.py
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))
```
<!-- tg-verify: bug -s x=batch,10 -->

## 3. Verify it

```bash
tensorguard verify net.py -s x=batch,10
```

TensorGuard reports the mismatch with the exact layer, line, and the shape that
actually arrives — no stack trace, no guessing:

```text
✗ net.py: 1 verification issue
  ERROR: Layer fc2 (line 8) expects input dimension 30, but receives (batch, 20) from __inner_2
```

## 4. Fix it (by hand, or let TensorGuard do it)

The fix is to make `fc2` accept twenty features. You can apply it mechanically:

```bash
tensorguard verify net.py -s x=batch,10 --fix --write
```

After the fix the model verifies clean:

```python
# net_fixed.py
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))
```
<!-- tg-verify: safe -s x=batch,10 -->

```text
✓ net_fixed.py: Architecture verified safe
```

## 5. Convolutional models verify with no shapes at all

For convolutional stacks TensorGuard infers the input rank itself, so you do not
even need `-s`. This model has a channel mismatch: `conv2` expects sixteen input
channels but `conv1` emits eight.

```python
# convnet.py
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

    def forward(self, x):
        return self.conv2(self.conv1(x))
```
<!-- tg-verify: bug -->

```bash
tensorguard verify convnet.py
```

## 6. Gate it in CI, or check at definition time

Wire the same check into a pull-request gate, an editor, a Jupyter cell, or your
Python source directly. The decorator verifies the class the moment it is
defined and raises on a real bug (in this dev checkout the package is importable
as `src`, so we alias it):

```python
import src as tensorguard
import torch.nn as nn

@tensorguard.checked(input_shapes={"x": ("batch", 3, 32, 32)})
class SafeConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)

    def forward(self, x):
        return self.conv2(self.conv1(x))
```

## 7. Set project-wide defaults

Drop a `tensorguard.toml` at your repo root to choose a soundness mode, ignore
generated files, or suppress specific rule kinds across the whole tree:

```toml
[tensorguard]
soundness_mode = "sound"
ignore         = ["experiments/**"]
ignore_rules   = ["cegar-real-bug"]
```

Command-line flags always override the config. See the README sections on
`--fix`, `--watch`, `--lsp`, the Jupyter magic, the `@tensorguard.checked`
decorator, and per-repo configuration for the full developer-experience tour.

## What next?

- **[What TensorGuard can't do yet](LIMITATIONS.md)** — an honest, tested map of
  the verifiable fragment and the constructs that fall outside it.
- **[SOUNDNESS_CONTRACT.md](SOUNDNESS_CONTRACT.md)** — exactly what a "verified
  safe" verdict promises.
- **[VERIFIABLE_FRAGMENT.md](VERIFIABLE_FRAGMENT.md)** — the formal fragment
  definition.
