# Property-based full-module-AST testing with shrinking (Step 114)

Seed `20240602` — **800** structured module ASTs drawn from a compositional algebra (Linear / Conv2d / ReLU / Flatten across the 2D-vector and 4D-image regimes).

## Soundness sweep vs the live torch dispatcher

- regimes generated: `{'img': 382, 'vec': 418}`
- eager-torch oracle: **374** clean, **426** raise
- TensorGuard verdicts: `{'SAFE': 374, 'UNSAFE': 426}`
- decided verdicts: **800** (abstentions: 0)
- soundness violations (SAFE but torch raises): **0**
- false alarms (UNSAFE but torch clean): **0**
- perfect agreement on all decided verdicts: **True** (Wilson 0.9952–1.0)

## Shrinking to a minimal counterexample

A deliberately large buggy module is reduced by a deterministic delta-debugging shrinker under the predicate *eager torch raises* (the cell an always-`SAFE` broken verifier would miss):

- start: **7** layers (dim-sum 96)
- minimal: **1** layer(s) (dim-sum 5); layer reduction factor **7.0×**
- locally minimal (no single further reduction preserves the bug): **True**
- the *real* TensorGuard verifier catches the shrunk witness: **True** (verdict `UNSAFE`)

Minimal counterexample source:

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(1, 1)
    def forward(self, x):
        x = self.l0(x)
        return x
```
