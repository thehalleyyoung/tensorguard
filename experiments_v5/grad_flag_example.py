"""Grad-flag refinement: a worked example.

Demonstrates the bug class TensorGuard catches with the
\varphi_grad component of its refinement-type system: a parameter
appears live in ``forward``, but the only path through it is wrapped
in ``with torch.no_grad():``.  After ``loss.backward()``, that
parameter's ``.grad`` is still ``None``, so ``optimizer.step()``
silently leaves it untrained.

Run with python3.11 to reproduce::

    $ python3.11 experiments_v5/grad_flag_example.py
    head.weight has grad?  True
    trunk.weight has grad? False  <-- bug: never updated by SGD
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(8, 8, bias=False)
        self.head = nn.Linear(8, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BUG: trunk is only ever evaluated under no_grad, so its
        # parameters are never reached by an autograd-tracked path.
        with torch.no_grad():
            h = self.trunk(x)
        # h.requires_grad is False here; head sits on top of a
        # detached activation, so only head.weight gets a gradient.
        return self.head(h)


def main() -> int:
    torch.manual_seed(0)
    model = MyModel()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    x = torch.randn(16, 8)
    y = torch.randn(16, 4)

    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()

    head_has = model.head.weight.grad is not None
    trunk_has = model.trunk.weight.grad is not None
    print(f"head.weight has grad?  {head_has}")
    print(f"trunk.weight has grad? {trunk_has}")

    # The bug: trunk parameters are silently frozen.
    assert head_has, "head.weight.grad should be populated"
    assert not trunk_has, (
        "expected trunk.weight.grad to be None due to no_grad context"
    )

    # Even after opt.step(), trunk.weight is unchanged.
    before = model.trunk.weight.detach().clone()
    opt.step()
    after = model.trunk.weight.detach().clone()
    assert torch.equal(before, after), "trunk should not be updated"
    print("confirmed: trunk parameters are never updated by SGD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
