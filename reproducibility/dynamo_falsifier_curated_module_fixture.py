
import torch
import torch.nn as nn

@torch.library.custom_op('repro::shape_branch', mutates_args=())
def shape_branch(x: torch.Tensor) -> torch.Tensor:
    # Reads x.shape[-1] -- a shape bit the surrounding nn.Module
    # never advertises through self.* refinement variables, so it
    # is outside catalogue(M) by construction.
    last = x.shape[-1]
    if last % 2 == 0:
        return x * 1.0
    else:
        return x + 1.0

@shape_branch.register_fake
def _fake_shape_branch(x):
    return torch.empty_like(x)


class ShapeGuardModule(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # The catalogue for this module advertises:
        #   x.shape[0] (batch), self.linear.weight.shape[0,1].
        # It does NOT advertise x.shape[-1] before the linear; the
        # custom op shape_branch reads x.shape[-1] downstream, which
        # would be a SHAPE guard outside catalogue(M) under Dynamo.
        h = self.linear(x)
        return torch.ops.repro.shape_branch(h)
