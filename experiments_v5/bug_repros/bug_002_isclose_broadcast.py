"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/174985
Expected Error: doesn't match the broadcast shape
"""

import torch
import torch.nn as nn

INPUT_SHAPES = {"scalar": (), "tensor": (13, 2, 12)}


class BuggyModule(nn.Module):
    def forward(self, scalar, tensor):
        return torch.isclose(scalar, tensor, rtol=1e-5, atol=1e-8, equal_nan=True)


if __name__ == '__main__':
    try:
        m = BuggyModule()
        scalar = torch.tensor(float('nan'), dtype=torch.float16)
        tensor = torch.zeros(13, 2, 12, dtype=torch.float16)
        m(scalar, tensor)
    except RuntimeError as e:
        print(f"Error: {e}")
