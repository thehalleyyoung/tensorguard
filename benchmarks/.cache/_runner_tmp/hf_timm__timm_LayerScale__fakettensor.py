from __future__ import annotations
import math
from typing import Any, Optional, Tuple, Callable, Dict, List, Union
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerScale(nn.Module):
    """ LayerScale on tensors with channels in last-dim.
    """

    def __init__(self, dim: int, init_values: float=1e-05, inplace: bool=False, device=None, dtype=None) -> None:
        super().__init__()
        self.init_values = init_values
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
