from __future__ import annotations
import math
from typing import Any, Optional, Tuple, Callable, Dict, List, Union
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims (for 2D NCHW tensors)
    """

    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, act_layer: Type[nn.Module]=nn.ReLU, norm_layer: Optional[Type[nn.Module]]=None, bias: Union[bool, Tuple[bool, bool]]=True, drop: float=0.0, device=None, dtype=None):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0], **dd)
        self.norm = norm_layer(hidden_features, **dd) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1], **dd)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
