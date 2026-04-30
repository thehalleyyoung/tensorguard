from __future__ import annotations
import math
from typing import Any, Optional, Tuple, Callable, Dict, List, Union
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """

    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, act_layer: Type[nn.Module]=nn.GELU, norm_layer: Optional[Type[nn.Module]]=None, bias: Union[bool, Tuple[bool, bool]]=True, drop: Union[float, Tuple[float, float]]=0.0, use_conv: bool=False, device=None, dtype=None):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features, **dd) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1], **dd)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
