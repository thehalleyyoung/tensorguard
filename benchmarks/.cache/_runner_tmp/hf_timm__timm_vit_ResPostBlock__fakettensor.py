from __future__ import annotations
import math
from typing import Any, Optional, Tuple, Callable, Dict, List, Union
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResPostBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=False, qk_norm: bool=False, scale_attn_norm: bool=False, scale_mlp_norm: bool=False, proj_bias: bool=True, proj_drop: float=0.0, attn_drop: float=0.0, init_values: Optional[float]=None, drop_path: float=0.0, act_layer: Type[nn.Module]=nn.GELU, norm_layer: Type[nn.Module]=LayerNorm, mlp_layer: Type[nn.Module]=Mlp, attn_layer: LayerType=Attention, depth: int=0, device=None, dtype=None) -> None:
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.init_values = init_values
        self.attn = _create_attn(attn_layer, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, scale_norm=scale_attn_norm, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer, depth=depth, **dd)
        self.norm1 = norm_layer(dim, **dd)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, norm_layer=norm_layer if scale_mlp_norm else None, bias=proj_bias, drop=proj_drop, **dd)
        self.norm2 = norm_layer(dim, **dd)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.init_weights()

    def init_weights(self) -> None:
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, is_causal: bool=False) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x, attn_mask=attn_mask, is_causal=is_causal)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x
