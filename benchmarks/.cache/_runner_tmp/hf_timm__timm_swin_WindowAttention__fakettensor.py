from __future__ import annotations
import math
from typing import Any, Optional, Tuple, Callable, Dict, List, Union
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.

    Supports both shifted and non-shifted windows.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int]=None, window_size: _int_or_tuple_2_t=7, qkv_bias: bool=True, attn_drop: float=0.0, proj_drop: float=0.0, device=None, dtype=None):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            window_size: The height and width of the window.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** (-0.5)
        self.fused_attn = use_fused_attn(experimental=True)
        self.relative_position_bias_table = nn.Parameter(torch.empty((2 * win_h - 1) * (2 * win_w - 1), num_heads, **dd))
        self.register_buffer('relative_position_index', torch.empty(win_h * win_w, win_h * win_w, device=device, dtype=torch.long), persistent=False)
        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias, **dd)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim, **dd)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters and buffers."""
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Compute and fill non-persistent buffer values."""
        win_h, win_w = self.window_size
        self.relative_position_index.copy_(get_relative_position_index(win_h, win_w, device=self.relative_position_index.device))

    def set_window_size(self, window_size: Tuple[int, int]) -> None:
        """Update window size & interpolate position embeddings
        Args:
            window_size (int): New window size
        """
        window_size = to_2tuple(window_size)
        if window_size == self.window_size:
            return
        self.window_size = window_size
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        with torch.no_grad():
            new_bias_shape = ((2 * win_h - 1) * (2 * win_w - 1), self.num_heads)
            self.relative_position_bias_table = nn.Parameter(resize_rel_pos_bias_table(self.relative_position_bias_table, new_window_size=self.window_size, new_bias_shape=new_bias_shape))
            self.register_buffer('relative_position_index', get_relative_position_index(win_h, win_w, device=self.relative_position_bias_table.device), persistent=False)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features with shape of (num_windows*B, N, C).
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None.

        Returns:
            Output features with shape of (num_windows*B, N, C).
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.fused_attn:
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def init_non_persistent_buffers(self) -> None:
        """Initialize non-persistent buffers."""
        self._init_buffers()
