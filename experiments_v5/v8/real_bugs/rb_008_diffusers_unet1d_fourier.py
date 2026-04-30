"""
Real Bug Repro: Diffusers UNet1DModel GaussianFourier time embedding mismatch

GitHub Issue: https://github.com/huggingface/diffusers/issues/12110
Fixed in PR:  https://github.com/huggingface/diffusers/pull/12111 (commit 751e250f)
Repository:   huggingface/diffusers
Model:        UNet1DModel

Bug: When `use_timestep_embedding=True` and `time_embedding_type="fourier"`, the
GaussianFourierProjection was initialized with a hardcoded `embedding_size=8`,
producing output of size `2 * 8 = 16`. However, `timestep_input_dim` was computed
as `2 * block_out_channels[0]` (e.g., `2 * 32 = 64` with default channels).
The TimestepEmbedding's first linear layer was then `nn.Linear(64, 128)`, but
received a 16-dimensional input from GaussianFourierProjection.

Original error:
  RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x16 and 64x128)
  Input: 16 features (from GaussianFourierProjection(embedding_size=8))
  Expected: 64 features (from timestep_input_dim = 2 * block_out_channels[0] = 64)

The fix changed `embedding_size=8` to `embedding_size=time_embed_dim // 2` so that
the projection output matches `timestep_input_dim`.

Substitution note: default block_out_channels=(32, 32, 64) giving timestep_input_dim=64;
hardcoded embedding_size=8 gives 16 output features.
"""
import torch
import torch.nn as nn

# GaussianFourierProjection with hardcoded embedding_size=8 outputs 16 features
INPUT_SHAPES = {"x": (1, 16)}


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # TimestepEmbedding.linear_1: built with in_channels=timestep_input_dim=64
        # but GaussianFourierProjection only outputs 16 features
        self.linear_1 = nn.Linear(64, 128)

    def forward(self, x):
        # Bug: x has 16 features but linear_1 expects 64
        # mat1 shape (1, 16) cannot multiply with weight shape (64, 128)
        return self.linear_1(x)
