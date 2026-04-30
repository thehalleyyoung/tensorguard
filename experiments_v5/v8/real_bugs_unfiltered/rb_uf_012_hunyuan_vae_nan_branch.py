"""
Unfiltered post-freeze repro #12 (out-of-fragment, data-dependent control flow):
HF diffusers - VAE NaN-recovery branch shape rewrite.

GitHub PR  : https://github.com/huggingface/diffusers/pull/13561  (merged 2026-04-22)
Repository : huggingface/diffusers
Buggy file : src/diffusers/models/autoencoders/autoencoder_kl_hunyuan.py
            (pre-#13561, encode() NaN-detection branch)

Root cause: when the latent tensor contained NaN, the recovery
branch reshapes to a tile-based layout with a hard-coded tile
size that did not match the runtime spatial dimensions. The
buggy reshape only fires under a runtime data-dependent check
(`if torch.isnan(z).any(): ...`) which TG cannot statically
take.

Out-of-fragment (data-dependent control flow): TG conservatively
takes both branches; under the no-NaN branch the reshape is
correct, so TG silently verifies. Expected verdict: silent
verified (TG never enters the buggy branch).
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, channels=4, tile_size=64):
        super().__init__()
        self.channels = channels
        self.tile_size = tile_size
        self.proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, z):
        # z: (B, channels, H, W).  No-NaN branch is the no-op
        # path; TG explores this branch and proves shape preservation.
        z = self.proj(z)
        if torch.isnan(z).any():
            # BUG (pre-#13561): tile_size hard-coded; H % tile_size
            # may be non-zero at runtime.  TG does not enter this
            # branch under the no-NaN assumption used at verification
            # time.
            B, C, H, W = z.shape
            z = z.view(B, C, H // self.tile_size, self.tile_size,
                       W // self.tile_size, self.tile_size)
            z = z.permute(0, 2, 4, 1, 3, 5).contiguous()
            z = z.view(B * (H // self.tile_size) * (W // self.tile_size),
                       C, self.tile_size, self.tile_size)
        return z


INPUT_SHAPES = {"z": (1, 4, 32, 32)}
