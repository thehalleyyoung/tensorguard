"""
Unfiltered post-freeze repro #8 (in-fragment, view total-size mismatch):
HF diffusers - Wan2VAE 3D conv decoder spatial reshape.

GitHub PR  : https://github.com/huggingface/diffusers/pull/13520  (merged 2026-04-19)
Repository : huggingface/diffusers
Buggy file : src/diffusers/models/autoencoders/autoencoder_kl_wan.py
            (pre-#13520, decoder upsample-3d block)

Root cause: depth*height*width product on input does not match
the post-upsample target product because the upsample factor
is hard-coded to 2 in the decoder while the encoder's stride
list ends with 3 for the temporal axis.  The buggy module
attempts to .view() to a tensor whose total size differs from
the input by a factor of 3/2.

In-fragment, expected verdict: RP at >= 0.99.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, channels=384, t_in=4, h_in=8, w_in=8):
        super().__init__()
        self.channels = channels
        self.t_in = t_in
        self.h_in = h_in
        self.w_in = w_in
        # Final spatial conv after the buggy upsample.
        self.out_conv = nn.Conv3d(channels, 3, kernel_size=1)

    def forward(self, x):
        # x: (B, channels, t_in, h_in, w_in).
        B, C, T, H, W = x.shape
        # BUG (pre-#13520): upsample factor 2 in (t,h,w) -- but the
        # decoder rebuilds with t-factor 3, so the .view() target
        # mismatches by 3/2 along the t axis.
        upsampled_total = B * C * (T * 2) * (H * 2) * (W * 2)
        # Hardcoded view target uses T*3 instead of T*2:
        target = (B, C, T * 3, H * 2, W * 2)
        # Manually flatten then view -- this is the buggy reshape.
        flat = x.repeat_interleave(8, dim=-1).reshape(B, C, T * 2, H * 2, W * 2)
        # The upstream code then does (now-incorrect) reshape to target:
        bad = flat.view(*target)
        return self.out_conv(bad)


INPUT_SHAPES = {"x": (1, 384, 4, 8, 8)}
