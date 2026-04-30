"""Targeted: Conv2d clean module with unresolved out_channels via config attribute.
Exercises the `out_c is None` branch (lines ~4946-4976) without triggering
the kernel-too-big guard, so mutations on the alternate spatial-dim arithmetic
on lines 4963-4964 can be detected as verdict changes."""
import torch, torch.nn as nn

class Cfg:
    pass

class M(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg if cfg is not None else Cfg()
        if not hasattr(cfg, "out_c"):
            cfg.out_c = 8
        self.conv = nn.Conv2d(3, cfg.out_c, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    M()(torch.randn(1, 3, 16, 16))
