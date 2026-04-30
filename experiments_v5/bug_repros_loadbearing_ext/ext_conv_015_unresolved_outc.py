"""Targeted: Conv2d whose out_channels comes from an unresolved attribute
(triggers the `out_c is None` branch at line ~4946 and the alternate
spatial-dim arithmetic at lines 4963-4964)."""
import torch, torch.nn as nn

class Cfg:
    pass

class M(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg if cfg is not None else Cfg()
        if not hasattr(cfg, "out_c"):
            cfg.out_c = 5
        self.conv = nn.Conv2d(3, cfg.out_c, kernel_size=7)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    try:
        M()(torch.randn(1, 3, 3, 3))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
