import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, image_size=384, patch_size=16):
        super().__init__()
        self.np = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, 64, kernel_size=patch_size,
                              stride=patch_size)
    def forward(self, x):
        return self.proj(x).view(x.shape[0], self.np, 64)
