# input_shape: (2, 64, 32, 32)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    """Sub-pixel convolution upsampler (×2)."""

    def __init__(self, in_ch: int = 64, out_ch: int = 3, scale: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return self.shuffle(x)
