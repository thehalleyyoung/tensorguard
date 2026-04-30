# input_shape: (2, 256, 16, 16)
# bug: classifier expects 32 input channels but follows up2 which produces
#      64 channels (in_ch wired wrong on classifier conv).
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, in_ch: int = 256, num_classes: int = 21):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 128, 3, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)  # outputs 64ch
        self.classifier = nn.Conv2d(32, num_classes, 1)  # WRONG: expects 32

    def forward(self, x):
        x = F.relu(self.bn(self.proj(x)))
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        return self.classifier(x)

