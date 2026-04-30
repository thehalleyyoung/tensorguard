# input_shape: (8, 32, 128)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    """Attention-pooling head: pool tokens to a single embedding."""

    def __init__(self, dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        scores = self.attn(x)
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)
        return self.head(pooled)
