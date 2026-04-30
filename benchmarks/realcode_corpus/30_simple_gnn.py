# input_shape: (32, 16)
import torch
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    """Simple GCN-style message passing layer.

    Forward takes node features X of shape (N, F).  The adjacency
    matrix A is treated as a fixed buffer for static analysis.
    """

    def __init__(self, in_features: int = 16, hidden: int = 32, num_nodes: int = 32):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.register_buffer("A", torch.eye(num_nodes))

    def forward(self, x):
        h = self.A @ x
        h = F.relu(self.lin1(h))
        h = self.A @ h
        return self.lin2(h)
