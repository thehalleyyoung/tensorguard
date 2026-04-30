import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self, num_experts=4, top_k=5):
        super().__init__()
        self.gate = nn.Linear(64, num_experts)
        self.k = top_k
    def forward(self, x):
        scores = self.gate(x)
        return torch.topk(scores, self.k, dim=-1).values
