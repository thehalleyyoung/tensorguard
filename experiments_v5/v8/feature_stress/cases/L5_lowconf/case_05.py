"""
L5-lowconf stress case 05: division by zero from bucket_size=0 in routing.

Target feature: low-confidence violations (L5).
Bug: bucket_size=0 → ZeroDivisionError when computing number of buckets:
num_experts // bucket_size. Detectable by flow-sensitive analyser.

Expected:
  WITHOUT L5: Verified
  WITH    L5: Refuted (division_by_zero on bucket_size)
"""
import torch
import torch.nn as nn


class MixtureOfExperts(nn.Module):
    def __init__(self, d_model: int = 64, num_experts: int = 8, bucket_size: int = 0):
        super().__init__()
        # BUG: bucket_size=0 → ZeroDivisionError
        self.num_buckets = num_experts // bucket_size
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.router(x).softmax(-1)
        idx = scores.argmax(-1)
        return self.experts[0](x)  # simplified


FEATURE = "L5_lowconf"
INPUT_SHAPES = {"x": ("batch", "seq", 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
