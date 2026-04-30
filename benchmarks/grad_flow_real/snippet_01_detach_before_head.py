# bug: detach() on stage1 output severs gradient to stage1 parameters;
#      fine-tuning stage1 via this path is silently broken.
# source: mirrors pattern seen in transfer-learning repos where a frozen
#         encoder is later unfrozen but the .detach() call is left in.
import torch
import torch.nn as nn


class TwoStageModel(nn.Module):
    """Two-stage encoder-head; stage1 grad killed by .detach()."""

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Linear(16, 8)
        self.stage2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stage1(x).detach()   # BUG: severs grad to stage1
        return self.stage2(feat)
