# bug: the backbone output is detached before being passed to the
#      trainable classification head, so although the head receives
#      a forward-pass activation, the head parameters still receive
#      gradients (from the head itself), but the backbone is guaranteed
#      frozen because the input gradient is severed.  If the intent was
#      to fine-tune the backbone, the .detach() is the bug.
# source: fine-tuning code where a frozen backbone was later unfrozen
#         but the .detach() guard was not removed.
import torch
import torch.nn as nn


class FineTuneClassifier(nn.Module):
    """Backbone + head; backbone grad severed by leftover .detach()."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(16, 8)
        self.head     = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).detach()   # BUG: backbone gets no gradient
        return self.head(features)
