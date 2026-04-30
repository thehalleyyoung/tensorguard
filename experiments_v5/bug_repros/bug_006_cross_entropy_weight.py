"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/174181
Expected Error: weight tensor should be defined
"""

import torch
import torch.nn as nn

INPUT_SHAPES = {"input": (3, 10), "target": (3,)}


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # weight has 5 elements but model has 10 classes
        self.loss = nn.CrossEntropyLoss(weight=torch.ones(5))

    def forward(self, input, target):
        return self.loss(input, target)


if __name__ == '__main__':
    try:
        m = BuggyModule()
        input = torch.randn(3, 10)
        target = torch.tensor([1, 0, 4])
        m(input, target)
    except RuntimeError as e:
        print(f"Error: {e}")
