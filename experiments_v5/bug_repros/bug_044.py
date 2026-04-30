"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172374
Expected Error substring: out of bounds
"""

def _run():
    import torch, torch.nn as nn
    m = nn.NLLLoss()
    input = torch.log_softmax(torch.randn(2, 3), dim=1)
    target = torch.tensor([5, 1])
    m(input, target)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
