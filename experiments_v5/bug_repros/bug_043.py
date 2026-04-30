"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172386
Expected Error substring: must match the size of tensor
"""

def _run():
    import torch, torch.nn as nn
    m = nn.MSELoss()
    input = torch.randn(3, 4)
    target = torch.randn(4, 3)
    m(input, target)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
