"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/179931
Expected Error substring: expected input
"""

def _run():
    import torch, torch.nn as nn
    m = nn.Conv2d(3, 16, 3)
    x = torch.randn(2, 4, 32, 32)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
