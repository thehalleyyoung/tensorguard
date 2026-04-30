"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/173902
Expected Error substring: expected input
"""

def _run():
    import torch, torch.nn as nn
    m = nn.Conv3d(2, 4, 3)
    x = torch.randn(1, 5, 8, 8, 8)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
