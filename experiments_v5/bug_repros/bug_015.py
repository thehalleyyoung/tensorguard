"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/177017
Expected Error substring: running_mean should contain 8 elements not 16
"""

def _run():
    import torch, torch.nn as nn
    m = nn.BatchNorm2d(16)
    x = torch.randn(4, 8, 10, 10)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
