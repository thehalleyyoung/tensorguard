"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/174339
Expected Error substring: running_mean should contain 16 elements not 32
"""

def _run():
    import torch, torch.nn as nn
    m = nn.BatchNorm1d(32)
    x = torch.randn(4, 16)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
