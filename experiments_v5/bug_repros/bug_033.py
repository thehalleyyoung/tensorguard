"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/173157
Expected Error substring: Expected
"""

def _run():
    import torch, torch.nn as nn
    m = nn.GroupNorm(4, 16)
    x = torch.randn(2, 8, 5, 5)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
