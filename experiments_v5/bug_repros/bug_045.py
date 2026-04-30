"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172364
Expected Error substring: Calculated padded input size per channel
"""

def _run():
    import torch, torch.nn as nn
    m = nn.Conv2d(3, 4, kernel_size=5)
    x = torch.randn(1, 3, 3, 3)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
