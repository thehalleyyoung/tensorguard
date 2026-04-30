"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172822
Expected Error substring: expected input
"""

def _run():
    import torch, torch.nn as nn
    m = nn.ConvTranspose2d(8, 4, 3)
    x = torch.randn(1, 4, 16, 16)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
