"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171858
Expected Error substring: expected
"""

def _run():
    import torch, torch.nn as nn
    m = nn.InstanceNorm2d(8, affine=True)
    x = torch.randn(2, 4, 6, 6)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
