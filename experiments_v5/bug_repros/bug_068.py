"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/170934
Expected Error substring: Calculated output size
"""

def _run():
    import torch, torch.nn as nn
    m = nn.MaxPool2d(5)
    x = torch.randn(1, 1, 3, 3)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
