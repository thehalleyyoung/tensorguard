"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171852
Expected Error substring: in_channels must be divisible by groups
"""

def _run():
    import torch.nn as nn
    nn.Conv2d(10, 6, 3, groups=3)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
