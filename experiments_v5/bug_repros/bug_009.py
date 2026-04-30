"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/179789
Expected Error substring: mat1 and mat2 shapes cannot be multiplied
"""

def _run():
    import torch, torch.nn as nn
    m = nn.Linear(128, 32)
    x = torch.randn(8, 64)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
