"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171669
Expected Error substring: Expected 2D
"""

def _run():
    import torch, torch.nn as nn
    m = nn.Conv1d(3, 4, 3)
    x = torch.randn(2, 3, 5, 5)
    m(x)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
