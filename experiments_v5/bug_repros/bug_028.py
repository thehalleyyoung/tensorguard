"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/173724
Expected Error substring: shape
"""

def _run():
    import torch
    x = torch.randn(16)
    x.view(4, -1, 3)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
