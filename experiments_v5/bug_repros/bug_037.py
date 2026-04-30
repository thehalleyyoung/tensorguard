"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172739
Expected Error substring: is invalid for input of size
"""

def _run():
    import torch
    x = torch.randn(2, 3, 4)
    x.flatten().view(5, 5)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
