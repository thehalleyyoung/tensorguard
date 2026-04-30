"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/175831
Expected Error substring: shape '[5, 3]' is invalid for input of size 12
"""

def _run():
    import torch
    x = torch.randn(3, 4)
    x.reshape(5, 3)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
