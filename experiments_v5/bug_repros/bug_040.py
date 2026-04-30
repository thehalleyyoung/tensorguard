"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172579
Expected Error substring: must match the size of tensor
"""

def _run():
    import torch
    a = torch.randn(2, 3, 4, 5)
    b = torch.randn(4, 5, 6)
    a @ b

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
