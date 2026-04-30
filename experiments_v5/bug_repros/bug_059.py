"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171704
Expected Error substring: must match the size of tensor
"""

def _run():
    import torch
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    torch.maximum(a, b)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
