"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171517
Expected Error substring: 1D tensors expected
"""

def _run():
    import torch
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    torch.dot(a, b)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
