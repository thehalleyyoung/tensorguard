"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/173709
Expected Error substring: must match the size of tensor
"""

def _run():
    import torch
    cond = torch.zeros(3, 4, dtype=torch.bool)
    x = torch.randn(5, 4)
    y = torch.randn(3, 4)
    torch.where(cond, x, y)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
