"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171850
Expected Error substring: must be a matrix
"""

def _run():
    import torch
    a = torch.randn(2, 3, 4)
    b = torch.randn(4, 5)
    torch.mm(a, b)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
