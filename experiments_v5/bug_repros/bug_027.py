"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/173765
Expected Error substring: Expected size for first two dimensions
"""

def _run():
    import torch
    a = torch.randn(3, 4, 5)
    b = torch.randn(5, 5, 6)
    torch.bmm(a, b)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
