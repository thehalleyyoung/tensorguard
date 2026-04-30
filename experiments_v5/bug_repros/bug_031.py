"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/173316
Expected Error substring: size
"""

def _run():
    import torch
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 5, 6)
    torch.einsum('abc,acd->abd', a, b)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
