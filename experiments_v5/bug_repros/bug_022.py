"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/175165
Expected Error substring: size
"""

def _run():
    import torch
    a = torch.randn(2, 3, 4)
    b = torch.randn(3, 4, 5)
    torch.einsum('bij,bjk->bik', a, b)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
