"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/175683
Expected Error substring: Sizes of tensors must match
"""

def _run():
    import torch
    a = torch.randn(2, 3)
    b = torch.randn(2, 4)
    torch.cat([a, b], dim=0)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
