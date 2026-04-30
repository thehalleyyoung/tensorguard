"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172712
Expected Error substring: stack expects each tensor to be equal size
"""

def _run():
    import torch
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    torch.stack([a, b])

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
