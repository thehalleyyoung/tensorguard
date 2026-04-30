"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171931
Expected Error substring: must match the existing size
"""

def _run():
    import torch
    x = torch.randn(3, 1, 4)
    x.expand(3, 2, 5)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
