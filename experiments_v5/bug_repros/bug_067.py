"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/170980
Expected Error substring: must be batches of square matrices
"""

def _run():
    import torch
    a = torch.randn(3, 4)
    torch.linalg.inv(a)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
