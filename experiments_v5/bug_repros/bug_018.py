"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/176230
Expected Error substring: mat1 and mat2 shapes cannot be multiplied
"""

def _run():
    import torch
    a = torch.randn(3, 4)
    b = torch.randn(5, 6)
    a @ b

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
