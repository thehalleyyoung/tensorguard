"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171994
Expected Error substring: Size does not match
"""

def _run():
    import torch
    src = torch.randn(3, 4)
    idx = torch.zeros(3, 5, dtype=torch.long)
    torch.gather(src, 0, idx)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
