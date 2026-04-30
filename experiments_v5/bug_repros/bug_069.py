"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/170666
Expected Error substring: size
"""

def _run():
    import torch
    x = torch.tensor([1.0, 2.0, 3.0])
    r = torch.tensor([2, 3])  # wrong length
    torch.repeat_interleave(x, r)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
