"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171764
Expected Error substring: split_with_sizes
"""

def _run():
    import torch
    x = torch.randn(6)
    torch.split(x, [2, 2, 3])

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
