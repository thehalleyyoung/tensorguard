"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171523
Expected Error substring: Dimension out of range
"""

def _run():
    import torch
    x = torch.randn(3, 4)
    idx = torch.tensor([0, 1])
    torch.index_select(x, 5, idx)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
