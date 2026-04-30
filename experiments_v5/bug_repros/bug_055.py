"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171853
Expected Error substring: out of range
"""

def _run():
    import torch
    import torch.nn.functional as F
    w = torch.randn(5, 3)
    idx = torch.tensor([1, 9])
    F.embedding(idx, w)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
