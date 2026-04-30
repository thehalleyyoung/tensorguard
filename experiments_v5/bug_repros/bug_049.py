"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172019
Expected Error substring: Given normalized_shape
"""

def _run():
    import torch
    import torch.nn.functional as F
    x = torch.randn(2, 3, 4)
    F.layer_norm(x, [5])

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
