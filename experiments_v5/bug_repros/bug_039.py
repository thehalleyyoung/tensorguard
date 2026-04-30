"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172684
Expected Error substring: Dimension out of range
"""

def _run():
    import torch
    import torch.nn.functional as F
    x = torch.randn(2, 3, 4)
    F.softmax(x, dim=5)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
