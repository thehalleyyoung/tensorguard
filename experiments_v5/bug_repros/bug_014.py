"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/177254
Expected Error substring: Dimension out of range
"""

def _run():
    import torch
    x = torch.randn(2, 3, 4)
    x.transpose(0, 5)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
