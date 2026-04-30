"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171622
Expected Error substring: view size is not compatible
"""

def _run():
    import torch
    x = torch.randn(2, 3, 4)
    x.transpose(0, 1).view(2, 12)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
