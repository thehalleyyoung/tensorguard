"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/171948
Expected Error substring: out of bound
"""

def _run():
    import torch
    out = torch.zeros(3, 4)
    idx = torch.tensor([[0,1,2,5]])
    src = torch.ones(1, 4)
    out.scatter_(0, idx, src)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
