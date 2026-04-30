"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172880
Expected Error substring: out of range
"""

def _run():
    import torch, torch.nn as nn
    m = nn.Embedding(10, 4)
    idx = torch.tensor([0, -3, 2])
    m(idx)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
