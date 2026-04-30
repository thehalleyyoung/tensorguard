"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/172419
Expected Error substring: Expected input batch_size
"""

def _run():
    import torch, torch.nn as nn
    m = nn.CrossEntropyLoss()
    input = torch.randn(4, 10)
    target = torch.tensor([1,2,3,4,5])
    m(input, target)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
