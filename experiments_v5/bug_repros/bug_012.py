"""
GitHub Issue: https://github.com/pytorch/pytorch/issues/178882
Expected Error substring: embed_dim must be divisible by num_heads
"""

def _run():
    import torch, torch.nn as nn
    m = nn.MultiheadAttention(embed_dim=10, num_heads=3)

if __name__ == '__main__':
    try:
        _run()
        print("NO_ERROR")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
