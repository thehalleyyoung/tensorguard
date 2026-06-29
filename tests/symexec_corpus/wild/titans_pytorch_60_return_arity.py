import torch

class NestedAttn:
    def __init__(self, dim):
        self.dim = dim
    def forward(self, x, cache=None, return_kv_cache=False):
        out = x
        if return_kv_cache:
            return out, cache
        return out

if __name__ == "__main__":
    nested_attn = NestedAttn(512)
    tokens = torch.randn(1, 1024, 512)
    out1, cache = nested_attn.forward(tokens)
