import torch

class M:
    def forward(self, x, return_kv_cache=False):
        if return_kv_cache:
            return x, None
        return x

if __name__ == "__main__":
    m = M()
    tokens = torch.randn(1, 1024, 512)
    out2, cache = m.forward(tokens, return_kv_cache=True)
