import torch

class Block:
    def __init__(self, dim):
        self.dim = dim
    def forward(self, x):
        h = x.transpose(1, 2)
        h = h.unsqueeze(0)
        h = h.squeeze(0)
        return h.transpose(1, 2)

if __name__ == "__main__":
    block = Block(64)
    x = torch.randn(2, 16, 64)
    y = block.forward(x)
    a = y[0, :, :]
    b = y[:, -1, :]
