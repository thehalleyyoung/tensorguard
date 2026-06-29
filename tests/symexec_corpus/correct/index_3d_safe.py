import torch

if __name__ == "__main__":
    y = torch.randn(4, 8, 16)
    z = y[-1, :, :]
