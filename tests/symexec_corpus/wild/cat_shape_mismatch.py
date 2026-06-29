import torch

def f():
    a = torch.zeros(2, 3)
    b = torch.zeros(2, 4)
    return torch.cat([a, b], dim=0)

if __name__ == "__main__":
    f()
