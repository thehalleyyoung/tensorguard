import torch

def f():
    a = torch.zeros(2, 3)
    b = torch.zeros(4, 5)
    return torch.einsum("ij,jk->ik", a, b)

if __name__ == "__main__":
    f()
