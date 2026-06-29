import torch

def f():
    a = torch.zeros(2, 3)
    b = torch.zeros(3, 4)
    return a @ b

if __name__ == "__main__":
    f()
