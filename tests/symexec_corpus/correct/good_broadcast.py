import torch

def f():
    a = torch.zeros(3, 1)
    b = torch.zeros(1, 4)
    return a + b

if __name__ == "__main__":
    f()
