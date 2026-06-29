import torch

def f():
    a = torch.zeros(3, 4)
    b = torch.zeros(5, 6)
    return a + b

if __name__ == "__main__":
    f()
