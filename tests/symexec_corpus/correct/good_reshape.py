import torch

def f():
    a = torch.zeros(3, 4)
    return a.reshape(2, 6)

if __name__ == "__main__":
    f()
