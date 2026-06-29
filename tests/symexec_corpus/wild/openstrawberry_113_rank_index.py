import torch

def monte_carlo_rollout(model, x):
    output = x
    output = output[-1, :, :]
    return output

if __name__ == "__main__":
    x = torch.randn(10, 32)
    monte_carlo_rollout(None, x)
