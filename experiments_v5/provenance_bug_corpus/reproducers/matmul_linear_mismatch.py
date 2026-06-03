"""Minimized PyTorch reproducer for a Linear/matmul feature mismatch."""

import torch


def main() -> None:
    torch.nn.Linear(4, 3)(torch.randn(2, 5))


if __name__ == "__main__":
    main()
