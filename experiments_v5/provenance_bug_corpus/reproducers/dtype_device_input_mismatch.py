"""Minimized PyTorch reproducer for an input/weight dtype mismatch."""

import torch


def main() -> None:
    layer = torch.nn.Conv2d(3, 4, kernel_size=3).double()
    layer(torch.randn(1, 3, 8, 8))


if __name__ == "__main__":
    main()
