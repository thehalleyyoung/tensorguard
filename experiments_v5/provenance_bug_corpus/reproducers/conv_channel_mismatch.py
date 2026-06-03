"""Minimized PyTorch reproducer for a convolution channel mismatch."""

import torch


def main() -> None:
    torch.nn.Conv2d(3, 4, kernel_size=3)(torch.randn(1, 2, 8, 8))


if __name__ == "__main__":
    main()
