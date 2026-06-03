"""Minimized PyTorch reproducer for an out-of-range dimension."""

import torch


def main() -> None:
    torch.randn(2, 3).transpose(0, 3)


if __name__ == "__main__":
    main()
