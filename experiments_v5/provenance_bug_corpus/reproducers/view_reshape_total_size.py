"""Minimized PyTorch reproducer for an invalid reshape total size."""

import torch


def main() -> None:
    torch.randn(2, 3).view(5)


if __name__ == "__main__":
    main()
