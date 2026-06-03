"""Minimized PyTorch reproducer for a broadcast shape mismatch."""

import torch


def main() -> None:
    torch.ones(2, 3) + torch.ones(2, 4)


if __name__ == "__main__":
    main()
