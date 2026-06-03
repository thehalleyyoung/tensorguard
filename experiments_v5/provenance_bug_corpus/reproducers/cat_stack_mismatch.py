"""Minimized PyTorch reproducer for a cat/stack shape mismatch."""

import torch


def main() -> None:
    torch.cat([torch.ones(2, 3), torch.ones(2, 4)], dim=0)


if __name__ == "__main__":
    main()
