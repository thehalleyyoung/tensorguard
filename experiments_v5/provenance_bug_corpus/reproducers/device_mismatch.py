"""Minimized PyTorch reproducer for a CPU/CUDA device mismatch."""

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this device-mismatch reproducer")
    torch.ones(2, device="cpu") + torch.ones(2, device="cuda")


if __name__ == "__main__":
    main()
