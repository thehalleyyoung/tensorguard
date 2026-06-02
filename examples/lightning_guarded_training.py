"""Step 169 — PyTorch Lightning adoption walkthrough (runnable).

A realistic, copy-pasteable example of guarding a real ``Trainer.fit`` with
TensorGuard.  ``GuardedCNN`` is a small but genuine convolutional image
classifier (the kind of model people actually train); ``BuggyCNN`` has a single
wrong dimension in its classifier head — exactly the mistake that today costs a
full ``Trainer`` spin-up and a mid-epoch crash to discover.

Add one callback and the bug is reported *at ``fit`` time, before the first
optimizer step*::

    import pytorch_lightning as pl
    from src.framework_hooks import TensorGuardCallback

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[TensorGuardCallback(input_shapes={"x": ("b", 3, 32, 32)})],
    )
    trainer.fit(model, dataloader)   # raises TensorGuardViolation if mis-shaped

Run ``python -m examples.lightning_guarded_training`` for a live demo: the clean
model trains for one ``fast_dev_run`` batch, the buggy one is blocked before any
batch is seen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl

INPUT_SHAPES = {"x": ("b", 3, 32, 32)}


class GuardedCNN(pl.LightningModule):
    """A small CIFAR-shaped CNN classifier (clean: provably shape-safe)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.head = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, 1)
        return self.head(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class BuggyCNN(pl.LightningModule):
    """Identical architecture, but the classifier head expects the wrong size.

    Self-contained (does not inherit ``forward``) so the mistake lives in this
    class's own source — exactly what a static checker reads.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        # 32*8*8 = 2048 features actually arrive; this head expects 32*16*16.
        self.head = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, 1)
        return self.head(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def make_loader(n: int = 8) -> DataLoader:
    ds = TensorDataset(torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,)))
    return DataLoader(ds, batch_size=4)


def _trainer():
    from src.framework_hooks import TensorGuardCallback

    return pl.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        callbacks=[TensorGuardCallback(input_shapes=INPUT_SHAPES)],
    )


def main() -> None:  # pragma: no cover - manual demo entry point
    import warnings

    from src.framework_hooks import TensorGuardViolation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("Training the clean CNN under a guarded Trainer ...")
        _trainer().fit(GuardedCNN(), make_loader())
        print("  -> trained one fast_dev_run batch, no violation.\n")

        print("Trying the buggy CNN ...")
        try:
            _trainer().fit(BuggyCNN(), make_loader())
        except TensorGuardViolation as exc:
            print(f"  -> blocked at fit() before any optimizer step:\n     {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
