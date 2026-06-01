"""
Domain: PHASE (diagnostic-only — see note)
Bug class: train/eval-dependent layer (BatchNorm1d followed by Dropout).
Real-world concern: BatchNorm/Dropout behave differently in train vs. eval;
            forgetting `model.eval()` at inference is a classic correctness
            bug.
Provenance: canonical phase-sensitive module.

NOTE (diagnostic-only): the shipped checker registers phase well-formedness
            constraints for BatchNorm/Dropout layers but does NOT refute this
            construct — the phase domain does not currently flip any model
            from SAFE to UNSAFE. It is therefore classified DIAGNOSTIC-ONLY
            (it records phase structure / surfaces phase-sensitive layers)
            rather than a VERIFICATION domain that contributes refutations.
            This file documents that honest classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SHAPES = {"x": (4, 10)}


class PhaseSensitiveModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        return self.drop(self.bn(x))
