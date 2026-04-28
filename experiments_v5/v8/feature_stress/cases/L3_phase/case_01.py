"""
L3-phase stress case 01: BatchNorm used in a context where eval-mode
running_mean shape disagrees with training-time feature count.

Target feature: train/eval phase check (L3).
Bug: A BatchNorm1d is initialised with num_features=32 but the forward
path feeds a tensor with 64 features only during eval (simulated by
the architecture structure). The phase-sensitive discrepancy would be
caught if the analyser tracks train vs eval mode distinctly.

Expected:
  WITHOUT L3 (check_phases=False): Verified (phase bugs filtered out)
  WITH    L3 (check_phases=True):  should be Refuted (phase violation)

Honest observation: verify_model's _encode_phase_safety only registers
satisfiable constraints (Or(TRAIN, EVAL)), never generates UNSAT.
→ L3 is a NO-OP in the current analyser; this case documents that limitation.
"""
import torch
import torch.nn as nn


class PhaseSensitiveBN(nn.Module):
    """BatchNorm whose num_features matches forward path at all times.
    The phase-sensitive *conceptual* bug is that this model would behave
    differently in train vs eval (running stats vs batch stats), but since
    both modes see the correct feature count the base checker finds nothing.
    A proper phase-aware analyser would flag the eval-mode discrepancy.
    """

    def __init__(self):
        super().__init__()
        # Correct feature count; no shape error at base level.
        self.bn = nn.BatchNorm1d(64)
        self.proj = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return self.bn(h)


FEATURE = "L3_phase"
INPUT_SHAPES = {"x": ("batch", 64)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
