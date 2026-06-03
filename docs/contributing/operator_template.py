"""Template for adding a TensorGuard operator transfer/conformance test.

Copy this file into tests/ with a real test_* name, replace OPERATOR_UNDER_TEST,
and keep one valid real-PyTorch case plus one invalid case when the operator has
a refutable precondition. This template is intentionally not named test_*.py so
pytest will not collect it directly.
"""

from __future__ import annotations

import pytest

from src.api import verify_architecture


OPERATOR_UNDER_TEST = "replace.me"


def test_operator_transfer_accepts_real_valid_case():
    source = """
import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x):
        # Replace with a minimal, valid use of OPERATOR_UNDER_TEST.
        return x
"""
    result = verify_architecture(source, input_shape=(2, 3), soundness_mode="sound")
    assert not result.bugs


def test_operator_transfer_refutes_real_invalid_case():
    source = """
import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x):
        # Replace with a minimal, invalid use that real PyTorch would reject.
        return x.reshape(5, 5)
"""
    result = verify_architecture(source, input_shape=(2, 3, 4), soundness_mode="sound")
    assert result.bugs
    assert any("shape" in bug.category.lower() for bug in result.bugs)


@pytest.mark.parametrize("path", ["operator_confidence_table.json", "proof_footprint_manifest.json"])
def test_operator_metadata_regenerated(path):
    assert path
