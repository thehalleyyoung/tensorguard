"""Tests for ast_tied_param_audit.py — synthetic positive and negative fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Allow importing from reproducibility/ without installing the package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "reproducibility"))

from ast_tied_param_audit import detect_tied_params, wilson_ci  # noqa: E402

OUTPUT_PATH = Path(__file__).parent.parent / "experiments_v5" / "ast_tied_param_prevalence.json"

# ---------------------------------------------------------------------------
# Positive fixtures — each should be flagged by the detector
# ---------------------------------------------------------------------------

POSITIVE_SHARED_MODULE_WEIGHT = """
import torch.nn as nn

class TiedEmbedDecoder(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.decoder = nn.Linear(dim, vocab, bias=False)
        # Tie input and output embeddings (common in language models)
        self.decoder.weight = self.embed.weight
"""

POSITIVE_SETATTR_ALIAS = """
import torch.nn as nn

class WeightTiedLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        # Expose weight via setattr alias
        setattr(self, 'w', self.linear.weight)
"""

POSITIVE_DIRECT_WEIGHT_EXTRACTION = """
import torch.nn as nn

class ParamExtractor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        # Extract weight as a named self-attribute
        self.w = self.proj.weight
"""

# ---------------------------------------------------------------------------
# Negative fixtures — should NOT be flagged
# ---------------------------------------------------------------------------

NEGATIVE_INDEPENDENT_LINEARS = """
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
"""

NEGATIVE_FRESH_PARAMETERS = """
import torch
import torch.nn as nn

class TwoParams(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
"""

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPositiveFixtures:
    def test_shared_module_weight_inplace_rebind(self):
        """Tied embed+decoder weights via self.decoder.weight = self.embed.weight (R3)."""
        assert detect_tied_params(POSITIVE_SHARED_MODULE_WEIGHT), (
            "Should detect in-place weight rebind tying embed and decoder"
        )

    def test_setattr_alias(self):
        """setattr(self, 'w', self.linear.weight) should be flagged (R7)."""
        assert detect_tied_params(POSITIVE_SETATTR_ALIAS), (
            "Should detect weight extraction via setattr"
        )

    def test_direct_weight_extraction(self):
        """self.w = self.proj.weight should be flagged (R1)."""
        assert detect_tied_params(POSITIVE_DIRECT_WEIGHT_EXTRACTION), (
            "Should detect direct weight attribute extraction"
        )


class TestNegativeFixtures:
    def test_independent_linears_not_flagged(self):
        """Two independent nn.Linear layers — no sharing, should not be flagged."""
        assert not detect_tied_params(NEGATIVE_INDEPENDENT_LINEARS), (
            "Independent linears should not be flagged as tied"
        )

    def test_fresh_parameters_not_flagged(self):
        """Two independent nn.Parameter instances — should not be flagged."""
        assert not detect_tied_params(NEGATIVE_FRESH_PARAMETERS), (
            "Independent Parameters should not be flagged as tied"
        )


class TestOutputArtifact:
    def test_json_exists_after_audit(self):
        """The JSON output file must exist (created by running the audit script)."""
        assert OUTPUT_PATH.exists(), (
            f"Expected {OUTPUT_PATH} to exist after running "
            "reproducibility/ast_tied_param_audit.py"
        )

    def test_json_has_required_keys(self):
        """JSON must contain all required keys with valid types."""
        assert OUTPUT_PATH.exists(), "Run the audit script first"
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        required = {"n_blocks", "n_flagged", "prevalence", "wilson_low", "wilson_high", "recomputed_bound"}
        assert required.issubset(data.keys()), f"Missing keys: {required - data.keys()}"

    def test_json_prevalence_in_unit_interval(self):
        """prevalence must be a float in [0, 1]."""
        assert OUTPUT_PATH.exists(), "Run the audit script first"
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        p = data["prevalence"]
        assert isinstance(p, (int, float)), "prevalence must be numeric"
        assert 0.0 <= p <= 1.0, f"prevalence {p} out of [0, 1]"

    def test_json_wilson_bounds_ordered(self):
        """Wilson CI must satisfy low <= prevalence <= high."""
        assert OUTPUT_PATH.exists(), "Run the audit script first"
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        lo, hi, p = data["wilson_low"], data["wilson_high"], data["prevalence"]
        assert lo <= p <= hi, f"Wilson CI [{lo}, {hi}] does not contain prevalence {p}"

    def test_json_bound_equals_wilson_high_times_025(self):
        """recomputed_bound must equal wilson_high * 0.25."""
        assert OUTPUT_PATH.exists(), "Run the audit script first"
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        expected = data["wilson_high"] * 0.25
        assert abs(data["recomputed_bound"] - expected) < 1e-9


class TestWilsonCI:
    def test_zero_successes(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert 0.0 < hi < 0.05  # upper bound small but positive

    def test_all_successes(self):
        lo, hi = wilson_ci(100, 100)
        assert lo > 0.95
        assert hi >= 1.0 - 1e-9

    def test_symmetric_midpoint(self):
        lo, hi = wilson_ci(50, 100)
        assert abs((lo + hi) / 2 - 0.5) < 0.02
