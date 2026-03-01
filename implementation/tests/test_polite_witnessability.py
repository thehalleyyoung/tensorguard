"""Tests for polite witnessability verification."""

import os
import sys

import pytest

IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, IMPL_ROOT)

from experiments.run_polite_witnessability import (
    device_witness,
    generate_arrangements,
    generate_equality_arrangements,
    phase_witness,
    stride_finite_witness,
    verify_equality_arrangement_consistency,
    verify_witnessability,
)


class TestArrangementGeneration:
    """Tests for arrangement generation."""

    def test_arrangement_count_binary(self):
        arrs = generate_arrangements(["A", "B"], 3)
        assert len(arrs) == 2 ** 3  # 8

    def test_arrangement_count_ternary(self):
        arrs = generate_arrangements(["A", "B", "C"], 2)
        assert len(arrs) == 3 ** 2  # 9

    def test_empty_domain(self):
        arrs = generate_arrangements([], 2)
        assert len(arrs) == 0

    def test_zero_vars(self):
        arrs = generate_arrangements(["A", "B"], 0)
        assert len(arrs) == 1  # single empty tuple


class TestEqualityArrangements:
    """Tests for equality arrangement (set partition) generation."""

    def test_bell_number_3(self):
        """Bell number B(3) = 5."""
        partitions = generate_equality_arrangements(3)
        assert len(partitions) == 5

    def test_bell_number_2(self):
        """Bell number B(2) = 2."""
        partitions = generate_equality_arrangements(2)
        assert len(partitions) == 2

    def test_single_var(self):
        partitions = generate_equality_arrangements(1)
        assert len(partitions) == 1
        assert partitions[0] == [frozenset({0})]


class TestWitnessProduction:
    """Tests for witness production functions."""

    def test_device_witness_valid(self):
        domain = ["CPU", "CUDA_0", "CUDA_1", "CUDA_2", "CUDA_3"]
        witness = device_witness(("CPU", "CUDA_0"), domain)
        assert witness is not None
        assert witness["dev_0"] == "CPU"
        assert witness["dev_1"] == "CUDA_0"

    def test_device_witness_invalid_element(self):
        domain = ["CPU", "CUDA_0"]
        witness = device_witness(("CPU", "INVALID"), domain)
        assert witness is None

    def test_phase_witness_valid(self):
        domain = ["TRAIN", "EVAL"]
        witness = phase_witness(("TRAIN", "EVAL"), domain)
        assert witness is not None
        assert witness["phase_0"] == "TRAIN"

    def test_stride_witness_valid(self):
        domain = ["1", "2", "4", "8", "16"]
        witness = stride_finite_witness(("4", "8"), domain)
        assert witness is not None


class TestWitnessabilityVerification:
    """Tests for full witnessability verification."""

    def test_device_theory_polite(self):
        domain = ["CPU", "CUDA_0", "CUDA_1", "CUDA_2", "CUDA_3"]
        result = verify_witnessability("T_device", domain, 2, device_witness)
        assert result["is_polite"] is True
        assert result["witness_rate"] == 1.0

    def test_phase_theory_polite(self):
        domain = ["TRAIN", "EVAL"]
        result = verify_witnessability("T_phase", domain, 2, phase_witness)
        assert result["is_polite"] is True

    def test_equality_arrangement_device(self):
        domain = ["CPU", "CUDA_0", "CUDA_1", "CUDA_2", "CUDA_3"]
        result = verify_equality_arrangement_consistency("T_device", domain, 3)
        assert result["all_realizable"] is True

    def test_equality_arrangement_phase_overflow(self):
        """3 vars over 2-element domain: some partitions unrealizable."""
        domain = ["TRAIN", "EVAL"]
        result = verify_equality_arrangement_consistency("T_phase", domain, 3)
        # B(3)=5 partitions, but only partitions with ≤2 classes are realizable
        assert result["unrealizable_count"] > 0
