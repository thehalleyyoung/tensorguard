"""
Tests for the held-out deep composition benchmark.

Verifies all 15 held-out models are parseable, TensorGuard produces
correct verdicts, and hop count annotations are accurate.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model, Device, Phase

# Import the benchmark definitions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from run_heldout_deep_composition import HELDOUT_BENCHMARKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _verify(bench):
    """Run verify_model on a benchmark entry and return the result."""
    return verify_model(
        bench["source"],
        input_shapes=bench["input_shapes"],
    )


def _bench_by_name(name):
    """Look up a benchmark by name."""
    for b in HELDOUT_BENCHMARKS:
        if b["name"] == name:
            return b
    raise KeyError(f"No benchmark named {name!r}")


# ---------------------------------------------------------------------------
# Structural / parseability tests
# ---------------------------------------------------------------------------
class TestBenchmarkStructure:
    """All 15 models are well-formed and parseable."""

    def test_benchmark_count(self):
        assert len(HELDOUT_BENCHMARKS) == 15

    def test_category_distribution(self):
        cats = [b["category"] for b in HELDOUT_BENCHMARKS]
        assert cats.count("reshape_chain") == 5
        assert cats.count("multi_branch") == 5
        assert cats.count("mixed_arithmetic") == 5

    def test_all_required_keys(self):
        required = {"name", "category", "num_hops", "has_bug",
                     "input_shapes", "source"}
        for b in HELDOUT_BENCHMARKS:
            missing = required - set(b.keys())
            assert not missing, f"{b['name']} missing keys: {missing}"


@pytest.mark.parametrize("bench", HELDOUT_BENCHMARKS,
                         ids=[b["name"] for b in HELDOUT_BENCHMARKS])
class TestParseability:
    """TensorGuard can parse and produce a verdict for every model."""

    def test_produces_verdict(self, bench):
        result = _verify(bench)
        assert result.safe is not None, (
            f"{bench['name']}: verify_model returned None for .safe")


# ---------------------------------------------------------------------------
# Reshape-chain models (category a)
# ---------------------------------------------------------------------------
class TestReshapeChain:
    """5 reshape/view chain models."""

    def test_reshape_chain_transpose_bug(self):
        b = _bench_by_name("reshape-chain-transpose-bug")
        result = _verify(b)
        assert not result.safe, "Should detect 50 vs 48 mismatch"

    def test_reshape_chain_safe_roundtrip(self):
        b = _bench_by_name("reshape-chain-safe-roundtrip")
        result = _verify(b)
        assert result.safe, "Roundtrip reshapes should be safe"

    def test_reshape_chain_view_squeeze_bug(self):
        b = _bench_by_name("reshape-chain-view-squeeze-bug")
        result = _verify(b)
        assert not result.safe, "Should detect 100 vs 120 mismatch"

    def test_reshape_chain_nested_safe(self):
        b = _bench_by_name("reshape-chain-nested-safe")
        result = _verify(b)
        assert result.safe, "Nested reshapes with matching dims should be safe"

    def test_reshape_chain_multi_view_bug(self):
        b = _bench_by_name("reshape-chain-multi-view-bug")
        result = _verify(b)
        assert not result.safe, "Should detect 64 vs 60 mismatch"


# ---------------------------------------------------------------------------
# Multi-branch models (category b)
# ---------------------------------------------------------------------------
class TestMultiBranch:
    """5 multi-branch merge models."""

    def test_multi_branch_triple_merge_bug(self):
        b = _bench_by_name("multi-branch-triple-merge-bug")
        result = _verify(b)
        assert not result.safe, "Should detect 32 vs 31 merge mismatch"

    def test_multi_branch_residual_safe(self):
        b = _bench_by_name("multi-branch-residual-safe")
        result = _verify(b)
        assert result.safe, "Matching residual branches should be safe"

    def test_multi_branch_asymmetric_bug(self):
        b = _bench_by_name("multi-branch-asymmetric-bug")
        result = _verify(b)
        assert not result.safe, "Should detect 16 vs 15 merge mismatch"

    def test_multi_branch_parallel_safe(self):
        b = _bench_by_name("multi-branch-parallel-safe")
        result = _verify(b)
        assert result.safe, "Matching parallel branches should be safe"

    def test_multi_branch_four_way_bug(self):
        b = _bench_by_name("multi-branch-four-way-bug")
        result = _verify(b)
        assert not result.safe, "Should detect 64 vs 63 four-way mismatch"


# ---------------------------------------------------------------------------
# Mixed-arithmetic models (category c)
# ---------------------------------------------------------------------------
class TestMixedArithmetic:
    """5 convolution output size arithmetic models."""

    def test_conv_stride_padding_bug(self):
        b = _bench_by_name("conv-stride-padding-bug")
        result = _verify(b)
        assert not result.safe, "Should detect 4*4 vs 3*3 spatial mismatch"

    def test_conv_pooling_safe(self):
        b = _bench_by_name("conv-pooling-safe")
        result = _verify(b)
        assert result.safe, "Correct pooling arithmetic should be safe"

    def test_conv_dilation_bug(self):
        b = _bench_by_name("conv-dilation-bug")
        result = _verify(b)
        assert not result.safe, "Should detect dilation spatial size error"

    def test_conv_residual_arithmetic_safe(self):
        b = _bench_by_name("conv-residual-arithmetic-safe")
        result = _verify(b)
        assert result.safe, "Correct conv residual should be safe"

    def test_conv_kernel_stride_mismatch_bug(self):
        b = _bench_by_name("conv-kernel-stride-mismatch-bug")
        result = _verify(b)
        assert not result.safe, "Should detect 3*3 vs 2*2 spatial mismatch"


# ---------------------------------------------------------------------------
# Hop count annotation tests
# ---------------------------------------------------------------------------
class TestHopCounts:
    """Hop count annotations are correct for each model."""

    def test_reshape_chain_transpose_bug_hops(self):
        assert _bench_by_name("reshape-chain-transpose-bug")["num_hops"] == 5

    def test_reshape_chain_safe_roundtrip_hops(self):
        assert _bench_by_name("reshape-chain-safe-roundtrip")["num_hops"] == 4

    def test_reshape_chain_view_squeeze_bug_hops(self):
        assert _bench_by_name("reshape-chain-view-squeeze-bug")["num_hops"] == 6

    def test_reshape_chain_nested_safe_hops(self):
        assert _bench_by_name("reshape-chain-nested-safe")["num_hops"] == 5

    def test_reshape_chain_multi_view_bug_hops(self):
        assert _bench_by_name("reshape-chain-multi-view-bug")["num_hops"] == 5

    def test_multi_branch_triple_merge_bug_hops(self):
        assert _bench_by_name("multi-branch-triple-merge-bug")["num_hops"] == 4

    def test_multi_branch_residual_safe_hops(self):
        assert _bench_by_name("multi-branch-residual-safe")["num_hops"] == 3

    def test_multi_branch_asymmetric_bug_hops(self):
        assert _bench_by_name("multi-branch-asymmetric-bug")["num_hops"] == 5

    def test_multi_branch_parallel_safe_hops(self):
        assert _bench_by_name("multi-branch-parallel-safe")["num_hops"] == 4

    def test_multi_branch_four_way_bug_hops(self):
        assert _bench_by_name("multi-branch-four-way-bug")["num_hops"] == 4

    def test_conv_stride_padding_bug_hops(self):
        assert _bench_by_name("conv-stride-padding-bug")["num_hops"] == 4

    def test_conv_pooling_safe_hops(self):
        assert _bench_by_name("conv-pooling-safe")["num_hops"] == 4

    def test_conv_dilation_bug_hops(self):
        assert _bench_by_name("conv-dilation-bug")["num_hops"] == 4

    def test_conv_residual_arithmetic_safe_hops(self):
        assert _bench_by_name("conv-residual-arithmetic-safe")["num_hops"] == 5

    def test_conv_kernel_stride_mismatch_bug_hops(self):
        assert _bench_by_name("conv-kernel-stride-mismatch-bug")["num_hops"] == 5
