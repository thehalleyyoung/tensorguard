"""Comprehensive tests for assume-guarantee compositional verification."""

import pytest
import time

from src.assume_guarantee import (
    decompose_graph,
    verify_compositional,
    verify_compositional_incremental,
    check_interface_compatibility,
    InterfaceContract,
    SubModule,
    CompositionalResult,
    InterfaceCheck,
    DecompositionStrategy,
    VerificationCache,
    _shapes_compatible,
    decompose_and_summarize,
    reset_default_cache,
)
from src.model_checker import (
    extract_computation_graph,
    verify_model,
    ComputationGraph,
    VerificationResult,
    Device,
    Phase,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Source code fixtures
# ═══════════════════════════════════════════════════════════════════════════════

SIMPLE_LINEAR = """\
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""

TWO_LAYER_MLP = """\
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
"""

THREE_LAYER_MLP = """\
import torch.nn as nn

class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
"""

CONV_MODEL = """\
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x
"""

SHAPE_MISMATCH_MODEL = """\
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

DROPOUT_MODEL = """\
import torch.nn as nn

class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""

RESIDUAL_MODEL = """\
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + identity
        return out
"""

EMPTY_FORWARD = """\
import torch.nn as nn

class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
"""

DEEP_MLP = """\
import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 80)
        self.fc2 = nn.Linear(80, 60)
        self.fc3 = nn.Linear(60, 40)
        self.fc4 = nn.Linear(40, 20)
        self.fc5 = nn.Linear(20, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
"""

CONV_FC_MODEL = """\
import torch.nn as nn

class ConvFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the default cache before each test."""
    reset_default_cache()
    yield
    reset_default_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Basic decomposition of simple MLP
# ═══════════════════════════════════════════════════════════════════════════════

class TestBasicDecomposition:
    def test_two_layer_mlp_decomposes(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        modules = decompose_graph(graph, strategy="auto")
        assert len(modules) >= 1
        for m in modules:
            assert isinstance(m, SubModule)

    def test_submodule_has_name(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        modules = decompose_graph(graph, strategy="auto")
        for m in modules:
            assert m.name is not None
            assert len(m.name) > 0

    def test_submodule_has_graph(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        modules = decompose_graph(graph, strategy="auto")
        for m in modules:
            assert isinstance(m.graph, ComputationGraph)
            assert m.graph.num_steps > 0

    def test_submodule_has_contracts(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        modules = decompose_graph(graph, strategy="auto")
        for m in modules:
            assert isinstance(m.input_contract, InterfaceContract)
            assert isinstance(m.output_contract, InterfaceContract)

    def test_decomposition_covers_all_steps(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        modules = decompose_graph(graph, strategy="auto")
        total_steps = sum(m.graph.num_steps for m in modules)
        assert total_steps == graph.num_steps

    def test_step_ranges_contiguous(self):
        graph = extract_computation_graph(THREE_LAYER_MLP)
        modules = decompose_graph(graph, strategy="auto")
        if len(modules) > 1:
            for i in range(len(modules) - 1):
                assert modules[i].step_range[1] == modules[i + 1].step_range[0]

    def test_three_layer_mlp_more_modules(self):
        graph = extract_computation_graph(THREE_LAYER_MLP)
        modules = decompose_graph(graph, strategy="auto")
        assert len(modules) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Decomposition of CNN model
# ═══════════════════════════════════════════════════════════════════════════════

class TestCNNDecomposition:
    def test_conv_model_decomposes(self):
        graph = extract_computation_graph(CONV_MODEL)
        modules = decompose_graph(graph, strategy="auto")
        assert len(modules) >= 1

    def test_conv_fc_model_decomposes(self):
        graph = extract_computation_graph(CONV_FC_MODEL)
        modules = decompose_graph(graph, strategy="auto")
        assert len(modules) >= 1
        total = sum(m.graph.num_steps for m in modules)
        assert total == graph.num_steps


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Decomposition of model with residual connections
# ═══════════════════════════════════════════════════════════════════════════════

class TestResidualDecomposition:
    def test_residual_model_decomposes(self):
        graph = extract_computation_graph(RESIDUAL_MODEL)
        modules = decompose_graph(graph, strategy="auto")
        assert len(modules) >= 1

    def test_residual_decomposition_covers_all_steps(self):
        graph = extract_computation_graph(RESIDUAL_MODEL)
        modules = decompose_graph(graph, strategy="auto")
        total = sum(m.graph.num_steps for m in modules)
        assert total == graph.num_steps


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Compositional verification of safe model
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompositionalSafe:
    def test_safe_mlp_compositional(self):
        result = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)
        assert result.safe is True

    def test_safe_mlp_agrees_with_monolithic(self):
        mono = verify_model(TWO_LAYER_MLP, input_shapes={"x": ("batch", 784)})
        comp = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert mono.safe == comp.safe

    def test_dropout_model_safe(self):
        result = verify_compositional(
            DROPOUT_MODEL,
            input_shapes={"x": ("batch", 10)},
            measure_monolithic=False,
        )
        assert result.safe is True

    def test_three_layer_safe(self):
        result = verify_compositional(
            THREE_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert result.safe is True

    def test_compositional_result_has_submodule_results(self):
        result = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert len(result.submodule_results) >= 1
        for name, vr in result.submodule_results.items():
            assert isinstance(vr, VerificationResult)

    def test_compositional_result_has_interface_checks(self):
        result = verify_compositional(
            THREE_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert isinstance(result.interface_checks, list)
        for ic in result.interface_checks:
            assert isinstance(ic, InterfaceCheck)

    def test_all_interfaces_compatible_for_safe_model(self):
        result = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        for ic in result.interface_checks:
            assert ic.compatible is True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Compositional verification of buggy model
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompositionalBuggy:
    def test_shape_mismatch_detected(self):
        result = verify_compositional(
            SHAPE_MISMATCH_MODEL,
            input_shapes={"x": ("batch", 10)},
            measure_monolithic=False,
        )
        assert result.safe is False

    def test_buggy_agrees_with_monolithic(self):
        mono = verify_model(SHAPE_MISMATCH_MODEL, input_shapes={"x": ("batch", 10)})
        comp = verify_compositional(
            SHAPE_MISMATCH_MODEL,
            input_shapes={"x": ("batch", 10)},
            measure_monolithic=False,
        )
        assert mono.safe == comp.safe


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Incremental verification with cache
# ═══════════════════════════════════════════════════════════════════════════════

class TestIncrementalVerification:
    def test_incremental_with_cache(self):
        cache = VerificationCache()
        # First run: populate cache
        r1 = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            cache=cache,
            measure_monolithic=False,
        )
        assert r1.safe is True
        assert cache.size > 0

        # Second run with same cache: should hit cache
        r2 = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            cache=cache,
            measure_monolithic=False,
        )
        assert r2.safe is True
        assert r2.cache_hits > 0

    def test_incremental_with_changed_module(self):
        cache = VerificationCache()
        r1 = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            cache=cache,
            measure_monolithic=False,
        )

        # Incremental with a changed module name
        r2 = verify_compositional_incremental(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            cache=cache,
            changed_modules={"block_0"},
        )
        assert isinstance(r2, CompositionalResult)
        assert r2.safe is True

    def test_incremental_no_changed_reverifies_all(self):
        cache = VerificationCache()
        r1 = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            cache=cache,
            measure_monolithic=False,
        )
        # Empty changed_modules means re-verify all
        r2 = verify_compositional_incremental(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            cache=cache,
            changed_modules=set(),
        )
        assert r2.safe is True

    def test_cache_put_get(self):
        cache = VerificationCache()
        result = VerificationResult(safe=True)
        cache.put("fp123", result, 1.0)
        assert cache.get("fp123") is result
        assert cache.get("nonexistent") is None

    def test_cache_invalidate(self):
        cache = VerificationCache()
        result = VerificationResult(safe=True)
        cache.put("fp123", result, 1.0)
        cache.invalidate("fp123")
        assert cache.get("fp123") is None

    def test_cache_clear(self):
        cache = VerificationCache()
        cache.put("a", VerificationResult(safe=True), 1.0)
        cache.put("b", VerificationResult(safe=True), 1.0)
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_layer_model(self):
        result = verify_compositional(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)
        assert result.safe is True
        # Single layer should produce 1 sub-module
        assert result.num_submodules >= 1

    def test_empty_forward_model(self):
        result = verify_compositional(
            EMPTY_FORWARD,
            input_shapes={"x": ("batch", 10)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)

    def test_single_layer_decomposition(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        modules = decompose_graph(graph, strategy="auto")
        assert len(modules) >= 1
        # All steps must be covered
        total = sum(m.graph.num_steps for m in modules)
        assert total == graph.num_steps

    def test_parse_error_returns_unsafe(self):
        result = verify_compositional(
            "this is not valid python class def",
            input_shapes={},
            measure_monolithic=False,
        )
        assert result.safe is False


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Interface contract generation and compatibility checking
# ═══════════════════════════════════════════════════════════════════════════════

class TestInterfaceContracts:
    def test_contract_has_name(self):
        contract = InterfaceContract(name="test_input")
        assert contract.name == "test_input"

    def test_contract_fingerprint_deterministic(self):
        c1 = InterfaceContract(
            name="test",
            input_shapes={"x": ("batch", 10)},
        )
        c2 = InterfaceContract(
            name="test",
            input_shapes={"x": ("batch", 10)},
        )
        assert c1.fingerprint() == c2.fingerprint()

    def test_contract_fingerprint_changes_with_shapes(self):
        c1 = InterfaceContract(name="test", input_shapes={"x": ("batch", 10)})
        c2 = InterfaceContract(name="test", input_shapes={"x": ("batch", 20)})
        assert c1.fingerprint() != c2.fingerprint()

    def test_contract_pretty(self):
        c = InterfaceContract(
            name="encoder_out",
            input_shapes={"x": ("batch", 784)},
            output_shapes={"y": ("batch", 256)},
            constraints=["dim 1 must match"],
        )
        pretty = c.pretty()
        assert "encoder_out" in pretty
        assert "784" in pretty

    def test_shapes_compatible_exact_match(self):
        ok, msg = _shapes_compatible(("batch", 10), ("batch", 10))
        assert ok is True

    def test_shapes_compatible_symbolic(self):
        ok, msg = _shapes_compatible(("batch", 10), ("N", 10))
        assert ok is True

    def test_shapes_incompatible_concrete(self):
        ok, msg = _shapes_compatible(("batch", 20), ("batch", 50))
        assert ok is False

    def test_shapes_incompatible_rank(self):
        ok, msg = _shapes_compatible(("batch", 10), ("batch", 10, 3))
        assert ok is False
        assert "rank mismatch" in msg

    def test_shapes_compatible_wildcard(self):
        ok, msg = _shapes_compatible(("*",), ("batch", 10, 3))
        assert ok is True

    def test_interface_check_compatible(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        modules = decompose_graph(
            graph, strategy="auto",
            input_shapes={"x": ("batch", 784)},
        )
        if len(modules) >= 2:
            check = check_interface_compatibility(modules[0], modules[1])
            assert isinstance(check, InterfaceCheck)
            assert check.compatible is True

    def test_interface_check_has_names(self):
        graph = extract_computation_graph(THREE_LAYER_MLP)
        modules = decompose_graph(
            graph, strategy="auto",
            input_shapes={"x": ("batch", 784)},
        )
        if len(modules) >= 2:
            check = check_interface_compatibility(modules[0], modules[1])
            assert check.producer == modules[0].name
            assert check.consumer == modules[1].name


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Speedup measurement
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedup:
    def test_speedup_field_present(self):
        result = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=True,
        )
        assert result.speedup_vs_monolithic > 0

    def test_total_time_positive(self):
        result = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert result.total_time_ms > 0

    def test_no_monolithic_speedup_is_one(self):
        result = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert result.speedup_vs_monolithic == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Different decomposition strategies
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecompositionStrategies:
    def test_layer_boundary_strategy(self):
        graph = extract_computation_graph(THREE_LAYER_MLP)
        modules = decompose_graph(graph, strategy="layer_boundary")
        assert len(modules) >= 1
        total = sum(m.graph.num_steps for m in modules)
        assert total == graph.num_steps

    def test_branch_merge_strategy(self):
        graph = extract_computation_graph(RESIDUAL_MODEL)
        modules = decompose_graph(graph, strategy="branch_merge")
        assert len(modules) >= 1

    def test_single_layer_strategy(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        modules = decompose_graph(graph, strategy="single_layer")
        # Each step should be its own module
        assert len(modules) == graph.num_steps

    def test_auto_strategy(self):
        graph = extract_computation_graph(THREE_LAYER_MLP)
        modules = decompose_graph(graph, strategy="auto")
        assert len(modules) >= 1

    def test_verify_with_layer_boundary_strategy(self):
        result = verify_compositional(
            THREE_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            strategy="layer_boundary",
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)
        assert result.decomposition_strategy == DecompositionStrategy.LAYER_BOUNDARY

    def test_verify_with_branch_merge_strategy(self):
        result = verify_compositional(
            RESIDUAL_MODEL,
            input_shapes={"x": ("batch", 64)},
            strategy="branch_merge",
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)
        assert result.decomposition_strategy == DecompositionStrategy.BRANCH_MERGE

    def test_deep_mlp_auto_decomposition(self):
        graph = extract_computation_graph(DEEP_MLP)
        modules = decompose_graph(graph, strategy="auto")
        assert len(modules) >= 1
        total = sum(m.graph.num_steps for m in modules)
        assert total == graph.num_steps


# ═══════════════════════════════════════════════════════════════════════════════
# Additional tests: SubModule, CompositionalResult, utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubModuleFingerprint:
    def test_fingerprint_deterministic(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        m1 = decompose_graph(graph, strategy="auto")
        m2 = decompose_graph(graph, strategy="auto")
        for a, b in zip(m1, m2):
            assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_differs_for_different_models(self):
        g1 = extract_computation_graph(SIMPLE_LINEAR)
        g2 = extract_computation_graph(THREE_LAYER_MLP)
        m1 = decompose_graph(g1, strategy="auto")
        m2 = decompose_graph(g2, strategy="auto")
        # Different models should produce different fingerprints
        # (use last module to avoid block_0 collisions on similar first layers)
        assert m1[-1].fingerprint() != m2[-1].fingerprint()


class TestCompositionalResultPretty:
    def test_pretty_output(self):
        result = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        pretty = result.pretty()
        assert "SAFE" in pretty or "UNSAFE" in pretty
        assert "Sub-modules" in pretty

    def test_unsafe_pretty(self):
        result = verify_compositional(
            SHAPE_MISMATCH_MODEL,
            input_shapes={"x": ("batch", 10)},
            measure_monolithic=False,
        )
        pretty = result.pretty()
        assert "UNSAFE" in pretty


class TestDecomposeAndSummarize:
    def test_summarize_mlp(self):
        summary = decompose_and_summarize(
            TWO_LAYER_MLP,
            strategy="auto",
            input_shapes={"x": ("batch", 784)},
        )
        assert "MLP" in summary
        assert "Total steps" in summary

    def test_summarize_parse_error(self):
        summary = decompose_and_summarize("not valid python")
        assert "Parse error" in summary or "error" in summary.lower()


class TestConvFCVerification:
    def test_conv_fc_compositional_safe(self):
        result = verify_compositional(
            CONV_FC_MODEL,
            input_shapes={"x": ("batch", 1, 28, 28)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)
        assert result.safe is True
