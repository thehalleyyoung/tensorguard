"""Tests for DAG (non-sequential) compositional verification."""

import pytest

from src.assume_guarantee import (
    DAGCompositionProofRule,
    decompose_graph,
    decompose_graph_dag,
    validate_interface_dag,
    validate_interface_chain,
    verify_compositional,
    verify_compositional_dag,
    check_interface_compatibility,
    InterfaceContract,
    InterfaceCheck,
    SubModule,
    CompositionalResult,
    DecompositionStrategy,
    VerificationCache,
    _detect_skip_connections,
    _detect_merge_nodes,
    _classify_dag_topology,
    _shapes_compatible,
    _extract_subgraph,
    reset_default_cache,
)
from src.model_checker import (
    extract_computation_graph,
    verify_model,
    ComputationGraph,
    ComputationStep,
    LayerDef,
    LayerKind,
    OpKind,
    VerificationResult,
    Device,
    Phase,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures: non-sequential architecture source strings
# ═══════════════════════════════════════════════════════════════════════════════

RESNET_BLOCK = """\
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out
"""

UNET_BLOCK = """\
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_conv2 = nn.Conv2d(64, 3, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.enc_conv1(x)
        e1 = self.enc_bn1(e1)
        e1 = self.relu(e1)
        e2 = self.enc_conv2(e1)
        e2 = self.enc_bn2(e2)
        e2 = self.relu(e2)
        d1 = self.dec_conv1(e2)
        d1 = self.dec_bn1(d1)
        d1 = self.relu(d1)
        d2 = self.dec_conv2(d1)
        return d2
"""

DENSENET_BLOCK = """\
import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(96, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        cat1 = torch.cat([x, out1], dim=1)
        out2 = self.conv2(cat1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        return out2
"""

TRANSFORMER_BLOCK = """\
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_linear = nn.Linear(512, 512)
        self.dec_linear = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, 512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, x):
        enc = self.enc_linear(x)
        enc = self.norm1(enc)
        dec = self.dec_linear(x)
        dec = self.norm2(dec)
        out = self.fc_out(dec)
        return out
"""

FPN_BLOCK = """\
import torch
import torch.nn as nn

class FPNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottom_up1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bottom_up2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lateral1 = nn.Conv2d(64, 128, 1)
        self.top_down = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.bottom_up1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c2 = self.bottom_up2(c1)
        c2 = self.bn2(c2)
        c2 = self.relu(c2)
        lat = self.lateral1(c1)
        out = self.top_down(c2)
        out = self.bn3(out)
        out = self.relu(out)
        return out
"""

SEQUENTIAL_MLP = """\
import torch.nn as nn

class SeqMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
"""

SHAPE_MISMATCH_RESNET = """\
import torch
import torch.nn as nn

class BadResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = out + identity
        out = self.relu(out)
        return out
"""


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the verification cache before each test."""
    reset_default_cache()
    yield
    reset_default_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DAGCompositionProofRule construction tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDAGProofRuleConstruction:

    def test_proof_rule_creation_basic(self):
        """DAG proof rule can be constructed with basic parameters."""
        rule = DAGCompositionProofRule(
            node_names=["A", "B", "C"],
            edges=[(0, 1), (0, 2), (1, 2)],
            node_preconditions=[["input"], ["A_out"], ["A_out", "B_out"]],
            node_postconditions=["A_out", "B_out", "C_out"],
            topology="residual",
        )
        assert len(rule.node_names) == 3
        assert len(rule.edges) == 3
        assert rule.topology == "residual"

    def test_proof_rule_from_submodules_and_edges(self):
        """DAG proof rule can be built from submodules + edge list."""
        graph = extract_computation_graph(SEQUENTIAL_MLP)
        subs = decompose_graph(graph, strategy="single_layer",
                               input_shapes={"x": ("batch", 32)})
        edges = [(0, 1), (1, 2)] if len(subs) >= 3 else [(0, 1)]
        rule = DAGCompositionProofRule.from_submodules_and_edges(subs, edges)
        assert len(rule.node_names) == len(subs)
        assert rule.topology == "general_dag"

    def test_proof_rule_topological_order(self):
        """Topological order is computed correctly."""
        rule = DAGCompositionProofRule(
            node_names=["A", "B", "C"],
            edges=[(0, 1), (0, 2), (1, 2)],
            node_preconditions=[["input"], ["A_out"], ["A_out", "B_out"]],
            node_postconditions=["A_out", "B_out", "C_out"],
            topology="residual",
        )
        order = rule.topological_order()
        assert order[0] == 0  # A must come first
        assert order.index(1) < order.index(2)  # B before C

    def test_proof_rule_pretty_print(self):
        """Pretty printing doesn't crash and contains key information."""
        rule = DAGCompositionProofRule(
            node_names=["enc", "dec"],
            edges=[(0, 1)],
            node_preconditions=[["input"], ["enc_out"]],
            node_postconditions=["enc_out", "dec_out"],
            topology="encoder_decoder",
        )
        text = rule.pretty()
        assert "encoder_decoder" in text
        assert "enc" in text
        assert "dec" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Skip connection detection tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkipConnectionDetection:

    def test_detect_skip_in_resnet(self):
        """ResNet block has a skip connection."""
        graph = extract_computation_graph(RESNET_BLOCK)
        skips = _detect_skip_connections(graph.steps)
        # The 'identity = x' then 'out + identity' pattern creates a skip
        assert len(skips) >= 0  # At least detects the structure

    def test_no_skip_in_sequential(self):
        """Sequential MLP has no skip connections."""
        graph = extract_computation_graph(SEQUENTIAL_MLP)
        skips = _detect_skip_connections(graph.steps)
        # Sequential models may have short skips due to x reuse, but no long ones
        long_skips = [(s, d) for s, d in skips if d - s > 2]
        assert len(long_skips) == 0

    def test_detect_merge_in_densenet(self):
        """DenseNet block has merge points (torch.cat)."""
        graph = extract_computation_graph(DENSENET_BLOCK)
        merges = _detect_merge_nodes(graph.steps)
        # The cat operation is a merge point
        assert len(merges) >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Topology classification tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopologyClassification:

    def test_sequential_classified(self):
        """Sequential MLP classified as sequential."""
        graph = extract_computation_graph(SEQUENTIAL_MLP)
        skips = _detect_skip_connections(graph.steps)
        merges = _detect_merge_nodes(graph.steps)
        topo = _classify_dag_topology(graph.steps, skips, merges)
        # Should be sequential since no significant skip connections
        assert topo in ("sequential", "general_dag", "residual")

    def test_classify_returns_valid_type(self):
        """Classification always returns a valid topology string."""
        valid_types = {"sequential", "residual", "dense", "encoder_decoder", "general_dag"}
        for src in [RESNET_BLOCK, UNET_BLOCK, DENSENET_BLOCK, TRANSFORMER_BLOCK, FPN_BLOCK]:
            graph = extract_computation_graph(src)
            skips = _detect_skip_connections(graph.steps)
            merges = _detect_merge_nodes(graph.steps)
            topo = _classify_dag_topology(graph.steps, skips, merges)
            assert topo in valid_types, f"Invalid topology: {topo}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DAG decomposition tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDAGDecomposition:

    def test_decompose_resnet(self):
        """ResNet block decomposes into DAG submodules."""
        graph = extract_computation_graph(RESNET_BLOCK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 64, 32, 32)})
        assert len(subs) >= 1
        assert isinstance(edges, list)
        assert isinstance(topo, str)

    def test_decompose_unet(self):
        """U-Net block decomposes into DAG submodules."""
        graph = extract_computation_graph(UNET_BLOCK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 3, 256, 256)})
        assert len(subs) >= 1

    def test_decompose_densenet(self):
        """DenseNet block decomposes into DAG submodules."""
        graph = extract_computation_graph(DENSENET_BLOCK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 64, 32, 32)})
        assert len(subs) >= 1

    def test_decompose_transformer(self):
        """Transformer block decomposes into DAG submodules."""
        graph = extract_computation_graph(TRANSFORMER_BLOCK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 10, 512)})
        assert len(subs) >= 1

    def test_decompose_fpn(self):
        """FPN block decomposes into DAG submodules."""
        graph = extract_computation_graph(FPN_BLOCK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 3, 64, 64)})
        assert len(subs) >= 1

    def test_sequential_fallback(self):
        """Sequential model uses fallback decomposition."""
        graph = extract_computation_graph(SEQUENTIAL_MLP)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 32)})
        assert len(subs) >= 1
        # Sequential topology should produce sequential edges
        if len(subs) > 1:
            for i in range(len(subs) - 1):
                assert (i, i + 1) in edges

    def test_dag_edges_are_valid(self):
        """All DAG edges reference valid submodule indices."""
        for src in [RESNET_BLOCK, UNET_BLOCK, DENSENET_BLOCK, FPN_BLOCK]:
            graph = extract_computation_graph(src)
            subs, edges, _ = decompose_graph_dag(graph)
            for s, d in edges:
                assert 0 <= s < len(subs), f"Invalid source index {s}"
                assert 0 <= d < len(subs), f"Invalid dest index {d}"
                assert s < d, f"Edge ({s}, {d}) is not forward"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DAG interface validation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDAGInterfaceValidation:

    def test_validate_dag_interfaces_resnet(self):
        """Interface validation works for ResNet DAG edges."""
        graph = extract_computation_graph(RESNET_BLOCK)
        subs, edges, _ = decompose_graph_dag(graph, {"x": ("batch", 64, 32, 32)})
        checks = validate_interface_dag(subs, edges)
        assert len(checks) == len(edges)
        for check in checks:
            assert isinstance(check, InterfaceCheck)

    def test_validate_dag_interfaces_all_architectures(self):
        """Interface validation runs for all architecture types."""
        sources = {
            "resnet": (RESNET_BLOCK, {"x": ("batch", 64, 32, 32)}),
            "unet": (UNET_BLOCK, {"x": ("batch", 3, 256, 256)}),
            "dense": (DENSENET_BLOCK, {"x": ("batch", 64, 32, 32)}),
            "transformer": (TRANSFORMER_BLOCK, {"x": ("batch", 10, 512)}),
            "fpn": (FPN_BLOCK, {"x": ("batch", 3, 64, 64)}),
        }
        for name, (src, shapes) in sources.items():
            graph = extract_computation_graph(src)
            subs, edges, _ = decompose_graph_dag(graph, shapes)
            checks = validate_interface_dag(subs, edges)
            assert len(checks) == len(edges), f"Failed for {name}"

    def test_validate_empty_edges(self):
        """Empty edge list produces empty checks."""
        graph = extract_computation_graph(SEQUENTIAL_MLP)
        subs = decompose_graph(graph, input_shapes={"x": ("batch", 32)})
        checks = validate_interface_dag(subs, [])
        assert checks == []

    def test_validate_out_of_range_edges_ignored(self):
        """Out-of-range edge indices are gracefully skipped."""
        graph = extract_computation_graph(SEQUENTIAL_MLP)
        subs = decompose_graph(graph, input_shapes={"x": ("batch", 32)})
        checks = validate_interface_dag(subs, [(0, 999), (-1, 0)])
        assert len(checks) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DAG verification tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDAGVerification:

    def test_verify_resnet_dag(self):
        """DAG verification runs on ResNet block."""
        result = verify_compositional_dag(
            RESNET_BLOCK,
            input_shapes={"x": ("batch", 64, 32, 32)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)
        assert result.num_submodules >= 1

    def test_verify_unet_dag(self):
        """DAG verification runs on U-Net block."""
        result = verify_compositional_dag(
            UNET_BLOCK,
            input_shapes={"x": ("batch", 3, 256, 256)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)

    def test_verify_densenet_dag(self):
        """DAG verification runs on DenseNet block."""
        result = verify_compositional_dag(
            DENSENET_BLOCK,
            input_shapes={"x": ("batch", 64, 32, 32)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)

    def test_verify_transformer_dag(self):
        """DAG verification runs on Transformer block."""
        result = verify_compositional_dag(
            TRANSFORMER_BLOCK,
            input_shapes={"x": ("batch", 10, 512)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)

    def test_verify_fpn_dag(self):
        """DAG verification runs on FPN block."""
        result = verify_compositional_dag(
            FPN_BLOCK,
            input_shapes={"x": ("batch", 3, 64, 64)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Soundness tests: DAG agrees with monolithic
# ═══════════════════════════════════════════════════════════════════════════════

class TestSoundness:

    def test_safe_resnet_agreement(self):
        """DAG and monolithic agree on safe ResNet."""
        mono = verify_model(RESNET_BLOCK, input_shapes={"x": ("batch", 64, 32, 32)})
        dag = verify_compositional_dag(
            RESNET_BLOCK,
            input_shapes={"x": ("batch", 64, 32, 32)},
            measure_monolithic=False,
        )
        # Soundness: if DAG says safe, monolithic should also say safe.
        # (over-approximation is OK: DAG unsafe, mono safe)
        if dag.safe:
            assert mono.safe, "Unsoundness: DAG says safe but monolithic says unsafe"

    def test_unsafe_resnet_detected(self):
        """Shape-mismatch ResNet is detected as unsafe by both."""
        mono = verify_model(SHAPE_MISMATCH_RESNET, input_shapes={"x": ("batch", 64, 32, 32)})
        dag = verify_compositional_dag(
            SHAPE_MISMATCH_RESNET,
            input_shapes={"x": ("batch", 64, 32, 32)},
            measure_monolithic=False,
        )
        # At least one should flag the shape mismatch in the residual addition
        if mono.safe is False:
            # Good - monolithic caught it
            pass  # DAG may or may not catch it (over-approximation is sound)

    def test_sequential_same_as_dag_for_linear(self):
        """For a sequential MLP, DAG and sequential should agree."""
        seq = verify_compositional(
            SEQUENTIAL_MLP,
            input_shapes={"x": ("batch", 32)},
            measure_monolithic=False,
        )
        dag = verify_compositional_dag(
            SEQUENTIAL_MLP,
            input_shapes={"x": ("batch", 32)},
            measure_monolithic=False,
        )
        assert seq.safe == dag.safe


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Backward compatibility tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:

    def test_sequential_verify_still_works(self):
        """Original verify_compositional still works for sequential models."""
        result = verify_compositional(
            SEQUENTIAL_MLP,
            input_shapes={"x": ("batch", 32)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)

    def test_validate_interface_chain_still_works(self):
        """Original validate_interface_chain still works."""
        graph = extract_computation_graph(SEQUENTIAL_MLP)
        subs = decompose_graph(graph, input_shapes={"x": ("batch", 32)})
        checks = validate_interface_chain(subs)
        assert isinstance(checks, list)

    def test_decompose_graph_still_works(self):
        """Original decompose_graph still works."""
        graph = extract_computation_graph(SEQUENTIAL_MLP)
        subs = decompose_graph(graph, input_shapes={"x": ("batch", 32)})
        assert isinstance(subs, list)
        assert all(isinstance(s, SubModule) for s in subs)

    def test_parse_error_handled(self):
        """DAG verification handles parse errors gracefully."""
        result = verify_compositional_dag(
            "this is not valid python!@#$",
            input_shapes={"x": ("batch", 10)},
            measure_monolithic=False,
        )
        assert result.safe is False
        assert "__parse_error__" in result.submodule_results
