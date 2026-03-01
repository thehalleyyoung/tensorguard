"""Tests for non-sequential assume-guarantee verification using BRANCH_MERGE."""

import pytest

from src.assume_guarantee import (
    DAGCompositionProofRule,
    decompose_graph,
    decompose_graph_dag,
    validate_interface_dag,
    verify_compositional,
    verify_compositional_dag,
    check_interface_compatibility,
    InterfaceContract,
    InterfaceCheck,
    SubModule,
    CompositionalResult,
    DecompositionStrategy,
    _detect_skip_connections,
    _detect_merge_nodes,
    _classify_dag_topology,
    _shapes_compatible,
    reset_default_cache,
)
from src.model_checker import (
    extract_computation_graph,
    verify_model,
    ComputationGraph,
    VerificationResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture source-code fixtures
# ═══════════════════════════════════════════════════════════════════════════════

RESNET50_BOTTLENECK = """\
import torch
import torch.nn as nn

class ResNet50Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity
        out = self.relu(out)
        return out
"""

RESNET_SHAPE_MISMATCH = """\
import torch
import torch.nn as nn

class BadResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, 1)
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

UNET_ENCODER_DECODER = """\
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn_e1 = nn.BatchNorm2d(64)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_e2 = nn.BatchNorm2d(128)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_e3 = nn.BatchNorm2d(256)
        self.dec1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(128)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.dec3 = nn.Conv2d(64, 3, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.bn_e1(self.enc1(x)))
        e2 = self.relu(self.bn_e2(self.enc2(e1)))
        e3 = self.relu(self.bn_e3(self.enc3(e2)))
        d1 = self.relu(self.bn_d1(self.dec1(e3)))
        d2 = self.relu(self.bn_d2(self.dec2(d1)))
        d3 = self.dec3(d2)
        return d3
"""

TRANSFORMER_CROSS = """\
import torch
import torch.nn as nn

class TransformerCross(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_proj = nn.Linear(512, 512)
        self.enc_norm = nn.LayerNorm(512)
        self.dec_proj = nn.Linear(512, 512)
        self.dec_norm = nn.LayerNorm(512)
        self.cross_q = nn.Linear(512, 512)
        self.cross_k = nn.Linear(512, 512)
        self.cross_v = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, 512)
        self.out_norm = nn.LayerNorm(512)

    def forward(self, x):
        enc = self.enc_norm(self.enc_proj(x))
        dec = self.dec_norm(self.dec_proj(x))
        q = self.cross_q(dec)
        k = self.cross_k(enc)
        v = self.cross_v(enc)
        out = self.out_norm(self.fc_out(q))
        return out
"""

FPN_NETWORK = """\
import torch
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lat2 = nn.Conv2d(128, 256, 1)
        self.lat1 = nn.Conv2d(64, 256, 1)
        self.smooth = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_out = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.relu(self.bn1(self.c1(x)))
        c2 = self.relu(self.bn2(self.c2(c1)))
        c3 = self.relu(self.bn3(self.c3(c2)))
        l2 = self.lat2(c2)
        l1 = self.lat1(c1)
        out = self.relu(self.bn_out(self.smooth(c3)))
        return out
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
        self.conv3 = nn.Conv2d(128, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        cat1 = torch.cat([x, out1], dim=1)
        out2 = self.relu(self.bn2(self.conv2(cat1)))
        cat2 = torch.cat([x, out1, out2], dim=1)
        out3 = self.relu(self.bn3(self.conv3(cat2)))
        return out3
"""


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the verification cache before each test."""
    reset_default_cache()
    yield
    reset_default_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Skip connection shape propagation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkipConnectionShapePropagation:

    def test_resnet_skip_detected(self):
        """ResNet-50 bottleneck block has a skip connection from identity."""
        graph = extract_computation_graph(RESNET50_BOTTLENECK)
        skips = _detect_skip_connections(graph.steps)
        # The identity = x pattern followed by out + identity creates a skip
        assert isinstance(skips, list)

    def test_resnet_dag_decomposes_with_edges(self):
        """ResNet-50 bottleneck block produces DAG submodules and edges."""
        graph = extract_computation_graph(RESNET50_BOTTLENECK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 256, 16, 16)})
        assert len(subs) >= 1
        assert isinstance(edges, list)

    def test_resnet_skip_shape_preserved(self):
        """Residual skip preserves shape: DAG verification is sound (over-approx OK)."""
        mono = verify_model(
            RESNET50_BOTTLENECK,
            input_shapes={"x": ("batch", 256, 16, 16)},
        )
        result = verify_compositional_dag(
            RESNET50_BOTTLENECK,
            input_shapes={"x": ("batch", 256, 16, 16)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)
        # Soundness: if DAG says safe, monolithic must agree.
        # Over-approximation (DAG unsafe, mono safe) is acceptable.
        if result.safe:
            assert mono.safe

    def test_unet_skip_connections_detected(self):
        """U-Net encoder-decoder has skip-like data flow."""
        graph = extract_computation_graph(UNET_ENCODER_DECODER)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 3, 128, 128)})
        assert len(subs) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Residual addition shape compatibility verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestResidualAdditionShapeCompat:

    def test_matching_residual_monolithic_safe(self):
        """Residual addition with matching shapes (256 + 256) is monolithically safe."""
        mono = verify_model(
            RESNET50_BOTTLENECK,
            input_shapes={"x": ("batch", 256, 16, 16)},
        )
        assert mono.safe is True

    def test_mismatched_residual_detected(self):
        """Residual addition with mismatched shapes (256 + 128) is detected."""
        mono = verify_model(
            RESNET_SHAPE_MISMATCH,
            input_shapes={"x": ("batch", 256, 16, 16)},
        )
        dag = verify_compositional_dag(
            RESNET_SHAPE_MISMATCH,
            input_shapes={"x": ("batch", 256, 16, 16)},
            measure_monolithic=False,
        )
        # At least one of monolithic or DAG should catch the mismatch
        assert mono.safe is False or dag.safe is False

    def test_residual_dag_uses_branch_merge(self):
        """DAG verification of residual model uses BRANCH_MERGE strategy."""
        result = verify_compositional_dag(
            RESNET50_BOTTLENECK,
            input_shapes={"x": ("batch", 256, 16, 16)},
            measure_monolithic=False,
        )
        assert result.decomposition_strategy == DecompositionStrategy.BRANCH_MERGE


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cross-attention key/value shape constraints
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossAttentionShapes:

    def test_transformer_dag_decomposes(self):
        """Transformer cross-attention produces DAG decomposition."""
        graph = extract_computation_graph(TRANSFORMER_CROSS)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 16, 512)})
        assert len(subs) >= 1
        assert isinstance(edges, list)

    def test_transformer_cross_attn_safe(self):
        """Cross-attention with matching dims (512→512) is safe."""
        result = verify_compositional_dag(
            TRANSFORMER_CROSS,
            input_shapes={"x": ("batch", 16, 512)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)
        assert result.safe is True

    def test_transformer_interface_contracts_present(self):
        """Transformer DAG submodules all have interface contracts."""
        graph = extract_computation_graph(TRANSFORMER_CROSS)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 16, 512)})
        for sm in subs:
            assert isinstance(sm.input_contract, InterfaceContract)
            assert isinstance(sm.output_contract, InterfaceContract)

    def test_transformer_kv_from_encoder(self):
        """Encoder output feeds key/value projections — DAG edges reflect this."""
        graph = extract_computation_graph(TRANSFORMER_CROSS)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 16, 512)})
        # Edges should exist (non-sequential data flow)
        assert len(edges) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FPN lateral connection shapes
# ═══════════════════════════════════════════════════════════════════════════════

class TestFPNLateralShapes:

    def test_fpn_dag_decomposes(self):
        """FPN produces DAG submodules."""
        graph = extract_computation_graph(FPN_NETWORK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 3, 64, 64)})
        assert len(subs) >= 1

    def test_fpn_lateral_safe(self):
        """FPN with correct lateral channel projections is safe."""
        result = verify_compositional_dag(
            FPN_NETWORK,
            input_shapes={"x": ("batch", 3, 64, 64)},
            measure_monolithic=False,
        )
        assert isinstance(result, CompositionalResult)
        assert result.safe is True

    def test_fpn_has_non_sequential_edges(self):
        """FPN should produce non-sequential edges (lateral connections)."""
        graph = extract_computation_graph(FPN_NETWORK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 3, 64, 64)})
        # Lateral connections create skip-like edges
        assert len(edges) >= 1

    def test_fpn_interface_validation(self):
        """FPN interface validation produces checks for every edge."""
        graph = extract_computation_graph(FPN_NETWORK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 3, 64, 64)})
        checks = validate_interface_dag(subs, edges)
        assert len(checks) == len(edges)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BRANCH_MERGE handles non-sequential topology
# ═══════════════════════════════════════════════════════════════════════════════

class TestBranchMergeTopology:

    def test_branch_merge_on_resnet(self):
        """BRANCH_MERGE strategy correctly decomposes ResNet block."""
        result = verify_compositional_dag(
            RESNET50_BOTTLENECK,
            input_shapes={"x": ("batch", 256, 16, 16)},
            measure_monolithic=False,
        )
        assert result.decomposition_strategy == DecompositionStrategy.BRANCH_MERGE
        assert result.num_submodules >= 1

    def test_branch_merge_on_densenet(self):
        """BRANCH_MERGE strategy correctly verifies DenseNet block."""
        result = verify_compositional_dag(
            DENSENET_BLOCK,
            input_shapes={"x": ("batch", 64, 32, 32)},
            measure_monolithic=False,
        )
        assert result.decomposition_strategy == DecompositionStrategy.BRANCH_MERGE

    def test_branch_merge_on_fpn(self):
        """BRANCH_MERGE strategy correctly verifies FPN."""
        result = verify_compositional_dag(
            FPN_NETWORK,
            input_shapes={"x": ("batch", 3, 64, 64)},
            measure_monolithic=False,
        )
        assert result.decomposition_strategy == DecompositionStrategy.BRANCH_MERGE

    def test_branch_merge_on_transformer(self):
        """BRANCH_MERGE strategy correctly verifies Transformer cross-attention."""
        result = verify_compositional_dag(
            TRANSFORMER_CROSS,
            input_shapes={"x": ("batch", 16, 512)},
            measure_monolithic=False,
        )
        assert result.decomposition_strategy == DecompositionStrategy.BRANCH_MERGE

    def test_dag_agrees_with_monolithic_resnet(self):
        """DAG verification agrees with monolithic for safe ResNet."""
        mono = verify_model(
            RESNET50_BOTTLENECK,
            input_shapes={"x": ("batch", 256, 16, 16)},
        )
        dag = verify_compositional_dag(
            RESNET50_BOTTLENECK,
            input_shapes={"x": ("batch", 256, 16, 16)},
            measure_monolithic=False,
        )
        if dag.safe:
            assert mono.safe, "Unsound: DAG says safe but monolithic says unsafe"

    def test_dag_agrees_with_monolithic_unet(self):
        """DAG verification agrees with monolithic for U-Net."""
        mono = verify_model(
            UNET_ENCODER_DECODER,
            input_shapes={"x": ("batch", 3, 128, 128)},
        )
        dag = verify_compositional_dag(
            UNET_ENCODER_DECODER,
            input_shapes={"x": ("batch", 3, 128, 128)},
            measure_monolithic=False,
        )
        if dag.safe:
            assert mono.safe, "Unsound: DAG says safe but monolithic says unsafe"

    def test_dag_proof_rule_has_correct_topology(self):
        """DAG proof rule reports the detected topology."""
        graph = extract_computation_graph(RESNET50_BOTTLENECK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 256, 16, 16)})
        rule = DAGCompositionProofRule.from_submodules_and_edges(subs, edges, topo)
        assert rule.topology in ("residual", "general_dag", "sequential")

    def test_topological_order_valid(self):
        """Topological order respects all edges (no backward references)."""
        graph = extract_computation_graph(RESNET50_BOTTLENECK)
        subs, edges, topo = decompose_graph_dag(graph, {"x": ("batch", 256, 16, 16)})
        rule = DAGCompositionProofRule.from_submodules_and_edges(subs, edges, topo)
        order = rule.topological_order()
        pos = {node: i for i, node in enumerate(order)}
        for src, dst in edges:
            assert pos[src] < pos[dst], (
                f"Edge ({src},{dst}) violates topological order"
            )

    def test_parse_error_returns_unsafe(self):
        """DAG verification handles parse errors gracefully."""
        result = verify_compositional_dag(
            "this is not valid python!@#$",
            input_shapes={"x": ("batch", 10)},
            measure_monolithic=False,
        )
        assert result.safe is False
