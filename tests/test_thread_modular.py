"""Tests for thread-modular verification of TorchDynamo graph-break composition."""

import pytest

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    LayerDef,
    LayerKind,
    OpKind,
)
from src.dynamo_gap_analysis import GapCategory
from src.thread_modular import (
    ShapeEnv,
    MonotonicityKind,
    MonotonicityConstraint,
    EnvironmentAssumption,
    SubgraphContract,
    InterBreakTransformer,
    TransformerKind,
    CompositionVerdict,
    CompositionResult,
    GapDetail,
    NonMonotonicPattern,
    ThreadModularVerifier,
    infer_contract,
    infer_inter_break_transformer,
    check_contract_chain,
    detect_non_monotonic_patterns,
    verify_thread_modular,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_graph(class_name, steps=None, inputs=None, outputs=None,
                layers=None, features=None):
    g = ComputationGraph(class_name=class_name)
    g.steps = steps or []
    g.input_names = inputs or []
    g.output_names = outputs or []
    if layers:
        g.layers = layers
    g.dynamic_features = features or {}
    return g


def _make_step(op, inputs, output, params=None, layer_ref=None):
    return ComputationStep(
        op=op, inputs=inputs, output=output,
        params=params or {}, layer_ref=layer_ref,
    )


def _make_linear_layer(name, in_f, out_f):
    return LayerDef(
        attr_name=name, kind=LayerKind.LINEAR,
        in_features=in_f, out_features=out_f,
    )


def _make_simple_linear_subgraph(name, inp_name, out_name, in_f, out_f):
    """Create a subgraph with a single linear layer."""
    layer = _make_linear_layer("fc", in_f, out_f)
    return _make_graph(
        name,
        steps=[_make_step(OpKind.LAYER_CALL, [inp_name], out_name, layer_ref="fc")],
        inputs=[inp_name],
        outputs=[out_name],
        layers={"fc": layer},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ShapeEnv tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapeEnv:
    def test_compatible_same_shapes(self):
        e1 = ShapeEnv(shapes={"x": (2, 3), "y": (4,)})
        e2 = ShapeEnv(shapes={"x": (2, 3)})
        assert e1.compatible_with(e2)

    def test_incompatible_different_dims(self):
        e1 = ShapeEnv(shapes={"x": (2, 3)})
        e2 = ShapeEnv(shapes={"x": (2, 4)})
        assert not e1.compatible_with(e2)

    def test_compatible_with_symbolic(self):
        e1 = ShapeEnv(shapes={"x": ("batch", 3)})
        e2 = ShapeEnv(shapes={"x": ("batch", 3)})
        assert e1.compatible_with(e2)

    def test_incompatible_rank_mismatch(self):
        e1 = ShapeEnv(shapes={"x": (2, 3)})
        e2 = ShapeEnv(shapes={"x": (2, 3, 4)})
        assert not e1.compatible_with(e2)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Contract inference tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestContractInference:
    def test_infer_linear_contract(self):
        sg = _make_simple_linear_subgraph("Net", "x", "out", 64, 32)
        contract = infer_contract(sg, 0, {"x": ("batch", 64)})

        assert contract.subgraph_index == 0
        assert "x" in contract.precondition.shapes
        assert contract.precondition.shapes["x"] == ("batch", 64)
        assert "out" in contract.postcondition.shapes
        assert contract.postcondition.shapes["out"] == ("batch", 32)

    def test_infer_activation_preserves_shape(self):
        sg = _make_graph(
            "ActNet",
            steps=[_make_step(OpKind.ACTIVATION, ["x"], "y")],
            inputs=["x"],
            outputs=["y"],
        )
        contract = infer_contract(sg, 0, {"x": ("batch", 128)})
        assert contract.postcondition.shapes["y"] == ("batch", 128)

    def test_infer_reshape_contract(self):
        sg = _make_graph(
            "ReshapeNet",
            steps=[
                _make_step(OpKind.RESHAPE, ["x"], "y",
                           params={"target_shape": ("batch", 4, 16)}),
            ],
            inputs=["x"],
            outputs=["y"],
        )
        contract = infer_contract(sg, 0, {"x": ("batch", 64)})
        assert contract.postcondition.shapes["y"] == ("batch", 4, 16)
        assert GapCategory.CONDITIONAL_RESHAPE in contract.gap_categories

    def test_infer_multi_step_contract(self):
        fc1 = _make_linear_layer("fc1", 64, 128)
        fc2 = _make_linear_layer("fc2", 128, 10)
        sg = _make_graph(
            "TwoLayer",
            steps=[
                _make_step(OpKind.LAYER_CALL, ["x"], "h", layer_ref="fc1"),
                _make_step(OpKind.ACTIVATION, ["h"], "h_relu"),
                _make_step(OpKind.LAYER_CALL, ["h_relu"], "out", layer_ref="fc2"),
            ],
            inputs=["x"],
            outputs=["out"],
            layers={"fc1": fc1, "fc2": fc2},
        )
        contract = infer_contract(sg, 0, {"x": ("batch", 64)})
        assert contract.postcondition.shapes["out"] == ("batch", 10)

    def test_infer_contract_symbolic_default(self):
        """Contract inference with no input shapes uses symbolic defaults."""
        sg = _make_graph(
            "Net",
            steps=[_make_step(OpKind.ACTIVATION, ["x"], "y")],
            inputs=["x"],
            outputs=["y"],
        )
        contract = infer_contract(sg, 0)
        assert "x" in contract.precondition.shapes
        assert contract.precondition.shapes["x"] == ("batch", "dim")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Inter-break abstract transformer tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInterBreakTransformer:
    def test_identity_transformer(self):
        t = InterBreakTransformer(
            kind=TransformerKind.IDENTITY, break_index=0,
        )
        env = ShapeEnv(shapes={"x": (2, 3)})
        result = t.apply(env)
        assert result.shapes == {"x": (2, 3)}

    def test_conservative_transformer(self):
        t = InterBreakTransformer(
            kind=TransformerKind.CONSERVATIVE, break_index=0,
        )
        env = ShapeEnv(shapes={"x": (2, 3)})
        result = t.apply(env)
        # All concrete dims become symbolic
        assert all(isinstance(d, str) for d in result.shapes["x"])

    def test_dimension_routing_transformer(self):
        t = InterBreakTransformer(
            kind=TransformerKind.DIMENSION_ROUTING, break_index=0,
            preserves_batch_dim=True,
        )
        env = ShapeEnv(shapes={"x": (8, 16, 32)})
        result = t.apply(env)
        assert result.shapes["x"][0] == 8  # batch preserved
        assert isinstance(result.shapes["x"][1], str)  # routed

    def test_infer_identity_transformer(self):
        """Two subgraphs with clean output→input chain."""
        sg1 = _make_graph(
            "G1",
            steps=[_make_step(OpKind.ACTIVATION, ["x"], "y")],
            inputs=["x"], outputs=["y"],
        )
        sg2 = _make_graph(
            "G2",
            steps=[_make_step(OpKind.ACTIVATION, ["y"], "z")],
            inputs=["y"], outputs=["z"],
        )
        c1 = infer_contract(sg1, 0, {"x": ("batch", 64)})
        c2 = infer_contract(sg2, 1, {"y": ("batch", 64)})
        t = infer_inter_break_transformer(sg1, sg2, c1, c2, 0)
        assert t.kind == TransformerKind.IDENTITY

    def test_infer_conservative_transformer_external_input(self):
        """Subgraph 2 has input not from subgraph 1's outputs."""
        sg1 = _make_graph(
            "G1",
            steps=[_make_step(OpKind.ACTIVATION, ["x"], "y")],
            inputs=["x"], outputs=["y"],
        )
        sg2 = _make_graph(
            "G2",
            steps=[_make_step(OpKind.ACTIVATION, ["z"], "w")],
            inputs=["z"], outputs=["w"],
        )
        c1 = infer_contract(sg1, 0, {"x": ("batch", 64)})
        c2 = infer_contract(sg2, 1, {"z": ("batch", 64)})
        t = infer_inter_break_transformer(sg1, sg2, c1, c2, 0)
        assert t.kind == TransformerKind.CONSERVATIVE


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Composition soundness checking tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompositionSoundness:
    def test_sound_identity_chain(self):
        pre_i = ShapeEnv(shapes={"x": ("batch", 64)})
        post_i = ShapeEnv(shapes={"y": ("batch", 64)})
        pre_j = ShapeEnv(shapes={"y": ("batch", 64)})
        post_j = ShapeEnv(shapes={"z": ("batch", 32)})
        c_i = SubgraphContract(0, pre_i, post_i,
                               EnvironmentAssumption(pre_i, post_i))
        c_j = SubgraphContract(1, pre_j, post_j,
                               EnvironmentAssumption(pre_j, post_j))
        t = InterBreakTransformer(kind=TransformerKind.IDENTITY, break_index=0)
        sound, gaps = check_contract_chain(c_i, c_j, t)
        assert sound
        assert len(gaps) == 0

    def test_gap_on_conservative_transformer(self):
        pre_i = ShapeEnv(shapes={"x": ("batch", 64)})
        post_i = ShapeEnv(shapes={"y": ("batch", 64)})
        pre_j = ShapeEnv(shapes={"y": ("batch", 64)})
        post_j = ShapeEnv(shapes={"z": ("batch", 32)})
        c_i = SubgraphContract(0, pre_i, post_i,
                               EnvironmentAssumption(pre_i, post_i))
        c_j = SubgraphContract(1, pre_j, post_j,
                               EnvironmentAssumption(pre_j, post_j))
        t = InterBreakTransformer(
            kind=TransformerKind.CONSERVATIVE, break_index=0,
        )
        sound, gaps = check_contract_chain(c_i, c_j, t)
        assert not sound
        assert len(gaps) >= 1

    def test_gap_on_dimension_mismatch(self):
        pre_i = ShapeEnv(shapes={"x": ("batch", 64)})
        post_i = ShapeEnv(shapes={"y": (2, 32)})
        pre_j = ShapeEnv(shapes={"y": (2, 64)})  # expects 64 but gets 32
        post_j = ShapeEnv(shapes={"z": (2, 10)})
        c_i = SubgraphContract(0, pre_i, post_i,
                               EnvironmentAssumption(pre_i, post_i))
        c_j = SubgraphContract(1, pre_j, post_j,
                               EnvironmentAssumption(pre_j, post_j))
        t = InterBreakTransformer(kind=TransformerKind.IDENTITY, break_index=0)
        sound, gaps = check_contract_chain(c_i, c_j, t)
        assert not sound
        assert any(g.category == GapCategory.CONSTRAINT_CHAIN for g in gaps)

    def test_gap_on_rank_mismatch(self):
        pre_i = ShapeEnv(shapes={"x": ("batch", 64)})
        post_i = ShapeEnv(shapes={"y": ("batch", 8, 8)})  # 3D
        pre_j = ShapeEnv(shapes={"y": ("batch", 64)})       # expects 2D
        post_j = ShapeEnv(shapes={"z": ("batch", 10)})
        c_i = SubgraphContract(0, pre_i, post_i,
                               EnvironmentAssumption(pre_i, post_i))
        c_j = SubgraphContract(1, pre_j, post_j,
                               EnvironmentAssumption(pre_j, post_j))
        t = InterBreakTransformer(kind=TransformerKind.IDENTITY, break_index=0)
        sound, gaps = check_contract_chain(c_i, c_j, t)
        assert not sound
        assert any(g.category == GapCategory.NON_MONOTONIC for g in gaps)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Non-monotonic pattern detection tests (false-negative categories)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNonMonotonicPatterns:
    def test_shape_inversion_pattern(self):
        """Category 1: Non-monotonic constraint — reshape in both subgraphs."""
        sg1 = _make_graph("G1", steps=[
            _make_step(OpKind.RESHAPE, ["x"], "y",
                       params={"target_shape": ("batch", 4, 16)}),
        ], inputs=["x"], outputs=["y"])
        sg2 = _make_graph("G2", steps=[
            _make_step(OpKind.RESHAPE, ["y"], "z",
                       params={"target_shape": ("batch", 64)}),
        ], inputs=["y"], outputs=["z"])
        c1 = infer_contract(sg1, 0, {"x": ("batch", 64)})
        c2 = infer_contract(sg2, 1, {"y": ("batch", 4, 16)})
        patterns = detect_non_monotonic_patterns(sg1, sg2, c1, c2)
        assert any(p[0] == NonMonotonicPattern.SHAPE_INVERSION for p in patterns)

    def test_dynamic_routing_pattern(self):
        """Category 2: Dynamic routing — subscript/where op."""
        sg1 = _make_graph("G1", steps=[
            _make_step(OpKind.SUBSCRIPT, ["x"], "y"),
        ], inputs=["x"], outputs=["y"])
        sg2 = _make_graph("G2", steps=[
            _make_step(OpKind.LAYER_CALL, ["y"], "z"),
        ], inputs=["y"], outputs=["z"])
        c1 = infer_contract(sg1, 0)
        c2 = infer_contract(sg2, 1)
        patterns = detect_non_monotonic_patterns(sg1, sg2, c1, c2)
        assert any(p[0] == NonMonotonicPattern.DIMENSION_ROUTING for p in patterns)

    def test_accumulator_pattern(self):
        """Category 3: Accumulator — cat/stack ops."""
        sg1 = _make_graph("G1", steps=[
            _make_step(OpKind.CAT, ["x", "y"], "z", params={"dim": 0}),
        ], inputs=["x", "y"], outputs=["z"])
        sg2 = _make_graph("G2", steps=[
            _make_step(OpKind.MATMUL, ["z", "w"], "out"),
        ], inputs=["z", "w"], outputs=["out"])
        c1 = infer_contract(sg1, 0)
        c2 = infer_contract(sg2, 1)
        patterns = detect_non_monotonic_patterns(sg1, sg2, c1, c2)
        assert any(p[0] == NonMonotonicPattern.ACCUMULATOR for p in patterns)

    def test_conditional_reshape_pattern(self):
        """Category 4: Conditional reshape — symbolic target shape."""
        sg1 = _make_graph("G1", steps=[
            _make_step(OpKind.RESHAPE, ["x"], "y",
                       params={"target_shape": ("batch", "N", "H")}),
        ], inputs=["x"], outputs=["y"])
        sg2 = _make_graph("G2", steps=[
            _make_step(OpKind.LAYER_CALL, ["y"], "z"),
        ], inputs=["y"], outputs=["z"])
        c1 = infer_contract(sg1, 0)
        c2 = infer_contract(sg2, 1)
        patterns = detect_non_monotonic_patterns(sg1, sg2, c1, c2)
        assert any(p[0] == NonMonotonicPattern.CONDITIONAL_RESHAPE for p in patterns)

    def test_data_dependent_dim_pattern(self):
        """Category 5: Data-dependent dimension selection."""
        sg1 = _make_graph("G1", steps=[
            _make_step(OpKind.SUBSCRIPT, ["x"], "y"),
        ], inputs=["x"], outputs=["y"])
        sg2 = _make_graph("G2", steps=[
            _make_step(OpKind.MATMUL, ["y", "w"], "z"),
        ], inputs=["y", "w"], outputs=["z"])
        c1 = infer_contract(sg1, 0)
        c2 = infer_contract(sg2, 1)
        patterns = detect_non_monotonic_patterns(sg1, sg2, c1, c2)
        assert any(p[0] == NonMonotonicPattern.DATA_DEPENDENT_DIM for p in patterns)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ThreadModularVerifier integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestThreadModularVerifier:
    def test_single_subgraph_monolithic(self):
        sg = _make_simple_linear_subgraph("Net", "x", "y", 64, 32)
        result = verify_thread_modular([sg], {"x": ("batch", 64)})
        assert result.verdict == CompositionVerdict.MONOLITHIC_SAFE
        assert result.num_subgraphs == 1

    def test_two_compatible_subgraphs_verified(self):
        """Two subgraphs with clean output→input chain should verify."""
        sg1 = _make_graph(
            "G1",
            steps=[_make_step(OpKind.ACTIVATION, ["x"], "y")],
            inputs=["x"], outputs=["y"],
        )
        sg2 = _make_graph(
            "G2",
            steps=[_make_step(OpKind.ACTIVATION, ["y"], "z")],
            inputs=["y"], outputs=["z"],
        )
        result = verify_thread_modular(
            [sg1, sg2], {"x": ("batch", 64)}
        )
        assert result.verdict == CompositionVerdict.COMPOSITION_VERIFIED
        assert len(result.gaps) == 0

    def test_three_compatible_subgraphs_verified(self):
        """Three subgraphs chaining cleanly."""
        sgs = []
        names = [("x", "y"), ("y", "z"), ("z", "w")]
        for i, (inp, out) in enumerate(names):
            sgs.append(_make_graph(
                f"G{i}",
                steps=[_make_step(OpKind.ACTIVATION, [inp], out)],
                inputs=[inp], outputs=[out],
            ))
        result = verify_thread_modular(sgs, {"x": ("batch", 64)})
        assert result.verdict == CompositionVerdict.COMPOSITION_VERIFIED

    def test_gap_detected_external_input(self):
        """Subgraph 2 has external input → gap detected."""
        sg1 = _make_graph(
            "G1",
            steps=[_make_step(OpKind.ACTIVATION, ["x"], "y")],
            inputs=["x"], outputs=["y"],
        )
        sg2 = _make_graph(
            "G2",
            steps=[_make_step(OpKind.ACTIVATION, ["ext_input"], "z")],
            inputs=["ext_input"], outputs=["z"],
        )
        result = verify_thread_modular(
            [sg1, sg2], {"x": ("batch", 64)}
        )
        assert result.verdict == CompositionVerdict.GAP_DETECTED
        assert len(result.gaps) >= 1

    def test_gap_detected_reshape_across_break(self):
        """Reshape in both subgraphs → non-monotonic gap."""
        sg1 = _make_graph("G1", steps=[
            _make_step(OpKind.RESHAPE, ["x"], "y",
                       params={"target_shape": ("batch", 4, 16)}),
        ], inputs=["x"], outputs=["y"])
        sg2 = _make_graph("G2", steps=[
            _make_step(OpKind.RESHAPE, ["y"], "z",
                       params={"target_shape": ("batch", 64)}),
        ], inputs=["y"], outputs=["z"])
        result = verify_thread_modular(
            [sg1, sg2], {"x": ("batch", 64)}
        )
        assert result.verdict == CompositionVerdict.GAP_DETECTED

    def test_empty_subgraphs(self):
        result = verify_thread_modular([], {})
        assert result.verdict == CompositionVerdict.MONOLITHIC_SAFE
        assert result.num_subgraphs == 0

    def test_composition_result_to_dict(self):
        sg = _make_simple_linear_subgraph("Net", "x", "y", 64, 32)
        result = verify_thread_modular([sg], {"x": ("batch", 64)})
        d = result.to_dict()
        assert d["verdict"] == "MONOLITHIC_SAFE"
        assert isinstance(d["gaps"], list)

    def test_verifier_contracts_accessible(self):
        sg1 = _make_graph("G1",
            steps=[_make_step(OpKind.ACTIVATION, ["x"], "y")],
            inputs=["x"], outputs=["y"])
        sg2 = _make_graph("G2",
            steps=[_make_step(OpKind.ACTIVATION, ["y"], "z")],
            inputs=["y"], outputs=["z"])
        verifier = ThreadModularVerifier([sg1, sg2], {"x": ("batch", 64)})
        verifier.verify()
        assert len(verifier.contracts) == 2
        assert len(verifier.transformers) == 1
