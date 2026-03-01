"""Tests for denotational_semantics module."""

from __future__ import annotations

import pytest

from src.denotational_semantics import (
    AbstractDevice,
    AbstractPhase,
    AbstractShape,
    AbstractTensorState,
    ConcreteTensorState,
    DenotationalStep,
    OpSemantics,
    OP_SEMANTICS,
    alpha,
    alpha_tensor,
    check_galois_connection,
    compose_graph_semantics,
    concrete_add,
    concrete_cat,
    concrete_flatten,
    concrete_identity,
    concrete_matmul,
    concrete_reshape,
    concrete_squeeze,
    concrete_transpose,
    concrete_unsqueeze,
    abstract_add,
    abstract_cat,
    abstract_flatten,
    abstract_identity,
    abstract_matmul,
    abstract_reshape,
    abstract_squeeze,
    abstract_transpose,
    abstract_unsqueeze,
    verify_soundness_for_concrete_input,
)
from src.model_checker import OpKind


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Abstract domain tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAbstractShape:
    def test_concrete_shape(self):
        s = AbstractShape(dims=(2, 3, 4))
        assert s.rank == 3
        assert s.is_concrete()

    def test_symbolic_shape(self):
        s = AbstractShape(dims=("batch", 3, 4))
        assert s.rank == 3
        assert not s.is_concrete()

    def test_leq_concrete(self):
        s1 = AbstractShape(dims=(2, 3))
        s2 = AbstractShape(dims=(2, 3))
        assert s1.leq(s2)

    def test_leq_symbolic_top(self):
        s1 = AbstractShape(dims=(2, 3))
        s2 = AbstractShape(dims=("batch", 3))
        assert s1.leq(s2)  # concrete ⊑ symbolic

    def test_not_leq_different(self):
        s1 = AbstractShape(dims=(2, 3))
        s2 = AbstractShape(dims=(4, 3))
        assert not s1.leq(s2)


class TestAbstractPhase:
    def test_join_same(self):
        assert AbstractPhase.TRAIN.join(AbstractPhase.TRAIN) == AbstractPhase.TRAIN

    def test_join_different(self):
        assert AbstractPhase.TRAIN.join(AbstractPhase.EVAL) == AbstractPhase.TOP

    def test_leq_top(self):
        assert AbstractPhase.TRAIN.leq(AbstractPhase.TOP)

    def test_leq_same(self):
        assert AbstractPhase.EVAL.leq(AbstractPhase.EVAL)

    def test_not_leq_different(self):
        assert not AbstractPhase.TRAIN.leq(AbstractPhase.EVAL)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Galois connection tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGaloisConnection:
    def test_alpha_tensor(self):
        c = ConcreteTensorState(shape=(2, 3, 4), device="cpu", phase="train")
        a = alpha_tensor(c)
        assert a.shape.dims == (2, 3, 4)
        assert a.device == AbstractDevice.CPU
        assert a.phase == AbstractPhase.TRAIN

    def test_alpha_cuda(self):
        c = ConcreteTensorState(shape=(2, 3), device="cuda", phase="eval")
        a = alpha_tensor(c)
        assert a.device == AbstractDevice.CUDA
        assert a.phase == AbstractPhase.EVAL

    def test_check_galois_sound(self):
        c = ConcreteTensorState(shape=(2, 3), device="cpu", phase="train")
        a = AbstractTensorState(
            shape=AbstractShape(dims=("batch", 3)),
            device=AbstractDevice.CPU,
            phase=AbstractPhase.TRAIN,
        )
        assert check_galois_connection(c, a)

    def test_check_galois_unsound(self):
        c = ConcreteTensorState(shape=(2, 3), device="cpu", phase="train")
        a = AbstractTensorState(
            shape=AbstractShape(dims=(4, 3)),  # wrong concrete dim
            device=AbstractDevice.CPU,
            phase=AbstractPhase.TRAIN,
        )
        assert not check_galois_connection(c, a)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Concrete semantics tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConcreteSemantics:
    def test_matmul(self):
        result = concrete_matmul([(2, 3, 4), (2, 4, 5)])
        assert result == (2, 3, 5)

    def test_matmul_inner_dim_mismatch(self):
        with pytest.raises(ValueError):
            concrete_matmul([(2, 3, 4), (2, 5, 6)])

    def test_add_broadcast(self):
        result = concrete_add([(2, 3, 4), (1, 4)])
        assert result == (2, 3, 4)

    def test_add_incompatible(self):
        with pytest.raises(ValueError):
            concrete_add([(2, 3), (2, 4)])

    def test_reshape(self):
        result = concrete_reshape([(2, 3, 4)], target=(6, 4))
        assert result == (6, 4)

    def test_reshape_mismatch(self):
        with pytest.raises(ValueError):
            concrete_reshape([(2, 3, 4)], target=(5, 5))

    def test_transpose(self):
        result = concrete_transpose([(2, 3, 4)])
        assert result == (2, 4, 3)

    def test_flatten(self):
        result = concrete_flatten([(2, 3, 4)], start_dim=1)
        assert result == (2, 12)

    def test_squeeze(self):
        result = concrete_squeeze([(2, 1, 4)])
        assert result == (2, 4)

    def test_squeeze_dim(self):
        result = concrete_squeeze([(2, 1, 4)], dim=1)
        assert result == (2, 4)

    def test_unsqueeze(self):
        result = concrete_unsqueeze([(2, 3)], dim=1)
        assert result == (2, 1, 3)

    def test_cat(self):
        result = concrete_cat([(2, 3), (2, 5)], dim=1)
        assert result == (2, 8)

    def test_identity(self):
        result = concrete_identity([(2, 3, 4)])
        assert result == (2, 3, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Abstract semantics tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAbstractSemantics:
    def test_abstract_matmul(self):
        a = AbstractShape(dims=("batch", "m", "k"))
        b = AbstractShape(dims=("batch", "k", "n"))
        result = abstract_matmul([a, b])
        assert result.dims == ("batch", "m", "n")

    def test_abstract_add_broadcast(self):
        a = AbstractShape(dims=("batch", 3, 4))
        b = AbstractShape(dims=(1, 4))
        result = abstract_add([a, b])
        assert result.rank == 3
        assert result.dims[2] == 4

    def test_abstract_reshape(self):
        a = AbstractShape(dims=("batch", 3, 4))
        result = abstract_reshape([a], target=("batch", 12))
        assert result.dims == ("batch", 12)

    def test_abstract_identity(self):
        a = AbstractShape(dims=("batch", 3, 4))
        result = abstract_identity([a])
        assert result.dims == ("batch", 3, 4)

    def test_abstract_transpose(self):
        a = AbstractShape(dims=("batch", "m", "n"))
        result = abstract_transpose([a])
        assert result.dims == ("batch", "n", "m")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OP_SEMANTICS registry tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpSemanticsRegistry:
    def test_matmul_registered(self):
        assert OpKind.MATMUL in OP_SEMANTICS

    def test_add_registered(self):
        assert OpKind.ADD in OP_SEMANTICS

    def test_reshape_registered(self):
        assert OpKind.RESHAPE in OP_SEMANTICS

    def test_all_have_descriptions(self):
        for op, sem in OP_SEMANTICS.items():
            assert sem.description
            assert sem.concrete_fn is not None
            assert sem.abstract_fn is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Graph-level composition tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestComposeGraphSemantics:
    def test_two_step_matmul_add(self):
        steps = [
            DenotationalStep(
                node_name="matmul_0",
                op=OpKind.MATMUL,
                input_names=["x", "w"],
                output_name="y",
            ),
            DenotationalStep(
                node_name="add_0",
                op=OpKind.ADD,
                input_names=["y", "b"],
                output_name="z",
            ),
        ]
        initial = {
            "x": AbstractTensorState(shape=AbstractShape(dims=(2, 3, 4))),
            "w": AbstractTensorState(shape=AbstractShape(dims=(2, 4, 5))),
            "b": AbstractTensorState(shape=AbstractShape(dims=(1, 5))),
        }
        result = compose_graph_semantics(steps, initial)
        assert "z" in result
        assert result["z"].shape.dims == (2, 3, 5)

    def test_identity_chain(self):
        steps = [
            DenotationalStep("relu", OpKind.ACTIVATION, ["x"], "y"),
            DenotationalStep("dropout", OpKind.DROPOUT, ["y"], "z"),
        ]
        initial = {
            "x": AbstractTensorState(shape=AbstractShape(dims=(2, 3))),
        }
        result = compose_graph_semantics(steps, initial)
        assert result["z"].shape.dims == (2, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Soundness verification tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSoundnessVerification:
    def test_matmul_soundness(self):
        steps = [
            DenotationalStep("matmul", OpKind.MATMUL, ["x", "w"], "y"),
        ]
        concrete_init = {
            "x": ConcreteTensorState(shape=(2, 3, 4)),
            "w": ConcreteTensorState(shape=(2, 4, 5)),
        }
        assert verify_soundness_for_concrete_input(steps, concrete_init)

    def test_add_broadcast_soundness(self):
        steps = [
            DenotationalStep("add", OpKind.ADD, ["a", "b"], "c"),
        ]
        concrete_init = {
            "a": ConcreteTensorState(shape=(2, 3, 4)),
            "b": ConcreteTensorState(shape=(1, 4)),
        }
        assert verify_soundness_for_concrete_input(steps, concrete_init)

    def test_chain_soundness(self):
        steps = [
            DenotationalStep("matmul", OpKind.MATMUL, ["x", "w"], "y"),
            DenotationalStep("relu", OpKind.ACTIVATION, ["y"], "z"),
        ]
        concrete_init = {
            "x": ConcreteTensorState(shape=(4, 3)),
            "w": ConcreteTensorState(shape=(3, 5)),
        }
        assert verify_soundness_for_concrete_input(steps, concrete_init)
