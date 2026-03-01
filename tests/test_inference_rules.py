"""Tests for formal inference rules in assume-guarantee reasoning."""

import pytest

from src.assume_guarantee import (
    InferenceRule,
    ASYMMETRIC_AG_RULE,
    SEQUENTIAL_COMPOSITION_RULE,
    ProofStep,
    ProofTree,
    _build_proof_tree,
    verify_compositional,
    CompositionalResult,
    InterfaceCheck,
    InterfaceContract,
    SubModule,
    reset_default_cache,
)
from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    OpKind,
    VerificationResult,
)


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


@pytest.fixture(autouse=True)
def _reset():
    reset_default_cache()
    yield
    reset_default_cache()


def _make_step(op, output, inputs=None):
    return ComputationStep(
        op=op, output=output, inputs=inputs or ["x"],
        layer_ref=None, params={},
    )


def _make_submodule(name):
    s = [_make_step(OpKind.ACTIVATION, f"{name}_out")]
    graph = ComputationGraph(
        class_name="TestModel", steps=s,
        input_names=["x"], output_names=[f"{name}_out"], layers={},
    )
    return SubModule(
        name=name, graph=graph,
        input_contract=InterfaceContract(
            name=f"I_{name}", input_shapes={"x": ("batch", 10)},
        ),
        output_contract=InterfaceContract(
            name=f"O_{name}", output_shapes={f"{name}_out": ("batch", 10)},
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# InferenceRule instantiation
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferenceRuleInstantiation:
    def test_asymmetric_rule_has_name(self):
        assert ASYMMETRIC_AG_RULE.name == "AG-Asym"

    def test_asymmetric_rule_has_premises(self):
        assert len(ASYMMETRIC_AG_RULE.premises) == 2

    def test_asymmetric_rule_has_conclusion(self):
        assert "M_1 \\circ M_2" in ASYMMETRIC_AG_RULE.conclusion

    def test_asymmetric_rule_has_side_conditions(self):
        assert len(ASYMMETRIC_AG_RULE.side_conditions) == 2

    def test_sequential_rule_has_name(self):
        assert SEQUENTIAL_COMPOSITION_RULE.name == "AG-Seq"

    def test_sequential_rule_has_premises(self):
        assert len(SEQUENTIAL_COMPOSITION_RULE.premises) == 2

    def test_sequential_rule_has_conclusion(self):
        assert "M_1 \\circ \\cdots \\circ M_n" in SEQUENTIAL_COMPOSITION_RULE.conclusion

    def test_sequential_rule_has_side_conditions(self):
        assert len(SEQUENTIAL_COMPOSITION_RULE.side_conditions) == 2

    def test_custom_rule(self):
        rule = InferenceRule(
            name="Custom",
            premises=["P1", "P2"],
            conclusion="C",
            side_conditions=["SC1"],
        )
        assert rule.name == "Custom"
        assert rule.premises == ["P1", "P2"]


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferenceRuleLatex:
    def test_asymmetric_latex_has_frac(self):
        latex = ASYMMETRIC_AG_RULE.to_latex()
        assert "\\frac{" in latex

    def test_asymmetric_latex_has_name(self):
        latex = ASYMMETRIC_AG_RULE.to_latex()
        assert "\\textsc{AG-Asym}" in latex

    def test_asymmetric_latex_has_side_conditions(self):
        latex = ASYMMETRIC_AG_RULE.to_latex()
        assert "\\text{where }" in latex

    def test_sequential_latex_has_frac(self):
        latex = SEQUENTIAL_COMPOSITION_RULE.to_latex()
        assert "\\frac{" in latex

    def test_sequential_latex_has_name(self):
        latex = SEQUENTIAL_COMPOSITION_RULE.to_latex()
        assert "\\textsc{AG-Seq}" in latex

    def test_rule_without_side_conditions(self):
        rule = InferenceRule(
            name="Simple", premises=["P"], conclusion="C",
            side_conditions=[],
        )
        latex = rule.to_latex()
        assert "\\text{where }" not in latex
        assert "\\frac{P}{C}" in latex

    def test_latex_premises_joined_with_quad(self):
        latex = ASYMMETRIC_AG_RULE.to_latex()
        assert "\\quad" in latex


# ═══════════════════════════════════════════════════════════════════════════════
# Proof tree construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestProofTree:
    def test_build_proof_tree_two_modules(self):
        sms = [_make_submodule("M0"), _make_submodule("M1")]
        results = {
            "M0": VerificationResult(safe=True),
            "M1": VerificationResult(safe=True),
        }
        checks = [InterfaceCheck("M0", "M1", True, "OK")]
        tree = _build_proof_tree(sms, results, checks)
        assert len(tree.steps) == 1
        assert tree.steps[0].rule.name == "AG-Asym"
        assert tree.steps[0].discharged is True

    def test_build_proof_tree_three_modules(self):
        sms = [_make_submodule("M0"), _make_submodule("M1"), _make_submodule("M2")]
        results = {
            "M0": VerificationResult(safe=True),
            "M1": VerificationResult(safe=True),
            "M2": VerificationResult(safe=True),
        }
        checks = [
            InterfaceCheck("M0", "M1", True, "OK"),
            InterfaceCheck("M1", "M2", True, "OK"),
        ]
        tree = _build_proof_tree(sms, results, checks)
        assert len(tree.steps) == 2
        assert tree.overall_rule.name == "AG-Seq"

    def test_proof_tree_failed_step(self):
        sms = [_make_submodule("M0"), _make_submodule("M1")]
        results = {
            "M0": VerificationResult(safe=True),
            "M1": VerificationResult(safe=False),
        }
        checks = [InterfaceCheck("M0", "M1", True, "OK")]
        tree = _build_proof_tree(sms, results, checks)
        assert tree.steps[0].discharged is False

    def test_proof_tree_pretty(self):
        sms = [_make_submodule("M0"), _make_submodule("M1")]
        results = {
            "M0": VerificationResult(safe=True),
            "M1": VerificationResult(safe=True),
        }
        checks = [InterfaceCheck("M0", "M1", True, "OK")]
        tree = _build_proof_tree(sms, results, checks)
        pretty = tree.pretty()
        assert "Proof Tree" in pretty
        assert "AG-Asym" in pretty

    def test_proof_tree_to_latex(self):
        sms = [_make_submodule("M0"), _make_submodule("M1")]
        results = {
            "M0": VerificationResult(safe=True),
            "M1": VerificationResult(safe=True),
        }
        checks = [InterfaceCheck("M0", "M1", True, "OK")]
        tree = _build_proof_tree(sms, results, checks)
        latex = tree.to_latex()
        assert "\\frac{" in latex


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: verify_compositional produces proof_tree
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompositionalProofTreeIntegration:
    def test_two_layer_has_proof_tree(self):
        result = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert result.proof_tree is not None
        assert isinstance(result.proof_tree, ProofTree)

    def test_three_layer_proof_tree_uses_seq_rule(self):
        result = verify_compositional(
            THREE_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert result.proof_tree is not None
        if result.num_submodules > 2:
            assert result.proof_tree.overall_rule.name == "AG-Seq"

    def test_safe_model_all_steps_discharged(self):
        result = verify_compositional(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            measure_monolithic=False,
        )
        assert result.safe is True
        if result.proof_tree and result.proof_tree.steps:
            for step in result.proof_tree.steps:
                assert step.discharged is True
