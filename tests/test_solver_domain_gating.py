"""Step 2: --no-*-check flags gate the solver, not just the verdict.

Verifies that disabling a domain prevents TensorGuard from generating and
solving that domain's constraints at all (rather than computing them and
filtering the result afterwards).  The proof is that the per-domain safety
encoders return no constraints when their domain is disabled, so no solver
work or cross-domain witnesses are produced for that domain.
"""

from src.model_checker import (
    verify_model,
    ConstraintVerifier,
    extract_computation_graph,
)


GRAD_SRC = '''
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x).detach()
'''


def _violation_kinds(result):
    if result.safe or not result.counterexample:
        return set()
    return {v.kind for v in result.counterexample.violations}


def test_disabled_domain_encoders_emit_no_constraints():
    g = extract_computation_graph(GRAD_SRC)
    cv = ConstraintVerifier(
        g, input_shapes={"x": (4,)},
        check_devices=False, check_phases=False, check_gradients=False,
    )
    for idx, step in enumerate(g.steps):
        k = cv._build_kripke_state(idx, cv._init_state)
        assert cv._encode_device_safety(k, step, cv._init_state, idx) == []
        assert cv._encode_phase_safety(k, step, cv._init_state, idx) == []
        assert cv._encode_gradient_safety(k, step, cv._init_state, idx) == []


def test_enabled_domain_encoders_emit_constraints():
    g = extract_computation_graph(GRAD_SRC)
    cv = ConstraintVerifier(g, input_shapes={"x": (4,)})
    produced = any(
        cv._encode_gradient_safety(
            cv._build_kripke_state(i, cv._init_state), s, cv._init_state, i)
        for i, s in enumerate(g.steps)
    )
    assert produced


def test_grad_bug_present_by_default_absent_when_disabled():
    full = verify_model(GRAD_SRC, input_shapes={"x": (4,)})
    assert "gradient_broken" in _violation_kinds(full)
    gated = verify_model(GRAD_SRC, input_shapes={"x": (4,)},
                         check_gradients=False)
    assert "gradient_broken" not in _violation_kinds(gated)


def test_filter_domain_checks_drops_disabled_kinds():
    g = extract_computation_graph(GRAD_SRC)
    cv = ConstraintVerifier(g, input_shapes={"x": (4,)}, check_devices=False)
    pairs = [("shape_incompatible", lambda: []),
             ("device_mismatch", lambda: []),
             ("phase_violation", lambda: []),
             ("gradient_violation", lambda: [])]
    kept = {k for k, _ in cv._filter_domain_checks(pairs)}
    assert "device_mismatch" not in kept
    assert "shape_incompatible" in kept  # shape always runs
    assert "phase_violation" in kept and "gradient_violation" in kept
