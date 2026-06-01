"""Regression tests for CEGAR refined-contract bug surfacing (Step 1).

These tests pin the behaviour that unsatisfiable refined shape contracts are
promoted to real ``Bug`` objects, and that the reported bug set now depends on
``max_cegar_iterations`` (closing the previously documented limitation where
discovered predicates were inert metadata).
"""

import pytest

from src.api import (
    verify_architecture,
    BugCategory,
    _cegar_refined_contract_bugs,
)
from src.shape_cegar import (
    run_shape_cegar,
    ShapePredicate,
    PredicateKind,
    ShapeCEGARResult,
    IterationRecord,
)


CONFLICT_SRC = '''
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(768, 10)
        self.b = nn.Linear(512, 10)

    def forward(self, x):
        return self.a(x) + self.b(x)
'''

CLEAN_SRC = '''
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)

    def forward(self, x):
        return self.fc(x)
'''


def _refined(src, iters):
    r = verify_architecture(src, input_shapes={"x": ("b", "f")},
                            max_cegar_iterations=iters)
    return [b for b in r.bugs if b.category == BugCategory.CEGAR_REFINED_CONTRACT]


def test_conflicting_in_features_is_surfaced_as_refined_contract():
    bugs = _refined(CONFLICT_SRC, 10)
    assert len(bugs) == 1
    msg = bugs[0].message
    assert "x" in msg
    assert "768" in msg and "512" in msg
    assert bugs[0].severity == "error"
    assert bugs[0].confidence == 1.0


def test_refined_contract_depends_on_cegar_iterations():
    # With CEGAR disabled no predicates are proposed, so the contract-level
    # conflict cannot be surfaced; enabling CEGAR reveals it.
    assert _refined(CONFLICT_SRC, 0) == []
    assert len(_refined(CONFLICT_SRC, 10)) == 1


def test_clean_model_has_no_refined_contract_bug():
    assert _refined(CLEAN_SRC, 10) == []


def test_helper_detects_infeasible_union_from_iteration_log():
    # Even when the surviving discovered set is feasible, a conflict proposed
    # across the iteration log must be detected (this is the real signal).
    p768 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768)
    p512 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=512)
    result = ShapeCEGARResult(
        discovered_predicates=[p768],  # surviving set looks fine
        iteration_log=[
            IterationRecord(iteration=0, num_violations=1, num_spurious=1,
                            num_real=0, predicates_added=[p768, p512]),
        ],
    )
    bugs = _cegar_refined_contract_bugs(result, "<test>")
    assert len(bugs) == 1
    assert bugs[0].category == BugCategory.CEGAR_REFINED_CONTRACT


def test_helper_no_bug_when_feasible():
    p768 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768)
    result = ShapeCEGARResult(
        discovered_predicates=[p768],
        iteration_log=[
            IterationRecord(iteration=0, num_violations=0, num_spurious=0,
                            num_real=0, predicates_added=[p768]),
        ],
    )
    assert _cegar_refined_contract_bugs(result, "<test>") == []
