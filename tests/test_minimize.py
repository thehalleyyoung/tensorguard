"""Step 17 regression tests -- minimal-reproducer shrinker.

Covers the minimizer machinery on real TensorGuard/runtime oracles (the
`agreement_bug` live demo) and exercises the generic ddmin/shrink algorithm on
the `false_positive`/`false_negative` paths via a *synthetic oracle*, so the
disagreement-minimization machinery is tested even though no natural
disagreement exists in the corpus yet.
"""

from __future__ import annotations

import os

from evaluation import minimize


# -- IR / emitter -----------------------------------------------------------
def test_emit_source_threads_running_shape():
    model = {"batch": 4, "in_features": 16, "ops": [
        {"k": "linear", "out": 8}, {"k": "act", "fn": "relu"},
        {"k": "layernorm"}, {"k": "linear", "out": 4}]}
    src = minimize.emit_source(model)
    assert "nn.Linear(16, 8)" in src      # first linear in = input feature
    assert "nn.Linear(8, 4)" in src       # second linear in = running (8)
    assert "nn.LayerNorm(8)" in src       # layernorm over running (8)


def test_fault_override_creates_a_mismatch():
    model = {"batch": 2, "in_features": 8, "ops": [
        {"k": "linear", "out": 4, "in_override": 99}]}
    assert minimize.runtime_raises(model)
    assert minimize.tensorguard_refutes(model)


# -- live demo on real oracles ---------------------------------------------
def test_live_agreement_bug_minimization():
    seed = minimize.demo_seed_model()
    assert minimize.pred_agreement_bug(seed)
    minimal = minimize.minimize(seed, minimize.pred_agreement_bug)
    assert len(minimal["ops"]) < len(seed["ops"])
    assert len(minimal["ops"]) == 1            # shrinks to a single faulted op
    assert minimize.pred_agreement_bug(minimal)
    assert minimize.runtime_raises(minimal)
    assert minimize.tensorguard_refutes(minimal)
    assert minimize._size(minimal) < minimize._size(seed)


# -- generic ddmin/shrink via synthetic oracle ------------------------------
def _synthetic_predicate_keep_marker(model):
    """A predicate independent of TensorGuard: holds iff a 'marker' op is
    present. Lets us test ddmin's reduction logic in isolation."""
    return any(op.get("marker") for op in model["ops"])


def test_ddmin_isolates_the_essential_op():
    ops = [{"k": "act", "fn": "relu"} for _ in range(9)]
    ops.insert(5, {"k": "act", "fn": "relu", "marker": True})
    model = {"batch": 4, "in_features": 8, "ops": ops}
    reduced = minimize.ddmin_ops(model, _synthetic_predicate_keep_marker)
    assert len(reduced["ops"]) == 1
    assert reduced["ops"][0].get("marker")


def test_minimizer_supports_false_positive_predicate_shape():
    # Synthetic 'false positive': pretend TG refutes a model that runs clean.
    # We model the predicate as 'a frozen marker op survives', proving the
    # generic minimize() path works for the disagreement predicates too.
    ops = [{"k": "linear", "out": 8} for _ in range(6)]
    ops[3]["marker"] = True
    model = {"batch": 2, "in_features": 8, "ops": ops}

    def pred(m):
        return any(op.get("marker") for op in m["ops"])

    reduced = minimize.minimize(model, pred)
    assert pred(reduced)
    assert len(reduced["ops"]) == 1


def test_builtin_predicates_are_registered():
    assert set(minimize.PREDICATES) == {
        "false_positive", "false_negative", "agreement_bug"}


# -- determinism / artifact -------------------------------------------------
def test_minimization_is_deterministic():
    seed = minimize.demo_seed_model()
    a = minimize.minimize(seed, minimize.pred_agreement_bug)
    b = minimize.minimize(seed, minimize.pred_agreement_bug)
    assert minimize.emit_source(a) == minimize.emit_source(b)


def test_committed_artifact_is_up_to_date():
    assert os.path.exists(minimize.OUT_JSON)
    minimize.run(check=True)
