"""Step 241 — CEGAR predicate serialization proof boundary.

The Lean proof mirrors the actual Python predicate-record format, not a
handwritten pretty string.  These tests tie the two sides together:

* every live ``PredicateKind.name`` and record key is present in the Lean mirror,
* real ``ShapePredicate`` objects round-trip through the stable record format,
* serialized CEGAR history is monotone under append-only refinement, and
* replayed infeasible serialized contracts abstain (`UNKNOWN`), matching the
  live API's CEGAR refined-contract bug surfacing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap

import pytest

from src.api import BugCategory, verify_architecture
from src.shape_cegar import (
    CEGARStatus,
    CEGARVerdict,
    HAS_Z3,
    IterationRecord,
    PredicateKind,
    SHAPE_PREDICATE_RECORD_FIELDS,
    SHAPE_PREDICATE_RECORD_SCHEMA,
    ShapeCEGARResult,
    ShapePredicate,
    run_shape_cegar,
    serialized_predicates_feasible,
    serialized_predicates_status,
    shape_predicate_from_record,
    shape_predicate_history_from_result,
    shape_predicate_history_is_monotone,
    shape_predicate_to_record,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "CegarPredicateSerialization.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.CegarSerialized.python_record_keys_match_v1",
    "TensorGuard.CegarSerialized.python_kind_names_cover_v1",
    "TensorGuard.CegarSerialized.serialized_append_preserves_membership",
    "TensorGuard.CegarSerialized.serialized_append_length_mono",
    "TensorGuard.CegarSerialized.serialized_history_step_prefix",
    "TensorGuard.CegarSerialized.serialized_infeasible_abstains",
    "TensorGuard.CegarSerialized.serialized_feasible_safe",
    "TensorGuard.CegarSerialized.decideSerialized_safeSound",
    "TensorGuard.CegarSerialized.infeasible_serialized_safeSound_any_bug",
]

CONFLICT_SRC = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(768, 10)
        self.b = nn.Linear(512, 10)

    def forward(self, x):
        return self.a(x) + self.b(x)
"""


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


def test_serializer_schema_matches_lean_mirror():
    assert os.path.exists(_FILE), "CegarPredicateSerialization.lean missing"
    text = open(_FILE).read()

    assert SHAPE_PREDICATE_RECORD_SCHEMA in text
    for field in SHAPE_PREDICATE_RECORD_FIELDS:
        assert f'"{field}"' in text
    for kind in PredicateKind:
        assert f'"{kind.name}"' in text

    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.CegarPredicateSerialization" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_predicate_record_roundtrip_preserves_every_kind():
    predicates = [
        ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=0),
        ShapePredicate(PredicateKind.DIM_GT, "x", axis=-1, value=1),
        ShapePredicate(PredicateKind.DIM_GE, "x", axis=1, value=32),
        ShapePredicate(PredicateKind.DIM_DIVISIBLE, "x", axis=2, divisor=8),
        ShapePredicate(
            PredicateKind.DIM_MATCH,
            "x",
            axis=-1,
            match_tensor="w",
            match_axis=0,
        ),
        ShapePredicate(PredicateKind.NDIM_EQ, "x", value=4),
        ShapePredicate(PredicateKind.SHAPE_EQ, "x", value=(32, "features")),
    ]

    for pred in predicates:
        record = shape_predicate_to_record(pred)
        assert list(record) == list(SHAPE_PREDICATE_RECORD_FIELDS)
        assert record["schema"] == SHAPE_PREDICATE_RECORD_SCHEMA
        assert shape_predicate_from_record(record) == pred

    shape_record = shape_predicate_to_record(predicates[-1])
    assert shape_record["value"] == [32, "features"]
    assert shape_predicate_from_record(shape_record).value == (32, "features")


def test_serialized_history_monotone_on_actual_records():
    p768 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768)
    p512 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=512)
    result = ShapeCEGARResult(
        iteration_log=[
            IterationRecord(0, 1, 1, 0, predicates_added=[p768]),
            IterationRecord(1, 1, 1, 0, predicates_added=[p768, p512]),
        ],
    )

    history = shape_predicate_history_from_result(result)
    assert shape_predicate_history_is_monotone(history)
    assert len(history[0]) == 1
    assert len(history[1]) == 2
    assert not shape_predicate_history_is_monotone([history[1], history[0]])


def test_real_cegar_result_history_serializes_monotonically():
    source = textwrap.dedent(
        """\
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(768, 10)
            def forward(self, x):
                return self.fc(x)
        """
    )
    result = run_shape_cegar(
        source,
        input_shapes={"x": ("batch", "features")},
        max_iterations=10,
    )
    history = shape_predicate_history_from_result(result)
    assert shape_predicate_history_is_monotone(history)


@pytest.mark.skipif(not HAS_Z3, reason="Z3 required for infeasibility replay")
def test_infeasible_serialized_contract_abstains():
    p768 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768)
    p512 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=512)
    records = [shape_predicate_to_record(p) for p in (p768, p512)]

    assert serialized_predicates_feasible(records) is False
    assert serialized_predicates_status(records) is CEGARStatus.INFEASIBLE_REFINEMENT
    assert ShapeCEGARResult(
        final_status=serialized_predicates_status(records),
    ).verdict is CEGARVerdict.UNKNOWN


@pytest.mark.skipif(not HAS_Z3, reason="Z3 required for live CEGAR refinement")
def test_live_api_surfaces_real_infeasible_refined_contract():
    result = verify_architecture(
        CONFLICT_SRC,
        input_shapes={"x": ("batch", "features")},
        max_cegar_iterations=10,
    )
    refined = [
        bug for bug in result.bugs
        if bug.category is BugCategory.CEGAR_REFINED_CONTRACT
    ]
    assert refined, [bug.message for bug in result.bugs]
    assert "768" in refined[0].message and "512" in refined[0].message


def test_lean_file_has_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


@pytest.mark.slow
def test_lean_serialization_module_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.CegarPredicateSerialization"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"lake build failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )


@pytest.mark.slow
def test_lean_serialization_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.CegarPredicateSerialization"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_CegarSerializationAxCheck.lean")
    body = "import TensorGuard.CegarPredicateSerialization\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS
    ) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        lean_path = os.path.join(_LEAN, ".lake", "build", "lib")
        env = dict(os.environ, LEAN_PATH=lean_path)
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_CegarSerializationAxCheck.lean"],
            cwd=_LEAN,
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, (
        f"axiom check failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    out = proc.stdout
    assert "sorryAx" not in out, f"a serialization proof depends on sorryAx:\n{out}"
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}:\n{out}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"axiom output missing {thm}:\n{out}"
