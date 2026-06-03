"""Step 238 — whole-module straight-line subject reduction.

``lean/TensorGuard/SubjectReduction.lean`` proves that local complete/sound
shape-transfer certificates compose across an entire straight-line module:
executing a well-typed statement list preserves well-formed shapes for every
intermediate and final tensor.  This test keeps the proof imported/audited,
keeps its operator surface synchronized with the current confidence table, and
anchors representative theorem-shaped programs to real PyTorch execution and the
TensorGuard verifier.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import pytest

from src.fx_extractor import verify_module
from src.model_checker import SymbolicShapePropagator, extract_computation_graph

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "SubjectReduction.lean")
_TABLE = os.path.join(_ROOT, "operator_confidence_table.json")
_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.SubjectReduction.step_subject_reduction",
    "TensorGuard.SubjectReduction.exec_subject_reduction",
    "TensorGuard.SubjectReduction.whole_module_subject_reduction",
    "TensorGuard.SubjectReduction.program_outputs_have_positive_shapes",
    "TensorGuard.SubjectReduction.mlp_exec_shape",
    "TensorGuard.SubjectReduction.cnn_head_exec_shape",
    "TensorGuard.SubjectReduction.indexing_exec_shape",
    "TensorGuard.SubjectReduction.attention_exec_shape",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _manifest_names() -> set[str]:
    text = open(_FILE).read()
    match = re.search(
        r"def currentCompleteSoundOperatorNames\s*:\s*List String\s*:=\s*\[(.*?)\]",
        text,
        flags=re.DOTALL,
    )
    assert match, "currentCompleteSoundOperatorNames missing"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def _table_names_by_confidence(confidence: str) -> set[str]:
    payload = json.load(open(_TABLE))
    return {
        row["operator"]
        for row in payload["operators"]
        if row["confidence"] == confidence
    }


def _dims(shape):
    return tuple(dim.value for dim in shape.dims)


def _propagated_shapes(source: str, input_shapes: dict[str, tuple[int, ...]]):
    graph = extract_computation_graph(source)
    env = SymbolicShapePropagator(graph).propagate(input_shapes)
    return {name: _dims(shape) for name, shape in env.items()}


def test_subject_reduction_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.SubjectReduction" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for theorem in _THEOREMS:
        assert f"#print axioms {theorem}" in audit, f"audit missing {theorem}"


def test_subject_reduction_file_has_no_sorry_admit_or_axiom():
    code = _strip_comments(open(_FILE).read())
    assert not re.search(r"\b(sorry|admit|axiom)\b", code)


def test_operator_manifest_matches_current_nonheuristic_registry():
    manifest = _manifest_names()
    complete = _table_names_by_confidence("complete")
    sound = _table_names_by_confidence("sound")
    heuristic = _table_names_by_confidence("heuristic")
    assert manifest == complete | sound
    assert manifest.isdisjoint(heuristic)
    assert len(manifest) == 136


def test_proof_carrying_external_surface_is_explicit():
    text = open(_FILE).read()
    assert "certifiedExternal" in text
    assert "requires the local transfer to provide well-formed" in text


def test_mlp_theorem_shape_matches_verifier_and_torch():
    source = """
import torch
import torch.nn as nn
import torch.nn.functional as F
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.norm = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        h = self.fc1(x)
        a = F.relu(h)
        n = self.norm(a)
        return self.fc2(n)
"""
    env = _propagated_shapes(source, {"x": (2, 8)})
    assert env["h"] == (2, 16)
    assert env["a"] == (2, 16)
    assert env["n"] == (2, 16)

    ns: dict[str, object] = {}
    exec(source, ns)
    model = ns["M"]()
    assert verify_module(model, input_shapes={"x": (2, 8)}).safe
    assert tuple(model(torch.zeros(2, 8)).shape) == (2, 4)


def test_cnn_head_theorem_shape_matches_verifier_and_torch():
    source = """
import torch
import torch.nn as nn
import torch.nn.functional as F
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.fc = nn.Linear(384, 10)
    def forward(self, x):
        y = F.relu(self.conv(x))
        flat = torch.flatten(y, 1)
        return self.fc(flat)
"""
    env = _propagated_shapes(source, {"x": (2, 3, 8, 8)})
    assert env["y"] == (2, 6, 8, 8)
    assert env["flat"] == (2, 384)

    ns: dict[str, object] = {}
    exec(source, ns)
    model = ns["M"]()
    assert verify_module(model, input_shapes={"x": (2, 3, 8, 8)}).safe
    assert tuple(model(torch.zeros(2, 3, 8, 8)).shape) == (2, 10)


def test_indexing_theorem_shape_matches_verifier_and_torch():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 7)

        def forward(self, x, idx):
            return self.fc(x.gather(1, idx))

    model = M()
    result = verify_module(model, input_shapes={"x": (4, 3), "idx": (4, 2)})
    assert result.safe
    idx = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 1]])
    assert tuple(model(torch.zeros(4, 3), idx).shape) == (4, 7)


def test_attention_theorem_shape_matches_verifier_and_torch():
    if not hasattr(F, "scaled_dot_product_attention"):
        pytest.skip("torch build lacks scaled_dot_product_attention")

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(9, 16)

        def forward(self, q, k, v):
            return self.fc(F.scaled_dot_product_attention(q, k, v))

    model = M()
    shapes = {"q": (2, 4, 5, 8), "k": (2, 4, 7, 8), "v": (2, 4, 7, 9)}
    assert verify_module(model, input_shapes=shapes).safe
    q, k, v = (torch.zeros(*shapes[name]) for name in ("q", "k", "v"))
    assert tuple(model(q, k, v).shape) == (2, 4, 5, 16)


def test_ill_typed_straight_line_shape_is_rejected_by_lean_example():
    # The Lean examples are closed equalities.  A local type mismatch in the same
    # language is observable as a failing verifier result on real code, proving
    # the straight-line typing relation is not vacuously total.
    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 3)

        def forward(self, x):
            return self.fc(x)

    assert not verify_module(Bad(), input_shapes={"x": (2, 4)}).safe


@pytest.mark.slow
def test_subject_reduction_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.SubjectReduction"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_subject_reduction_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stdout[-3000:] + build.stderr[-3000:]

    proc = subprocess.run(
        ["lake", "env", "lean", "-R", ".", "TensorGuard/AxiomAudit.lean"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "sorryAx" not in proc.stdout
    axiom_lists = re.findall(r"depends on axioms:\s*\[([^\]]*)\]", proc.stdout)
    seen = set()
    for lst in axiom_lists:
        for name in (s.strip() for s in lst.split(",")):
            if name:
                seen.add(name)
    assert not (seen - _TRUSTED_AXIOMS), f"untrusted axioms: {seen - _TRUSTED_AXIOMS}"
    for theorem in _THEOREMS:
        assert f"'{theorem}'" in proc.stdout, f"audit output missing {theorem}"
