"""Step 239 — conditional subject reduction and branch abstention.

The Lean proof extends whole-module subject reduction from straight-line
programs to supported conditionals: both branches are checked from the same
incoming environment and must agree at the join before downstream code can run.
The real API side must remain conservative for unsupported tensor-value branch
conditions, returning UNKNOWN in sound mode rather than silently SAFE.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap

import pytest

from src.api import verify_architecture
from src.verifiable_fragment import (
    UnsupportedCategory,
    _analyze_source,
    analyze_source,
)

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_SUBJECT_REDUCTION = os.path.join(_LEAN, "TensorGuard", "SubjectReduction.lean")
_AXIOM_AUDIT = os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")
_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_CONDITIONAL_THEOREMS = [
    "TensorGuard.SubjectReduction.envJoin_preserves_wf",
    "TensorGuard.SubjectReduction.condExec_some_iff_supported_join",
    "TensorGuard.SubjectReduction.sound_safe_implies_supported_branch",
    "TensorGuard.SubjectReduction.unsupported_branch_cannot_silently_safe",
    "TensorGuard.SubjectReduction.supported_conditional_subject_reduction",
    "TensorGuard.SubjectReduction.conditional_then_program_subject_reduction",
    "TensorGuard.SubjectReduction.conditional_join_exec_shape",
    "TensorGuard.SubjectReduction.conditional_tail_exec_shape",
    "TensorGuard.SubjectReduction.divergent_branch_join_rejected",
    "TensorGuard.SubjectReduction.unsupported_branch_abstains_example",
]

_SUPPORTED_TRAINING_BRANCH = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_fc = nn.Linear(8, 8)
        self.eval_fc = nn.Linear(8, 8)

    def forward(self, x):
        if self.training:
            y = self.train_fc(x)
        else:
            y = self.eval_fc(x)
        return y
"""

_UNSUPPORTED_JOINABLE_TENSOR_BRANCH = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        if x.sum() > 0:
            y = self.fc(x)
        else:
            y = x + 0
        return y
"""


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _exec_model(source: str):
    ns: dict[str, object] = {}
    exec(textwrap.dedent(source), ns)
    return ns["M"]()


def test_conditional_theorems_are_audited_and_finished():
    code = _strip_comments(open(_SUBJECT_REDUCTION).read())
    assert not re.search(r"\b(sorry|admit|axiom)\b", code)

    audit = open(_AXIOM_AUDIT).read()
    for theorem in _CONDITIONAL_THEOREMS:
        assert f"#print axioms {theorem}" in audit


def test_supported_training_branch_is_not_static_fragment_violation():
    assert analyze_source(_SUPPORTED_TRAINING_BRANCH) == []
    result = verify_architecture(
        _SUPPORTED_TRAINING_BRANCH,
        input_shapes={"x": (2, 8)},
        max_cegar_iterations=0,
        soundness_mode="sound",
    )
    assert result.verdict == "SAFE"

    model = _exec_model(_SUPPORTED_TRAINING_BRANCH)
    model.train()
    assert tuple(model(torch.ones(2, 8)).shape) == (2, 8)
    model.eval()
    assert tuple(model(torch.ones(2, 8)).shape) == (2, 8)


def test_tensor_value_branch_is_blocking_even_when_branches_join():
    issues, _warnings = _analyze_source(_UNSUPPORTED_JOINABLE_TENSOR_BRANCH)
    blocking = [issue for issue in issues if issue.severity == "blocking"]
    assert [issue.category for issue in blocking] == [
        UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW
    ]

    model = _exec_model(_UNSUPPORTED_JOINABLE_TENSOR_BRANCH)
    assert tuple(model(torch.ones(2, 8)).shape) == (2, 8)
    assert tuple(model(-torch.ones(2, 8)).shape) == (2, 8)


def test_sound_mode_abstains_on_joinable_tensor_value_branch():
    sound = verify_architecture(
        _UNSUPPORTED_JOINABLE_TENSOR_BRANCH,
        input_shapes={"x": (2, 8)},
        max_cegar_iterations=0,
        soundness_mode="sound",
    )
    assert sound.verdict == "UNKNOWN"
    assert sound.abstained is True
    assert any("DATA_DEPENDENT_CONTROL_FLOW" in r for r in sound.unknown_reasons)

    balanced = verify_architecture(
        _UNSUPPORTED_JOINABLE_TENSOR_BRANCH,
        input_shapes={"x": (2, 8)},
        max_cegar_iterations=0,
        soundness_mode="balanced",
    )
    assert balanced.verdict == "SAFE"


@pytest.mark.slow
def test_conditional_subject_reduction_lean_builds():
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
def test_conditional_subject_reduction_axiom_clean():
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
    for theorem in _CONDITIONAL_THEOREMS:
        assert f"'{theorem}'" in proc.stdout, f"audit output missing {theorem}"
