"""Step 240 — bounded-loop and ModuleList subject reduction.

The Lean side proves that bounded unrolling composes subject reduction and that
over-budget / unsupported loops cannot silently produce a SAFE environment.  The
Python side wires the same bound into the real graph extractor: statically
resolved ModuleList/Sequential and literal range loops are unrolled up to the
limit, and sound mode reports UNKNOWN beyond it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap

import pytest

from src.api import verify_architecture
from src.model_checker import extract_computation_graph
from src.verifiable_fragment import UnsupportedCategory, _analyze_source

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_SUBJECT_REDUCTION = os.path.join(_LEAN, "TensorGuard", "SubjectReduction.lean")
_AXIOM_AUDIT = os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")
_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_LOOP_THEOREMS = [
    "TensorGuard.SubjectReduction.exec_program_list_subject_reduction",
    "TensorGuard.SubjectReduction.bounded_unroll_exec_some_implies_supported_and_within_limit",
    "TensorGuard.SubjectReduction.modulelist_beyond_unroll_limit_abstains",
    "TensorGuard.SubjectReduction.static_range_beyond_unroll_limit_abstains",
    "TensorGuard.SubjectReduction.unsupported_loop_cannot_silently_safe",
    "TensorGuard.SubjectReduction.bounded_unroll_subject_reduction",
    "TensorGuard.SubjectReduction.modulelist_unroll_exec_shape",
    "TensorGuard.SubjectReduction.static_range_unroll_exec_shape",
    "TensorGuard.SubjectReduction.modulelist_beyond_limit_rejected",
    "TensorGuard.SubjectReduction.static_range_beyond_limit_rejected",
    "TensorGuard.SubjectReduction.unsupported_loop_abstains_example",
]

_UNDER_LIMIT_MODULELIST = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
"""

_OVER_LIMIT_MODULELIST = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
"""

_UNDER_LIMIT_STATIC_RANGE = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        for _ in range(3):
            x = self.fc(x)
        return x
"""

_OVER_LIMIT_STATIC_RANGE = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        for _ in range(4):
            x = self.fc(x)
        return x
"""

_STATIC_LEN_RANGE = """
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, x):
        for i in range(len(self.layers)):
            x = x + 0
        return x
"""

_DATA_DEPENDENT_RANGE = """
import torch
import torch.nn as nn

class M(nn.Module):
    def forward(self, x):
        for _ in range(int(x.sum().item())):
            x = x + 0
        return x
"""


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _exec_model(source: str):
    ns: dict[str, object] = {}
    exec(textwrap.dedent(source), ns)
    return ns["M"]()


def test_loop_theorems_are_audited_and_finished():
    code = _strip_comments(open(_SUBJECT_REDUCTION).read())
    assert not re.search(r"\b(sorry|admit|axiom)\b", code)

    audit = open(_AXIOM_AUDIT).read()
    for theorem in _LOOP_THEOREMS:
        assert f"#print axioms {theorem}" in audit


def test_modulelist_under_limit_is_unrolled_and_safe_on_real_code():
    graph = extract_computation_graph(
        _UNDER_LIMIT_MODULELIST,
        max_loop_unrolls=3,
    )
    assert graph.loop_abstentions == []
    assert {"__layers_unroll_0", "__layers_unroll_1", "__layers_unroll_2"}.issubset(
        graph.layers
    )

    result = verify_architecture(
        _UNDER_LIMIT_MODULELIST,
        input_shapes={"x": (2, 8)},
        max_cegar_iterations=0,
        max_loop_unrolls=3,
        soundness_mode="sound",
    )
    assert result.verdict == "SAFE"
    assert result.unknown_reasons == []

    model = _exec_model(_UNDER_LIMIT_MODULELIST)
    assert tuple(model(torch.zeros(2, 8)).shape) == (2, 8)


def test_modulelist_beyond_limit_is_unknown_in_sound_mode_only():
    graph = extract_computation_graph(
        _OVER_LIMIT_MODULELIST,
        max_loop_unrolls=3,
    )
    assert graph.loop_abstentions
    assert graph.loop_abstentions[0]["iterations"] == 4
    assert graph.loop_abstentions[0]["limit"] == 3
    assert "__layers_unroll_0" not in graph.layers

    sound = verify_architecture(
        _OVER_LIMIT_MODULELIST,
        input_shapes={"x": (2, 8)},
        max_cegar_iterations=0,
        max_loop_unrolls=3,
        soundness_mode="sound",
    )
    assert sound.bugs == []
    assert sound.verdict == "UNKNOWN"
    assert any("max_loop_unrolls=3" in r for r in sound.unknown_reasons)

    balanced = verify_architecture(
        _OVER_LIMIT_MODULELIST,
        input_shapes={"x": (2, 8)},
        max_cegar_iterations=0,
        max_loop_unrolls=3,
        soundness_mode="balanced",
    )
    assert balanced.verdict == "SAFE"


def test_literal_static_range_under_and_over_limit():
    under = verify_architecture(
        _UNDER_LIMIT_STATIC_RANGE,
        input_shapes={"x": (2, 8)},
        max_cegar_iterations=0,
        max_loop_unrolls=3,
        soundness_mode="sound",
    )
    assert under.verdict == "SAFE"

    over = verify_architecture(
        _OVER_LIMIT_STATIC_RANGE,
        input_shapes={"x": (2, 8)},
        max_cegar_iterations=0,
        max_loop_unrolls=3,
        soundness_mode="sound",
    )
    assert over.verdict == "UNKNOWN"
    assert any("static_range" in r for r in over.unknown_reasons)


def test_range_len_is_not_misclassified_as_data_dependent_iteration():
    issues, _warnings = _analyze_source(_STATIC_LEN_RANGE)
    assert UnsupportedCategory.DATA_DEPENDENT_ITERATION not in {
        issue.category for issue in issues
    }


def test_data_dependent_range_still_abstains_in_sound_mode():
    issues, _warnings = _analyze_source(_DATA_DEPENDENT_RANGE)
    assert UnsupportedCategory.DATA_DEPENDENT_ITERATION in {
        issue.category for issue in issues
    }

    result = verify_architecture(
        _DATA_DEPENDENT_RANGE,
        input_shapes={"x": (2, 8)},
        max_cegar_iterations=0,
        max_loop_unrolls=3,
        soundness_mode="sound",
    )
    assert result.verdict == "UNKNOWN"
    assert any("DATA_DEPENDENT_ITERATION" in r for r in result.unknown_reasons)


@pytest.mark.slow
def test_loop_subject_reduction_lean_builds():
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
def test_loop_subject_reduction_axiom_clean():
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
    for theorem in _LOOP_THEOREMS:
        assert f"'{theorem}'" in proc.stdout, f"audit output missing {theorem}"
