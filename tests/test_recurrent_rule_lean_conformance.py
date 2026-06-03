"""Step 237 — Lean recurrent hidden-state contracts checked against real torch.

``lean/TensorGuard/RecurrentRule.lean`` mechanizes the shape-only
RNN/GRU/LSTM contract used by TensorGuard: output features are
``num_directions * H_out``, hidden-state depth is ``num_directions *
num_layers``, ``batch_first`` selects the state batch axis, and projected LSTM
``c_n`` keeps ``hidden_size`` while output/``h_n`` use ``proj_size``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from src.model_checker import SymbolicShapePropagator, extract_computation_graph

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "RecurrentRule.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.RecurrentRule.batch_first_output_preserves_layout",
    "TensorGuard.RecurrentRule.time_major_output_preserves_layout",
    "TensorGuard.RecurrentRule.batch_first_state_selects_dim0",
    "TensorGuard.RecurrentRule.time_major_state_selects_dim1",
    "TensorGuard.RecurrentRule.bidirectional_output_feature_doubles",
    "TensorGuard.RecurrentRule.bidirectional_state_depth_doubles",
    "TensorGuard.RecurrentRule.lstm_cell_state_uses_hidden_size_under_projection",
    "TensorGuard.RecurrentRule.gru_cell_state_rejected",
    "TensorGuard.RecurrentRule.rnn_cell_state_rejected",
    "TensorGuard.RecurrentRule.projected_bilstm_output_shape",
    "TensorGuard.RecurrentRule.projected_bilstm_h_state_shape",
    "TensorGuard.RecurrentRule.projected_bilstm_c_state_shape",
    "TensorGuard.RecurrentRule.time_major_bigru_output_shape",
    "TensorGuard.RecurrentRule.time_major_bigru_h_state_shape",
    "TensorGuard.RecurrentRule.unbatched_rnn_output_shape",
    "TensorGuard.RecurrentRule.unbatched_rnn_h_state_shape",
    "TensorGuard.RecurrentRule.wrong_input_size_rejected",
    "TensorGuard.RecurrentRule.bad_rank_rejected",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _dims(shape):
    return tuple(dim.value for dim in shape.dims)


def _propagated_shapes(source: str, input_shape):
    graph = extract_computation_graph(source)
    env = SymbolicShapePropagator(graph).propagate({"x": input_shape})
    return {name: _dims(shape) for name, shape in env.items()}


def test_lean_recurrent_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.RecurrentRule" in fh.read()
    with open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")) as fh:
        audit = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_recurrent_file_has_no_sorry_or_admit():
    with open(_FILE) as fh:
        code = _strip_comments(fh.read())
    assert not re.search(r"\b(sorry|admit)\b", code)


def test_projected_bilstm_batch_first_contract_matches_torch_and_verifier():
    source = """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=11, hidden_size=9, num_layers=2,
            batch_first=True, bidirectional=True, proj_size=5,
        )
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, h_n, c_n
"""
    env = _propagated_shapes(source, (3, 5, 11))
    assert env["output"] == (3, 5, 10)
    assert env["h_n"] == (4, 3, 5)
    assert env["c_n"] == (4, 3, 9)

    ns = {}
    exec(source, ns)
    output, h_n, c_n = ns["M"]()(torch.zeros(3, 5, 11))
    assert tuple(output.shape) == env["output"]
    assert tuple(h_n.shape) == env["h_n"]
    assert tuple(c_n.shape) == env["c_n"]


def test_time_major_bigru_selects_second_dim_for_state_batch():
    source = """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=8, hidden_size=6, num_layers=3,
            batch_first=False, bidirectional=True,
        )
    def forward(self, x):
        output, h_n = self.gru(x)
        return output, h_n
"""
    env = _propagated_shapes(source, (7, 4, 8))
    assert env["output"] == (7, 4, 12)
    assert env["h_n"] == (6, 4, 6)

    ns = {}
    exec(source, ns)
    output, h_n = ns["M"]()(torch.zeros(7, 4, 8))
    assert tuple(output.shape) == env["output"]
    assert tuple(h_n.shape) == env["h_n"]


def test_unbatched_bidirectional_rnn_has_no_batch_axis_in_state():
    source = """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=6, hidden_size=5, num_layers=2,
            bidirectional=True,
        )
    def forward(self, x):
        output, h_n = self.rnn(x)
        return output, h_n
"""
    env = _propagated_shapes(source, (9, 6))
    assert env["output"] == (9, 10)
    assert env["h_n"] == (4, 5)

    ns = {}
    exec(source, ns)
    output, h_n = ns["M"]()(torch.zeros(9, 6))
    assert tuple(output.shape) == env["output"]
    assert tuple(h_n.shape) == env["h_n"]


@pytest.mark.slow
def test_recurrent_rule_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.RecurrentRule"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_recurrent_rule_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    # AxiomAudit imports the umbrella TensorGuard module, so rebuild the root
    # after adding the new import rather than relying on a stale lake cache.
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
    illegal = seen - _TRUSTED_AXIOMS
    assert not illegal, f"untrusted axioms in recurrent proofs: {illegal}"

    for thm in _THEOREMS:
        assert f"'{thm}'" in proc.stdout, f"audit output missing {thm}"
