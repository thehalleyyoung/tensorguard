"""Step 37 -- tests for the torch.export frontend and fx/export reconciliation."""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.export_extractor import (  # noqa: E402
    HAS_EXPORT,
    export_trace_to_graph,
    verify_module_export,
)
from src.fx_extractor import verify_module  # noqa: E402
from src.model_checker import OpKind  # noqa: E402
from evaluation import frontend_reconciliation as FR  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_EXPORT, reason="torch.export unavailable")


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.mp = nn.MaxPool2d(2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = torch.relu(self.bn(self.c1(x)))
        x = self.mp(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class _MLPBad(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(99, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---- lowering ------------------------------------------------------------
def test_export_lowers_mlp_to_layer_calls():
    g = export_trace_to_graph(_MLP().eval(), example_inputs=(torch.randn(2, 8),))
    ops = [s.op for s in g.steps]
    assert ops.count(OpKind.LAYER_CALL) == 2
    assert OpKind.ACTIVATION in ops
    # Two recognised Linear layers, recovered from lifted parameters.
    assert len(g.layers) == 2


def test_export_recovers_layer_dims_from_lifted_params():
    g = export_trace_to_graph(_MLP().eval(), example_inputs=(torch.randn(2, 8),))
    linears = [l for l in g.layers.values() if l.in_features is not None]
    dims = sorted((l.in_features, l.out_features) for l in linears)
    assert dims == [(8, 16), (16, 4)]


def test_export_pooling_is_precise_not_unsupported():
    g = export_trace_to_graph(_CNN().eval(),
                              example_inputs=(torch.randn(1, 3, 16, 16),))
    assert OpKind.UNSUPPORTED not in [s.op for s in g.steps]
    # conv + bn + adaptive/max pool + linear all surface as LAYER_CALLs.
    assert sum(1 for s in g.steps if s.op == OpKind.LAYER_CALL) >= 5


def test_export_verifies_cnn_safe():
    r = verify_module_export(_CNN().eval(), input_shapes={"x": (1, 3, 16, 16)},
                            example_inputs=(torch.randn(1, 3, 16, 16),))
    assert r.safe is True
    assert r.errors == []


def test_export_capture_failure_is_unsafe_not_crash():
    # torch.export validates shapes eagerly; a buggy model fails to capture and
    # is reported unsafe rather than raising.
    r = verify_module_export(_MLPBad().eval(), input_shapes={"x": (2, 8)},
                            example_inputs=(torch.randn(2, 8),))
    assert r.safe is False
    assert any("extraction failed" in e for e in r.errors)


# ---- reconciliation against fx ------------------------------------------
def test_fx_and_export_agree_on_mlp():
    rf = verify_module(_MLP().eval(), input_shapes={"x": (2, 8)}, backend="fx")
    re = verify_module_export(_MLP().eval(), input_shapes={"x": (2, 8)},
                             example_inputs=(torch.randn(2, 8),))
    assert rf.safe == re.safe is True


def test_reconciliation_corpus_has_zero_divergences():
    rows = FR.evaluate_corpus()
    summ = FR._summarise(rows)
    assert summ["divergences"] == 0
    assert summ["fx_correct"] == summ["n_models"]
    assert summ["export_correct"] == summ["n_models"]


def test_reconciliation_gate_passes():
    assert FR.gate() == 0


def test_reconciliation_artifact_fresh_or_qualified():
    assert os.path.exists(FR.JSON_PATH)
    assert FR.run(check=True) == 0


def test_committed_reconciliation_reports_zero_divergences():
    rep = json.load(open(FR.JSON_PATH))
    assert rep["summary"]["divergences"] == 0
