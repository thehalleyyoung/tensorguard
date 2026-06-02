"""Tests for quantization & export safety checks (Step 98)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import reproducibility.quant_export_safety as harness  # noqa: E402
from src.quant_export_checks import (  # noqa: E402
    Confidence,
    ExportHazardKind,
    QuantHazardKind,
    analyze_export_safety,
    analyze_quantization,
    summarize,
)

QUANT_ADD_BAD = '''
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, a, b):
        a = self.quant(a)
        b = self.quant(b)
        return self.dequant(a + b)
'''

QUANT_ADD_GOOD = '''
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.nn.quantized import FloatFunctional
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.ff = FloatFunctional()
    def forward(self, a, b):
        a = self.quant(a)
        b = self.quant(b)
        return self.dequant(self.ff.add(a, b))
'''

PLAIN_FLOAT = '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)
    def forward(self, x):
        return self.l(x) + self.l(x)
'''


# --- quantization analyzer -------------------------------------------------
def test_quant_arith_without_floatfunctional_flagged():
    hz = analyze_quantization(QUANT_ADD_BAD)
    kinds = {h.kind for h in hz}
    assert QuantHazardKind.QUANT_ARITH_WITHOUT_FLOATFUNCTIONAL in kinds


def test_quant_arith_with_floatfunctional_clean():
    hz = analyze_quantization(QUANT_ADD_GOOD)
    kinds = {h.kind for h in hz}
    assert QuantHazardKind.QUANT_ARITH_WITHOUT_FLOATFUNCTIONAL not in kinds


def test_plain_float_model_not_flagged_as_quant():
    # No QuantStub/DeQuantStub => not a quantization module => no quant hazards.
    assert analyze_quantization(PLAIN_FLOAT) == []


def test_missing_dequantstub_flagged():
    src = '''
import torch.nn as nn
from torch.ao.quantization import QuantStub
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.l = nn.Linear(4, 4)
    def forward(self, x):
        return self.l(self.quant(x))
'''
    kinds = {h.kind for h in analyze_quantization(src)}
    assert QuantHazardKind.MISSING_DEQUANTSTUB in kinds


def test_missing_quantstub_flagged():
    src = '''
import torch.nn as nn
from torch.ao.quantization import DeQuantStub
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.dequant = DeQuantStub()
        self.l = nn.Linear(4, 4)
    def forward(self, x):
        return self.dequant(self.l(x))
'''
    kinds = {h.kind for h in analyze_quantization(src)}
    assert QuantHazardKind.MISSING_QUANTSTUB in kinds


def test_quant_scalar_mul_not_flagged():
    # ``x * 2`` is a tensor-scalar op, legal on quantized tensors path; only
    # tensor-tensor arithmetic is the FloatFunctional hazard.
    src = '''
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        return self.dequant(x * 2)
'''
    kinds = {h.kind for h in analyze_quantization(src)}
    assert QuantHazardKind.QUANT_ARITH_WITHOUT_FLOATFUNCTIONAL not in kinds


def test_quant_hazard_confidence_is_heuristic():
    for h in analyze_quantization(QUANT_ADD_BAD):
        assert h.confidence is Confidence.HEURISTIC


# --- export analyzer -------------------------------------------------------
def test_export_data_dependent_branch_flagged():
    src = '''
import torch.nn as nn
class M(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        return x
'''
    kinds = {h.kind for h in analyze_export_safety(src)}
    assert ExportHazardKind.DATA_DEPENDENT_CONTROL_FLOW in kinds


def test_export_item_flagged():
    src = '''
import torch.nn as nn
class M(nn.Module):
    def forward(self, x):
        k = int(x.sum().item())
        return x + k
'''
    kinds = {h.kind for h in analyze_export_safety(src)}
    assert ExportHazardKind.TENSOR_TO_SCALAR in kinds


def test_export_clean_linear_no_hazard():
    src = '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)
    def forward(self, x):
        return self.l(x)
'''
    assert analyze_export_safety(src) == []


def test_export_hazard_confidence_is_sound():
    src = '''
import torch.nn as nn
class M(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        return x * 2
'''
    for h in analyze_export_safety(src):
        assert h.confidence is Confidence.SOUND


def test_summarize_keys():
    s = summarize(QUANT_ADD_BAD)
    assert set(s) >= {
        "quant_hazards",
        "export_hazards",
        "n_quant_hazards",
        "n_export_hazards",
        "export_safe",
        "quant_safe",
    }
    assert s["quant_safe"] is False


# --- live consistency vs real torch ---------------------------------------
def test_live_export_consistency():
    data = harness.measure()
    assert data["all_export_consistent"] is True
    # at least one clean and one failing export case exercised
    cleans = [r for r in data["export_cases"] if r["live_exports_clean"]]
    fails = [r for r in data["export_cases"] if not r["live_exports_clean"]]
    assert cleans and fails


def test_live_quant_add_raises_consistency():
    data = harness.measure()
    assert data["all_quant_consistent"] is True
    bad = next(
        r for r in data["quant_cases"]
        if r["name"] == "quant_add_no_floatfunctional"
    )
    assert bad["static_quant_hazard"] is True
    assert bad["live_raises"] is True


# --- determinism -----------------------------------------------------------
_VOLATILE_SUBSTRINGS = (
    "time", "elapsed", "timestamp", "wall", "clock",
    "_ms", "seconds", "duration", "date",
)


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_artifact_has_no_volatile_fields():
    data = harness.measure()
    for key in _walk_keys(data):
        low = key.lower()
        for bad in _VOLATILE_SUBSTRINGS:
            assert bad not in low, f"volatile substring {bad!r} in key {key!r}"


def test_artifact_is_byte_deterministic():
    assert harness.run(check=True) == 0
