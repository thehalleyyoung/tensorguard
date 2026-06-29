"""Step (even_more #4) — dtype/device lattice: device-type mismatch detector.

A binary op (``+``, ``@``, …) on two tensors whose *device types* are statically
known and different (e.g. ``cpu`` vs ``cuda``) is a forced ``RuntimeError`` at
runtime.  The detector is sound: it fires only when both device types are pinned
and genuinely differ; an unknown device, or the same type with a different
ordinal (``cuda`` vs ``cuda:0``), abstains.
"""

from src.symexec.bugs import SymBugKind
from src.symexec.engine import analyze_source, analyze_file


def _kinds(src):
    return [b.kind.value for b in analyze_source(src).bugs]


def _main(body):
    return 'import torch\nif __name__ == "__main__":\n' + "".join(
        "    " + line + "\n" for line in body
    )


# ---- device kwarg captured on the constructor --------------------------------

def test_device_kwarg_captured_matmul():
    src = _main([
        'a = torch.randn(2, 3, device="cuda")',
        'b = torch.randn(3, 4, device="cpu")',
        "c = a @ b",
    ])
    assert "device_mismatch" in _kinds(src)


def test_device_kwarg_captured_add():
    src = _main([
        'a = torch.zeros(2, 3, device="cuda")',
        'b = torch.ones(2, 3, device="cpu")',
        "c = a + b",
    ])
    assert "device_mismatch" in _kinds(src)


# ---- .cuda()/.cpu()/.to() update the device metadata -------------------------

def test_cuda_cpu_methods_set_device():
    src = _main([
        "a = torch.randn(2, 3).cuda()",
        "b = torch.randn(2, 3).cpu()",
        "c = a + b",
    ])
    assert "device_mismatch" in _kinds(src)


def test_to_method_sets_device():
    src = _main([
        'a = torch.randn(2, 3).to("cuda")',
        'b = torch.randn(2, 3).to("cpu")',
        "c = a + b",
    ])
    assert "device_mismatch" in _kinds(src)


# ---- soundness: abstain when a device is unknown -----------------------------

def test_abstains_when_one_device_unknown():
    src = _main([
        'a = torch.randn(2, 3).to("cuda")',
        "b = torch.randn(2, 3)",  # device unknown
        "c = a + b",
    ])
    assert "device_mismatch" not in _kinds(src)


def test_abstains_both_devices_unknown():
    src = _main([
        "a = torch.randn(2, 3)",
        "b = torch.randn(2, 3)",
        "c = a + b",
    ])
    assert "device_mismatch" not in _kinds(src)


# ---- soundness: same device type, different ordinal does NOT fire -------------

def test_cuda_vs_cuda0_no_false_positive():
    src = _main([
        'a = torch.randn(2, 3, device="cuda")',
        'b = torch.randn(2, 3, device="cuda:0")',
        "c = a + b",
    ])
    assert "device_mismatch" not in _kinds(src)


def test_same_device_clean():
    src = _main([
        "a = torch.randn(2, 3).cuda()",
        "b = torch.randn(2, 3).cuda()",
        "c = a + b",
    ])
    assert "device_mismatch" not in _kinds(src)


# ---- fires on matmul as well as elementwise ----------------------------------

def test_fires_on_matmul_path():
    src = _main([
        'a = torch.randn(2, 3, device="cpu")',
        'b = torch.randn(3, 4, device="cuda")',
        "c = torch.matmul(a, b)",
    ])
    kinds = _kinds(src)
    assert "device_mismatch" in kinds


# ---- the new kind carries the TYPE_ERROR category ----------------------------

def test_kind_category():
    from src.symexec.bugs import _API_CATEGORY

    assert _API_CATEGORY[SymBugKind.DEVICE_MISMATCH] == "TYPE_ERROR"


def test_message_and_fix_present():
    src = _main([
        'a = torch.randn(2, 3, device="cuda")',
        'b = torch.randn(2, 3, device="cpu")',
        "c = a + b",
    ])
    bugs = [b for b in analyze_source(src).bugs if b.kind is SymBugKind.DEVICE_MISMATCH]
    assert bugs
    bug = bugs[0]
    assert "cuda" in bug.message and "cpu" in bug.message
    assert bug.fix_suggestion


# ---- corpus fingerprint is unaffected by the new kind ------------------------

def test_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )
