"""Step 134 — machine-checked **non-shape transfer functions** (device / dtype /
phase / gradient), cross-checked against the live verifier.

`lean/TensorGuard/DeviceDtype.lean` models the four scalar algebras TensorGuard
runs alongside the shape algebra and proves, for each, *no false positive* (an
``unknown`` operand never flags) and *refutation soundness* (a flagged bug
witnesses a genuine torch runtime error), plus the structural laws the frontends
rely on and that the reduced product over the four inherits both properties.

This test guards two halves:

1. **The Lean proof** — a fast lexical ``sorry`` guard is always on; the build /
   axiom-audit checks are toolchain-gated and skip when ``lake`` is absent.

2. **The live cross-check** — a Python mirror of each Lean ``*Bug`` predicate is
   evaluated and the **real** ``verify_module`` verdict on a curated set of
   ``nn.Module``s must agree with the Lean model's prediction. The
   device/dtype/phase/gradient-*consistent* cases are additionally **executed**
   against real torch to prove they do not raise.
"""

import os
import re
import shutil
import subprocess

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.fx_extractor import verify_module  # noqa: E402
from src.model_checker import Phase  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "DeviceDtype.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.DevDtype.devBug_no_false_positive",
    "TensorGuard.DevDtype.devBug_refutation_sound",
    "TensorGuard.DevDtype.cuda_then_cpu_roundtrip",
    "TensorGuard.DevDtype.pinMemory_preserves",
    "TensorGuard.DevDtype.dtMatmulBug_no_false_positive",
    "TensorGuard.DevDtype.dtMatmulBug_refutation_sound",
    "TensorGuard.DevDtype.dtFloatParamBug_refutation_sound",
    "TensorGuard.DevDtype.dtElementwise_never_bug",
    "TensorGuard.DevDtype.dtPromote_comm",
    "TensorGuard.DevDtype.phaseBug_count_gt_one",
    "TensorGuard.DevDtype.phaseBug_eval_tracking_safe",
    "TensorGuard.DevDtype.phaseBug_refutation_sound",
    "TensorGuard.DevDtype.gradBrokenBug_refutation_sound",
    "TensorGuard.DevDtype.productBug_false_iff",
    "TensorGuard.DevDtype.productBug_refutation_sound",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# --------------------------------------------------------------------------- #
# Python mirrors of the Lean `*Bug` predicates (None == Lean `unknown`).
# --------------------------------------------------------------------------- #
_FLOAT = {"float16", "bfloat16", "float32", "float64"}


def _dev_bug(a, b):
    if a is None or b is None:
        return False
    return a != b


def _dtype_matmul_bug(a, b):
    if a is None or b is None:
        return False
    return a != b


def _dtype_float_param_bug(inp):
    if inp is None:
        return False
    return inp not in _FLOAT


def _dtype_elementwise_bug(a, b):
    return False


def _phase_bug(phase, track_running_stats, count_per_channel):
    uses_batch_stats = (phase == "train") or (not track_running_stats)
    return uses_batch_stats and count_per_channel == 1


def _grad_broken_bug(input_requires_grad):
    return input_requires_grad


def _kinds(res):
    if res.safe or res.counterexample is None:
        return set()
    return {v.kind for v in res.counterexample.violations}


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_exists_imported_and_audited():
    assert os.path.exists(_FILE), "DeviceDtype.lean missing"
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.DeviceDtype" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check: device algebra.
# --------------------------------------------------------------------------- #
def test_device_consistent_matches_lean_and_runs():
    class M(nn.Module):
        def forward(self, x, y):
            return x.cuda() + y.cuda()        # both cuda -> devBug = False

    res = verify_module(M(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert ("device_mismatch" in _kinds(res)) == _dev_bug("cuda:0", "cuda:0")

    # Executable surrogate (CI is CPU-only): a same-device add never raises.
    class Mcpu(nn.Module):
        def forward(self, x, y):
            return x.cpu() + y.cpu()

    rcpu = verify_module(Mcpu(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert ("device_mismatch" in _kinds(rcpu)) == _dev_bug("cpu", "cpu")
    Mcpu()(torch.randn(2, 4), torch.randn(2, 4))


def test_device_mismatch_matches_lean():
    class M(nn.Module):
        def forward(self, x, y):
            return x.to("cuda") + y           # cuda vs cpu -> devBug = True

    res = verify_module(M(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert ("device_mismatch" in _kinds(res)) == _dev_bug("cuda:0", "cpu")
    assert _dev_bug("cuda:0", "cpu") is True


def test_device_unknown_abstains():
    # x.to(y.device) -- target device taken from another tensor is *unknown*,
    # so the verifier inherits and never invents a mismatch (no false positive).
    class M(nn.Module):
        def forward(self, x, y):
            return x.to(y.device) + y

    res = verify_module(M(), input_shapes={"x": (2, 4), "y": (2, 4)})
    assert ("device_mismatch" in _kinds(res)) == _dev_bug(None, "cpu")
    assert _dev_bug(None, "cpu") is False


# --------------------------------------------------------------------------- #
# 3. Live cross-check: dtype algebra.
# --------------------------------------------------------------------------- #
def test_dtype_elementwise_promotes_matches_lean_and_runs():
    class M(nn.Module):
        def forward(self, x, y):
            return x + y                      # promotion -> never flags

    res = verify_module(
        M(), input_shapes={"x": (2, 4), "y": (2, 4)},
        input_dtypes={"x": "float32", "y": "float16"})
    assert ("dtype_error" in _kinds(res)) == _dtype_elementwise_bug(
        "float32", "float16")
    # executable: f32 + f16 type-promotes in torch, does not raise.
    M()(torch.randn(2, 4), torch.randn(2, 4).half())


def test_dtype_matmul_mismatch_matches_lean():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    res = verify_module(
        M(), input_shapes={"x": (2, 4), "y": (4, 3)},
        input_dtypes={"x": "float16", "y": "float32"})
    assert ("dtype_error" in _kinds(res)) == _dtype_matmul_bug(
        "float16", "float32")
    assert _dtype_matmul_bug("float16", "float32") is True


def test_dtype_linear_int_input_matches_lean():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 3)

        def forward(self, x):
            return self.lin(x)

    res = verify_module(
        M(), input_shapes={"x": (2, 4)}, input_dtypes={"x": "int64"})
    assert ("dtype_error" in _kinds(res)) == _dtype_float_param_bug("int64")
    assert _dtype_float_param_bug("int64") is True


def test_dtype_unknown_abstains():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 3)

        def forward(self, x):
            return self.lin(x)

    res = verify_module(M(), input_shapes={"x": (2, 4)})  # no input_dtypes
    assert ("dtype_error" in _kinds(res)) == _dtype_float_param_bug(None)
    assert _dtype_float_param_bug(None) is False


# --------------------------------------------------------------------------- #
# 4. Live cross-check: phase algebra (BatchNorm batch-stats count-1).
# --------------------------------------------------------------------------- #
def test_phase_bn_count_one_train_matches_lean():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(4)

        def forward(self, x):
            return self.bn(x)

    res = verify_module(
        M(), input_shapes={"x": (1, 4)}, default_phase=Phase.TRAIN)
    # BatchNorm1d at (1, 4): per-channel count = 1, default tracks running stats.
    assert ("phase_error" in _kinds(res)) == _phase_bug("train", True, 1)
    assert _phase_bug("train", True, 1) is True


def test_phase_bn_count_one_eval_safe_matches_lean():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(4)

        def forward(self, x):
            return self.bn(x)

    res = verify_module(
        M(), input_shapes={"x": (1, 4)}, default_phase=Phase.EVAL)
    assert ("phase_error" in _kinds(res)) == _phase_bug("eval", True, 1)
    assert _phase_bug("eval", True, 1) is False
    # executable in eval: running stats used, no ValueError.
    m = M().eval()
    m(torch.randn(1, 4))


# --------------------------------------------------------------------------- #
# 5. Live cross-check: gradient-flow (`detach`) algebra.
# --------------------------------------------------------------------------- #
def test_gradient_detach_breaks_flow_matches_lean():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x).detach() + 1   # detach a grad-carrying tensor

    res = verify_module(M(), input_shapes={"x": (2, 4)})
    assert ("gradient_broken" in _kinds(res)) == _grad_broken_bug(True)
    assert _grad_broken_bug(True) is True


def test_gradient_no_detach_safe_matches_lean():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x) + 1

    res = verify_module(M(), input_shapes={"x": (2, 4)})
    assert ("gradient_broken" in _kinds(res)) == _grad_broken_bug(False)
    assert _grad_broken_bug(False) is False


# --------------------------------------------------------------------------- #
# 6. Reduced product: a clean model flags nothing across all four algebras.
# --------------------------------------------------------------------------- #
def test_product_clean_model_no_flags():
    class Clean(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x) + 1

    res = verify_module(
        Clean(), input_shapes={"x": (8, 4)},
        input_dtypes={"x": "float32"}, default_phase=Phase.TRAIN)
    ks = _kinds(res)
    for k in ("device_mismatch", "dtype_error", "phase_error",
              "gradient_broken"):
        assert k not in ks, (k, ks)


# --------------------------------------------------------------------------- #
# 7. Toolchain-gated Lean build / axiom-audit.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.DeviceDtype"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"lake build failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.DeviceDtype"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_DeviceDtypeAxCheck.lean")
    body = "import TensorGuard.DeviceDtype\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        lean_path = os.path.join(_LEAN, ".lake", "build", "lib")
        env = dict(os.environ, LEAN_PATH=lean_path)
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_DeviceDtypeAxCheck.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=900, env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, (
        f"axiom check failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    out = proc.stdout
    assert "sorryAx" not in out, f"a transfer proof depends on sorryAx:\n{out}"
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}:\n{out}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"axiom output missing {thm}:\n{out}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
