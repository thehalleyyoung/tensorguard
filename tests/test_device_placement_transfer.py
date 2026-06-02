"""Step 140 — **device-placement chain transfer**, machine-checked in Lean and
cross-checked against real torch across ``cpu`` and ``mps``/``cuda``.

``lean/TensorGuard/DevicePlacement.lean`` lifts the single-tensor device tag to a
transfer function over a *chain* of ``.to(...)`` moves: ``keep`` propagates the
device tag, ``toCpu``/``toAccel`` move it (absorbing, last move wins).  It proves
the chain is compositional (``devRun_append``), a move is absorbing
(``run_after_move``, ``run_ends_at_target``), device is preserved on the move-free
fragment (``run_noMove_id``) and a binary op is valid iff both operands share a
device tag (``binValid_iff_eq`` — cross-device is *always* flagged).

This test mirrors ``devStep``/``devRun`` in Python and replays **every** chain on
a real tensor moved between ``cpu`` and the available accelerator, asserting
``tensor.device.type`` equals the Lean ``devRun`` prediction; and that a binary
op on two real tensors raises in eager torch exactly when the Lean ``binValid``
predicate is ``false``.
"""

import itertools
import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "DevicePlacement.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.DevicePlacement.keep_id",
    "TensorGuard.DevicePlacement.move_absorbing",
    "TensorGuard.DevicePlacement.devRun_append",
    "TensorGuard.DevicePlacement.run_after_move",
    "TensorGuard.DevicePlacement.run_ends_at_target",
    "TensorGuard.DevicePlacement.run_noMove_id",
    "TensorGuard.DevicePlacement.binValid_refl",
    "TensorGuard.DevicePlacement.binValid_iff_eq",
    "TensorGuard.DevicePlacement.cpu_accel_invalid",
    "TensorGuard.DevicePlacement.chain_binValid_iff",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# Python mirror of the Lean transfer.  Tags: "cpu", "accel".
def _dev_step(d, op):
    return {"keep": d, "toCpu": "cpu", "toAccel": "accel"}[op]


def _dev_run(d0, ops):
    d = d0
    for op in ops:
        d = _dev_step(d, op)
    return d


def _bin_valid(a, b):
    return a == b


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.DevicePlacement" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against real torch device placement.
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


def _accel_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return None


_ACCEL = _accel_device()


def _tag_to_device(tag):
    return "cpu" if tag == "cpu" else _ACCEL


def _torch_run(d0, ops):
    cur = torch.ones(4, device=_tag_to_device(d0))
    for op in ops:
        if op == "keep":
            cur = cur + 1.0
        elif op == "toCpu":
            cur = cur.to("cpu")
        elif op == "toAccel":
            cur = cur.to(_ACCEL)
    return cur.device.type


@pytest.mark.skipif(_ACCEL is None, reason="no accelerator (cuda/mps) available")
def test_dev_run_matches_real_torch_all_chains():
    ops_alphabet = ["keep", "toCpu", "toAccel"]
    accel_type = torch.device(_ACCEL).type
    checked = 0
    for d0 in ("cpu", "accel"):
        for length in range(0, 4):
            for chain in itertools.product(ops_alphabet, repeat=length):
                chain = list(chain)
                predicted_tag = _dev_run(d0, chain)
                predicted_type = "cpu" if predicted_tag == "cpu" else accel_type
                assert _torch_run(d0, chain) == predicted_type, (d0, chain)
                checked += 1
    assert checked == 2 * sum(3 ** n for n in range(4))


@pytest.mark.skipif(_ACCEL is None, reason="no accelerator (cuda/mps) available")
def test_binary_op_raises_exactly_on_cross_device():
    for a_tag in ("cpu", "accel"):
        for b_tag in ("cpu", "accel"):
            a = torch.ones(4, device=_tag_to_device(a_tag))
            b = torch.ones(4, device=_tag_to_device(b_tag))
            predicted_valid = _bin_valid(a_tag, b_tag)
            try:
                _ = a + b
                raised = False
            except RuntimeError:
                raised = True
            # eager torch raises iff the Lean binValid predicate is false
            assert (not raised) == predicted_valid, (a_tag, b_tag)


def test_move_free_chain_preserves_device():
    # run_noMove_id: a chain of only `keep` ops preserves the device.
    cur = torch.ones(4)  # cpu
    for _ in range(3):
        cur = cur + 1.0
    assert cur.device.type == _dev_run("cpu", ["keep", "keep", "keep"])


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.DevicePlacement"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.DevicePlacement"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_DevicePlacementAxCheck.lean")
    body = "import TensorGuard.DevicePlacement\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_DevicePlacementAxCheck.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=900, env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    out = proc.stdout
    assert "sorryAx" not in out
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"axiom output missing {thm}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
