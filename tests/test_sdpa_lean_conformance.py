"""Step 230 — Lean-checked SDPA broadcasting, masks, and GQA caveat.

``lean/TensorGuard/SDPA.lean`` mechanizes the shape contract used by
``src.sdpa_verify.verify_sdpa``:

* ordinary SDPA right-aligns and broadcasts query/key/value leading dimensions;
* masks broadcast against the post-q/k score tensor ``(..., L_q, L_k)``;
* explicit ``enable_gqa=True`` is scoped to PyTorch's ``-3`` head axis: key and
  value head counts must divide query heads, and the output uses query heads.

The concrete cases below are generated from those theorem shapes and checked
against real ``torch.nn.functional.scaled_dot_product_attention``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.sdpa_verify import verify_sdpa  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "SDPA.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.SDPA.bcDim_none_iff",
    "TensorGuard.SDPA.bcShape_same",
    "TensorGuard.SDPA.bcShape_suffix_same",
    "TensorGuard.SDPA.standard_output_shape",
    "TensorGuard.SDPA.standard_output_rank",
    "TensorGuard.SDPA.standard_equal_leads",
    "TensorGuard.SDPA.mask_exact_valid",
    "TensorGuard.SDPA.mask_trailing_valid",
    "TensorGuard.SDPA.gqaHeadsValid_iff",
    "TensorGuard.SDPA.gqa_key_repetition_count",
    "TensorGuard.SDPA.gqa_value_repetition_count",
    "TensorGuard.SDPA.gqa_nondivisible_key_flagged",
    "TensorGuard.SDPA.gqa_nondivisible_value_flagged",
    "TensorGuard.SDPA.gqa_output_shape",
    "TensorGuard.SDPA.gqa_output_rank",
    "TensorGuard.SDPA.gqa_output_uses_query_heads",
    "TensorGuard.SDPA.gqa_prefix_broadcast_required",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _real_sdpa(q, k, v, mask=None, *, enable_gqa=False):
    tq = torch.randn(*q)
    tk = torch.randn(*k)
    tv = torch.randn(*v)
    tm = None if mask is None else torch.zeros(*mask, dtype=torch.float32)
    kwargs = {"attn_mask": tm}
    if enable_gqa:
        kwargs["enable_gqa"] = True
    try:
        out = F.scaled_dot_product_attention(tq, tk, tv, **kwargs)
    except Exception:
        return "err", None
    return "ok", tuple(out.shape)


def _supports_gqa() -> bool:
    try:
        F.scaled_dot_product_attention(
            torch.randn(1, 1, 2, 4),
            torch.randn(1, 1, 3, 4),
            torch.randn(1, 1, 3, 4),
            enable_gqa=True,
        )
    except TypeError:
        return False
    except Exception:
        return True
    return True


def _check(q, k, v, mask=None, *, enable_gqa=False):
    real_status, real_shape = _real_sdpa(q, k, v, mask, enable_gqa=enable_gqa)
    verdict = verify_sdpa(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)
    static_status = "ok" if verdict.ok else "err"
    assert static_status == real_status, (
        f"q={q} k={k} v={v} mask={mask} gqa={enable_gqa}: "
        f"real={real_status} static={static_status} ({verdict.error})"
    )
    if real_status == "ok":
        assert verdict.output_shape == real_shape


def _ordinary_cases():
    yield (2, 4, 5, 8), (2, 4, 7, 8), (2, 4, 7, 16), None
    yield (1, 4, 5, 8), (2, 4, 7, 8), (2, 4, 7, 12), None
    yield (3, 5, 8), (3, 7, 8), (3, 7, 9), None
    yield (2, 4, 5, 8), (2, 4, 7, 8), (2, 4, 7, 16), (2, 4, 5, 7)
    yield (2, 4, 5, 8), (2, 4, 7, 8), (2, 4, 7, 16), (5, 7)
    yield (2, 8, 5, 16), (2, 2, 7, 16), (2, 2, 7, 16), None
    yield (2, 4, 5, 8), (2, 4, 7, 9), (2, 4, 7, 16), None
    yield (2, 4, 5, 8), (2, 5, 7, 8), (2, 5, 7, 16), None
    yield (2, 4, 5, 8), (2, 4, 7, 8), (2, 4, 7, 16), (2, 4, 5, 6)


def _gqa_cases():
    yield (8, 5, 16), (2, 7, 16), (2, 7, 16), None
    yield (1, 5, 16), (3, 7, 16), (3, 7, 16), None
    yield (2, 8, 5, 16), (2, 4, 7, 16), (2, 2, 7, 32), None
    yield (2, 8, 5, 16), (2, 2, 7, 16), (2, 3, 7, 32), None
    yield (1, 8, 5, 16), (3, 2, 7, 16), (3, 2, 7, 16), None
    yield (3, 8, 5, 16), (2, 2, 7, 16), (3, 2, 7, 16), None
    yield (3, 8, 5, 16), (1, 2, 7, 16), (3, 2, 7, 16), (3, 8, 5, 7)
    yield (3, 8, 5, 16), (1, 2, 7, 16), (3, 2, 7, 16), (5, 7)
    yield (3, 8, 5, 16), (1, 2, 7, 16), (3, 2, 7, 16), (3, 2, 5, 7)
    yield (2, 4, 5, 8), (2, 4, 7, 8), (2, 4, 7, 12), None


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.SDPA" in fh.read()
    with open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")) as fh:
        audit = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry_or_admit():
    with open(_FILE) as fh:
        code = _strip_comments(fh.read())
    assert not re.search(r"\b(sorry|admit)\b", code)


# --------------------------------------------------------------------------- #
# 2. Generated conformance: ordinary SDPA broadcast + masks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q,k,v,mask", list(_ordinary_cases()))
def test_generated_ordinary_sdpa_cases_match_real_torch(q, k, v, mask):
    _check(q, k, v, mask, enable_gqa=False)


def test_symbolic_ordinary_sdpa_abstains_without_false_refutation():
    verdict = verify_sdpa(
        ("Bq", 4, "Lq", 16),
        ("Bk", 4, "Lk", 16),
        ("Bk", 4, "Lk", 32),
        attn_mask=("Bq", 4, "Lq", "Lk"),
    )
    assert verdict.ok
    assert verdict.output_shape == ("Bq", 4, "Lq", 32)


# --------------------------------------------------------------------------- #
# 3. Generated conformance: scoped enable_gqa head caveat
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q,k,v,mask", list(_gqa_cases()))
def test_generated_gqa_cases_match_real_torch(q, k, v, mask):
    if not _supports_gqa():
        pytest.skip("installed torch lacks enable_gqa")
    _check(q, k, v, mask, enable_gqa=True)


def test_symbolic_gqa_uses_query_heads_and_abstains_on_divisibility():
    verdict = verify_sdpa(
        ("B", "Hq", "Lq", 16),
        ("B", "Hk", "Lk", 16),
        ("B", "Hv", "Lk", 32),
        enable_gqa=True,
    )
    assert verdict.ok
    assert verdict.output_shape == ("B", "Hq", "Lq", 32)


def test_gqa_flag_changes_rank3_head_axis_from_broadcast_to_divisibility():
    if not _supports_gqa():
        pytest.skip("installed torch lacks enable_gqa")
    ordinary = verify_sdpa((1, 5, 16), (3, 7, 16), (3, 7, 16))
    gqa = verify_sdpa((1, 5, 16), (3, 7, 16), (3, 7, 16), enable_gqa=True)
    assert ordinary.ok
    assert ordinary.output_shape == (3, 5, 16)
    assert not gqa.ok
    assert gqa.error_kind == "gqa_heads"
    _check((1, 5, 16), (3, 7, 16), (3, 7, 16), enable_gqa=True)


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.SDPA"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.SDPA"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_SDPAAxCheck.lean")
    body = "import TensorGuard.SDPA\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS
    ) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_SDPAAxCheck.lean"],
            cwd=_LEAN,
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
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
