"""Step 149 — **`nn.Embedding` shape + index-range rule**, machine-checked in Lean
and cross-checked against a real `nn.Embedding`.

``lean/TensorGuard/EmbeddingRule.lean`` proves the lookup raises the rank by one
with trailing dim ``embedding_dim`` (``emb_rank`` / ``emb_trailing`` /
``emb_prefix``), scales numel by ``embedding_dim`` (``emb_numel``), and that the
index guard passes **iff** every index is ``< num_embeddings``
(``allValid_iff`` / ``outOfRange_flagged``), with range monotonicity
(``allValid_mono``).

This test replays the rule on a **real** ``nn.Embedding`` and the verifier's own
``_propagate_embedding``: the output shape equals ``input.shape +
(embedding_dim,)`` and an **out-of-range index makes torch raise** — exactly when
the Lean guard flags it.
"""

import os
import re
import shutil
import subprocess

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "EmbeddingRule.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.EmbeddingRule.prod_append",
    "TensorGuard.EmbeddingRule.emb_rank",
    "TensorGuard.EmbeddingRule.emb_trailing",
    "TensorGuard.EmbeddingRule.emb_prefix",
    "TensorGuard.EmbeddingRule.emb_numel",
    "TensorGuard.EmbeddingRule.idxValid_iff",
    "TensorGuard.EmbeddingRule.allValid_iff",
    "TensorGuard.EmbeddingRule.outOfRange_flagged",
    "TensorGuard.EmbeddingRule.allValid_mono",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def _emb_shape(input_shape, emb_dim):
    return list(input_shape) + [emb_dim]


def _all_valid(idxs, num):
    return all(0 <= i < num for i in idxs)


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.EmbeddingRule" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against a real nn.Embedding.
# --------------------------------------------------------------------------- #
def test_embedding_shape_matches_real_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    checked = 0
    for num, dim, ishape in [(10, 4, (3,)), (100, 8, (2, 5)), (7, 16, (4, 2, 3))]:
        emb = nn.Embedding(num, dim)
        idx = torch.randint(0, num, ishape)
        y = emb(idx)
        predicted = _emb_shape(ishape, dim)
        assert list(y.shape) == predicted, (num, dim, ishape)
        assert len(y.shape) == len(ishape) + 1              # emb_rank
        assert y.shape[-1] == dim                            # emb_trailing
        assert list(y.shape[:-1]) == list(ishape)            # emb_prefix
        assert y.numel() == _prod(ishape) * dim              # emb_numel
        checked += 1
    assert checked > 0


def test_embedding_out_of_range_raises_iff_invalid():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    num, dim = 5, 3
    emb = nn.Embedding(num, dim)
    # in-range => Lean guard passes => no raise
    ok = torch.tensor([0, 1, 4])
    assert _all_valid(ok.tolist(), num)
    emb(ok)  # must not raise
    # out-of-range => Lean guard flags => torch raises
    bad = torch.tensor([0, 5])  # 5 == num is out of range
    assert not _all_valid(bad.tolist(), num)
    with pytest.raises((RuntimeError, IndexError)):
        emb(bad)
    # lower-bound: a real negative index is also rejected by the engine.
    neg = torch.tensor([0, -1])
    assert not _all_valid(neg.tolist(), num)
    with pytest.raises((RuntimeError, IndexError)):
        emb(neg)


def test_embedding_range_monotonicity():
    # allValid_mono: an index set valid for num stays valid for any num' >= num.
    idxs = [0, 3, 4]
    assert _all_valid(idxs, 5)
    for bigger in (5, 6, 100):
        assert _all_valid(idxs, bigger)


def test_embedding_matches_verifier_propagator():
    pytest.importorskip("torch")
    from src.tensor_shapes import ShapeDim, TensorShape
    from src.model_checker import _propagate_embedding, LayerDef, LayerKind

    for dim, ishape in [(4, (3,)), (8, (2, 5))]:
        ts = TensorShape(tuple(ShapeDim(int(d)) for d in ishape))
        ld = LayerDef(attr_name="emb", kind=LayerKind.EMBEDDING)
        ld.embedding_dim = dim
        pred, err = _propagate_embedding(ts, ld)
        assert err is None
        assert [d.value for d in pred.dims] == _emb_shape(ishape, dim)


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.EmbeddingRule"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.EmbeddingRule"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_EmbeddingAxCheck.lean")
    body = "import TensorGuard.EmbeddingRule\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_EmbeddingAxCheck.lean"],
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
