"""Step 75 — differential testing against real ``torch`` execution.

The engine's soundness claim is concrete: when it *reports* a bug it is claiming
the program would actually fail at runtime, and when it stays *silent* it is
claiming the program runs.  This test checks that claim against ground truth by
**executing** the curated corpus (``tests/symexec_corpus/``) under real PyTorch
and comparing the verdicts:

* every ``wild/`` repro must (a) be flagged by the engine **and** (b) actually
  raise when run — a confirmed *true positive*, and the raised exception family
  must match the reported bug kind (e.g. a ``rank_index_error`` really raises
  ``IndexError``; a shape mismatch really raises ``RuntimeError``);
* every ``correct/`` model must (a) be silent **and** (b) run to completion — a
  confirmed *true negative*.

The whole module is skipped where torch is unavailable.  Corpus files are
executed in an isolated globals namespace with ``__name__ == "__main__"`` so their
demo harnesses run; the parse-error repro is caught at ``compile`` time.
"""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("torch")

from src.symexec import analyze_source

_CORPUS = pathlib.Path(__file__).resolve().parent / "symexec_corpus"
_MANIFEST = json.loads((_CORPUS / "manifest.json").read_text(encoding="utf-8"))

# Which runtime exception family each reported bug kind predicts.  A tuple allows
# more than one acceptable type; ``Exception`` would be too weak to be meaningful.
_KIND_TO_EXC = {
    "rank_index_error": (IndexError,),
    "tensor_index_oob": (IndexError,),
    "return_arity_contract": (ValueError,),
    "unpack_arity_mismatch": (ValueError,),
    "broadcast_mismatch": (RuntimeError,),
    "matmul_dim_mismatch": (RuntimeError,),
    "reshape_size_mismatch": (RuntimeError,),
    "layer_dim_mismatch": (RuntimeError,),
    "cat_shape_mismatch": (RuntimeError,),
    "einsum_dim_mismatch": (RuntimeError,),
}


def _execute(path: pathlib.Path):
    """Run a corpus file under real torch in an isolated namespace.

    Returns ``None`` if it completes cleanly, otherwise the raised exception
    instance (a ``SyntaxError`` is returned when the file does not even
    compile)."""
    src = path.read_text(encoding="utf-8")
    try:
        code = compile(src, str(path), "exec")
    except SyntaxError as exc:
        return exc
    try:
        exec(code, {"__name__": "__main__"})
        return None
    except Exception as exc:  # noqa: BLE001 — differential ground truth
        return exc


# -- Step 75: wild repros — engine report ⇔ real failure ----------------


@pytest.mark.parametrize("name", sorted(_MANIFEST["wild"]))
def test_wild_repro_really_fails_and_is_flagged(name):
    spec = _MANIFEST["wild"][name]
    path = _CORPUS / "wild" / name

    # (a) the engine flags it
    result = analyze_source(path.read_text(encoding="utf-8"), str(path))
    assert result.bugs, f"{name}: engine reported no bug"

    # (b) it really fails at runtime
    exc = _execute(path)
    assert exc is not None, f"{name}: expected a runtime failure, but it ran clean"

    # (c) the raised exception family matches the reported kind
    if spec["expect"] == "parse_error":
        assert isinstance(exc, SyntaxError)
    else:
        expected = _KIND_TO_EXC.get(spec["kind"])
        if expected is not None:
            assert isinstance(exc, expected), (
                f"{name}: kind {spec['kind']} predicted {expected}, "
                f"got {type(exc).__name__}: {exc}"
            )


# -- Step 75: correct models — engine silent ⇔ real success -------------


@pytest.mark.parametrize("name", sorted(_MANIFEST["correct"]))
def test_correct_model_is_silent_and_runs(name):
    path = _CORPUS / "correct" / name

    # (a) the engine stays silent
    result = analyze_source(path.read_text(encoding="utf-8"), str(path))
    assert result.bugs == [], (
        f"{name}: false positive {[b.kind.value for b in result.bugs]}"
    )

    # (b) it really runs to completion
    exc = _execute(path)
    assert exc is None, f"{name}: expected clean run, but raised {exc!r}"
