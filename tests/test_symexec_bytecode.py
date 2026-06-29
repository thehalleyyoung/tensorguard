"""Tests for the Step-81 bytecode fallback (``src/symexec/bytecode.py``).

The fallback must (a) recover precise concrete values for straight-line pure
expressions the AST modeler abstains on, (b) stay strictly inside a
side-effect-free fragment — never executing a call, attribute load, import or
control-flow jump from the analyzed source — and (c) preserve soundness and the
zero-regression contract when wired into the interpreter (no new false reports,
fingerprints of unaffected code unchanged).
"""

import ast

import pytest

import src.symexec as symexec
from src.symexec import bytecode as bc
from src.symexec import values as v


def _fold(src: str, env=None):
    node = ast.parse(src, mode="eval").body
    return bc.fold_expr(node, env or {})


# --------------------------------------------------------------------------- #
# Constant folding inside the safe fragment                                   #
# --------------------------------------------------------------------------- #

def test_folds_set_literal():
    assert _fold("{1, 2, 3}") == {1, 2, 3}


def test_folds_fstring_over_known_constant():
    assert _fold('f"dim-{n}"', {"n": 5}) == "dim-5"


def test_folds_arithmetic_over_names():
    assert _fold("a * b - 1", {"a": 3, "b": 4}) == 11


def test_folds_tuple_list_dict():
    assert _fold("(a, 2, a + 1)", {"a": 7}) == (7, 2, 8)
    assert _fold("[a, a]", {"a": 1}) == [1, 1]
    assert _fold('{"x": 1, "y": n}', {"n": 9}) == {"x": 1, "y": 9}


def test_folds_comparison_and_boolean_unary():
    assert _fold("a == 3", {"a": 3}) is True
    assert _fold("not a", {"a": 0}) is True
    assert _fold("-a", {"a": 5}) == -5


def test_folds_constant_subscript():
    assert _fold("(10, 20, 30)[i]", {"i": 1}) == 20
    assert _fold('"abcd"[k]', {"k": 0}) == "a"


# --------------------------------------------------------------------------- #
# Refusal to leave the safe fragment                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "src,env",
    [
        ("len(x)", {"x": (1, 2)}),         # CALL
        ("obj.attr", {"obj": 1}),          # LOAD_ATTR
        ("a + z", {"a": 1}),               # unknown name z
        ("a if a else 0", {"a": 1}),       # control-flow jump
        ("a and b", {"a": 1, "b": 2}),     # short-circuit jump
        ("[i for i in r]", {"r": (1, 2)}), # comprehension / nested code
        ("(lambda: 1)", {}),               # lambda / nested code
    ],
)
def test_abstains_outside_safe_fragment(src, env):
    assert _fold(src, env) is bc.NOT_CONCRETE


def test_abstains_on_runtime_error():
    # A genuine evaluation error is not a sound constant -> abstain.
    assert _fold("1 // z", {"z": 0}) is bc.NOT_CONCRETE


def test_dos_guards_reject_oversized_pow_and_shift():
    assert _fold("a ** b", {"a": 10, "b": 100000}) is bc.NOT_CONCRETE
    assert _fold("a << b", {"a": 1, "b": 100000}) is bc.NOT_CONCRETE


def test_safe_eval_never_calls_user_code():
    # A name bound to an object whose methods would raise if invoked: folding an
    # expression that *reads* it as a container element must not call anything.
    class Boom:
        def __getitem__(self, k):  # pragma: no cover - must never run
            raise AssertionError("user code executed")

    # Building a tuple that merely *contains* the object is fine (no call);
    # subscripting it would be refused because Boom is not a constant container.
    assert _fold("(x, 1)", {"x": Boom()})[1] == 1
    assert _fold("x[0]", {"x": Boom()}) is bc.NOT_CONCRETE


# --------------------------------------------------------------------------- #
# Lifting to the abstract domain                                              #
# --------------------------------------------------------------------------- #

def test_fold_to_abstract_lifts_values():
    node = ast.parse("{1, 2}", mode="eval").body
    val = bc.fold_to_abstract(node, {})
    assert isinstance(val, v.SetVal)
    s = ast.parse('f"{n}"', mode="eval").body
    assert bc.fold_to_abstract(s, {"n": 7}) == v.StrVal(const="7")


def test_fold_to_abstract_returns_none_on_abstain():
    node = ast.parse("len(x)", mode="eval").body
    assert bc.fold_to_abstract(node, {"x": (1,)}) is None


# --------------------------------------------------------------------------- #
# abstract_to_concrete projection                                             #
# --------------------------------------------------------------------------- #

def test_abstract_to_concrete_known_constants():
    assert bc.abstract_to_concrete(v.int_const(4)) == 4
    assert bc.abstract_to_concrete(v.BoolVal(const=True)) is True
    assert bc.abstract_to_concrete(v.StrVal(const="hi")) == "hi"
    assert bc.abstract_to_concrete(v.NoneVal()) is None
    t = v.TupleVal(elems=(v.int_const(1), v.int_const(2)), exact_len=True)
    assert bc.abstract_to_concrete(t) == (1, 2)


def test_abstract_to_concrete_abstains_on_unknown():
    assert bc.abstract_to_concrete(v.TOP) is bc.NOT_CONCRETE
    assert bc.abstract_to_concrete(v.IntVal(sym=None)) is bc.NOT_CONCRETE
    assert bc.abstract_to_concrete(v.TensorVal(rank=2)) is bc.NOT_CONCRETE
    inexact = v.TupleVal(elems=(v.int_const(1),), exact_len=False)
    assert bc.abstract_to_concrete(inexact) is bc.NOT_CONCRETE


# --------------------------------------------------------------------------- #
# End-to-end: the interpreter now models previously-abstained constructs       #
# --------------------------------------------------------------------------- #

def _eval_in_engine(expr_src: str, setup: str = ""):
    """Evaluate ``expr_src`` inside a function body and return its abstract value."""
    import src.symexec.interpreter as interp
    import src.symexec.state as state_mod

    interpreter = interp.Interpreter(ast.parse(""))
    st = state_mod.State()
    body = ast.parse(setup) if setup else ast.parse("")
    for stmt in body.body:
        st = interpreter.exec_stmt(stmt, st)
    node = ast.parse(expr_src, mode="eval").body
    return interpreter.eval_expr(node, st)


def test_interpreter_models_set_literal_via_fallback():
    val = _eval_in_engine("{1, 2, 3}")
    assert isinstance(val, v.SetVal)
    assert not val.is_top()


def test_interpreter_models_fstring_via_fallback():
    val = _eval_in_engine('f"layer-{n}"', setup="n = 4")
    assert val == v.StrVal(const="layer-4")


def test_fallback_does_not_introduce_false_positives():
    # A set/f-string heavy function must still produce zero reports.
    src = (
        "def f():\n"
        "    n = 2\n"
        "    cfg = {1, 2, 3}\n"
        "    name = f'dim-{n}'\n"
        "    return name\n"
        "if __name__ == '__main__':\n"
        "    f()\n"
    )
    result = symexec.analyze_source(src, "x.py")
    assert result.bugs == []


def test_corpus_fingerprints_unchanged_by_fallback():
    # The fallback must not alter the verdict on the curated corpus.
    import os

    base = os.path.join("tests", "symexec_corpus", "wild")
    src = open(os.path.join(base, "matmul_dim_mismatch.py")).read()
    result = symexec.analyze_source(src, "m.py")
    kinds = sorted(b.kind.name for b in result.bugs)
    assert kinds == ["MATMUL_DIM_MISMATCH"]
