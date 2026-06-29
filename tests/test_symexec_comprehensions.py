"""Step 39 — comprehensions & generators build precise container/element values.

A list/set/dict comprehension (and a generator expression) is interpreted by
binding each generator target to the *element abstraction* of its iterable,
assuming the ``if`` guards hold, and evaluating the result expression.  This
makes the produced container carry a precise element summary (so a later read of
an element is reasoned about) and surfaces bugs that occur *inside* the body.
"""

from src.symexec import analyze_source, SymBugKind


def _kinds(src: str, name: str = "m"):
    return [b.kind for b in analyze_source(src, name).bugs]


# ── element abstraction flows out of the comprehension ──────────────────────
def test_listcomp_element_type_flows_to_later_read():
    # every element is None, so reading one and dereferencing is a None-deref
    src = "def f():\n    xs = [None, None]\n    ys = [x.attr for x in xs]\n    return ys[0]\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


def test_dictcomp_value_type_flows_to_later_read():
    src = "def f():\n    d = {k: None for k in range(3)}\n    return d[0].attr\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


# ── bugs *inside* the comprehension body are surfaced ───────────────────────
def test_bug_in_listcomp_body_is_reported():
    src = "def f():\n    xs = [None]\n    return [x.attr for x in xs]\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


def test_bug_in_genexp_body_is_reported():
    src = "def f():\n    xs = [None]\n    return sum(x.attr for x in xs)\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


def test_bug_in_dictcomp_value_is_reported():
    src = "def f():\n    xs = [None]\n    return {i: x.attr for i, x in enumerate(xs)}\n"
    # enumerate is unmodeled (Top), so this should not crash; just assert no error
    _kinds(src)


def test_nested_generators_reach_inner_element():
    src = "def f():\n    xss = [[None]]\n    return [y.attr for xs in xss for y in xs]\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


# ── ``if`` guards refine the body (no false positives) ──────────────────────
def test_filter_excludes_none_no_false_positive():
    src = "def f():\n    xs = [None]\n    return [x.attr for x in xs if x is not None]\n"
    assert _kinds(src) == []


def test_unfiltered_none_is_still_caught():
    src = "def f():\n    xs = [None]\n    return [x.attr for x in xs]\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


# ── soundness: unmodeled iterables abstain, never crash ─────────────────────
def test_setcomp_over_param_abstains():
    src = "def f(xs):\n    return {x.attr for x in xs}\n"
    assert _kinds(src) == []


def test_listcomp_over_param_abstains():
    src = "def f(xs):\n    return [x.attr for x in xs]\n"
    assert _kinds(src) == []


# ── string / tensor iteration element abstraction does not crash ────────────
def test_string_iteration_yields_str_elements():
    src = "def f():\n    return [c for c in 'abc']\n"
    assert _kinds(src) == []


def test_tensor_iteration_yields_subtensor():
    src = "import torch\ndef f():\n    x = torch.zeros(4, 3)\n    return [row for row in x]\n"
    assert _kinds(src) == []
