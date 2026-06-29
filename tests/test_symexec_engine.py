"""Regression tests for the symbolic-execution engine (``src.symexec``).

These pin the engine's ability to catch the Python-semantics bug classes that
the FX + SMT shape path is blind to — the same classes we agent-confirmed and
filed against real open-source repositories:

* titans-pytorch #60  — unpacking a single-tensor return into ``(out, cache)``
* OpenStrawberry #113  — indexing a 2-D tensor with three index dimensions
* vector-quantize-pytorch #248 — a file that does not parse

and, just as importantly, that correct code produces **zero** reports.
"""

import pytest

from src.symexec import analyze_source, SymBugKind


def _kinds(result):
    return [b.kind for b in result.bugs]


def test_structural_return_arity_with_opaque_value():
    """Mirrors the real titans file: forward returns a single value produced by
    an *opaque* op (abstract value TOP), so the bug is caught structurally —
    from the fact that every reachable ``return`` yields one value — not from
    the value's type."""
    src = '''
import torch

def exists(v):
    return v is not None

class Attn:
    def forward(self, x, cache=None, return_kv_cache=False):
        out = some_unknown_global_op(x)   # value is unknown / TOP
        if not return_kv_cache:
            return out
        return out, cache

if __name__ == "__main__":
    attn = Attn()
    tokens = torch.randn(1, 1024, 512)
    out1, cache = attn(tokens)
'''
    r = analyze_source(src, "attn.py")
    assert SymBugKind.RETURN_ARITY_CONTRACT in _kinds(r)
    bug = next(b for b in r.bugs if b.kind == SymBugKind.RETURN_ARITY_CONTRACT)
    assert bug.line == 17


# ── titans-pytorch #60 class: unpack-arity / return-arity contract ──────────
def test_unpack_single_tensor_return_into_two_targets():
    src = '''
import torch

class NestedAttn:
    def __init__(self, dim):
        self.dim = dim
    def forward(self, x, cache=None, return_kv_cache=False):
        out = x
        if return_kv_cache:
            return out, cache
        return out

if __name__ == "__main__":
    nested_attn = NestedAttn(512)
    tokens = torch.randn(1, 1024, 512)
    out1, cache = nested_attn.forward(tokens)
'''
    r = analyze_source(src, "titans.py")
    assert SymBugKind.RETURN_ARITY_CONTRACT in _kinds(r)
    bug = next(b for b in r.bugs if b.kind == SymBugKind.RETURN_ARITY_CONTRACT)
    assert bug.line == 16  # the unpack line in the demo


def test_unpack_passing_flag_is_safe():
    """When the kwarg is passed, forward returns a 2-tuple → no report."""
    src = '''
import torch
class M:
    def forward(self, x, return_kv_cache=False):
        if return_kv_cache:
            return x, None
        return x
if __name__ == "__main__":
    m = M()
    tokens = torch.randn(1, 1024, 512)
    out2, cache = m.forward(tokens, return_kv_cache=True)
'''
    r = analyze_source(src, "ok.py")
    assert r.bugs == []


def test_unpack_none_into_two():
    src = '''
def f():
    return None
def g():
    a, b = f()
'''
    r = analyze_source(src, "none.py")
    kinds = _kinds(r)
    assert SymBugKind.NONE_PROPAGATION in kinds or SymBugKind.UNPACK_ARITY_MISMATCH in kinds


# ── OpenStrawberry #113 class: rank-dependent indexing ──────────────────────
def test_index_2d_tensor_with_three_index_dims():
    src = '''
import torch

def monte_carlo_rollout(model, x):
    output = x
    output = output[-1, :, :]
    return output

if __name__ == "__main__":
    x = torch.randn(10, 32)
    monte_carlo_rollout(None, x)
'''
    r = analyze_source(src, "ostraw.py")
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(r)
    bug = next(b for b in r.bugs if b.kind == SymBugKind.RANK_INDEX_ERROR)
    assert "2-D tensor" in bug.message and "3 index" in bug.message


def test_index_3d_tensor_with_three_index_dims_is_safe():
    src = '''
import torch
if __name__ == "__main__":
    y = torch.randn(4, 8, 16)
    z = y[-1, :, :]
'''
    r = analyze_source(src, "ok_index.py")
    assert r.bugs == []


def test_ellipsis_index_does_not_false_report():
    src = '''
import torch
if __name__ == "__main__":
    y = torch.randn(4, 8)
    z = y[..., 0, 0]
'''
    r = analyze_source(src, "ellipsis.py")
    assert r.bugs == []


# ── vector-quantize-pytorch #248 class: file does not parse ─────────────────
def test_syntax_error_is_reported():
    r = analyze_source("def f(:\n    pass\n", "bad.py")
    assert len(r.bugs) == 1
    assert "does not parse" in r.bugs[0].message
    assert r.bugs[0].confidence == 1.0


# ── soundness: a realistic correct module must be silent ────────────────────
def test_correct_module_no_false_positives():
    src = '''
import torch

class Block:
    def __init__(self, dim):
        self.dim = dim
    def forward(self, x):
        h = x.transpose(1, 2)
        h = h.unsqueeze(0)
        h = h.squeeze(0)
        out = h.transpose(1, 2)
        return out

if __name__ == "__main__":
    block = Block(64)
    x = torch.randn(2, 16, 64)
    y = block.forward(x)
    a = y[0, :, :]
    b = y[:, -1, :]
'''
    r = analyze_source(src, "correct.py")
    assert r.bugs == []


def test_unknown_rank_does_not_report():
    """If the receiver's rank is unknown, indexing must not be reported."""
    src = '''
def f(x):
    return x[0, 0, 0, 0]
'''
    r = analyze_source(src, "unknown.py")
    assert r.bugs == []


def test_engine_ran_main_flag():
    src = 'if __name__ == "__main__":\n    x = 1\n'
    r = analyze_source(src, "m.py")
    assert r.ran_main is True


def test_to_api_bug_maps_categories():
    src = '''
import torch
def f(x):
    return x[0, 0, 0]
if __name__ == "__main__":
    f(torch.randn(2, 3))
'''
    r = analyze_source(src, "api.py")
    assert r.bugs
    api_bug = r.bugs[0].to_api_bug("api.py")
    assert api_bug.location.line > 0
    assert api_bug.category is not None
