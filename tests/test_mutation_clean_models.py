"""Regression tests for the clean-model mutation-testing harness (Step 112)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproducibility import mutation_clean_models as mut  # noqa: E402
from corpus_extended import model_mutators as ops  # noqa: E402

_VOLATILE = ("time", "elapsed", "timestamp", "wall", "clock",
             "_ms", "seconds", "duration", "date")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


@pytest.fixture(scope="module")
def data():
    return mut.measure()


def test_no_volatile_fields(data):
    for key in _walk_keys(data):
        low = str(key).lower()
        assert not any(tok in low for tok in _VOLATILE), f"volatile key: {key}"


def test_byte_deterministic(tmp_path, data):
    # The committed artifact must reproduce byte-for-byte.
    assert mut.run(check=True) == 0


def test_operators_produce_genuine_bugs():
    # A multi-layer Linear/Conv model: width bumps must yield real runtime bugs.
    src = (
        "import torch\nimport torch.nn as nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(16, 16)\n"
        "        self.b = nn.Linear(16, 4)\n"
        "    def forward(self, x):\n"
        "        return self.b(self.a(x))\n"
    )
    shapes = {"x": (8, 16)}
    assert mut._runs_clean(src, shapes)
    for name in ("linear_out_bump", "linear_in_bump", "dtype_long_cast"):
        mutant = ops.OPERATORS[name](src)
        assert mutant is not None and mutant != src
        assert mut._raises(mutant, shapes), f"{name} did not produce a runtime bug"


def test_inapplicable_operator_returns_none():
    src = (
        "import torch\nimport torch.nn as nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(8, 8)\n"
        "    def forward(self, x):\n"
        "        return self.a(x)\n"
    )
    # No Conv2d present -> conv operators are inapplicable.
    assert ops.conv_out_bump(src) is None
    assert ops.conv_in_bump(src) is None


def test_corpus_scale(data):
    assert data["n_clean_parents"] >= 100
    assert data["n_genuine_bug_mutants"] >= 100
    assert data["n_operators"] == len(ops.OPERATORS)


def test_sound_mode_never_passes_a_genuine_bug(data):
    # The headline soundness property of mutation testing: a genuine bug is
    # never silently reported SAFE in sound mode.
    assert data["sound_mode_zero_false_safe"] is True
    assert data["per_mode"]["sound"]["n_survived_safe"] == 0
    assert data["per_mode"]["sound"]["survived_safe_ids"] == []


def test_high_kill_rate(data):
    kr = data["per_mode"]["sound"]["kill_rate"]
    assert kr["n"] == data["n_genuine_bug_mutants"]
    assert kr["point"] >= 0.95
    # Wilson interval is well-formed.
    assert 0.0 <= kr["low"] <= kr["point"] <= kr["high"] <= 1.0


def test_per_operator_and_domain_present(data):
    sm = data["per_mode"]["sound"]
    for op in ops.OPERATORS:
        assert op in sm["per_operator"]
        d = sm["per_operator"][op]
        assert d["n_genuine_bugs"] >= 0
        assert 0 <= d["n_killed"] <= d["n_genuine_bugs"]
        assert d["domain"] == ops.OPERATOR_DOMAIN[op]
    for dom in ("shape", "dtype"):
        assert dom in sm["per_domain"]
