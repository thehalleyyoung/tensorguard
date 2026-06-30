"""Tests for verified auto-repair (even_more.md Tier 1, idea #2)."""

from __future__ import annotations

import ast

from src.symexec import (
    FixCandidate,
    VerifiedFix,
    propose_fix,
    repair,
    verify_fix,
)
from src.symexec.bugs import SymBug, SymBugKind
from src.symexec.engine import analyze_source

RESHAPE = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    x = torch.randn(2, 3)\n"
    "    y = x.reshape(5, 5)\n"
)
NEGDIM = (
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    x = torch.randn(2, -3)\n"
)
MATMUL = (  # no repair strategy registered
    "import torch\n"
    "if __name__ == '__main__':\n"
    "    a = torch.randn(2, 3); b = torch.randn(4, 5); c = a @ b\n"
)


# --------------------------------------------------------------------------- #
# Proposal.                                                                    #
# --------------------------------------------------------------------------- #
def test_propose_reshape_flatten():
    bug = analyze_source(RESHAPE).bugs[0]
    cand = propose_fix(bug, RESHAPE)
    assert cand is not None
    assert cand.strategy == "reshape-flatten"
    assert ".reshape(-1)" in cand.patched_source
    ast.parse(cand.patched_source)  # still valid Python


def test_propose_negdim_abs():
    bug = analyze_source(NEGDIM).bugs[0]
    cand = propose_fix(bug, NEGDIM)
    assert cand is not None
    assert cand.strategy == "negdim-abs"
    assert "randn(2, 3)" in cand.patched_source
    assert "-3" not in cand.patched_source


def test_no_strategy_returns_none():
    bug = analyze_source(MATMUL).bugs[0]
    assert propose_fix(bug, MATMUL) is None


def test_propose_edit_is_line_local():
    bug = analyze_source(RESHAPE).bugs[0]
    cand = propose_fix(bug, RESHAPE)
    # only the offending line changed; line count preserved.
    before = RESHAPE.splitlines()
    after = cand.patched_source.splitlines()
    assert len(before) == len(after)
    changed = [i for i in range(len(before)) if before[i] != after[i]]
    assert changed == [bug.line - 1]


# --------------------------------------------------------------------------- #
# End-to-end repair (proposal + re-verification + diff).                       #
# --------------------------------------------------------------------------- #
def test_repair_reshape_is_verified():
    fixes = repair(RESHAPE, filename="m.py")
    assert len(fixes) == 1
    f = fixes[0]
    assert isinstance(f, VerifiedFix)
    assert f.verified
    assert f.kind == "reshape_size_mismatch"
    assert "reshape(-1)" in f.patched_source
    assert f.diff and "--- a/m.py" in f.diff and "+++ b/m.py" in f.diff
    # the patched source really is clean per the engine.
    assert analyze_source(f.patched_source).bugs == []


def test_repair_negdim_is_verified():
    fixes = repair(NEGDIM)
    assert len(fixes) == 1 and fixes[0].verified
    assert analyze_source(fixes[0].patched_source).bugs == []


def test_repair_returns_nothing_without_strategy():
    assert repair(MATMUL) == []


def test_repair_clean_source_is_empty():
    clean = (
        "import torch\n"
        "if __name__ == '__main__':\n"
        "    x = torch.randn(2, 3); y = x.reshape(6)\n"
    )
    assert repair(clean) == []


# --------------------------------------------------------------------------- #
# Re-verification gating (the core guarantee): reject bad candidates.          #
# --------------------------------------------------------------------------- #
def _bug(kind_value, line):
    return SymBug(kind=SymBugKind(kind_value), message="m", line=line, col=0,
                  function="")


def test_verify_rejects_when_target_still_fires():
    # patched source still contains the same reshape bug on the same line.
    cand = FixCandidate(
        kind="reshape_size_mismatch", line=4, strategy="noop",
        description="does nothing", patched_source=RESHAPE,
    )
    original = analyze_source(RESHAPE).bugs
    vf = verify_fix(cand, original, filename="m.py")
    assert not vf.verified
    assert "still fires" in vf.detail


def test_verify_rejects_when_new_bug_introduced():
    # original program has only a reshape bug; the "fix" removes it but the
    # patched source introduces a brand-new matmul bug.
    patched = (
        "import torch\n"
        "if __name__ == '__main__':\n"
        "    x = torch.randn(2, 3)\n"
        "    a = torch.randn(2, 3); b = torch.randn(4, 5); c = a @ b\n"
    )
    cand = FixCandidate(
        kind="reshape_size_mismatch", line=4, strategy="bad",
        description="introduces a new bug", patched_source=patched,
    )
    original = [_bug("reshape_size_mismatch", 4)]
    vf = verify_fix(cand, original, filename="m.py")
    assert not vf.verified
    assert "new bug kind" in vf.detail


def test_repair_unverified_only_flag_surfaces_rejections():
    # A source where the reshape flatten is fine, but we also show that
    # verified_only=False would include unverified candidates if any existed.
    fixes_all = repair(RESHAPE, verified_only=False)
    fixes_verified = repair(RESHAPE, verified_only=True)
    assert [f.verified for f in fixes_verified] == [True]
    assert len(fixes_all) >= len(fixes_verified)


# --------------------------------------------------------------------------- #
# Expanded strategies: intent + layer repairs, each machine re-verified.       #
# --------------------------------------------------------------------------- #
from src.symexec import SymConfig  # noqa: E402

_H = SymConfig.heuristic()

MISSING_SUPER = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        self.fc = nn.Linear(3, 4)\n"
    "    def forward(self, x):\n"
    "        return self.fc(x)\n"
)
FORWARD_CALL = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(3, 4)\n"
    "    def forward(self, x):\n"
    "        return self.fc.forward(x)\n"
)
DATA_ACCESS = (
    "import torch\n"
    "def f():\n"
    "    x = torch.randn(3, 4)\n"
    "    return x.data\n"
)
LAYER_DIM = (
    "import torch\n"
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.a = nn.Linear(10, 20)\n"
    "        self.b = nn.Linear(30, 5)\n"
    "    def forward(self, x):\n"
    "        return self.b(self.a(x))\n"
    "def main():\n"
    "    m = Net()\n"
    "    m(torch.randn(2, 10))\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)


def test_repair_missing_super_init_verified():
    fixes = repair(MISSING_SUPER, filename="m.py", config=_H)
    assert [f.kind for f in fixes] == ["missing_super_init"]
    f = fixes[0]
    assert f.verified
    assert f.strategy == "insert-super-init"
    assert "super().__init__()" in f.patched_source
    # The patched source no longer triggers the bug under the same config.
    assert not any(
        b.kind.value == "missing_super_init"
        for b in analyze_source(f.patched_source, config=_H).bugs
    )


def test_repair_missing_super_init_not_duplicated():
    # Already-correct __init__ must not get a second super() call.
    good = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(3, 4)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    assert repair(good, filename="m.py", config=_H) == []


def test_repair_direct_forward_call_verified():
    fixes = repair(FORWARD_CALL, filename="m.py", config=_H)
    f = next(x for x in fixes if x.kind == "direct_forward_call")
    assert f.verified
    assert f.strategy == "forward-to-call"
    assert "self.fc(x)" in f.patched_source
    assert ".forward(" not in f.patched_source


def test_repair_tensor_data_access_verified():
    fixes = repair(DATA_ACCESS, filename="m.py", config=_H)
    f = next(x for x in fixes if x.kind == "tensor_data_access")
    assert f.verified
    assert f.strategy == "data-to-detach"
    assert ".detach()" in f.patched_source
    assert ".data" not in f.patched_source


def test_repair_layer_dim_mismatch_verified():
    fixes = repair(LAYER_DIM, filename="m.py")
    f = next(x for x in fixes if x.kind == "layer_dim_mismatch")
    assert f.verified
    assert f.strategy == "layer-in-size"
    # in_features rewritten 30 -> 20 (the dim actually flowing in).
    assert "nn.Linear(20, 5)" in f.patched_source


def test_layer_dim_mismatch_ambiguous_definition_abstains():
    # Two layers declared with the same in_features 30 -> the rewrite target is
    # ambiguous, so the strategy yields nothing rather than a risky edit.
    ambiguous = (
        "import torch\n"
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(10, 20)\n"
        "        self.b = nn.Linear(30, 5)\n"
        "        self.c = nn.Linear(30, 7)\n"
        "    def forward(self, x):\n"
        "        return self.b(self.a(x))\n"
        "def main():\n"
        "    m = Net()\n"
        "    m(torch.randn(2, 10))\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    fixes = repair(ambiguous, filename="m.py")
    assert all(f.kind != "layer_dim_mismatch" for f in fixes)


CONV_DIM = (
    "import torch\n"
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.c = nn.Conv2d(3, 8, 3)\n"
    "    def forward(self, x):\n"
    "        return self.c(x)\n"
    "def main():\n"
    "    m = Net()\n"
    "    m(torch.randn(1, 1, 16, 16))\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)

BN_DIM = (
    "import torch\n"
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.b = nn.BatchNorm2d(16)\n"
    "    def forward(self, x):\n"
    "        return self.b(x)\n"
    "def main():\n"
    "    m = Net()\n"
    "    m(torch.randn(2, 8, 4, 4))\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)


def test_repair_conv_in_channels_verified():
    # R3: the layer fixer generalizes beyond Linear to Conv in_channels.
    fixes = repair(CONV_DIM, filename="m.py")
    f = next(x for x in fixes if x.kind == "layer_dim_mismatch")
    assert f.verified
    assert f.strategy == "layer-in-size"
    assert "nn.Conv2d(1, 8, 3)" in f.patched_source


def test_repair_batchnorm_num_features_verified():
    # R3: also BatchNorm's num_features (first constructor argument).
    fixes = repair(BN_DIM, filename="m.py")
    f = next(x for x in fixes if x.kind == "layer_dim_mismatch")
    assert f.verified
    assert f.strategy == "layer-in-size"
    assert "nn.BatchNorm2d(8)" in f.patched_source


def test_propose_preserves_trailing_newline():
    cand = propose_fix(
        next(b for b in analyze_source(LAYER_DIM).bugs
             if b.kind.value == "layer_dim_mismatch"),
        LAYER_DIM,
    )
    assert cand is not None
    assert cand.patched_source.endswith("\n")


# Autograd-correctness repairs: numpy-on-grad and tensor copy-construct.        #
NUMPY_GRAD = (
    "import torch\n"
    "def f():\n"
    "    x = torch.randn(3, requires_grad=True)\n"
    "    return x.numpy()\n"
)

COPY_CONSTRUCT = (
    "import torch\n"
    "def f():\n"
    "    a = torch.randn(3)\n"
    "    b = torch.tensor(a)\n"
    "    return b\n"
)


def test_repair_numpy_on_grad_inserts_detach():
    fixes = repair(NUMPY_GRAD, filename="m.py")
    f = next(x for x in fixes if x.kind == "numpy_on_grad")
    assert f.verified
    assert f.strategy == "numpy-detach"
    assert "x.detach().numpy()" in f.patched_source


def test_repair_tensor_copy_construct_to_clone_detach():
    fixes = repair(COPY_CONSTRUCT, filename="m.py", config=_H)
    f = next(x for x in fixes if x.kind == "tensor_copy_construct")
    assert f.verified
    assert f.strategy == "tensor-copy-to-clone"
    assert "a.clone().detach()" in f.patched_source
    assert "torch.tensor(" not in f.patched_source.splitlines()[3]


def test_repair_tensor_copy_construct_complex_arg_abstains():
    # A non-simple argument (an expression) is not safely method-chainable, so
    # the strategy abstains rather than emit `(a + a).clone()...` without parens.
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.randn(3)\n"
        "    b = torch.tensor(a + a)\n"
        "    return b\n"
    )
    fixes = repair(src, filename="m.py", config=_H)
    assert all(f.kind != "tensor_copy_construct" for f in fixes)
