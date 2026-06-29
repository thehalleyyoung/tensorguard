"""Intent-based (non-crashing) detector — discarded pure tensor transform.

A bare statement such as ``x.to(device)`` / ``x.cuda()`` / ``x.reshape(...)``
returns a *new* tensor and has no in-place effect, so discarding the result is a
silent no-op — almost always a bug (the author expected an in-place mutation).

Because this never raises at runtime, it is a **heuristic-only** finding: it must
appear in ``heuristic`` mode and must be suppressed in ``balanced``/``sound`` so
their zero-false-positive guarantee is preserved.
"""

from src.symexec.bugs import SymBugKind
from src.symexec.config import SymConfig
from src.symexec.engine import analyze_source, analyze_file


def _kinds(src, cfg=None):
    r = analyze_source(src, config=cfg) if cfg is not None else analyze_source(src)
    return [b.kind.value for b in r.bugs]


def _main(body):
    return 'import torch\nif __name__ == "__main__":\n' + "".join(
        "    " + line + "\n" for line in body
    )


HEUR = SymConfig.heuristic()


# ---- fires in heuristic mode on a discarded pure transform -------------------

def test_discarded_cuda_flagged_heuristic():
    src = _main(["x = torch.randn(3)", "x.cuda()"])
    assert "discarded_tensor_result" in _kinds(src, HEUR)


def test_discarded_to_flagged_heuristic():
    src = _main(["x = torch.randn(3)", 'x.to("cuda")'])
    assert "discarded_tensor_result" in _kinds(src, HEUR)


def test_discarded_reshape_flagged_heuristic():
    src = _main(["x = torch.randn(4)", "x.reshape(2, 2)"])
    assert "discarded_tensor_result" in _kinds(src, HEUR)


def test_discarded_chained_flagged_heuristic():
    src = _main(["x = torch.randn(3)", "x.detach().cpu()"])
    assert "discarded_tensor_result" in _kinds(src, HEUR)


# ---- not a no-op when the result is used ------------------------------------

def test_assigned_result_clean():
    src = _main(["x = torch.randn(3)", "y = x.cuda()"])
    assert "discarded_tensor_result" not in _kinds(src, HEUR)


def test_reassigned_result_clean():
    src = _main(["x = torch.randn(3)", "x = x.cuda()"])
    assert "discarded_tensor_result" not in _kinds(src, HEUR)


# ---- in-place statements are not flagged (they DO have an effect) -----------

def test_inplace_statement_not_flagged():
    src = _main(["x = torch.randn(3)", "x.add_(1)"])
    assert "discarded_tensor_result" not in _kinds(src, HEUR)


# ---- soundness: NEVER fires in balanced or sound modes ----------------------

def test_suppressed_in_balanced_default():
    src = _main(["x = torch.randn(3)", "x.cuda()"])
    assert "discarded_tensor_result" not in _kinds(src)  # default = balanced


def test_suppressed_in_sound_mode():
    src = _main(["x = torch.randn(3)", "x.cuda()"])
    assert "discarded_tensor_result" not in _kinds(src, SymConfig.sound())


# ---- severity is a warning, not an error ------------------------------------

def test_severity_is_warning():
    src = _main(["x = torch.randn(3)", "x.cuda()"])
    bugs = [
        b for b in analyze_source(src, config=HEUR).bugs
        if b.kind is SymBugKind.DISCARDED_TENSOR_RESULT
    ]
    assert bugs and bugs[0].severity == "warning"
    assert bugs[0].fix_suggestion


# ---- corpus fingerprint (balanced) is unaffected ----------------------------

def test_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )


# ---- direct module.forward(x) call (anti-pattern, heuristic-only) ------------

_NET = (
    "import torch\n"
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(3, 4)\n"
    "    def forward(self, x):\n"
    "        return self.fc(x)\n"
    "def run():\n"
    "    m = Net()\n"
    "    x = torch.randn(2, 3)\n"
    "    {call}\n"
    "    return y\n"
)


def test_direct_forward_flagged_heuristic():
    src = _NET.format(call="y = m.forward(x)")
    assert "direct_forward_call" in _kinds(src, HEUR)


def test_module_call_clean_heuristic():
    src = _NET.format(call="y = m(x)")
    assert "direct_forward_call" not in _kinds(src, HEUR)


def test_direct_forward_suppressed_in_balanced():
    src = _NET.format(call="y = m.forward(x)")
    assert "direct_forward_call" not in _kinds(src)


def test_direct_forward_suppressed_in_sound():
    src = _NET.format(call="y = m.forward(x)")
    assert "direct_forward_call" not in _kinds(src, SymConfig.sound())


def test_direct_forward_severity_warning():
    src = _NET.format(call="y = m.forward(x)")
    bugs = [
        b for b in analyze_source(src, config=HEUR).bugs
        if b.kind is SymBugKind.DIRECT_FORWARD_CALL
    ]
    assert bugs and bugs[0].severity == "warning"
    assert bugs[0].fix_suggestion


# ---- tensor.data access (autograd footgun, heuristic-only) -------------------

def test_data_access_flagged_heuristic():
    src = _main(["x = torch.randn(3)", "y = x.data"])
    assert "tensor_data_access" in _kinds(src, HEUR)


def test_detach_not_flagged():
    src = _main(["x = torch.randn(3)", "y = x.detach()"])
    assert "tensor_data_access" not in _kinds(src, HEUR)


def test_data_access_suppressed_in_balanced():
    src = _main(["x = torch.randn(3)", "y = x.data"])
    assert "tensor_data_access" not in _kinds(src)


def test_data_access_suppressed_in_sound():
    src = _main(["x = torch.randn(3)", "y = x.data"])
    assert "tensor_data_access" not in _kinds(src, SymConfig.sound())


# ---- missing super().__init__() in nn.Module subclass (heuristic-only) --------

HEUR = SymConfig.heuristic()

_BAD_INIT = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        self.fc = nn.Linear(3, 4)\n"
    "    def forward(self, x):\n"
    "        return self.fc(x)\n"
)


def test_missing_super_init_flagged_heuristic():
    assert "missing_super_init" in _kinds(_BAD_INIT, HEUR)


def test_missing_super_init_suppressed_in_balanced():
    assert "missing_super_init" not in _kinds(_BAD_INIT)


def test_missing_super_init_suppressed_in_sound():
    assert "missing_super_init" not in _kinds(_BAD_INIT, SymConfig.sound())


def test_super_init_present_not_flagged():
    src = _BAD_INIT.replace(
        "        self.fc", "        super().__init__()\n        self.fc"
    )
    assert "missing_super_init" not in _kinds(src, HEUR)


def test_base_init_call_not_flagged():
    src = _BAD_INIT.replace(
        "        self.fc", "        nn.Module.__init__(self)\n        self.fc"
    )
    assert "missing_super_init" not in _kinds(src, HEUR)


def test_no_init_not_flagged():
    src = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )
    assert "missing_super_init" not in _kinds(src, HEUR)


def test_non_module_class_not_flagged():
    src = (
        "class Foo:\n"
        "    def __init__(self):\n"
        "        self.x = 1\n"
    )
    assert "missing_super_init" not in _kinds(src, HEUR)


def test_missing_super_init_severity_and_fix():
    bugs = [
        b for b in analyze_source(_BAD_INIT, config=HEUR).bugs
        if b.kind is SymBugKind.MISSING_SUPER_INIT
    ]
    assert bugs and bugs[0].severity == "warning"
    assert bugs[0].fix_suggestion


def test_missing_super_init_corpus_fingerprint_unchanged():
    fp = analyze_file("tests/symexec_corpus/wild/matmul_dim_mismatch.py").fingerprint()
    assert fp == (
        "de466b6f54018384cb5b3c27b5b3f7be178001535d59bb785c46fdba83ead9e0"
    )


# ---- torch.tensor copy-construct from a tensor (heuristic-only) ---------------

def test_copy_construct_flagged_heuristic():
    src = _main(["x = torch.randn(3)", "y = torch.tensor(x)"])
    assert "tensor_copy_construct" in _kinds(src, HEUR)


def test_copy_construct_from_list_not_flagged():
    src = _main(["y = torch.tensor([1, 2, 3])"])
    assert "tensor_copy_construct" not in _kinds(src, HEUR)


def test_copy_construct_suppressed_in_balanced():
    src = _main(["x = torch.randn(3)", "y = torch.tensor(x)"])
    assert "tensor_copy_construct" not in _kinds(src)


def test_copy_construct_suppressed_in_sound():
    src = _main(["x = torch.randn(3)", "y = torch.tensor(x)"])
    assert "tensor_copy_construct" not in _kinds(src, SymConfig.sound())


def test_copy_construct_severity_and_fix():
    src = _main(["x = torch.randn(3)", "y = torch.tensor(x)"])
    bugs = [
        b for b in analyze_source(src, config=HEUR).bugs
        if b.kind is SymBugKind.TENSOR_COPY_CONSTRUCT
    ]
    assert bugs and bugs[0].severity == "warning"
    assert bugs[0].fix_suggestion
