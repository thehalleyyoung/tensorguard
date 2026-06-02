"""Step 62 — Jupyter / IPython integration.

The pure core (detect a model in a cell, verify it, render a verdict) is tested
directly against real torch models, and the IPython layer is exercised with a
real (test) InteractiveShell to prove the post-run-cell hook and the
``%%tensorguard`` cell magic work end to end.
"""

import io

import torch  # noqa: F401

from src.jupyter_integration import (
    check_cell,
    find_module_classes,
    format_cell_report,
    run_cell_check,
)


BUG = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
"""

SAFE = BUG.replace("nn.Linear(30, 5)", "nn.Linear(20, 5)")

NOT_A_MODEL = "x = 1\ny = [i*i for i in range(x)]\n"

CONV_BUG = """
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(16, 4, 3, padding=1)
    def forward(self, x):
        return self.c2(self.c1(x))
"""


def test_find_module_classes():
    assert find_module_classes(BUG) == ["Net"]
    assert find_module_classes(NOT_A_MODEL) == []
    # multiple + various base spellings
    multi = (
        "import torch.nn as nn\n"
        "from torch.nn import Module\n"
        "class A(nn.Module):\n    pass\n"
        "class B(Module):\n    pass\n"
        "class C:\n    pass\n"
    )
    assert find_module_classes(multi) == ["A", "B"]


def test_find_module_classes_tolerates_syntax_error():
    assert find_module_classes("class Net(nn.Module):\n    def forward(self") == []


def test_check_cell_no_model_is_silent():
    o = check_cell(NOT_A_MODEL)
    assert o.checked is False
    assert o.module_names == []
    assert format_cell_report(o) == ""


def test_check_cell_detects_linear_bug_with_shapes():
    o = check_cell(BUG, input_shapes={"x": ("batch", 10)})
    assert o.checked and not o.safe
    assert o.bug_count == 1
    assert "Net" in o.headline and "issue" in o.headline
    rep = format_cell_report(o)
    assert "expects input dimension 30" in rep


def test_check_cell_safe_model():
    o = check_cell(SAFE, input_shapes={"x": ("batch", 10)})
    assert o.checked and o.safe
    assert o.bug_count == 0
    assert "verified safe" in format_cell_report(o)


def test_check_cell_conv_bug_autoinfers_without_shapes():
    # Conv first layer pins rank/channels (Step 56), so the bug is caught even
    # without explicit input shapes.
    o = check_cell(CONV_BUG)
    assert o.checked and not o.safe
    assert o.bug_count >= 1


def test_run_cell_check_prints_via_printer():
    lines = []
    o = run_cell_check(
        BUG, input_shapes={"x": ("batch", 10)}, printer=lines.append
    )
    assert not o.safe
    assert any("tensorguard" in ln for ln in lines)


def test_color_report_wraps_ansi():
    o = check_cell(SAFE, input_shapes={"x": ("batch", 10)})
    colored = format_cell_report(o, use_color=True)
    assert "\033[32m" in colored  # green for safe


def test_ipython_extension_hook_and_magic():
    from IPython.testing.globalipapp import get_ipython

    from src import jupyter_integration as ji

    ip = get_ipython()
    ji.load_ipython_extension(ip)
    assert hasattr(ip, "_tensorguard_hook")

    # The cell magic with shapes catches the Linear bug AND defines the class.
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ip.run_cell_magic("tensorguard", "x=batch,10", BUG)
    out = buf.getvalue()
    assert "issue" in out
    assert "Net" in ip.user_ns  # the class really was defined

    ji.unload_ipython_extension(ip)
