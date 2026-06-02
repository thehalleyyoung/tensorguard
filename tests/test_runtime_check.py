"""Step 63 — the ``@tensorguard.checked`` decorator.

Verifies that decorating an nn.Module subclass checks it at definition time:
bugs raise (or warn / log), safe models pass through unchanged and remain fully
usable, and the decorator is robust to bare/parameterized use and missing
source.
"""

import warnings

import pytest
import torch
import torch.nn as nn

import src as tensorguard
from src.runtime_check import TensorGuardCheckError, checked


def test_bug_raises_at_definition():
    with pytest.raises(TensorGuardCheckError) as ei:
        @checked(input_shapes={"x": ("batch", 10)})
        class Bad(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(30, 5)

            def forward(self, x):
                return self.fc2(self.fc1(x))

    msg = str(ei.value)
    assert "Bad" in msg
    assert "expects input dimension 30" in msg
    assert ei.value.result is not None


def test_safe_model_passes_and_is_usable():
    @checked(input_shapes={"x": ("batch", 10)})
    class Good(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    m = Good()
    out = m(torch.randn(4, 10))
    assert tuple(out.shape) == (4, 5)
    assert Good.__tensorguard_result__ is not None
    assert not Good.__tensorguard_result__.bugs


def test_on_fail_warn_does_not_raise():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @checked(input_shapes={"x": ("batch", 10)}, on_fail="warn")
        class Warned(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(30, 5)

            def forward(self, x):
                return self.fc2(self.fc1(x))

    assert any("tensorguard" in str(x.message) for x in w)
    # class still defined and usable as a normal module object
    assert Warned.__tensorguard_result__ is not None


def test_on_fail_log_prints(capsys):
    @checked(input_shapes={"x": ("batch", 10)}, on_fail="log")
    class Logged(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(30, 5)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    captured = capsys.readouterr()
    assert "Logged" in captured.out
    assert "issue" in captured.out


def test_conv_bug_autoinfers_without_shapes():
    with pytest.raises(TensorGuardCheckError):
        @checked
        class ConvBad(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(3, 8, 3, padding=1)
                self.c2 = nn.Conv2d(16, 4, 3, padding=1)

            def forward(self, x):
                return self.c2(self.c1(x))


def test_bare_decorator_on_safe_conv():
    @checked
    class ConvGood(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)
            self.c2 = nn.Conv2d(8, 4, 3, padding=1)

        def forward(self, x):
            return self.c2(self.c1(x))

    out = ConvGood()(torch.randn(1, 3, 16, 16))
    assert tuple(out.shape)[1] == 4


def test_missing_source_abstains():
    # A class built without recoverable source must not raise.
    src = (
        "import torch.nn as nn\n"
        "class Dyn(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(10, 20)\n"
        "        self.fc2 = nn.Linear(30, 5)\n"
        "    def forward(self, x):\n"
        "        return self.fc2(self.fc1(x))\n"
    )
    ns: dict = {}
    exec(src, ns)
    Dyn = ns["Dyn"]
    decorated = checked(input_shapes={"x": ("batch", 10)})(Dyn)
    # inspect.getsource fails for exec'd classes -> abstains (no raise).
    assert decorated is Dyn
    assert decorated.__tensorguard_result__ is None


def test_exported_from_package():
    assert hasattr(tensorguard, "checked")
    assert hasattr(tensorguard, "TensorGuardCheckError")
    assert tensorguard.checked is checked
