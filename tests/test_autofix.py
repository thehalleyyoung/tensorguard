"""Step 59 — mechanical autofix suggestions (`tensorguard verify --fix`).

These tests prove, against real torch models, that:
  * a wrong nn.Linear in_features and a wrong nn.Conv2d in_channels each yield a
    concrete single-line edit suggestion,
  * applying the suggestion produces source that the verifier (and eager torch)
    accept,
  * keyword-argument constructor forms are handled,
  * safe models and unfixable bugs yield no suggestions,
  * the suggestion builder is fully defensive.
"""

import torch

from src.api import verify_architecture
from src.autofix import (
    AutoFix,
    apply_autofixes,
    build_autofixes,
    format_autofixes_ansi,
    format_autofixes_plain,
)


LINEAR_BUG = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
"""

LINEAR_BUG_KW = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(in_features=30, out_features=5)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
"""

CONV_BUG = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
    def forward(self, x):
        h = self.conv1(x)
        return self.conv2(h)
"""

SAFE = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
"""


def _exec_runs(source: str, x: torch.Tensor) -> tuple:
    ns: dict = {}
    exec(source, ns)
    model = ns["Net"]()
    return tuple(model(x).shape)


def test_linear_in_features_fix_suggested():
    r = verify_architecture(LINEAR_BUG, input_shapes={"x": ("batch", 10)})
    assert r.bugs
    fixes = r.autofixes
    assert len(fixes) == 1
    f = fixes[0]
    assert f.layer == "fc2"
    assert f.kind == "linear_in_features"
    assert f.old_value == 30 and f.new_value == 20
    assert "nn.Linear(20, 5)" in f.suggested


def test_applying_linear_fix_makes_it_verify_and_run():
    r = verify_architecture(LINEAR_BUG, input_shapes={"x": ("batch", 10)})
    fixed = apply_autofixes(LINEAR_BUG, r.autofixes)
    assert "nn.Linear(20, 5)" in fixed
    r2 = verify_architecture(fixed, input_shapes={"x": ("batch", 10)})
    assert not r2.bugs
    # Eager torch agrees the repaired model is valid.
    assert _exec_runs(fixed, torch.randn(4, 10)) == (4, 5)


def test_keyword_form_fix():
    r = verify_architecture(LINEAR_BUG_KW, input_shapes={"x": ("batch", 10)})
    assert len(r.autofixes) == 1
    f = r.autofixes[0]
    assert "in_features=20" in f.suggested
    fixed = apply_autofixes(LINEAR_BUG_KW, r.autofixes)
    r2 = verify_architecture(fixed, input_shapes={"x": ("batch", 10)})
    assert not r2.bugs


def test_conv_in_channels_fix():
    r = verify_architecture(CONV_BUG, input_shapes={"x": (1, 3, 32, 32)})
    assert len(r.autofixes) == 1
    f = r.autofixes[0]
    assert f.kind == "conv_in_channels"
    assert f.old_value == 16 and f.new_value == 8
    fixed = apply_autofixes(CONV_BUG, r.autofixes)
    r2 = verify_architecture(fixed, input_shapes={"x": (1, 3, 32, 32)})
    assert not r2.bugs
    assert _exec_runs(fixed, torch.randn(1, 3, 32, 32))[1] == 4


def test_safe_model_no_fixes():
    r = verify_architecture(SAFE, input_shapes={"x": ("batch", 10)})
    assert not r.bugs
    assert r.autofixes == []


def test_apply_is_noop_on_stale_fix():
    f = AutoFix(
        layer="fc2",
        kind="linear_in_features",
        line=6,
        original="        self.fc2 = nn.Linear(99, 5)",  # does not match source
        suggested="        self.fc2 = nn.Linear(20, 5)",
        description="stale",
        old_value=99,
        new_value=20,
    )
    out = apply_autofixes(SAFE, [f])
    assert out == SAFE  # nothing replaced because original line did not match


def test_render_plain_and_ansi():
    r = verify_architecture(LINEAR_BUG, input_shapes={"x": ("batch", 10)})
    plain = format_autofixes_plain(r.autofixes)
    ansi = format_autofixes_ansi(r.autofixes)
    assert "Suggested fixes (1)" in plain
    assert "nn.Linear(20, 5)" in plain
    assert "\033[" in ansi  # contains ANSI codes
    assert format_autofixes_plain([]) == ""
    assert format_autofixes_ansi([]) == ""


def test_build_autofixes_defensive_on_garbage():
    class Bad:
        kind = "shape_incompatible"

    # No graph, malformed violations: must not raise, must return [].
    assert build_autofixes("", [Bad(), None, 123], None) == []
