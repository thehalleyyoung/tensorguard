"""Step 50 — higher-order & callback handling.

``nn.Sequential`` (container of child layers), ``module.apply(fn)``,
``map``/``filter`` with a named function or lambda callback, and registered
hooks.

Key soundness design for ``nn.Sequential``:
- When the input rank is *known* (a real demo tensor flows in), the input is
  threaded through each modeled child so per-layer contracts and shape
  propagation fire exactly as for a hand-written forward.
- When the input rank is *unknown* (a plain ``forward(self, x)`` entry), an
  input-independent **structural same-kind feature-chain** check fires: adjacent
  layers of the same kind (Linear↔Linear via the feature dim, Conv↔Conv via the
  channel dim) whose out/in feature dims are both known and unequal are a
  guaranteed runtime error regardless of the input. Cross-kind neighbours and
  unknown modules break the chain (no false positives).
"""

from src.symexec.engine import analyze_source
from src.symexec.bugs import SymBugKind
from src.symexec.interpreter import Interpreter

import ast


LAYER = SymBugKind.LAYER_DIM_MISMATCH
RANK = SymBugKind.RANK_INDEX_ERROR


def _kinds(src):
    return [b.kind for b in analyze_source(src).bugs]


def _messages(src):
    return [b.message for b in analyze_source(src).bugs]


# --------------------------------------------------------------------------
# nn.Sequential — structural feature-chain check (input rank unknown)
# --------------------------------------------------------------------------

def test_sequential_linear_mismatch_structural():
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(30, 5))
    def forward(self, x):
        return self.seq(x)
"""
    assert LAYER in _kinds(src)


def test_sequential_linear_valid_no_false_positive():
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    def forward(self, x):
        return self.seq(x)
"""
    assert _kinds(src) == []


def test_sequential_transparent_layers_pass_feature_through():
    # The feature dim survives ReLU/Dropout/BatchNorm; the 20->30 break must
    # still be detected across them.
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(30, 5),
        )
    def forward(self, x):
        return self.seq(x)
"""
    assert LAYER in _kinds(src)


def test_sequential_conv_mismatch_structural():
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Conv2d(8, 32, 3))
    def forward(self, x):
        return self.seq(x)
"""
    assert LAYER in _kinds(src)


def test_sequential_conv_valid_no_false_positive():
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Conv2d(16, 32, 3))
    def forward(self, x):
        return self.seq(x)
"""
    assert _kinds(src) == []


def test_sequential_cross_kind_no_false_positive():
    # Linear out_features (20) must NOT be compared against Conv2d in_channels
    # (3): different kinds, so the chain breaks and nothing is reported.
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(10, 20), nn.Conv2d(3, 8, 3))
    def forward(self, x):
        return self.seq(x)
"""
    assert _kinds(src) == []


def test_sequential_unknown_module_breaks_chain():
    # A custom/unmodeled module between two Linears breaks the chain: we cannot
    # know whether it preserves the feature dim, so we must abstain.
    src = """
import torch.nn as nn
class Custom(nn.Module):
    def forward(self, x):
        return x
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(10, 20), Custom(), nn.Linear(30, 5))
    def forward(self, x):
        return self.seq(x)
"""
    assert _kinds(src) == []


# --------------------------------------------------------------------------
# nn.Sequential — starred construction
# --------------------------------------------------------------------------

def test_sequential_starred_list_mismatch():
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(10, 20), nn.Linear(30, 5)]
        self.seq = nn.Sequential(*layers)
    def forward(self, x):
        return self.seq(x)
"""
    assert LAYER in _kinds(src)


def test_sequential_starred_list_valid():
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(10, 20), nn.Linear(20, 5)]
        self.seq = nn.Sequential(*layers)
    def forward(self, x):
        return self.seq(x)
"""
    assert _kinds(src) == []


# --------------------------------------------------------------------------
# nn.Sequential — threaded with a known input tensor (demo)
# --------------------------------------------------------------------------

def test_sequential_threaded_known_input_mismatch():
    src = """
import torch
import torch.nn as nn
if __name__ == "__main__":
    seq = nn.Sequential(nn.Linear(10, 20), nn.Linear(30, 5))
    x = torch.zeros(4, 10)
    y = seq(x)
"""
    assert LAYER in _kinds(src)


def test_sequential_threaded_known_input_valid():
    src = """
import torch
import torch.nn as nn
if __name__ == "__main__":
    seq = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
    x = torch.zeros(4, 10)
    y = seq(x)
"""
    assert _kinds(src) == []


def test_sequential_output_shape_propagates_downstream():
    # The Sequential output feeds a reshape with a wrong element count; the
    # threaded output shape (4, 5) must flow so the reshape check can fire.
    src = """
import torch
import torch.nn as nn
if __name__ == "__main__":
    seq = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
    x = torch.zeros(4, 10)
    y = seq(x)
    z = y.view(4, 7)
"""
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in _kinds(src)


# --------------------------------------------------------------------------
# map / filter callbacks
# --------------------------------------------------------------------------

def test_map_named_function_surfaces_body_bug():
    src = """
import torch
def f(t):
    return t[0, 0, 0]
def g():
    xs = [torch.zeros(2, 3)]
    return list(map(f, xs))
"""
    assert RANK in _kinds(src)


def test_map_lambda_surfaces_body_bug():
    src = """
import torch
def g():
    xs = [torch.zeros(2, 3)]
    return list(map(lambda t: t[0, 0, 0], xs))
"""
    assert RANK in _kinds(src)


def test_map_valid_no_false_positive():
    src = """
import torch
def f(t):
    return t[0, 0]
def g():
    xs = [torch.zeros(2, 3)]
    return list(map(f, xs))
"""
    assert _kinds(src) == []


# --------------------------------------------------------------------------
# module.apply(fn) and registered hooks
# --------------------------------------------------------------------------

def test_apply_callback_surfaces_body_bug():
    src = """
import torch
import torch.nn as nn
def init_weights(m):
    z = torch.zeros(2, 3)
    return z[0, 0, 0]
def build():
    net = nn.Linear(10, 20)
    net.apply(init_weights)
"""
    assert RANK in _kinds(src)


def test_register_hook_is_benign_no_crash():
    src = """
import torch.nn as nn
def build():
    net = nn.Linear(3, 4)
    h = net.register_forward_hook(lambda m, i, o: None)
    return h
"""
    # No modeled failure; importantly, analysis must not raise.
    assert _kinds(src) == []


# --------------------------------------------------------------------------
# constructor / child-storage unit checks
# --------------------------------------------------------------------------

def _build_seq(src):
    mod = ast.parse(src)
    it = Interpreter(mod, {})
    from src.symexec.state import State

    st = State()
    it.exec_block(mod.body, st)
    return st.get("seq")


def test_sequential_children_stored_index_keyed():
    src = """
import torch.nn as nn
seq = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
"""
    seq = _build_seq(src)
    assert seq.class_name == "Sequential"
    keys = [k for k, _ in seq.attrs]
    assert keys == ["0", "1", "2"]
