"""Step 29 (completion) — BatchNorm / Embedding / LayerNorm nn-layer transfers.

- ``BatchNorm{1,2,3}d``: channel dim (dim 1) must equal ``num_features``.
- ``Embedding``: output is the index shape with ``embedding_dim`` appended
  (rank + 1), so the embedding dim flows into downstream layers.
- ``LayerNorm``: the input's trailing dims must equal ``normalized_shape``.

All fire only on concrete mismatches and abstain on symbolic dims / unknown rank.
"""

from src.symexec import analyze_source, SymBugKind

LD = SymBugKind.LAYER_DIM_MISMATCH
N = "import torch\nimport torch.nn as nn\n"


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


# ── BatchNorm ────────────────────────────────────────────────────────────────
def test_batchnorm2d_ok():
    src = N + "def f():\n c = nn.BatchNorm2d(16)\n x = torch.zeros(4, 16, 8, 8)\n return c(x)\n"
    assert _kinds(src) == []


def test_batchnorm2d_channel_mismatch():
    src = N + "def f():\n c = nn.BatchNorm2d(16)\n x = torch.zeros(4, 8, 8, 8)\n return c(x)\n"
    assert LD in _kinds(src)


def test_batchnorm1d_ok():
    src = N + "def f():\n c = nn.BatchNorm1d(10)\n x = torch.zeros(32, 10)\n return c(x)\n"
    assert _kinds(src) == []


def test_batchnorm_symbolic_abstains():
    src = N + "def f(x):\n c = nn.BatchNorm2d(16)\n return c(x)\n"
    assert _kinds(src) == []


# ── Embedding ────────────────────────────────────────────────────────────────
def test_embedding_dim_flows_to_linear_ok():
    src = N + (
        "def f():\n e = nn.Embedding(1000, 64)\n l = nn.Linear(64, 5)\n"
        " x = torch.zeros(4, 10)\n return l(e(x))\n"
    )
    assert _kinds(src) == []


def test_embedding_dim_flows_to_linear_mismatch():
    src = N + (
        "def f():\n e = nn.Embedding(1000, 64)\n l = nn.Linear(99, 5)\n"
        " x = torch.zeros(4, 10)\n return l(e(x))\n"
    )
    assert LD in _kinds(src)


# ── LayerNorm ────────────────────────────────────────────────────────────────
def test_layernorm_int_ok():
    src = N + "def f():\n c = nn.LayerNorm(128)\n x = torch.zeros(4, 10, 128)\n return c(x)\n"
    assert _kinds(src) == []


def test_layernorm_int_mismatch():
    src = N + "def f():\n c = nn.LayerNorm(128)\n x = torch.zeros(4, 10, 256)\n return c(x)\n"
    assert LD in _kinds(src)


def test_layernorm_tuple_ok():
    src = N + "def f():\n c = nn.LayerNorm((10, 128))\n x = torch.zeros(4, 10, 128)\n return c(x)\n"
    assert _kinds(src) == []


def test_layernorm_tuple_mismatch():
    src = N + "def f():\n c = nn.LayerNorm((10, 128))\n x = torch.zeros(4, 10, 256)\n return c(x)\n"
    assert LD in _kinds(src)


def test_layernorm_symbolic_abstains():
    src = N + "def f(x):\n c = nn.LayerNorm(128)\n return c(x)\n"
    assert _kinds(src) == []


# ── self.<layer> in __init__ → forward ───────────────────────────────────────
def test_batchnorm_self_forward_mismatch():
    src = N + (
        "class Net(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n"
        "        self.bn = nn.BatchNorm2d(16)\n"
        "    def forward(self, x):\n        return self.bn(x)\n"
        "if __name__ == '__main__':\n"
        "    m = Net()\n    x = torch.zeros(4, 8, 8, 8)\n    m(x)\n"
    )
    assert LD in _kinds(src)
