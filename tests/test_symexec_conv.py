"""Step 108 — Conv{1,2,3}d in-channels mismatch.

The channel axis is ``rank - spatial - 1`` (covers batched ``(N,C,*)`` and
unbatched ``(C,*)``).  Output channels flow to downstream layers; spatial extents
become symbolic.  Fires only on a concrete channel-count mismatch; abstains on
symbolic channels or unknown rank.
"""

from src.symexec import analyze_source, SymBugKind

LD = SymBugKind.LAYER_DIM_MISMATCH


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


def test_conv2d_in_channels_ok():
    src = (
        "import torch\nimport torch.nn as nn\n"
        "def f():\n    c = nn.Conv2d(3, 16, 3)\n"
        "    x = torch.zeros(4, 3, 32, 32)\n    return c(x)\n"
    )
    assert _kinds(src) == []


def test_conv2d_in_channels_mismatch():
    src = (
        "import torch\nimport torch.nn as nn\n"
        "def f():\n    c = nn.Conv2d(3, 16, 3)\n"
        "    x = torch.zeros(4, 8, 32, 32)\n    return c(x)\n"
    )
    assert LD in _kinds(src)


def test_conv2d_unbatched_ok():
    src = (
        "import torch\nimport torch.nn as nn\n"
        "def f():\n    c = nn.Conv2d(3, 16, 3)\n"
        "    x = torch.zeros(3, 32, 32)\n    return c(x)\n"
    )
    assert _kinds(src) == []


def test_conv2d_output_channels_flow_ok():
    src = (
        "import torch\nimport torch.nn as nn\n"
        "def f():\n    a = nn.Conv2d(3, 16, 3)\n    b = nn.Conv2d(16, 8, 3)\n"
        "    x = torch.zeros(4, 3, 32, 32)\n    return b(a(x))\n"
    )
    assert _kinds(src) == []


def test_conv2d_output_channels_flow_mismatch():
    src = (
        "import torch\nimport torch.nn as nn\n"
        "def f():\n    a = nn.Conv2d(3, 16, 3)\n    b = nn.Conv2d(99, 8, 3)\n"
        "    x = torch.zeros(4, 3, 32, 32)\n    return b(a(x))\n"
    )
    assert LD in _kinds(src)


def test_conv2d_self_forward_mismatch():
    src = (
        "import torch\nimport torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n"
        "        self.conv = nn.Conv2d(3, 16, 3)\n"
        "    def forward(self, x):\n        return self.conv(x)\n"
        "if __name__ == '__main__':\n"
        "    m = Net()\n    x = torch.zeros(4, 8, 32, 32)\n    m(x)\n"
    )
    assert LD in _kinds(src)


def test_conv1d_in_channels_mismatch():
    src = (
        "import torch\nimport torch.nn as nn\n"
        "def f():\n    c = nn.Conv1d(3, 16, 3)\n"
        "    x = torch.zeros(4, 8, 100)\n    return c(x)\n"
    )
    assert LD in _kinds(src)


def test_conv2d_symbolic_abstains():
    src = (
        "import torch\nimport torch.nn as nn\n"
        "def f(x):\n    c = nn.Conv2d(3, 16, 3)\n    return c(x)\n"
    )
    assert _kinds(src) == []
