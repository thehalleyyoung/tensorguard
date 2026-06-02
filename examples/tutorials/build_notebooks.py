"""Step 175 — generate the Colab-runnable tutorial notebooks.

Run ``python examples/tutorials/build_notebooks.py`` to (re)generate the
``*.ipynb`` files in this directory from a single source of truth. Keeping the
generator in-repo makes the notebooks reproducible and reviewable (a diff of
this file is the diff of every notebook), and lets CI assert they are in sync.

Each notebook is intentionally small and self-contained so it runs end-to-end
on a free Colab CPU runtime in seconds, demonstrating one TensorGuard
integration against *real* PyTorch.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/thehalleyyoung/tensorguard/blob/"
    "main/examples/tutorials/{name})"
)

INSTALL = "%pip install -q tensorguard  # on Colab; locally: pip install -e ."


def _make_nb(name: str, title: str, intro: str, cells: List[Tuple[str, str]]):
    nb = nbf.v4.new_notebook()
    out = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{COLAB_BADGE.format(name=name)}\n\n{intro}"),
        nbf.v4.new_code_cell(INSTALL),
    ]
    for kind, body in cells:
        out.append(nbf.v4.new_markdown_cell(body) if kind == "md"
                   else nbf.v4.new_code_cell(body))
    nb.cells = out
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    return nb


# ── notebook definitions (single source of truth) ────────────────────────────
_DEFS: Dict[str, dict] = {
    "01_quickstart.ipynb": dict(
        title="TensorGuard quickstart — catch a shape bug before you run",
        intro="Verify a model *statically* and *batch-polymorphically* from its "
              "source. The buggy variant below only raises at call time in stock "
              "PyTorch; TensorGuard flags it first, for every batch size at once.",
        cells=[
            ("code",
             "from tensorguard import verify_architecture\n\n"
             "good = '''\n"
             "import torch, torch.nn as nn\n"
             "class MLP(nn.Module):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.fc1 = nn.Linear(784, 256)\n"
             "        self.fc2 = nn.Linear(256, 10)\n"
             "    def forward(self, x):\n"
             "        return self.fc2(torch.relu(self.fc1(x)))\n"
             "'''\n"
             "r = verify_architecture(good, input_shapes={'x': ('batch', 784)})\n"
             "print('good model:', r.status, '| bugs:', len(r.bugs))\n"
             "assert r.status == 'SAFE'"),
            ("md", "A one-line dimension typo (`256` → `255`) is caught:"),
            ("code",
             "bad = good.replace('nn.Linear(256, 10)', 'nn.Linear(255, 10)')\n"
             "r = verify_architecture(bad, input_shapes={'x': ('batch', 784)})\n"
             "print('bad model:', r.status, '| bugs:', len(r.bugs))\n"
             "assert r.status == 'UNSAFE'\n"
             "print(r.bugs[0].message)"),
        ],
    ),
    "02_torch_compile.ipynb": dict(
        title="Gate `torch.compile` with TensorGuard",
        intro="`guarded_compile` verifies a module and only then hands it to "
              "`torch.compile`, so a verified model is compiled while a shape bug "
              "is caught *before* a single graph is built.",
        cells=[
            ("code",
             "import torch, torch.nn as nn\n"
             "from tensorguard import guarded_compile\n\n"
             "class Net(nn.Module):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.a = nn.Linear(32, 16)\n"
             "        self.b = nn.Linear(16, 4)\n"
             "    def forward(self, x):\n"
             "        return self.b(torch.relu(self.a(x)))\n\n"
             "compiled = guarded_compile(Net(), input_shapes={'x': ('batch', 32)})\n"
             "print('compiled output shape:', tuple(compiled(torch.randn(8, 32)).shape))"),
            ("md", "The same shape contract, checked statically, rejects a buggy net:"),
            ("code",
             "from tensorguard import verify_architecture\n"
             "bad = '''\n"
             "import torch, torch.nn as nn\n"
             "class Net(nn.Module):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.a = nn.Linear(32, 16)\n"
             "        self.b = nn.Linear(15, 4)\n"
             "    def forward(self, x):\n"
             "        return self.b(torch.relu(self.a(x)))\n"
             "'''\n"
             "r = verify_architecture(bad, input_shapes={'x': ('batch', 32)})\n"
             "print('buggy net:', r.status, '->', r.bugs[0].message)\n"
             "assert r.status == 'UNSAFE'"),
        ],
    ),
    "03_onnx_export.ipynb": dict(
        title="Gate ONNX export with TensorGuard",
        intro="`guarded_onnx_export` verifies the module, exports it, and parses the "
              "proto back through `onnx.checker` — a structurally invalid graph fails "
              "at export time, not at downstream load time.",
        cells=[
            ("code",
             "import torch, torch.nn as nn\n"
             "from tensorguard import guarded_onnx_export\n\n"
             "class Net(nn.Module):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.a = nn.Linear(16, 8)\n"
             "        self.b = nn.Linear(8, 2)\n"
             "    def forward(self, x):\n"
             "        return self.b(torch.relu(self.a(x)))\n\n"
             "guarded_onnx_export(Net(), (torch.randn(1, 16),), 'model.onnx',\n"
             "                    input_shapes={'x': ('batch', 16)})\n"
             "import onnx; onnx.checker.check_model(onnx.load('model.onnx'))\n"
             "print('exported + checked OK')"),
        ],
    ),
    "04_lightning.ipynb": dict(
        title="Verify a PyTorch Lightning module",
        intro="TensorGuard verifies the network inside a `LightningModule`, so "
              "training never starts on a shape-broken net.",
        cells=[
            ("code",
             "import torch, torch.nn as nn\n"
             "import pytorch_lightning as pl\n"
             "from tensorguard import verify_architecture\n\n"
             "class LitMLP(pl.LightningModule):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.net = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))\n"
             "    def forward(self, x):\n"
             "        return self.net(x)\n\n"
             "lit = LitMLP()\n"
             "print('Lightning module ready:', type(lit).__name__)"),
            ("md", "Verify the net's shape contract before `Trainer.fit`:"),
            ("code",
             "net_src = '''\n"
             "import torch, torch.nn as nn\n"
             "class Net(nn.Module):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.net = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))\n"
             "    def forward(self, x):\n"
             "        return self.net(x)\n"
             "'''\n"
             "r = verify_architecture(net_src, input_shapes={'x': ('batch', 20)})\n"
             "print('Lightning net:', r.status)\n"
             "assert r.status == 'SAFE'"),
        ],
    ),
    "05_ci_precommit.ipynb": dict(
        title="Use TensorGuard in CI / pre-commit",
        intro="TensorGuard ships a CLI, a pytest plugin, a pre-commit hook, and a "
              "GitHub Action. This notebook shows the programmatic check you would "
              "wire into a CI job to fail the build on a shape regression.",
        cells=[
            ("code",
             "from tensorguard import verify_architecture\n\n"
             "def ci_gate(source, shapes):\n"
             "    r = verify_architecture(source, input_shapes=shapes)\n"
             "    if r.status != 'SAFE':\n"
             "        raise SystemExit('TensorGuard: shape regression\\n' +\n"
             "                         r.bugs[0].message)\n"
             "    print('TensorGuard: OK')\n\n"
             "ci_gate('''\n"
             "import torch, torch.nn as nn\n"
             "class M(nn.Module):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.a = nn.Linear(8, 4); self.b = nn.Linear(4, 2)\n"
             "    def forward(self, x):\n"
             "        return self.b(torch.relu(self.a(x)))\n"
             "''', {'x': ('batch', 8)})"),
            ("md",
             "On the command line the same gate is:\n\n"
             "```bash\n"
             "tensorguard path/to/model.py        # exits non-zero on a bug\n"
             "pytest --tensorguard                 # pytest plugin\n"
             "pre-commit run tensorguard --all-files\n"
             "```"),
        ],
    ),
}


def build_notebooks() -> Dict[str, object]:
    """Return ``{filename: NotebookNode}`` for every tutorial (no file I/O)."""
    return {
        name: _make_nb(name, d["title"], d["intro"], d["cells"])
        for name, d in _DEFS.items()
    }


def write_notebooks(out_dir: str = HERE) -> List[str]:
    paths = []
    for name, nb in build_notebooks().items():
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in write_notebooks():
        print("wrote", os.path.relpath(p))
