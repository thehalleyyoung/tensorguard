"""Step 175/277 — generate the Colab-runnable tutorial notebooks.

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
import json
from types import SimpleNamespace
from typing import Dict, List, Tuple

try:
    import nbformat as nbf
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    class _MiniV4:
        @staticmethod
        def new_notebook():
            return SimpleNamespace(
                cells=[],
                metadata={},
                nbformat=4,
                nbformat_minor=5,
            )

        @staticmethod
        def new_markdown_cell(source):
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": source,
            }

        @staticmethod
        def new_code_cell(source):
            return {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }

    class _MiniNbf:
        v4 = _MiniV4()

        @staticmethod
        def write(nb, fh):
            json.dump(
                {
                    "cells": nb.cells,
                    "metadata": nb.metadata,
                    "nbformat": nb.nbformat,
                    "nbformat_minor": nb.nbformat_minor,
                },
                fh,
                indent=1,
            )
            fh.write("\n")

    nbf = _MiniNbf()

HERE = os.path.dirname(os.path.abspath(__file__))

COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/thehalleyyoung/tensorguard/blob/"
    "main/examples/tutorials/{name})"
)

INSTALL = '%pip install -q "git+https://github.com/thehalleyyoung/tensorguard.git"'

TUTORIAL_TRACKS: Dict[str, List[str]] = {
    "shapes": ["01_quickstart.ipynb"],
    "attention": ["06_attention.ipynb"],
    "export": ["03_onnx_export.ipynb"],
    "distributed": ["07_distributed.ipynb"],
    "quantization": ["08_quantization.ipynb"],
    "stubs": ["09_community_stubs.ipynb"],
    "formal_certificates": ["10_formal_certificates.ipynb"],
    "notebooks": ["11_jupyter_magic.ipynb"],
}


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
    "06_attention.ipynb": dict(
        title="Verify attention and transformer shape contracts",
        intro="Scaled-dot-product attention is a real PyTorch kernel with "
               "multi-axis contracts: query/key embedding dims must agree and "
               "key/value sequence lengths must agree, while value width controls "
               "the output width. TensorGuard checks those facts statically.",
        cells=[
             ("code",
              "import torch, torch.nn as nn\n"
              "import torch.nn.functional as F\n"
              "from src.fx_extractor import verify_module\n\n"
              "class AttentionBlock(nn.Module):\n"
              "    def forward(self, q, k, v):\n"
              "        return F.scaled_dot_product_attention(q, k, v)\n\n"
              "good = verify_module(AttentionBlock(),\n"
              "                     input_shapes={'q': (2, 4, 5, 8),\n"
              "                                   'k': (2, 4, 7, 8),\n"
              "                                   'v': (2, 4, 7, 9)})\n"
              "print('good attention safe:', good.safe)\n"
              "assert good.safe"),
             ("md", "A key/value sequence mismatch is caught before the kernel runs:"),
             ("code",
              "bad = verify_module(AttentionBlock(),\n"
              "                    input_shapes={'q': (2, 4, 5, 8),\n"
              "                                  'k': (2, 4, 7, 8),\n"
              "                                  'v': (2, 4, 6, 9)})\n"
              "print('bad attention safe:', bad.safe)\n"
              "assert not bad.safe\n"
              "assert any(v.kind == 'shape_incompatible'\n"
              "           for v in bad.counterexample.violations)"),
        ],
    ),
    "07_distributed.ipynb": dict(
        title="Verify distributed-training shape invariants",
        intro="FSDP and DeepSpeed change physical parameter layout but must "
               "preserve the logical tensor shapes the forward pass expects. "
               "TensorGuard checks those sharding contracts without launching a "
               "distributed job.",
        cells=[
             ("code",
              "from src.distributed_verification import FSDPConfig, verify_distributed\n\n"
              "source = '''\n"
              "import torch.nn as nn\n"
              "class SimpleNet(nn.Module):\n"
              "    def __init__(self):\n"
              "        super().__init__()\n"
              "        self.fc1 = nn.Linear(256, 128)\n"
              "        self.fc2 = nn.Linear(128, 10)\n"
              "    def forward(self, x):\n"
              "        return self.fc2(self.fc1(x))\n"
              "'''\n"
              "result = verify_distributed(source, input_shapes={'x': ('batch', 256)},\n"
              "                            fsdp_config=FSDPConfig(world_size=4))\n"
              "print(result.pretty())\n"
              "assert result.safe\n"
              "assert result.fsdp_result is not None and result.fsdp_result.safe"),
        ],
    ),
    "08_quantization.ipynb": dict(
        title="Verify quantization and autocast gates",
        intro="Quantized and mixed-precision deployments fail when calibration, "
               "boundary, dtype, or backend assumptions drift. TensorGuard exposes "
               "small gates that can run before conversion or export.",
        cells=[
             ("code",
              "import torch, torch.nn as nn\n"
              "import torch.ao.quantization as tq\n"
              "from tensorguard import verify_mixed_precision, verify_quantization_eager\n\n"
              "class QuantReady(nn.Module):\n"
              "    def __init__(self):\n"
              "        super().__init__()\n"
              "        self.quant = tq.QuantStub()\n"
              "        self.fc = nn.Linear(4, 3)\n"
              "        self.dequant = tq.DeQuantStub()\n"
              "    def forward(self, x):\n"
              "        return self.dequant(self.fc(self.quant(x)))\n\n"
              "model = QuantReady().eval()\n"
              "model.qconfig = tq.default_qconfig\n"
              "prepared = tq.prepare(model, inplace=False)\n"
              "uncalibrated = verify_quantization_eager(prepared)\n"
              "print('uncalibrated quantization OK:', uncalibrated.ok)\n"
              "assert not uncalibrated.ok\n"
              "with torch.no_grad():\n"
              "    prepared(torch.randn(2, 4))\n"
              "calibrated = verify_quantization_eager(prepared)\n"
              "print('calibrated quantization OK:', calibrated.ok)\n"
              "assert calibrated.ok"),
             ("md", "The same deployment pass can check autocast policy choices:"),
             ("code",
              "mp = verify_mixed_precision(nn.Linear(4, 3).eval(),\n"
              "                            backend='cpu', autocast_dtype=torch.bfloat16)\n"
              "print('mixed precision OK:', mp.ok)\n"
              "assert mp.ok"),
        ],
    ),
    "09_community_stubs.ipynb": dict(
        title="Use governed community stubs safely",
        intro="Community stubs are declarative manifests, not executable Python. "
               "TensorGuard validates provenance and executes conformance cases "
               "before a third-party layer can turn UNKNOWN into a precise check.",
        cells=[
             ("code",
              "from src.stub_governance import load_community_stubs, validate_directory\n"
              "from src.shape_stub_registry import clear_user_stubs, get_shape_stub\n"
              "from src.tensor_shapes import ShapeDim, TensorShape\n\n"
              "reports = validate_directory('../../community_stubs')\n"
              "print('valid manifests:', len([r for r in reports if r.ok]), '/', len(reports))\n"
              "assert reports and all(r.ok for r in reports)\n\n"
              "loaded = load_community_stubs('../../community_stubs')\n"
              "print('loaded stubs:', loaded)\n"
              "assert 'Linear8bitLt' in loaded\n"
              "stub = get_shape_stub('Linear8bitLt')\n"
              "params = stub.bind_params((768, 3072), {})\n"
              "out, err = stub.transfer(TensorShape((ShapeDim('batch'), ShapeDim(768))), params)\n"
              "assert err is None and out.dims[-1].value == 3072\n"
              "_, err = stub.transfer(TensorShape((ShapeDim('batch'), ShapeDim(512))), params)\n"
              "assert err and '768' in err\n"
              "clear_user_stubs()"),
        ],
    ),
    "10_formal_certificates.ipynb": dict(
        title="Produce and replay formal safety certificates",
        intro="For safe models, TensorGuard can attach a replayable safety "
               "certificate and an optional proof-certificate DAG to the same "
               "top-level result used by CI.",
        cells=[
             ("code",
              "from tensorguard import verify_architecture\n\n"
              "source = '''\n"
              "import torch.nn as nn\n"
              "class M(nn.Module):\n"
              "    def __init__(self):\n"
              "        super().__init__()\n"
              "        self.fc = nn.Linear(10, 5)\n"
              "    def forward(self, x):\n"
              "        return self.fc(x)\n"
              "'''\n"
              "r = verify_architecture(source, input_shapes={'x': ('batch', 10)},\n"
              "                        produce_certificates=True, max_cegar_iterations=0)\n"
              "print('verdict:', r.verdict)\n"
              "print('certificate replay:', r.certificate_replay.ok,\n"
              "      r.certificate_replay.verification_conditions)\n"
              "assert r.verdict == 'SAFE'\n"
              "assert r.safety_certificate is not None\n"
              "assert r.proof_certificate is not None\n"
              "assert r.proof_certificate.verify_locally()\n"
              "assert r.certificate_replay.ok"),
             ("md", "Unsafe or out-of-fragment models deliberately do not receive a safe certificate:"),
             ("code",
              "bad = source.replace('nn.Linear(10, 5)', 'nn.Linear(9, 5)')\n"
              "unsafe = verify_architecture(bad, input_shapes={'x': ('batch', 10)},\n"
              "                             produce_certificates=True)\n"
              "print('unsafe verdict:', unsafe.verdict)\n"
              "assert unsafe.verdict == 'UNSAFE'\n"
              "assert unsafe.safety_certificate is None"),
        ],
    ),
    "11_jupyter_magic.ipynb": dict(
        title="Use `%%tensorguard` inside a notebook",
        intro="Load the TensorGuard IPython extension and verify a model cell in "
              "place. The magic checks the cell source before executing it, so a "
              "notebook experiment can surface tensor-contract bugs exactly where "
              "the model is written.",
        cells=[
            ("code",
             "%load_ext src.jupyter_integration"),
            ("md",
             "The next cell defines a broken attention-style block: the projection "
             "from 16 hidden units is followed by a head that incorrectly expects "
             "12. `%%tensorguard` checks the cell with a symbolic batch shape before "
             "the class is left behind in the notebook namespace."),
            ("code",
             "%%tensorguard x=batch,8\n"
             "import torch\n"
             "import torch.nn as nn\n\n"
             "class NotebookBlock(nn.Module):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.embed = nn.Linear(8, 16)\n"
             "        self.head = nn.Linear(12, 4)\n\n"
             "    def forward(self, x):\n"
             "        return self.head(torch.relu(self.embed(x)))"),
            ("md",
             "The pure helper is asserted below so CI proves the same model cell "
             "really produces a TensorGuard finding when the notebook is executed."),
            ("code",
             "from src.jupyter_integration import check_cell, format_cell_report\n\n"
             "cell = '''\n"
             "import torch\n"
             "import torch.nn as nn\n"
             "class NotebookBlock(nn.Module):\n"
             "    def __init__(self):\n"
             "        super().__init__()\n"
             "        self.embed = nn.Linear(8, 16)\n"
             "        self.head = nn.Linear(12, 4)\n"
             "    def forward(self, x):\n"
             "        return self.head(torch.relu(self.embed(x)))\n"
             "'''\n"
             "outcome = check_cell(cell, input_shapes={'x': ('batch', 8)})\n"
             "print(format_cell_report(outcome))\n"
             "assert outcome.checked\n"
             "assert not outcome.safe\n"
             "assert outcome.bug_count >= 1\n"
             "assert 'NotebookBlock' in outcome.headline"),
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
