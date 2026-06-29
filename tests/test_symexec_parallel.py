"""Tests for the parallel analysis driver (roadmap Step 85).

The driver must produce output **identical** to a serial whole-package run for
every backend (serial / thread / process) — same files, same per-file proof
fingerprints — since each module is analysed independently and the merge is
order-independent.  Also covers worker count selection, backend validation, and
the process-backend serial fallback.
"""

import os

import pytest

from src.symexec.package import analyze_package
from src.symexec.parallel import analyze_package_parallel


# --------------------------------------------------------------------------- #
# Fixture                                                                     #
# --------------------------------------------------------------------------- #

def _write(root, files):
    for rel, txt in files.items():
        p = os.path.join(str(root), rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(txt)


def _project(root):
    _write(root, {
        "mypkg/__init__.py": "",
        "mypkg/layers/__init__.py": "",
        "mypkg/layers/encoder.py": (
            "import torch.nn as nn\n"
            "class Encoder(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(10, 5)\n"
            "    def forward(self, x):\n"
            "        return self.fc(x)\n"
        ),
        "mypkg/model.py": (
            "import torch\n"
            "from mypkg.layers.encoder import Encoder\n"
            "def run():\n"
            "    enc = Encoder()\n"
            "    x = torch.randn(3, 8)\n"
            "    return enc(x)\n"
        ),
        "mypkg/extra.py": (
            "import torch\n"
            "def g():\n"
            "    a = torch.randn(2, 3)\n"
            "    b = torch.randn(7, 4)\n"
            "    return torch.matmul(a, b)\n"
        ),
    })
    return str(root)


def _fps(pr):
    return {os.path.basename(p): r.fingerprint() for p, r in pr.results.items()}


def _bugs(pr):
    return sorted(b.kind.name for _p, b in pr.all_bugs())


# --------------------------------------------------------------------------- #
# Backend equivalence                                                         #
# --------------------------------------------------------------------------- #

def test_serial_backend_matches_analyze_package(tmp_path):
    root = _project(tmp_path)
    assert _fps(analyze_package_parallel(root, backend="serial")) == _fps(
        analyze_package(root)
    )


def test_thread_backend_matches_analyze_package(tmp_path):
    root = _project(tmp_path)
    pr = analyze_package_parallel(root, backend="thread", workers=4)
    assert _fps(pr) == _fps(analyze_package(root))
    assert _bugs(pr) == ["LAYER_DIM_MISMATCH", "MATMUL_DIM_MISMATCH"]


def test_process_backend_matches_analyze_package(tmp_path):
    root = _project(tmp_path)
    pr = analyze_package_parallel(root, backend="process", workers=4)
    assert _fps(pr) == _fps(analyze_package(root))
    assert _bugs(pr) == ["LAYER_DIM_MISMATCH", "MATMUL_DIM_MISMATCH"]


def test_all_files_present(tmp_path):
    root = _project(tmp_path)
    pr = analyze_package_parallel(root, backend="thread")
    assert len(pr.results) == 5  # 3 packages' files + model + extra


# --------------------------------------------------------------------------- #
# Worker selection / config                                                   #
# --------------------------------------------------------------------------- #

def test_workers_one_uses_serial_path(tmp_path):
    root = _project(tmp_path)
    # workers==1 forces the serial path regardless of backend; result unchanged.
    pr = analyze_package_parallel(root, backend="process", workers=1)
    assert _fps(pr) == _fps(analyze_package(root))


def test_single_module_project(tmp_path):
    _write(tmp_path, {"solo.py": "import torch\ndef f():\n    return torch.randn(2, 3)\n"})
    pr = analyze_package_parallel(str(tmp_path), backend="process")
    assert len(pr.results) == 1


def test_unknown_backend_raises(tmp_path):
    root = _project(tmp_path)
    with pytest.raises(ValueError):
        analyze_package_parallel(root, backend="quantum", workers=2)


def test_default_workers_thread(tmp_path):
    root = _project(tmp_path)
    pr = analyze_package_parallel(root, backend="thread")  # workers=None -> auto
    assert _fps(pr) == _fps(analyze_package(root))


def test_budget_is_threaded_through(tmp_path):
    root = _project(tmp_path)
    # A generous budget must not change the result vs unbounded.
    pr = analyze_package_parallel(root, backend="thread", budget_ms=60000)
    assert _fps(pr) == _fps(analyze_package(root))
