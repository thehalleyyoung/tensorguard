"""Step 175/277 — the tutorial notebooks actually run (Colab-runnable, CI-proven).

Each notebook in ``examples/tutorials/`` is executed end-to-end with nbclient
against the *real* libraries it targets (torch, onnx, pytorch_lightning).
Notebooks whose optional dependency is missing are skipped, never silently
passed. We also assert the committed notebooks are in sync with their
single-source generator, so a docs drift fails CI.
"""

from __future__ import annotations

import glob
import os

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

from nbclient import NotebookClient  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TUT_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "tutorials"))

# Which third-party import each notebook needs beyond torch/tensorguard.
_EXTRA_DEP = {
    "03_onnx_export.ipynb": "onnx",
    "04_lightning.ipynb": "pytorch_lightning",
    "08_quantization.ipynb": "torch.ao.quantization",
}

_REQUIRED_TRACKS = {
    "shapes",
    "attention",
    "export",
    "distributed",
    "quantization",
    "stubs",
    "formal_certificates",
    "notebooks",
}


def _notebooks():
    return sorted(glob.glob(os.path.join(TUT_DIR, "*.ipynb")))


def test_there_are_tutorial_notebooks():
    nbs = _notebooks()
    names = {os.path.basename(p) for p in nbs}
    assert {
        "01_quickstart.ipynb",
        "02_torch_compile.ipynb",
        "03_onnx_export.ipynb",
        "04_lightning.ipynb",
        "05_ci_precommit.ipynb",
        "06_attention.ipynb",
        "07_distributed.ipynb",
        "08_quantization.ipynb",
        "09_community_stubs.ipynb",
        "10_formal_certificates.ipynb",
        "11_jupyter_magic.ipynb",
    } <= names


def test_tutorial_tracks_cover_requested_surfaces():
    import importlib.util

    gen = os.path.join(TUT_DIR, "build_notebooks.py")
    spec = importlib.util.spec_from_file_location("build_notebooks", gen)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    notebooks = {os.path.basename(path) for path in _notebooks()}
    assert _REQUIRED_TRACKS <= set(mod.TUTORIAL_TRACKS)
    for track in _REQUIRED_TRACKS:
        members = mod.TUTORIAL_TRACKS[track]
        assert members, f"{track} track has no notebooks"
        assert set(members) <= notebooks
        assert any(
            any(cell.cell_type == "code" and cell.source.strip()
                for cell in mod.build_notebooks()[name].cells)
            for name in members
        ), f"{track} track has no smoke-testable code cells"


@pytest.mark.parametrize("nb_path", _notebooks(), ids=lambda p: os.path.basename(p))
def test_notebook_runs_end_to_end(nb_path):
    pytest.importorskip("torch")
    name = os.path.basename(nb_path)
    extra = _EXTRA_DEP.get(name)
    if extra:
        pytest.importorskip(extra)

    nb = nbformat.read(nb_path, as_version=4)
    # Drop the `%pip install` bootstrap cell: locally we use the editable install.
    nb.cells = [
        c for c in nb.cells
        if not (c.cell_type == "code" and c.source.strip().startswith("%pip"))
    ]
    # Make the in-repo package importable from the kernel (no pip install needed
    # locally / in CI). The kernel inherits this process's environment.
    repo_root = os.path.normpath(os.path.join(HERE, ".."))
    old_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = repo_root + (os.pathsep + old_pp if old_pp else "")
    try:
        client = NotebookClient(
            nb,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": TUT_DIR}},
        )
        # Raises CellExecutionError if any assertion/cell fails — that is the proof.
        client.execute()
    finally:
        os.environ["PYTHONPATH"] = old_pp


def test_notebooks_match_generator():
    """The committed notebooks must equal a fresh generation (no drift)."""
    import importlib.util

    gen = os.path.join(TUT_DIR, "build_notebooks.py")
    spec = importlib.util.spec_from_file_location("build_notebooks", gen)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fresh = mod.build_notebooks()
    for path in _notebooks():
        name = os.path.basename(path)
        assert name in fresh, f"committed notebook {name} not produced by generator"
        committed_src = [c.source for c in nbformat.read(path, as_version=4).cells]
        generated_src = [c.source for c in fresh[name].cells]
        assert committed_src == generated_src, (
            f"{name} is out of sync with build_notebooks.py — "
            f"run `python examples/tutorials/build_notebooks.py`"
        )
