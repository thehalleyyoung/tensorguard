"""Tests for incremental analysis (roadmap Step 84).

Verifies that incremental re-analysis is (a) **correct** — byte-identical
fingerprints to a fresh whole-package / single-file run — and (b) genuinely
*incremental* — only files whose source or directly-imported project symbols
changed are recomputed, everything else is reused verbatim.  Also covers the
function-level change-detection helpers (``unit_index`` / ``diff_units``) and the
cache lifecycle (reuse, dependency invalidation, stale eviction).
"""

import ast
import os

import src.symexec as s
from src.symexec.package import analyze_package
from src.symexec.incremental import (
    IncrementalCache,
    IncrementalStats,
    UnitChange,
    analyze_package_incremental,
    analyze_source_incremental,
    diff_units,
    unit_index,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

ENC_BUG = (
    "import torch.nn as nn\n"
    "class Encoder(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc = nn.Linear(10, 5)\n"
    "    def forward(self, x):\n"
    "        return self.fc(x)\n"
)
ENC_OK = ENC_BUG.replace("nn.Linear(10, 5)", "nn.Linear(8, 5)")
MODEL = (
    "import torch\n"
    "from mypkg.layers.encoder import Encoder\n"
    "def run():\n"
    "    enc = Encoder()\n"
    "    x = torch.randn(3, 8)\n"
    "    return enc(x)\n"
)


def _write(root, files):
    for rel, txt in files.items():
        p = os.path.join(str(root), rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(txt)


def _build(root, enc=ENC_BUG, model=MODEL):
    _write(root, {
        "mypkg/__init__.py": "",
        "mypkg/layers/__init__.py": "",
        "mypkg/layers/encoder.py": enc,
        "mypkg/model.py": model,
    })


def _fps(pr):
    return {os.path.basename(p): r.fingerprint() for p, r in pr.results.items()}


def _bn(paths):
    return sorted(os.path.basename(p) for p in paths)


# --------------------------------------------------------------------------- #
# unit_index / diff_units                                                     #
# --------------------------------------------------------------------------- #

def test_unit_index_keys_top_level_units():
    idx = unit_index(ast.parse(MODEL))
    assert "run" in idx
    assert "<module>" in idx  # imports folded here


def test_diff_units_detects_modification():
    a = unit_index(ast.parse(ENC_BUG))
    b = unit_index(ast.parse(ENC_OK))
    d = diff_units(a, b)
    assert d.any
    assert "Encoder" in d.modified
    assert "Encoder" in d.affected


def test_diff_units_added_and_removed():
    a = unit_index(ast.parse("def f():\n    return 1\n"))
    b = unit_index(ast.parse("def g():\n    return 2\n"))
    d = diff_units(a, b)
    assert "g" in d.added
    assert "f" in d.removed
    assert not diff_units(a, a).any


def test_diff_units_module_level_change():
    a = unit_index(ast.parse("import os\n"))
    b = unit_index(ast.parse("import sys\n"))
    assert "<module>" in diff_units(a, b).modified


def test_unit_index_line_shift_is_a_change():
    # A blank line inserted above shifts positions -> different hash (bug lines move).
    a = unit_index(ast.parse("def f():\n    return 1\n"))
    b = unit_index(ast.parse("\ndef f():\n    return 1\n"))
    assert a["f"] != b["f"]


# --------------------------------------------------------------------------- #
# Correctness: incremental == fresh                                           #
# --------------------------------------------------------------------------- #

def test_first_run_matches_fresh_and_analyzes_all(tmp_path):
    _build(tmp_path)
    fresh = analyze_package(str(tmp_path))
    cache = IncrementalCache()
    inc, stats = analyze_package_incremental(str(tmp_path), cache)
    assert stats.reused == ()
    assert len(stats.reanalyzed) == 4
    assert _fps(fresh) == _fps(inc)


def test_no_edit_reuses_everything(tmp_path):
    _build(tmp_path)
    cache = IncrementalCache()
    inc1, _ = analyze_package_incremental(str(tmp_path), cache)
    inc2, stats = analyze_package_incremental(str(tmp_path), cache)
    assert stats.reanalyzed == ()
    assert stats.total == 4
    assert _fps(inc1) == _fps(inc2)


def test_edit_importer_only_recomputes_importer(tmp_path):
    _build(tmp_path)
    cache = IncrementalCache()
    analyze_package_incremental(str(tmp_path), cache)
    _build(tmp_path, model=MODEL + "\n# a comment\n")
    inc, stats = analyze_package_incremental(str(tmp_path), cache)
    assert _bn(stats.reanalyzed) == ["model.py"]
    assert _fps(analyze_package(str(tmp_path))) == _fps(inc)


def test_edit_imported_symbol_invalidates_dependents(tmp_path):
    # Fixing the Linear width in encoder.py must re-analyse BOTH encoder.py and
    # model.py (whose augmented analysis inlines Encoder), and clear the bug.
    _build(tmp_path, enc=ENC_BUG)
    cache = IncrementalCache()
    inc0, _ = analyze_package_incremental(str(tmp_path), cache)
    assert any(b.kind.name == "LAYER_DIM_MISMATCH" for _p, b in inc0.all_bugs())

    _build(tmp_path, enc=ENC_OK)
    inc, stats = analyze_package_incremental(str(tmp_path), cache)
    assert _bn(stats.reanalyzed) == ["encoder.py", "model.py"]
    assert _fps(analyze_package(str(tmp_path))) == _fps(inc)
    assert inc.all_bugs() == []


def test_unrelated_file_edit_does_not_invalidate_dependents(tmp_path):
    # Editing the leaf importer must NOT trigger re-analysis of the imported lib.
    _build(tmp_path)
    cache = IncrementalCache()
    analyze_package_incremental(str(tmp_path), cache)
    _build(tmp_path, model=MODEL.replace("randn(3, 8)", "randn(3, 8)  # noop"))
    _, stats = analyze_package_incremental(str(tmp_path), cache)
    assert "encoder.py" not in _bn(stats.reanalyzed)


def test_stale_entry_evicted_when_file_removed(tmp_path):
    _build(tmp_path)
    cache = IncrementalCache()
    analyze_package_incremental(str(tmp_path), cache)
    n_before = len(cache)
    # Remove model.py from the project.
    os.remove(os.path.join(str(tmp_path), "mypkg", "model.py"))
    _, stats = analyze_package_incremental(str(tmp_path), cache)
    assert len(cache) == n_before - 1
    assert all("model.py" not in k for k in [_p for _p in stats.reused + stats.reanalyzed])


def test_none_cache_equals_fresh(tmp_path):
    _build(tmp_path)
    inc, stats = analyze_package_incremental(str(tmp_path))  # no cache supplied
    assert len(stats.reanalyzed) == 4
    assert _fps(inc) == _fps(analyze_package(str(tmp_path)))


# --------------------------------------------------------------------------- #
# Single-file incremental                                                     #
# --------------------------------------------------------------------------- #

def test_source_incremental_reuse_and_recompute():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.randn(4, 8)\n"
        "    w = torch.randn(5, 3)\n"
        "    return torch.matmul(x, w)\n"
    )
    cache = IncrementalCache()
    r1, reused1 = analyze_source_incremental(src, "f.py", cache)
    assert reused1 is False
    r2, reused2 = analyze_source_incremental(src, "f.py", cache)
    assert reused2 is True
    assert r1 is r2  # exact cached object
    assert r1.fingerprint() == s.analyze_source(src, "f.py").fingerprint()


def test_source_incremental_recomputes_on_change():
    cache = IncrementalCache()
    src1 = "def f():\n    return 1\n"
    src2 = "def f():\n    return 2\n"
    _, reused1 = analyze_source_incremental(src1, "f.py", cache)
    r2, reused2 = analyze_source_incremental(src2, "f.py", cache)
    assert reused1 is False and reused2 is False  # changed source -> recompute


def test_cache_helpers():
    cache = IncrementalCache()
    analyze_source_incremental("x = 1\n", "g.py", cache)
    assert "g.py" in cache and len(cache) == 1
    cache.invalidate("g.py")
    assert "g.py" not in cache
    analyze_source_incremental("x = 1\n", "g.py", cache)
    cache.clear()
    assert len(cache) == 0


def test_stats_dataclass():
    st = IncrementalStats(reused=("a", "b"), reanalyzed=("c",))
    assert st.total == 3
    assert isinstance(UnitChange().any, bool)
