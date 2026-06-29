"""Tests for whole-package cross-file analysis (roadmap Step 82).

Exercises qualname/relative-import resolution, the symbol/import index, the
``resolve`` import-chain walk (re-exports + cycle guard), import-augmented module
construction, the cross-file call graph, and end-to-end cross-file bug detection
with no false positives on a correct project.  All fixtures are written into
``tmp_path`` so the suite is hermetic and torch-free.
"""

import ast
import os

import pytest

from src.symexec.package import (
    ModuleInfo,
    PackageIndex,
    PackageResult,
    _package_of,
    _qualname_for,
    _resolve_relative,
    analyze_package,
)


# --------------------------------------------------------------------------- #
# Fixture helpers                                                             #
# --------------------------------------------------------------------------- #

def _write(root, relpath, text):
    path = os.path.join(root, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _build_pkg(tmp_path, files):
    root = str(tmp_path)
    for rel, text in files.items():
        _write(root, rel, text)
    return root


# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #

def test_qualname_for_module_and_package(tmp_path):
    root = str(tmp_path)
    assert _qualname_for(root, os.path.join(root, "a", "b", "c.py")) == "a.b.c"
    assert _qualname_for(root, os.path.join(root, "a", "b", "__init__.py")) == "a.b"
    assert _qualname_for(root, os.path.join(root, "top.py")) == "top"


def test_package_of():
    assert _package_of("a.b.c", is_package=False) == "a.b"
    assert _package_of("a.b", is_package=True) == "a.b"
    assert _package_of("top", is_package=False) == ""


def test_resolve_relative_levels():
    # level 0 == absolute
    assert _resolve_relative("a.b", 0, "x.y") == "x.y"
    # level 1 == current package
    assert _resolve_relative("a.b", 1, "m") == "a.b.m"
    assert _resolve_relative("a.b", 1, None) == "a.b"
    # level 2 strips one package level
    assert _resolve_relative("a.b", 2, "m") == "a.m"
    # escaping the root returns None
    assert _resolve_relative("a", 3, "m") is None


# --------------------------------------------------------------------------- #
# Index construction                                                          #
# --------------------------------------------------------------------------- #

def test_index_discovers_modules_and_symbols(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "class Foo:\n    pass\n\ndef bar():\n    return 1\n",
    })
    idx = PackageIndex.build(root)
    assert "pkg" in idx.modules
    assert "pkg.a" in idx.modules
    assert idx.modules["pkg"].is_package is True
    assert idx.modules["pkg.a"].is_package is False
    assert set(idx.modules["pkg.a"].symbols) == {"Foo", "bar"}


def test_index_skips_unparseable_files(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/broken.py": "def (:\n",  # syntax error
        "pkg/ok.py": "x = 1\n",
    })
    idx = PackageIndex.build(root)
    assert "pkg.broken" not in idx.modules
    assert "pkg.ok" in idx.modules


def test_index_imports_absolute_and_from(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/layers.py": "class Enc:\n    pass\n",
        "pkg/model.py": (
            "import os\n"
            "import numpy as np\n"
            "from pkg.layers import Enc\n"
            "from pkg.layers import Enc as E2\n"
        ),
    })
    idx = PackageIndex.build(root)
    imports = idx.modules["pkg.model"].imports
    assert imports["os"] == ("os", None)
    assert imports["np"] == ("numpy", None)
    assert imports["Enc"] == ("pkg.layers", "Enc")
    assert imports["E2"] == ("pkg.layers", "Enc")


def test_index_relative_imports(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/sub/__init__.py": "",
        "pkg/sub/enc.py": "class Enc:\n    pass\n",
        "pkg/sub/model.py": "from .enc import Enc\nfrom . import enc\n",
    })
    idx = PackageIndex.build(root)
    imports = idx.modules["pkg.sub.model"].imports
    assert imports["Enc"] == ("pkg.sub.enc", "Enc")
    # ``from . import enc`` binds the module ``pkg.sub.enc`` (target_name == enc)
    assert imports["enc"] == ("pkg.sub", "enc")


# --------------------------------------------------------------------------- #
# Resolution                                                                  #
# --------------------------------------------------------------------------- #

def test_resolve_local_symbol(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "class Foo:\n    pass\n",
    })
    idx = PackageIndex.build(root)
    mod, node = idx.resolve("pkg.a", "Foo")
    assert mod == "pkg.a"
    assert isinstance(node, ast.ClassDef)


def test_resolve_through_import(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/defs.py": "class Foo:\n    pass\n",
        "pkg/use.py": "from pkg.defs import Foo\n",
    })
    idx = PackageIndex.build(root)
    mod, node = idx.resolve("pkg.use", "Foo")
    assert mod == "pkg.defs"
    assert isinstance(node, ast.ClassDef)


def test_resolve_reexport_chain(tmp_path):
    # __init__ re-exports Foo from a submodule; another module imports it
    # through the package — resolution must follow the chain to the real def.
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "from pkg.defs import Foo\n",
        "pkg/defs.py": "class Foo:\n    pass\n",
        "pkg/use.py": "from pkg import Foo\n",
    })
    idx = PackageIndex.build(root)
    mod, node = idx.resolve("pkg.use", "Foo")
    assert mod == "pkg.defs"
    assert isinstance(node, ast.ClassDef)


def test_resolve_third_party_unresolved(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/m.py": "import numpy as np\nfrom torch import nn\n",
    })
    idx = PackageIndex.build(root)
    assert idx.resolve("pkg.m", "np") is None  # module-only binding
    assert idx.resolve("pkg.m", "nn") is None  # third-party, not in project
    assert idx.resolve("pkg.m", "missing") is None


def test_resolve_cycle_guard(tmp_path):
    # a re-exports from b, b re-exports from a: resolution must not loop.
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "from pkg.b import Thing\n",
        "pkg/b.py": "from pkg.a import Thing\n",
    })
    idx = PackageIndex.build(root)
    assert idx.resolve("pkg.a", "Thing") is None  # unresolvable, terminates


# --------------------------------------------------------------------------- #
# Import-augmented module                                                     #
# --------------------------------------------------------------------------- #

def test_augmented_module_injects_imported_def(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/defs.py": "class Foo:\n    pass\n",
        "pkg/use.py": "from pkg.defs import Foo\n\ndef run():\n    return Foo()\n",
    })
    idx = PackageIndex.build(root)
    aug, injected = idx.augmented_module("pkg.use")
    names = [n.name for n in aug.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    assert "Foo" in names  # injected
    assert "run" in names  # original
    # the injected node's id is recorded
    foo_nodes = [n for n in aug.body if isinstance(n, ast.ClassDef) and n.name == "Foo"]
    assert id(foo_nodes[0]) in injected


def test_augmented_module_renames_alias(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/defs.py": "class Foo:\n    pass\n",
        "pkg/use.py": "from pkg.defs import Foo as Bar\n",
    })
    idx = PackageIndex.build(root)
    aug, injected = idx.augmented_module("pkg.use")
    cls = [n for n in aug.body if isinstance(n, ast.ClassDef)]
    assert [c.name for c in cls] == ["Bar"]  # bound under the alias, not Foo


def test_augmented_module_local_shadows_import(tmp_path):
    # A local def with the same name as an import must NOT be overridden.
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/defs.py": "class Foo:\n    x = 1\n",
        "pkg/use.py": "from pkg.defs import Foo\n\nclass Foo:\n    y = 2\n",
    })
    idx = PackageIndex.build(root)
    aug, injected = idx.augmented_module("pkg.use")
    foos = [n for n in aug.body if isinstance(n, ast.ClassDef) and n.name == "Foo"]
    assert len(foos) == 1  # not injected (shadowed)
    assert not injected


def test_augmented_module_third_party_not_injected(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/use.py": "from torch import nn\n",
    })
    idx = PackageIndex.build(root)
    aug, injected = idx.augmented_module("pkg.use")
    assert not injected  # nothing project-local to inject


# --------------------------------------------------------------------------- #
# Call graph                                                                  #
# --------------------------------------------------------------------------- #

def test_call_graph_name_edge(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/defs.py": "class Enc:\n    pass\n",
        "pkg/model.py": "from pkg.defs import Enc\n\ndef run():\n    return Enc()\n",
    })
    idx = PackageIndex.build(root)
    g = idx.call_graph()
    assert g["pkg.model:run"] == ["pkg.defs:Enc"]


def test_call_graph_module_attr_edge(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/defs.py": "class Enc:\n    pass\n",
        "pkg/model.py": "import pkg.defs as d\n\ndef run():\n    return d.Enc()\n",
    })
    idx = PackageIndex.build(root)
    g = idx.call_graph()
    assert g["pkg.model:run"] == ["pkg.defs:Enc"]


def test_call_graph_no_intrafile_edges(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/m.py": "def a():\n    return b()\n\ndef b():\n    return 1\n",
    })
    idx = PackageIndex.build(root)
    g = idx.call_graph()
    assert g == {}  # intra-file calls are not cross-file edges


# --------------------------------------------------------------------------- #
# End-to-end analysis                                                         #
# --------------------------------------------------------------------------- #

def test_cross_file_bug_detected(tmp_path):
    # model.py feeds an 8-wide tensor into an Encoder whose Linear expects 10.
    root = _build_pkg(tmp_path, {
        "mypkg/__init__.py": "",
        "mypkg/layers/__init__.py": "",
        "mypkg/layers/encoder.py": (
            "import torch.nn as nn\n\n"
            "class Encoder(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(10, 5)\n"
            "    def forward(self, x):\n"
            "        return self.fc(x)\n"
        ),
        "mypkg/model.py": (
            "import torch\n"
            "from mypkg.layers.encoder import Encoder\n\n"
            "def run():\n"
            "    enc = Encoder()\n"
            "    x = torch.randn(3, 8)\n"
            "    return enc(x)\n\n"
            "if __name__ == '__main__':\n"
            "    run()\n"
        ),
    })
    pr = analyze_package(root)
    bugs = pr.all_bugs()
    kinds = {b.kind.name for _p, b in bugs}
    assert "LAYER_DIM_MISMATCH" in kinds
    # and the cross-file edge is present
    g = pr.call_graph()
    assert any("encoder:Encoder" in t for ts in g.values() for t in ts)


def test_correct_project_no_false_positive(tmp_path):
    root = _build_pkg(tmp_path, {
        "mypkg/__init__.py": "",
        "mypkg/layers/__init__.py": "",
        "mypkg/layers/encoder.py": (
            "import torch.nn as nn\n\n"
            "class Encoder(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(8, 5)\n"
            "    def forward(self, x):\n"
            "        return self.fc(x)\n"
        ),
        "mypkg/model.py": (
            "import torch\n"
            "from mypkg.layers.encoder import Encoder\n\n"
            "def run():\n"
            "    enc = Encoder()\n"
            "    x = torch.randn(3, 8)\n"
            "    return enc(x)\n\n"
            "if __name__ == '__main__':\n"
            "    run()\n"
        ),
    })
    pr = analyze_package(root)
    assert pr.all_bugs() == []


def test_package_result_aggregates(tmp_path):
    root = _build_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "def f():\n    return 1\n",
        "pkg/b.py": "def g():\n    return 2\n",
    })
    pr = analyze_package(root)
    assert isinstance(pr, PackageResult)
    assert pr.files_analyzed == 3  # __init__, a, b
    assert pr.functions_analyzed >= 2
    assert pr.all_bugs() == []


def test_single_file_misses_cross_file_bug(tmp_path):
    # Establishes the value of whole-package analysis: the same model analysed
    # standalone (no Encoder body visible) yields no bug.
    import src.symexec as s
    model_src = (
        "import torch\n"
        "from mypkg.layers.encoder import Encoder\n\n"
        "def run():\n"
        "    enc = Encoder()\n"
        "    x = torch.randn(3, 8)\n"
        "    return enc(x)\n"
    )
    r = s.analyze_source(model_src, "model.py")
    assert r.bugs == []
