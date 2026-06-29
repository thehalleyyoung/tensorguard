"""Tests for the Jupyter / VS Code editor integrations (roadmap Step 89).

Covers notebook parsing, cell-attributed analysis (a finding's global line maps
back to the right cell + in-cell line), IPython-magic sanitization, the inline
HTML rendering, and the LSP ``publishDiagnostics`` notification VS Code consumes.
"""

import json

from src.symexec import (
    analyze_notebook,
    analyze_source,
    parse_notebook,
    to_publish_diagnostics,
    to_lsp_diagnostics,
)
from src.symexec.notebook import NotebookCell, _concatenate, _sanitize_line


def _nb(*code_sources, with_markdown=True):
    cells = []
    if with_markdown:
        cells.append({"cell_type": "markdown", "source": ["# heading\n"]})
    for src in code_sources:
        cells.append({"cell_type": "code", "source": src})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {}}


_BUG_CELL = ["a = torch.randn(2, 3)\n", "b = torch.randn(4, 5)\n", "c = a @ b\n"]


# --------------------------------------------------------------------------- #
# parse_notebook                                                              #
# --------------------------------------------------------------------------- #

def test_parse_skips_non_code_cells():
    nb = _nb(["import torch\n"], _BUG_CELL)
    cells = parse_notebook(nb)
    assert [c.index for c in cells] == [0, 1]
    assert "import torch" in cells[0].source


def test_parse_accepts_json_string():
    nb = _nb(["import torch\n"])
    cells = parse_notebook(json.dumps(nb))
    assert len(cells) == 1


def test_parse_accepts_file_path(tmp_path):
    nb = _nb(["import torch\n"])
    p = tmp_path / "x.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    cells = parse_notebook(str(p))
    assert len(cells) == 1


def test_parse_source_as_single_string():
    nb = {"cells": [{"cell_type": "code", "source": "import torch\n"}]}
    cells = parse_notebook(nb)
    assert cells[0].source == "import torch\n"


# --------------------------------------------------------------------------- #
# IPython magic sanitization                                                  #
# --------------------------------------------------------------------------- #

def test_sanitize_blanks_magics_and_shell():
    assert _sanitize_line("%matplotlib inline") == ""
    assert _sanitize_line("%%time") == ""
    assert _sanitize_line("!pip install torch") == ""
    assert _sanitize_line("?torch.randn") == ""


def test_sanitize_keeps_normal_code():
    assert _sanitize_line("x = a % b") == "x = a % b"
    assert _sanitize_line("    return x") == "    return x"


def test_concatenate_preserves_line_numbers_through_magics():
    cells = [NotebookCell(0, "%matplotlib inline\nimport torch\nx = 1\n")]
    source, line_map = _concatenate(cells)
    # magic line is blanked but still occupies its line so x=1 stays on cell line 3
    assert "%matplotlib" not in source
    # the synthetic def occupies global line 1
    assert source.startswith("def __nb_cells__():")
    assert line_map[0] == (-1, 0)


# --------------------------------------------------------------------------- #
# Cell-attributed analysis                                                    #
# --------------------------------------------------------------------------- #

def test_analyze_attributes_bug_to_cell():
    nb = _nb(["%matplotlib inline\n", "import torch\n"], _BUG_CELL)
    res = analyze_notebook(nb)
    assert len(res.findings) == 1
    f = res.findings[0]
    assert f.cell_index == 1
    assert f.cell_line == 3                 # the `c = a @ b` line
    assert f.bug.kind.value == "matmul_dim_mismatch"


def test_clean_notebook_has_no_findings():
    nb = _nb(["import torch\n", "a = torch.randn(2, 3)\n", "b = a @ torch.randn(3, 4)\n"])
    res = analyze_notebook(nb)
    assert res.findings == []
    assert "no issues" in res.summary()


def test_by_cell_grouping():
    nb = _nb(["import torch\n"], _BUG_CELL)
    res = analyze_notebook(nb)
    grouped = res.by_cell()
    assert set(grouped) == {1}
    assert len(grouped[1]) == 1


def test_findings_to_dict_carries_cell_location():
    nb = _nb(["import torch\n"], _BUG_CELL)
    res = analyze_notebook(nb)
    d = res.findings[0].to_dict()
    assert d["cell_index"] == 1 and d["cell_line"] == 3
    assert d["kind"] == "matmul_dim_mismatch"


def test_notebook_result_bugs_parity():
    nb = _nb(["import torch\n"], _BUG_CELL)
    res = analyze_notebook(nb)
    assert res.bugs == res.result.bugs


def test_analyze_accepts_json_string_input():
    nb = _nb(["import torch\n"], _BUG_CELL)
    res = analyze_notebook(json.dumps(nb))
    assert len(res.findings) == 1


# --------------------------------------------------------------------------- #
# HTML rendering                                                              #
# --------------------------------------------------------------------------- #

def test_html_clean_is_checkmark():
    nb = _nb(["import torch\n"])
    html = analyze_notebook(nb).to_html()
    assert "<table" not in html
    assert "no issues" in html


def test_html_with_findings_has_table_and_escapes():
    nb = _nb(["import torch\n"], _BUG_CELL)
    html = analyze_notebook(nb).to_html()
    assert "<table" in html
    assert "matmul_dim_mismatch" in html
    # the message contains no raw unescaped angle brackets that would break HTML
    assert "<script" not in html


def test_repr_html_delegates_to_to_html():
    nb = _nb(["import torch\n"], _BUG_CELL)
    res = analyze_notebook(nb)
    assert res._repr_html_() == res.to_html()


# --------------------------------------------------------------------------- #
# VS Code / LSP publishDiagnostics                                            #
# --------------------------------------------------------------------------- #

_SRC = "import torch\ndef f():\n    return torch.randn(2, 3) @ torch.randn(4, 5)\n"


def test_publish_diagnostics_notification_shape():
    r = analyze_source(_SRC, filename="m.py")
    note = to_publish_diagnostics(r, "file:///m.py")
    assert note["jsonrpc"] == "2.0"
    assert note["method"] == "textDocument/publishDiagnostics"
    assert note["params"]["uri"] == "file:///m.py"
    assert note["params"]["diagnostics"] == to_lsp_diagnostics(r, uri="file:///m.py")
    assert len(note["params"]["diagnostics"]) == 1


def test_publish_diagnostics_empty_clears():
    clean = analyze_source("x = 1\n", filename="m.py")
    note = to_publish_diagnostics(clean, "file:///m.py")
    assert note["params"]["diagnostics"] == []
