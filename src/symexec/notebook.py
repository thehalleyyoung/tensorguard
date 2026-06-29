"""Jupyter / notebook integration for the symbolic-execution engine (Step 89).

The engine is **torch-free**, so it can run inside a Jupyter kernel — exactly
where a data-scientist writes the throwaway shape bugs the engine is best at
catching.  This module makes that ergonomic:

* :func:`parse_notebook` reads a ``.ipynb`` document (path, JSON string, or the
  already-parsed dict) and returns its **code** cells.
* :func:`analyze_notebook` concatenates the code cells into one virtual module,
  analyses it once, and **maps every finding back to its originating cell**
  (cell index + in-cell line), so a bug reported on global line 57 is surfaced
  as "cell 4, line 3".  IPython-only lines (``%magic``, ``%%cellmagic``,
  ``!shell``, ``?help``) are blanked *in place* so the virtual module still
  parses while line numbers are preserved.
* :class:`NotebookResult` carries the per-cell findings and renders an
  ``_repr_html_`` so it displays as a table inline in a notebook.
* :func:`load_ipython_extension` registers a ``%%tensorguard`` cell magic
  (``%load_ext src.symexec.notebook``) that analyses the cell it decorates and
  displays the findings — no effect on the cell's normal execution.

Pure and self-contained: IPython is imported lazily and only when the magic is
actually used, so importing this module never requires Jupyter to be installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .engine import SymResult, analyze_source
from .config import SymConfig

__all__ = [
    "NotebookCell",
    "CellFinding",
    "NotebookResult",
    "parse_notebook",
    "analyze_notebook",
    "load_ipython_extension",
]

# Leading tokens that mark a line as IPython-only (not valid Python).  Such a
# line is replaced by an empty line so the virtual module parses while every
# subsequent line keeps its number.
_IPYTHON_PREFIXES = ("%", "!", "?")


@dataclass(frozen=True)
class NotebookCell:
    """One code cell: its zero-based ``index`` among code cells and ``source``
    (a single string, newline-joined)."""

    index: int
    source: str


@dataclass(frozen=True)
class CellFinding:
    """A :class:`~src.symexec.bugs.SymBug` mapped back to its notebook cell.

    ``cell_index`` is the zero-based code-cell number; ``cell_line`` is the
    1-based line within that cell; ``bug`` is the original finding (whose
    ``line`` is the global line in the concatenated virtual module)."""

    cell_index: int
    cell_line: int
    bug: Any

    def to_dict(self) -> dict:
        d = {"cell_index": self.cell_index, "cell_line": self.cell_line}
        d.update(self.bug.to_dict())
        return d


@dataclass
class NotebookResult:
    """The outcome of analysing a notebook: the underlying :class:`SymResult`
    over the concatenated virtual module plus per-cell finding attribution."""

    result: SymResult
    findings: List[CellFinding] = field(default_factory=list)
    cells: List[NotebookCell] = field(default_factory=list)

    @property
    def bugs(self) -> List[Any]:
        """The raw findings (global line numbers), for parity with SymResult."""
        return self.result.bugs

    def by_cell(self) -> Dict[int, List[CellFinding]]:
        """Findings grouped by cell index (ascending)."""
        out: Dict[int, List[CellFinding]] = {}
        for f in self.findings:
            out.setdefault(f.cell_index, []).append(f)
        return out

    def summary(self) -> str:
        n = len(self.findings)
        if n == 0:
            return f"tensorguard: no issues in {len(self.cells)} code cell(s)"
        cells = len({f.cell_index for f in self.findings})
        return f"tensorguard: {n} issue(s) across {cells} cell(s)"

    def _repr_html_(self) -> str:  # pragma: no cover - exercised via to_html
        return self.to_html()

    def to_html(self) -> str:
        """An HTML table of findings for inline notebook display."""
        if not self.findings:
            return (
                '<div style="font-family:monospace">'
                f"✅ {self.summary()}</div>"
            )
        rows = []
        for f in self.findings:
            b = f.bug
            conf = getattr(b, "confidence", None)
            conf_s = "" if conf is None else f"{float(conf):.2f}"
            rows.append(
                "<tr>"
                f"<td>{f.cell_index}</td>"
                f"<td>{f.cell_line}</td>"
                f"<td>{_html_escape(getattr(getattr(b,'kind',None),'value',''))}</td>"
                f"<td>{_html_escape(getattr(b,'message','') or '')}</td>"
                f"<td>{conf_s}</td>"
                "</tr>"
            )
        return (
            '<div style="font-family:monospace">'
            f"⚠️ {_html_escape(self.summary())}"
            '<table style="border-collapse:collapse" border="1">'
            "<tr><th>cell</th><th>line</th><th>kind</th>"
            "<th>message</th><th>conf</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )


def _html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _cell_source_to_str(source: Any) -> str:
    """Notebook cell ``source`` is a list of strings or a single string."""
    if isinstance(source, list):
        return "".join(source)
    return str(source or "")


def parse_notebook(nb: Any) -> List[NotebookCell]:
    """Return the code cells of a notebook.

    ``nb`` may be a path to a ``.ipynb`` file, a JSON string, or an
    already-parsed notebook dict.  Non-code cells (markdown/raw) are skipped;
    code cells are numbered 0..n in document order."""
    if isinstance(nb, dict):
        doc = nb
    else:
        text = str(nb)
        if "\n" not in text and text.endswith(".ipynb"):
            with open(text, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        else:
            doc = json.loads(text)
    cells: List[NotebookCell] = []
    idx = 0
    for cell in doc.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cells.append(NotebookCell(index=idx, source=_cell_source_to_str(cell.get("source"))))
        idx += 1
    return cells


def _sanitize_line(line: str) -> str:
    """Blank an IPython-only line (magic / shell / help) in place, preserving
    its presence so subsequent line numbers don't shift.  A normal Python line
    is returned unchanged."""
    stripped = line.lstrip()
    if stripped[:1] in _IPYTHON_PREFIXES:
        return ""
    return line


def _concatenate(cells: List[NotebookCell]) -> Tuple[str, List[Tuple[int, int]]]:
    """Join cells into one virtual *module* the engine will actually analyse.

    Notebook cells are top-level statements, but the engine only executes the
    ``__main__`` harness, free functions and methods — not bare module-level
    code.  So the cells are wrapped as the body of a synthetic free function
    (``def __nb_cells__():``), which the engine runs exactly like a script,
    following calls into any functions/classes the cells define.

    Returns ``(source, line_map)`` where ``line_map[i]`` is the
    ``(cell_index, in_cell_line)`` for global 1-based line ``i + 1`` in the
    *wrapped* source (the synthetic ``def`` line maps to ``(-1, 0)``)."""
    line_map: List[Tuple[int, int]] = [(-1, 0)]  # global line 1 = the `def`
    body: List[str] = []
    for cell in cells:
        lines = cell.source.split("\n")
        for in_cell_line, raw in enumerate(lines, start=1):
            sanitized = _sanitize_line(raw)
            # Indent into the synthetic function body; keep blank lines blank.
            body.append(("    " + sanitized) if sanitized else "")
            line_map.append((cell.index, in_cell_line))
    if not body:
        body = ["    pass"]
        line_map.append((-1, 0))
    source = "def __nb_cells__():\n" + "\n".join(body)
    return source, line_map


def analyze_notebook(
    nb: Any,
    filename: str = "<notebook>",
    *,
    budget_ms: Optional[float] = None,
    config: Optional[SymConfig] = None,
) -> NotebookResult:
    """Analyse a notebook's code cells and attribute every finding to its cell.

    ``nb`` is anything :func:`parse_notebook` accepts.  The code cells are
    concatenated into one virtual module (IPython-only lines blanked), analysed
    once, and each finding's global line is mapped back to ``(cell, line)``."""
    cells = parse_notebook(nb)
    source, line_map = _concatenate(cells)
    result = analyze_source(source, filename=filename, budget_ms=budget_ms, config=config)
    findings: List[CellFinding] = []
    for bug in result.bugs:
        gline = int(getattr(bug, "line", 0) or 0)
        if 1 <= gline <= len(line_map):
            cell_index, cell_line = line_map[gline - 1]
        else:
            cell_index, cell_line = (-1, gline)
        findings.append(CellFinding(cell_index=cell_index, cell_line=cell_line, bug=bug))
    return NotebookResult(result=result, findings=findings, cells=cells)


# --------------------------------------------------------------------------- #
# IPython cell magic: %%tensorguard                                           #
# --------------------------------------------------------------------------- #

def _analyze_single_cell(cell_source: str, config: Optional[SymConfig] = None) -> NotebookResult:
    """Analyse one cell of source as a one-cell notebook (used by the magic)."""
    cells = [NotebookCell(index=0, source=cell_source)]
    source, line_map = _concatenate(cells)
    result = analyze_source(source, filename="<cell>", config=config)
    findings = []
    for bug in result.bugs:
        gline = int(getattr(bug, "line", 0) or 0)
        cell_index, cell_line = line_map[gline - 1] if 1 <= gline <= len(line_map) else (0, gline)
        findings.append(CellFinding(cell_index=cell_index, cell_line=cell_line, bug=bug))
    return NotebookResult(result=result, findings=findings, cells=cells)


def load_ipython_extension(ipython) -> None:  # pragma: no cover - requires IPython
    """Register the ``%%tensorguard`` cell magic.

    Enable with ``%load_ext src.symexec.notebook``.  Decorating a cell with
    ``%%tensorguard`` analyses that cell and displays the findings table, then
    runs the cell normally so the magic is non-intrusive.  Use
    ``%%tensorguard --no-run`` to analyse without executing, or
    ``%%tensorguard --mode heuristic`` to pick a soundness mode."""
    from IPython.display import display

    def tensorguard_magic(line, cell):
        args = (line or "").split()
        mode = "balanced"
        run = True
        if "--no-run" in args:
            run = False
            args.remove("--no-run")
        if "--mode" in args:
            i = args.index("--mode")
            if i + 1 < len(args):
                mode = args[i + 1]
        nb_result = _analyze_single_cell(cell, config=SymConfig.for_mode(mode))
        display(nb_result)
        if run:
            ipython.run_cell(cell)

    ipython.register_magic_function(tensorguard_magic, "cell", "tensorguard")
