"""Generate the formal TensorGuard soundness appendix.

The appendix is intentionally not hand-written.  It is rendered from three
sources that already gate the repository's trust story:

* Lean theorem and lemma declarations under ``lean/``;
* the live proof-footprint manifest from ``src.proof_footprint``; and
* the verifiable-fragment grammar/tables from ``src.verifiable_fragment``.

Regenerate with::

    python -m src.formal_soundness_appendix > formal_soundness_appendix.tex
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
LEAN_ROOT = REPO / "lean"
APPENDIX_TEX = REPO / "formal_soundness_appendix.tex"
APPENDIX_PDF = REPO / "formal_soundness_appendix.pdf"


_DECL_RE = re.compile(
    r"^\s*(?:(?:private|protected)\s+)?(?P<kind>theorem|lemma)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_']*)\b"
)
_NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_'.]*)\b")
_END_RE = re.compile(r"^\s*end(?:\s+([A-Za-z_][A-Za-z0-9_'.]*))?\b")
_AXIOM_RE = re.compile(r"^\s*#print\s+axioms\s+([A-Za-z_][A-Za-z0-9_'.]*)\b")


@dataclass(frozen=True)
class LeanDeclaration:
    """A theorem/lemma statement mined from a Lean source file."""

    kind: str
    name: str
    qualified_name: str
    module: str
    path: str
    line: int
    statement: str
    doc: str


@dataclass(frozen=True)
class AppendixModel:
    """All deterministic data needed to render the appendix."""

    declarations: Sequence[LeanDeclaration]
    proof_payload: Mapping[str, object]
    audited_theorem_names: Sequence[str]
    grammar: str
    supported_counts: Mapping[str, int]
    supported_lists: Mapping[str, Sequence[str]]


def lean_source_files(repo: Path = REPO) -> List[Path]:
    """Return tracked Lean source paths in deterministic module order."""

    lean_dir = repo / "lean"
    return sorted(
        p for p in lean_dir.rglob("*.lean")
        if ".lake" not in p.parts and p.is_file()
    )


def module_name_for(path: Path, repo: Path = REPO) -> str:
    rel = path.relative_to(repo / "lean").with_suffix("")
    return ".".join(rel.parts)


def _mask_block_comments(source: str) -> str:
    """Replace Lean block comments with spaces while preserving line numbers."""

    out: List[str] = []
    i = 0
    depth = 0
    while i < len(source):
        nxt = source[i:i + 2]
        if nxt == "/-":
            depth += 1
            out.extend("  ")
            i += 2
            continue
        if depth and nxt == "-/":
            depth -= 1
            out.extend("  ")
            i += 2
            continue
        ch = source[i]
        if depth:
            out.append("\n" if ch == "\n" else " ")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _strip_line_comment(line: str) -> str:
    return re.sub(r"--.*$", "", line)


def _clean_doc(raw_parts: Sequence[str]) -> str:
    text = " ".join(part.strip() for part in raw_parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_statement(clean_lines: Sequence[str], start: int) -> str:
    parts: List[str] = []
    for line in clean_lines[start:start + 40]:
        body = _strip_line_comment(line).strip()
        if not body:
            continue
        if ":=" in body:
            before = body.split(":=", 1)[0].rstrip()
            if before:
                parts.append(before)
            break
        parts.append(body)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _pop_namespace(stack: List[str], name: str | None) -> None:
    if not stack:
        return
    if not name:
        stack.pop()
        return
    parts = name.split(".")
    if stack[-len(parts):] == parts:
        del stack[-len(parts):]
    elif stack[-1] == parts[-1]:
        stack.pop()


def scan_lean_declarations(repo: Path = REPO) -> List[LeanDeclaration]:
    """Scan Lean files for namespace-qualified theorem and lemma statements."""

    declarations: List[LeanDeclaration] = []
    for path in lean_source_files(repo):
        source = path.read_text(encoding="utf-8")
        raw_lines = source.splitlines()
        clean_lines = _mask_block_comments(source).splitlines()
        module = module_name_for(path, repo)
        rel_path = path.relative_to(repo).as_posix()
        namespace: List[str] = []
        pending_doc = ""
        doc_parts: List[str] | None = None

        for i, (raw, clean) in enumerate(zip(raw_lines, clean_lines), start=1):
            raw_stripped = raw.strip()
            if doc_parts is not None:
                end = raw_stripped.find("-/")
                piece = raw_stripped[:end] if end >= 0 else raw_stripped
                doc_parts.append(piece)
                if end >= 0:
                    pending_doc = _clean_doc(doc_parts)
                    doc_parts = None
                continue
            if raw_stripped.startswith("/--"):
                after = raw_stripped[3:]
                end = after.find("-/")
                if end >= 0:
                    pending_doc = _clean_doc([after[:end]])
                else:
                    doc_parts = [after]
                continue

            code = _strip_line_comment(clean)
            stripped = code.strip()
            if not stripped:
                continue

            ns_match = _NAMESPACE_RE.match(code)
            if ns_match:
                namespace.extend(ns_match.group(1).split("."))
                pending_doc = ""
                continue

            end_match = _END_RE.match(code)
            if end_match:
                _pop_namespace(namespace, end_match.group(1))
                pending_doc = ""
                continue

            decl_match = _DECL_RE.match(code)
            if decl_match:
                name = decl_match.group("name")
                qualified = ".".join(namespace + [name]) if namespace else name
                declarations.append(
                    LeanDeclaration(
                        kind=decl_match.group("kind"),
                        name=name,
                        qualified_name=qualified,
                        module=module,
                        path=rel_path,
                        line=i,
                        statement=_extract_statement(clean_lines, i - 1),
                        doc=pending_doc,
                    )
                )
                pending_doc = ""
                continue

            # Attribute lines may sit between a doc comment and theorem.  Other
            # code means the doc belonged to a different declaration.
            if not stripped.startswith("@["):
                pending_doc = ""

    return sorted(declarations, key=lambda d: (d.module, d.line, d.qualified_name))


def audited_theorem_names(repo: Path = REPO) -> List[str]:
    audit = repo / "lean" / "TensorGuard" / "AxiomAudit.lean"
    if not audit.exists():
        return []
    names: List[str] = []
    for line in audit.read_text(encoding="utf-8").splitlines():
        match = _AXIOM_RE.match(line)
        if match:
            names.append(match.group(1))
    return names


def build_model(repo: Path = REPO) -> AppendixModel:
    from src.proof_footprint import to_payload
    from src.verifiable_fragment import (
        SUPPORTED_F_FUNCTIONS,
        SUPPORTED_LAYER_TYPES,
        SUPPORTED_TENSOR_METHODS,
        SUPPORTED_TORCH_FUNCTIONS,
        SUPPORTED_TORCHVISION_V2_TRANSFORMS,
        VERIFIABLE_FRAGMENT_GRAMMAR,
    )

    supported_lists = {
        "nn layers": sorted(SUPPORTED_LAYER_TYPES),
        "torchvision v2 transforms": sorted(SUPPORTED_TORCHVISION_V2_TRANSFORMS),
        "tensor methods": sorted(SUPPORTED_TENSOR_METHODS),
        "torch functions": sorted(SUPPORTED_TORCH_FUNCTIONS),
        "F functions": sorted(SUPPORTED_F_FUNCTIONS),
    }
    return AppendixModel(
        declarations=scan_lean_declarations(repo),
        proof_payload=to_payload(),
        audited_theorem_names=audited_theorem_names(repo),
        grammar=VERIFIABLE_FRAGMENT_GRAMMAR.rstrip(),
        supported_counts={k: len(v) for k, v in supported_lists.items()},
        supported_lists=supported_lists,
    )


_UNICODE_TO_ASCII = {
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "∀": "forall",
    "∃": "exists",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "∧": "/\\",
    "∨": "\\/",
    "¬": "not ",
    "∈": "in",
    "∉": "notin",
    "⊆": "subseteq",
    "⊂": "subset",
    "⊢": "|-",
    "⊤": "top",
    "⊥": "bottom",
    "⟨": "<",
    "⟩": ">",
    "·": ".",
    "×": "x",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "λ": "lambda",
    "μ": "mu",
    "σ": "sigma",
}


def _ascii(text: str) -> str:
    for src, dst in _UNICODE_TO_ASCII.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def _tex(text: object) -> str:
    s = _ascii(str(text))
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _csv(items: Iterable[str]) -> str:
    return ", ".join(_tex(item) for item in items)


def _status_order(summary: Mapping[str, int]) -> List[str]:
    preferred = ["lean_theorem", "pen_and_paper_rule", "tested_only_rule", "heuristic"]
    return [s for s in preferred if s in summary] + sorted(set(summary) - set(preferred))


def _module_summary(declarations: Sequence[LeanDeclaration]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for decl in declarations:
        row = summary.setdefault(decl.module, {"theorem": 0, "lemma": 0})
        row[decl.kind] += 1
    return dict(sorted(summary.items()))


def _decl_index(declarations: Sequence[LeanDeclaration]) -> Dict[str, LeanDeclaration]:
    return {d.qualified_name: d for d in declarations}


def _proof_rows_by_operator(model: AppendixModel) -> Sequence[Mapping[str, object]]:
    rows = model.proof_payload.get("operators", [])
    assert isinstance(rows, list)
    return rows


def _lean_backed_rows(model: AppendixModel) -> Sequence[Mapping[str, object]]:
    return [
        row for row in _proof_rows_by_operator(model)
        if row.get("proof_status") == "lean_theorem"
    ]


def render_latex(model: AppendixModel | None = None) -> str:
    """Render a deterministic LaTeX appendix."""

    model = model or build_model()
    declarations = list(model.declarations)
    decls_by_name = _decl_index(declarations)
    proof_summary = model.proof_payload["summary"]
    assert isinstance(proof_summary, dict)
    total_ops = model.proof_payload["total"]
    theorem_count = sum(1 for d in declarations if d.kind == "theorem")
    lemma_count = sum(1 for d in declarations if d.kind == "lemma")
    module_summary = _module_summary(declarations)

    lines: List[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[letterpaper,margin=0.72in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{array}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\hypersetup{hidelinks}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.42em}",
        r"\sloppy",
        "",
        r"\title{\textbf{TensorGuard Formal Soundness Appendix}}",
        r"\author{Generated from Lean, proof-footprint, and fragment sources}",
        r"\date{Deterministic repository artifact}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section{Generated Provenance}",
        (
            "This appendix is generated by "
            r"\texttt{python -m src.formal\_soundness\_appendix}. "
            "It contains no hand-maintained theorem ledger: Lean declarations "
            "are scanned from repository source files, operator evidence is "
            "loaded from the live proof-footprint generator, and the accepted "
            "program fragment is rendered from the canonical grammar in "
            r"\texttt{src.verifiable\_fragment}."
        ),
        "",
        r"\begin{longtable}{|p{0.33\linewidth}|p{0.56\linewidth}|}",
        r"\hline",
        r"\textbf{Generated input} & \textbf{Observed value} \\",
        r"\hline",
        "Lean source files & "
        f"{len(lean_source_files())} files under \\texttt{{lean/}}, excluding \\texttt{{.lake}}. \\\\",
        r"\hline",
        "Lean declarations & "
        f"{theorem_count} theorem statements and {lemma_count} lemma statements. \\\\",
        r"\hline",
        "Axiom-audited declarations & "
        f"{len(model.audited_theorem_names)} \\texttt{{\\#print axioms}} entries in "
        r"\texttt{lean/TensorGuard/AxiomAudit.lean}. \\",
        r"\hline",
        "Operator proof footprint & "
        f"{total_ops} registered transfer functions from "
        r"\texttt{src.proof\_footprint}. \\",
        r"\hline",
        "Verifiable fragment & "
        + ", ".join(
            f"{count} {_tex(name)}" for name, count in model.supported_counts.items()
        )
        + r". \\",
        r"\hline",
        r"\end{longtable}",
        "",
        r"\section{Verifiable Fragment Grammar}",
        (
            "The strict soundness mode may report a confident safe verdict only "
            "inside this generated fragment. Programs outside it must abstain "
            "with UNKNOWN unless a concrete bug has already been refuted."
        ),
        r"\begin{verbatim}",
        model.grammar,
        r"\end{verbatim}",
        "",
        r"\subsection{Supported Construct Tables}",
    ]

    for name, items in model.supported_lists.items():
        lines.extend([
            f"\\paragraph{{{_tex(name.title())} ({len(items)})}}",
            r"{\footnotesize " + _csv(items) + r"}",
            "",
        ])

    lines.extend([
        r"\section{Proof-Footprint Summary}",
        (
            "The following table is a direct projection of the live "
            r"\texttt{tensorguard.proof\_footprint/v1} payload. "
            "Only rows that name concrete Lean declarations claim the "
            r"\texttt{lean\_theorem} tier; other transfer functions are labeled "
            "as paper rules, tested-only rules, or heuristics."
        ),
        r"\begin{longtable}{|p{0.36\linewidth}|p{0.18\linewidth}|p{0.36\linewidth}|}",
        r"\hline",
        r"\textbf{Evidence tier} & \textbf{Count} & \textbf{Meaning} \\",
        r"\hline",
    ])
    meanings = {
        "lean_theorem": "A manifest row names imported Lean theorem evidence.",
        "pen_and_paper_rule": "A static mathematical rule is specified and tested, but not per-operator Lean-backed.",
        "tested_only_rule": "The transfer is validated by registry or differential tests without a compact proof rule.",
        "heuristic": "The output shape is data-dependent or intentionally approximate.",
    }
    for status in _status_order(proof_summary):
        lines.append(
            f"{_tex(status)} & {proof_summary[status]} & {_tex(meanings.get(status, 'Manifest-defined status.'))} \\\\"
        )
        lines.append(r"\hline")
    lines.extend([
        r"\end{longtable}",
        "",
        r"\subsection{Lean-Backed Operator Rows}",
        r"\begin{longtable}{|p{0.20\linewidth}|p{0.23\linewidth}|p{0.47\linewidth}|}",
        r"\hline",
        r"\textbf{Operator} & \textbf{Lean module(s)} & \textbf{Lean theorem(s)} \\",
        r"\hline",
    ])
    for row in _lean_backed_rows(model):
        lines.append(
            f"{_tex(row['operator'])} & {_csv(row['lean_modules'])} & "
            f"{_csv(row['lean_theorems'])} \\\\"
        )
        lines.append(r"\hline")
    lines.extend([
        r"\end{longtable}",
        "",
        r"\subsection{Full Operator Evidence Ledger}",
        r"\begin{longtable}{|p{0.20\linewidth}|p{0.18\linewidth}|p{0.16\linewidth}|p{0.36\linewidth}|}",
        r"\hline",
        r"\textbf{Operator} & \textbf{Proof status} & \textbf{Confidence} & \textbf{Rule / Lean theorem evidence} \\",
        r"\hline",
    ])
    for row in _proof_rows_by_operator(model):
        theorem_list = row.get("lean_theorems") or []
        evidence = theorem_list if theorem_list else [str(row.get("rule", ""))]
        lines.append(
            f"{_tex(row['operator'])} & {_tex(row['proof_status'])} & "
            f"{_tex(row['confidence'])} & {_csv(evidence)} \\\\"
        )
        lines.append(r"\hline")
    lines.extend([
        r"\end{longtable}",
        "",
        r"\section{Axiom-Audited Lean Statements}",
        (
            r"\texttt{AxiomAudit.lean} is the executable audit surface: each row "
            r"is sent through \texttt{\#print axioms}, and the regression suite "
            "rejects any dependency on unfinished proofs such as "
            r"\texttt{sorryAx}. The statements below are mined from the Lean "
            "files that define those names."
        ),
        r"\begin{longtable}{|p{0.30\linewidth}|p{0.14\linewidth}|p{0.46\linewidth}|}",
        r"\hline",
        r"\textbf{Declaration} & \textbf{Source} & \textbf{Statement} \\",
        r"\hline",
    ])
    for name in model.audited_theorem_names:
        decl = decls_by_name.get(name)
        if decl is None:
            source = "unresolved"
            statement = "The audit entry did not resolve to a scanned theorem."
        else:
            source = f"{decl.path}:{decl.line}"
            statement = decl.statement
        lines.append(f"{_tex(name)} & {_tex(source)} & {_tex(statement)} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\end{longtable}",
        "",
        r"\section{All Lean Theorem and Lemma Statements}",
        (
            "This census is intentionally broad: it lists every theorem and lemma "
            "statement found in committed Lean source.  The namespace-qualified "
            "names are reconstructed from Lean namespace blocks rather than from "
            "file paths, so declarations such as "
            r"\texttt{TensorGuard.V5.applyOp\_sound\_clamp} resolve even though "
            r"they live in \texttt{lean/TensorGuard/SoundnessV5.lean}."
        ),
        r"\subsection{Declaration Count by Module}",
        r"\begin{longtable}{|p{0.50\linewidth}|p{0.16\linewidth}|p{0.16\linewidth}|}",
        r"\hline",
        r"\textbf{Module} & \textbf{Theorems} & \textbf{Lemmas} \\",
        r"\hline",
    ])
    for module, row in module_summary.items():
        lines.append(f"{_tex(module)} & {row['theorem']} & {row['lemma']} \\\\")
        lines.append(r"\hline")
    lines.extend([
        r"\end{longtable}",
        "",
        r"\subsection{Declaration Ledger}",
        r"\begin{longtable}{|p{0.30\linewidth}|p{0.13\linewidth}|p{0.47\linewidth}|}",
        r"\hline",
        r"\textbf{Qualified declaration} & \textbf{Source} & \textbf{Statement} \\",
        r"\hline",
    ])
    for decl in declarations:
        lines.append(
            f"{_tex(decl.qualified_name)} & {_tex(f'{decl.path}:{decl.line}')} & "
            f"{_tex(decl.statement)} \\\\"
        )
        lines.append(r"\hline")
    lines.extend([
        r"\end{longtable}",
        "",
        r"\section{Trust Boundary}",
        (
            "The appendix deliberately separates what is machine checked from "
            "what is only tested or specified.  A Lean-backed operator row names "
            "the exact declarations in this ledger.  A pen-and-paper row names a "
            "static rule and evidence files but is not upgraded to Lean status. "
            "A tested-only row is useful empirical evidence without a compact "
            "proof rule.  A heuristic row is outside the strict proof boundary "
            "and therefore cannot justify a sound-mode safe verdict by itself."
        ),
        "",
        r"\end{document}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    import sys

    sys.stdout.write(render_latex())


if __name__ == "__main__":  # pragma: no cover
    main()
