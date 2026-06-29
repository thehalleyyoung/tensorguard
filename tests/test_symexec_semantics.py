"""Step 91 — pin the operational-semantics spec for the modeled fragment.

These tests pin three things:
  1. ``docs/symexec/semantics.md`` is in sync with ``src/symexec/semantics.py``;
  2. the spec is internally well-formed (every form cites an ``ast`` node and a
     code location; the store is fully described);
  3. the spec is *faithful* — every ``ast`` node the spec claims to model is in
     fact dispatched on in ``src/symexec/interpreter.py`` (the spec cannot drift
     to describe unmodeled syntax), and the documented store components match the
     real ``State`` dataclass fields.
"""

from __future__ import annotations

import ast
import dataclasses as dc
from pathlib import Path

from src.symexec import semantics as sem
from src.symexec.state import State

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "symexec" / "semantics.md"
INTERP = REPO / "src" / "symexec" / "interpreter.py"


def test_doc_in_sync_with_module():
    assert DOC.exists(), "docs/symexec/semantics.md must be committed"
    expected = sem.render_markdown()
    actual = DOC.read_text(encoding="utf-8")
    assert actual.strip() == expected.strip(), (
        "docs/symexec/semantics.md is stale; regenerate with "
        "`python -m src.symexec.semantics > docs/symexec/semantics.md`"
    )


def test_every_form_is_well_formed():
    forms = sem.STATEMENT_FORMS + sem.EXPRESSION_FORMS
    assert forms, "the spec must enumerate syntactic forms"
    for f in forms:
        assert f.form and f.ast_node and f.rule and f.code, f"incomplete form: {f}"
        # the named ast node must be a real ast class.
        assert hasattr(ast, f.ast_node), f"unknown ast node: ast.{f.ast_node}"


def test_store_describes_real_state_fields():
    described = " ".join(c.name for c in sem.STORE)
    for field in dc.fields(State):
        assert field.name in described, (
            f"State.{field.name} is not described in the semantics store section"
        )


def test_forms_are_faithful_to_the_interpreter():
    """Every modeled ast node must actually be dispatched on in interpreter.py."""
    src = INTERP.read_text(encoding="utf-8")
    nodes = {f.ast_node for f in sem.STATEMENT_FORMS + sem.EXPRESSION_FORMS}
    missing = [n for n in nodes if f"ast.{n}" not in src]
    assert not missing, (
        f"semantics claims to model ast nodes the interpreter never dispatches "
        f"on: {missing}"
    )


def test_core_statement_forms_present():
    stmts = {f.ast_node for f in sem.STATEMENT_FORMS}
    for required in ("Assign", "If", "For", "While", "Return", "Expr"):
        assert required in stmts, f"core statement form missing: {required}"


def test_core_expression_forms_present():
    exprs = {f.ast_node for f in sem.EXPRESSION_FORMS}
    for required in ("Name", "Constant", "BinOp", "Call", "Subscript", "Attribute"):
        assert required in exprs, f"core expression form missing: {required}"


def test_abstraction_notes_mention_join_widen_and_abstain():
    blob = " ".join(sem.ABSTRACTION_NOTES).lower()
    assert "widen" in blob
    assert "join" in blob
    assert "abstain" in blob


def test_render_is_deterministic():
    assert sem.render_markdown() == sem.render_markdown()
