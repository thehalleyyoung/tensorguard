"""Step 57 -- world-class, source-mapped diagnostics.

A reported shape bug must render as a rustc-quality diagnostic: the offending
layer/op, the inferred-vs-expected shape, a source snippet with a caret, a
related location (where the layer is defined), and a suggested fix.  These are
produced by ``src.source_mapped_errors`` and surfaced on ``AnalysisResult`` by
``verify_architecture`` so the CLI can print them.

These tests prove (a) the caret aligns under the offending column, (b)
``verify_architecture`` attaches rich diagnostics for reported error bugs and
none for safe models, (c) duplicate violations at one location collapse to the
single richest diagnostic, and (d) the diagnostic carries the
inferred-vs-expected shapes and a fix.
"""
import textwrap

import pytest

from src.api import verify_architecture
from src.source_mapped_errors import (
    SourceMappedDiagnostic,
    format_plain,
    format_ansi,
    _caret_line,
)


BAD_LINEAR = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(30, 5)   # expects 30, gets 20
        def forward(self, x):
            x = self.fc1(x)
            return self.fc2(x)
"""

GOOD_LINEAR = BAD_LINEAR.replace("nn.Linear(30, 5)", "nn.Linear(20, 5)")


# ---------------------------------------------------------------------------
# 1. Caret rendering.
# ---------------------------------------------------------------------------
def test_caret_aligns_under_column():
    snippet = "        return self.fc2(x)"
    caret = _caret_line(snippet, 8)
    assert caret == " " * 8 + "^"
    # The caret sits exactly under the first non-space character.
    assert snippet[8] == "r"


def test_caret_preserves_tabs_for_alignment():
    snippet = "\t\treturn x"
    caret = _caret_line(snippet, 2)
    assert caret == "\t\t^"


def test_caret_minimum_width_one():
    assert _caret_line("x = y", 0) == "^"


# ---------------------------------------------------------------------------
# 2. Formatters include the snippet + caret for located diagnostics.
# ---------------------------------------------------------------------------
def _diag():
    return SourceMappedDiagnostic(
        message="Layer fc2 expects input dimension 30, but receives (batch, 20)",
        severity="error",
        source_line=10,
        source_col=8,
        source_snippet="        return self.fc2(x)",
        fix_suggestion="change fc2 to accept (batch, 20)",
    )


def test_format_plain_has_caret_and_fix():
    out = format_plain([_diag()])
    assert "-->" in out and "line 10" in out
    assert "return self.fc2(x)" in out
    assert "^" in out
    assert "fix:" in out


def test_format_ansi_has_caret_and_color_codes():
    out = format_ansi([_diag()])
    assert "\033[" in out          # color codes present
    assert "^" in out              # caret present
    assert "fix" in out


def test_no_caret_when_column_unknown():
    d = _diag()
    d.source_col = 0
    out = format_plain([d])
    # No caret line is emitted when the column is unknown (0).
    assert "^" not in out


# ---------------------------------------------------------------------------
# 3. End-to-end: verify_architecture attaches rich diagnostics.
# ---------------------------------------------------------------------------
def test_verify_architecture_attaches_rich_diagnostics():
    src = textwrap.dedent(BAD_LINEAR)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    assert result.bugs  # the bug is reported
    assert result.diagnostics  # and rich diagnostics are attached
    d = result.diagnostics[0]
    # Offending layer + inferred-vs-expected shapes in the message.
    assert "fc2" in d.message
    assert "30" in d.message            # expected
    assert "20" in d.message            # received
    # Source-mapped with a snippet and a fix.
    assert d.source_line > 0
    assert d.source_snippet
    assert d.fix_suggestion
    # A related location points at the layer definition.
    assert any("fc2" in r.message for r in d.related_locations)


def test_safe_model_has_no_diagnostics():
    src = textwrap.dedent(GOOD_LINEAR)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    assert not result.bugs
    assert result.diagnostics == []


def test_duplicate_violations_collapse_to_richest():
    # The BAD model produces several raw violations at the same location; the
    # attached diagnostics must collapse them to one rich entry per location.
    src = textwrap.dedent(BAD_LINEAR)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    locs = [(d.source_line, d.source_col) for d in result.diagnostics]
    assert len(locs) == len(set(locs))            # no duplicate locations
    # The kept diagnostic at the failing line carries a fix suggestion.
    assert all(
        d.fix_suggestion is not None
        for d in result.diagnostics
        if "expects input dimension" in d.message
    )


def test_no_infer_path_still_produces_diagnostics_when_bug_present():
    # Explicit shapes + a real bug: diagnostics must still be attached.
    src = textwrap.dedent(BAD_LINEAR)
    result = verify_architecture(
        src, input_shapes={"x": ("batch", 10)}, infer_inputs=False
    )
    assert result.bugs
    assert result.diagnostics
    assert result.diagnostics[0].source_snippet
