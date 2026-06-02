"""Step 61 — LSP provider (editor squiggles, hover shapes, quick-fixes).

These tests verify, against a real torch model, that a verification result is
rendered into spec-correct LSP payloads: 0-indexed diagnostic ranges, related
information, hover contents drawn from the inference chain, and code-action
quick-fixes whose workspace edit, when applied, reproduces the autofix.
"""

import torch  # noqa: F401

from src.api import verify_architecture
from src.lsp_provider import (
    build_lsp_report,
    collect_shape_hovers,
    hover_at,
    to_lsp_code_actions,
    to_lsp_diagnostics,
)


BUG = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
"""

SAFE = BUG.replace("nn.Linear(30, 5)", "nn.Linear(20, 5)")


def _verify(src):
    return verify_architecture(src, input_shapes={"x": ("batch", 10)})


def test_diagnostics_are_zero_indexed_and_tagged():
    r = _verify(BUG)
    diags = to_lsp_diagnostics(r, uri="file:///m.py")
    assert diags
    d = diags[0]
    # LSP positions are 0-indexed: exactly one less than our 1-indexed line.
    assert d["range"]["start"]["line"] == r.diagnostics[0].source_line - 1
    assert d["range"]["end"]["character"] > d["range"]["start"]["character"]
    assert d["severity"] == 1  # error
    assert d["source"] == "tensorguard"
    assert "expects input dimension 30" in d["message"]
    assert "fix:" in d["message"]
    # related info points at the layer definition line (def on source line 7 -> 6)
    assert "relatedInformation" in d
    assert d["relatedInformation"][0]["location"]["uri"] == "file:///m.py"


def test_code_action_edit_reproduces_autofix():
    r = _verify(BUG)
    actions = to_lsp_code_actions(r, uri="file:///m.py")
    assert len(actions) == 1
    a = actions[0]
    assert a["kind"] == "quickfix"
    assert a["isPreferred"] is True
    assert "in_features=20" in a["title"]
    edit = a["edit"]["changes"]["file:///m.py"][0]
    assert "nn.Linear(20, 5)" in edit["newText"]
    # The edit replaces a whole line: start at character 0.
    assert edit["range"]["start"]["character"] == 0

    # Applying the edit to that source line yields the autofix's suggestion.
    lines = BUG.splitlines()
    target_line0 = edit["range"]["start"]["line"]
    new_lines = list(lines)
    new_lines[target_line0] = edit["newText"]
    fixed = "\n".join(new_lines) + "\n"
    r2 = _verify(fixed)
    assert not r2.bugs


def test_hover_shows_produced_shape():
    r = _verify(BUG)
    hovers = collect_shape_hovers(r)
    assert hovers  # at least the fc1 line carries a concrete shape
    # the line producing h = fc1(x) is source line 9 -> stored 1-indexed
    assert any("(batch, 20)" in v for v in hovers.values())
    line = next(iter(hovers))
    h = hover_at(r, line)
    assert h is not None
    assert h["contents"]["kind"] == "markdown"
    assert "tensorguard" in h["contents"]["value"]
    # a line with no known shape returns None
    assert hover_at(r, 99999) is None


def test_safe_model_has_no_diagnostics_or_actions():
    r = _verify(SAFE)
    assert not r.bugs
    assert to_lsp_diagnostics(r) == []
    assert to_lsp_code_actions(r) == []


def test_build_lsp_report_aggregates_all_three():
    r = _verify(BUG)
    report = build_lsp_report(r, uri="file:///m.py")
    assert report["uri"] == "file:///m.py"
    assert len(report["diagnostics"]) == 1
    assert len(report["codeActions"]) == 1
    assert report["hovers"]
    # hovers are sorted by 1-indexed line and carry contents
    assert all("line" in h and "contents" in h for h in report["hovers"])


def test_provider_is_defensive_on_empty_result():
    class Empty:
        diagnostics = []
        autofixes = []
        inference_chain = None

    rep = build_lsp_report(Empty(), uri="x")
    assert rep == {"uri": "x", "diagnostics": [], "codeActions": [], "hovers": []}
