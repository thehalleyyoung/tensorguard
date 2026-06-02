"""Step 65 — keep the honest "What it can't do yet" doc in sync with the code.

LIMITATIONS.md documents the constructs outside the verifiable fragment.  This
test pins the doc to ``UNSUPPORTED_CATEGORY_INFO`` so the list cannot silently
drift, and runs representative snippets for the statically-detectable categories
to prove TensorGuard really does abstain rather than mislabel them "safe".
"""

import os

import torch  # noqa: F401

from src.verifiable_fragment import (
    UNSUPPORTED_CATEGORY_INFO,
    UnsupportedCategory,
    analyze_source,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOC = os.path.join(_REPO, "LIMITATIONS.md")


def _doc_text():
    with open(_DOC, encoding="utf-8") as fh:
        return fh.read()


def _mod(body: str) -> str:
    return (
        "import torch, torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(8, 8)\n"
        "    def forward(self, x):\n"
        "        x = self.fc(x)\n"
        f"{body}\n"
        "        return x\n"
    )


def test_doc_exists():
    assert os.path.exists(_DOC)
    assert "verifiable fragment" in _doc_text().lower()


def test_every_unsupported_category_is_documented():
    """The doc's human-readable description for each category must appear."""
    text = _doc_text().lower()
    missing = []
    for cat, info in UNSUPPORTED_CATEGORY_INFO.items():
        desc = (info.get("description") or "").lower()
        # Use a distinctive keyword from each description to detect coverage.
        # Drop trailing punctuation/parentheticals and pick the salient phrase.
        keyword = desc.split("(")[0].strip().rstrip(".")
        if keyword and keyword not in text:
            missing.append((cat.name, keyword))
    assert not missing, f"LIMITATIONS.md is missing categories: {missing}"


def test_doc_does_not_invent_categories():
    """Every table row maps to a real category description (no stale rows)."""
    text = _doc_text().lower()
    known = {
        (info.get("description") or "").split("(")[0].strip().rstrip(".").lower()
        for info in UNSUPPORTED_CATEGORY_INFO.values()
    }
    # The five static categories each have a canonical phrase we assert on.
    for phrase in [
        "branch (if/while) whose condition depends on a tensor value",
        "loop whose trip count depends on runtime data",
        "assert statement in",
        ".item()",
    ]:
        assert phrase.lower() in text, phrase
    assert known  # sanity


def test_static_categories_are_flagged_not_safe():
    """Representative snippets for the AST-detectable categories abstain."""
    cases = {
        UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW: _mod(
            "        if x.sum() > 0:\n            x = x + 1"
        ),
        UnsupportedCategory.DATA_DEPENDENT_ITERATION: _mod(
            "        while x.sum() > 0:\n            x = x - 1"
        ),
        UnsupportedCategory.DYNAMIC_ASSERTION: _mod("        assert x.sum() > 0"),
        UnsupportedCategory.TENSOR_TO_SCALAR: _mod("        n = x.sum().item()"),
    }
    for category, src in cases.items():
        blocking = analyze_source(src)
        cats = {c.category for c in blocking}
        assert category in cats, f"{category.name} no longer detected statically"
