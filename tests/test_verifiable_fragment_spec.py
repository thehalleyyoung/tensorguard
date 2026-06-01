"""Regression tests for the formal verifiable-fragment spec (Step 8)."""

import os
import subprocess
import sys

import pytest

from src.api import verify_architecture
from src.verifiable_fragment import (
    SUPPORTED_F_FUNCTIONS,
    SUPPORTED_LAYER_TYPES,
    SUPPORTED_TENSOR_METHODS,
    SUPPORTED_TORCH_FUNCTIONS,
    UNSUPPORTED_CATEGORY_INFO,
    UnsupportedCategory,
    analyze_source,
    render_spec_markdown,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


_CLEAN = _mod("        x = torch.relu(x)")
_DDCF = _mod("        if x.sum() > 0:\n            x = x + 1")
_DDIT = _mod("        while x.sum() > 0:\n            x = x - 1")
_ASSERT = _mod("        assert x.sum() > 0")
_ITEM = _mod("        n = x.sum().item()")


# ── analyze_source: the instance-free fallback ─────────────────────────────

def test_analyze_source_clean_is_empty():
    assert analyze_source(_CLEAN) == []


@pytest.mark.parametrize("src,category", [
    (_DDCF, UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW),
    (_DDIT, UnsupportedCategory.DATA_DEPENDENT_ITERATION),
    (_ASSERT, UnsupportedCategory.DYNAMIC_ASSERTION),
    (_ITEM, UnsupportedCategory.TENSOR_TO_SCALAR),
])
def test_analyze_source_flags_each_static_category(src, category):
    blocking = analyze_source(src)
    assert blocking, f"expected a blocking construct for {category.name}"
    assert category in {c.category for c in blocking}


def test_analyze_source_handles_syntax_error_gracefully():
    assert analyze_source("def (((") == []


# ── Fallback proof: unsupported -> UNKNOWN, never silent SAFE ───────────────

@pytest.mark.parametrize("src", [_DDCF, _DDIT, _ASSERT, _ITEM])
def test_out_of_fragment_yields_unknown_in_sound_mode(src):
    r = verify_architecture(
        src, input_shapes={"x": (4, 8)}, max_cegar_iterations=0,
        soundness_mode="sound",
    )
    assert r.verdict == "UNKNOWN"
    assert r.unknown_reasons


def test_clean_module_still_safe_in_sound_mode():
    r = verify_architecture(
        _CLEAN, input_shapes={"x": (4, 8)}, max_cegar_iterations=0,
        soundness_mode="sound",
    )
    assert r.verdict == "SAFE"


# ── Spec document ──────────────────────────────────────────────────────────

def test_every_unsupported_category_is_documented():
    for cat in UnsupportedCategory:
        assert cat in UNSUPPORTED_CATEGORY_INFO, cat.name
        info = UNSUPPORTED_CATEGORY_INFO[cat]
        assert info["description"].strip()
        assert info["detected_by"] in ("static", "fx", "static+fx")


def test_spec_markdown_contains_grammar_and_tables():
    md = render_spec_markdown()
    assert "## Grammar" in md
    assert "Module      ::=" in md
    # supported tables with correct counts
    assert f"Layer types ({len(SUPPORTED_LAYER_TYPES)})" in md
    assert f"Tensor methods ({len(SUPPORTED_TENSOR_METHODS)})" in md
    assert f"torch.* functions ({len(SUPPORTED_TORCH_FUNCTIONS)})" in md
    assert f"(F.*) functions ({len(SUPPORTED_F_FUNCTIONS)})" in md
    # every category appears in the excluded table
    for cat in UnsupportedCategory:
        assert f"`{cat.name}`" in md
    # the fallback policy section
    assert "never a silent pass" in md


def test_committed_spec_in_sync_with_code():
    path = os.path.join(_REPO, "VERIFIABLE_FRAGMENT.md")
    assert os.path.exists(path), "VERIFIABLE_FRAGMENT.md missing"
    with open(path, encoding="utf-8") as f:
        committed = f.read()
    fresh = render_spec_markdown()
    assert committed == fresh, (
        "VERIFIABLE_FRAGMENT.md is stale; regenerate with "
        "`python -m src.verifiable_fragment > VERIFIABLE_FRAGMENT.md`"
    )


def test_spec_module_runs_as_script():
    out = subprocess.run(
        [sys.executable, "-m", "src.verifiable_fragment"],
        cwd=_REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("# TensorGuard Verifiable Fragment")
