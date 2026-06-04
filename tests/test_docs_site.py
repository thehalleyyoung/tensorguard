"""Step 285 — generated public documentation site."""

from __future__ import annotations

import json
import re
from pathlib import Path

from reproducibility import docs_site
from src.operator_confidence import confidence_table
from src.proof_footprint import ProofStatus, proof_footprint_table, summary_for

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "docs" / "site"


def _manifest():
    return json.loads((SITE / "site_manifest.json").read_text(encoding="utf-8"))


def test_site_has_required_textbook_sections():
    manifest = _manifest()
    paths = {page["path"] for page in manifest["pages"]}
    assert {
        "index.html",
        "concepts/soundness.html",
        "concepts/verifiable-fragment.html",
        "concepts/evidence.html",
        "operators/index.html",
        "proof-footprint/index.html",
        "migration/runtime-assertions.html",
        "migration/pytea.html",
    } <= paths
    categories = {page["category"] for page in manifest["pages"]}
    assert {"concept chapter", "operator reference", "proof reference", "migration guide"} <= categories


def test_manifest_summaries_are_generated_from_real_code():
    conf = {"complete": 0, "sound": 0, "heuristic": 0}
    for row in confidence_table():
        conf[row["confidence"]] += 1
    assert _manifest()["operator_confidence_summary"] == conf

    proof_summary = summary_for(proof_footprint_table())
    assert _manifest()["proof_footprint_summary"] == proof_summary
    assert proof_summary[ProofStatus.LEAN_THEOREM.value] > 0


def test_pages_have_no_broken_internal_links():
    known = {page["path"] for page in _manifest()["pages"]}
    for page in _manifest()["pages"]:
        src = page["path"]
        text = (SITE / src).read_text(encoding="utf-8")
        for href in re.findall(r'href="([^"]+)"', text):
            if href.startswith(("http:", "https:", "#", "mailto:")):
                continue
            target = ((SITE / src).parent / href).resolve().relative_to(SITE.resolve()).as_posix()
            assert target in known, f"{src} links to missing {href} -> {target}"


def test_operator_and_proof_pages_surface_real_counts():
    manifest = _manifest()
    op_text = (SITE / "operators/index.html").read_text(encoding="utf-8")
    proof_text = (SITE / "proof-footprint/index.html").read_text(encoding="utf-8")
    total_ops = sum(manifest["operator_confidence_summary"].values())
    assert f"{total_ops} operators" in (SITE / "index.html").read_text(encoding="utf-8")
    assert "torch.matmul" in proof_text
    assert "heuristic" in op_text
    assert "complete" in op_text


def test_homepage_surfaces_symbolic_shape_examples():
    text = (SITE / "index.html").read_text(encoding="utf-8")
    assert "symbolic module verification" in text
    assert "100 / 100 cases" in text
    assert "verify_architecture(source)" in text
    assert "inferred_input_shapes" in text
    assert "Float[Tensor, &quot;batch 10&quot;]" in text
    assert "verify_module(model) infers (batch, 3, height, width)" in text
    assert "input_shapes={&quot;x&quot;: (&quot;batch&quot;, 10)}" not in text


def test_make_and_pages_workflow_are_wired():
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    workflow = (REPO / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "\ndocs-site:" in makefile
    assert "reproducibility/docs_site.py" in makefile
    assert "docs-site" in makefile.split(".PHONY", 1)[1].split("\n", 1)[0]
    assert "python reproducibility/docs_site.py --check" in workflow
    assert "path: docs/site" in workflow


def test_reproduce_all_and_artifact_index_own_site():
    import reproducibility.reproduce_all as ra

    assert "docs/site/index.html" in ra.GENERATED_DETERMINISTIC
    assert "docs/site/site_manifest.json" in ra.GENERATED_DETERMINISTIC


def test_generator_check_mode_is_byte_identical():
    assert docs_site.check() == 0
