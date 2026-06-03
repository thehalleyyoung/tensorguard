"""Step 264 -- tamper-evident generated-artifact hash index."""

from __future__ import annotations

import copy
from pathlib import Path

import reproducibility.artifact_index as ai

REPO = Path(__file__).resolve().parent.parent


def _by_path(data):
    return {entry["path"]: entry for entry in data["artifacts"]}


def test_index_covers_every_generated_artifact():
    import reproducibility.reproduce_all as ra

    expected = set(
        ra.GENERATED_DETERMINISTIC
        + ra._corpus_repro_paths()
        + ra._corpus_extended_paths()
        + ra.VOLATILE_REGENERATED
        + ai.EXTRA_GENERATED_ARTIFACTS
    )
    assert set(_by_path(ai.build_index())) == expected


def test_every_non_self_artifact_has_current_sha256_and_size():
    data = ai.build_index()
    assert data["all_hashed_artifacts_present"]
    for rel, entry in _by_path(data).items():
        if rel in ai.SELF_PATHS:
            assert entry["sha256"] is None
            assert entry["category"] == "self_index"
            continue
        path = REPO / rel
        assert entry["present"], rel
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == ai._sha256(path)


def test_root_digest_is_content_sensitive():
    data = ai.build_index()
    mutated = copy.deepcopy(data["artifacts"])
    target = next(entry for entry in mutated if entry["path"] not in ai.SELF_PATHS)
    target["sha256"] = "0" * 64
    assert ai._root_digest(data["artifacts"]) == data["artifact_root_sha256"]
    assert ai._root_digest(mutated) != data["artifact_root_sha256"]


def test_make_target_exists():
    makefile = (REPO / "Makefile").read_text()
    assert "\nartifact-index:" in makefile
    assert "reproducibility/artifact_index.py" in makefile
    assert "artifact-index" in makefile.split(".PHONY", 1)[1].split("\n", 1)[0]


def test_markdown_renders_hash_table():
    md = ai.render_markdown(ai.build_index())
    assert "artifact root sha256" in md
    assert "| path | category | bytes | sha256 |" in md
    assert "`reproducibility/artifact_index.json`" in md
    assert "`tool_paper.pdf`" in md


def test_check_mode_byte_identical():
    assert ai.run(check=True) == 0


def test_reproduce_all_owns_index_artifacts():
    import reproducibility.reproduce_all as ra

    assert "reproducibility/artifact_index.json" in ra.GENERATED_DETERMINISTIC
    assert "reproducibility/artifact_index.md" in ra.GENERATED_DETERMINISTIC
