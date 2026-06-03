"""Step 276 -- the copyable model gallery is executable and verifier-backed."""

from __future__ import annotations

import json

import torch

from examples import model_gallery as mg
from src.api import verify_architecture


def _load_class(source: str, class_name: str):
    namespace = {}
    exec(source, namespace)
    return namespace[class_name]


def _concrete_shape(shape):
    return tuple(2 if dim == "batch" else dim for dim in shape)


def test_gallery_has_at_least_25_copyable_real_model_cases():
    cases = mg.gallery_cases()
    assert len(cases) >= 25
    assert len({case.slug for case in cases}) == len(cases)
    assert all(case.clean_source != case.buggy_source for case in cases)
    assert all(case.copy_config.startswith(f"tensorguard verify {case.slug}.py") for case in cases)


def test_committed_gallery_artifacts_are_fresh_and_complete():
    assert mg.run(check=True) == 0
    with open(mg.JSON_PATH, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["schema"] == "tensorguard.model_gallery.v1"
    assert payload["case_count"] == len(mg.gallery_cases())
    assert len(payload["families"]) >= 10
    assert len(payload["cases"]) == payload["case_count"]

    with open(mg.MD_PATH, "r", encoding="utf-8") as fh:
        markdown = fh.read()
    assert "Twenty-five pasteable PyTorch" in markdown
    assert markdown.count("| `") >= 25


def test_each_clean_gallery_model_executes_and_verifies_safe():
    for case in mg.gallery_cases():
        cls = _load_class(case.clean_source, case.clean_class)
        model = cls()
        kwargs = {
            name: torch.randn(*_concrete_shape(shape))
            for name, shape in case.input_shapes.items()
        }
        with torch.no_grad():
            output = model(**kwargs)
        assert output.shape[0] == 2, case.slug

        result = verify_architecture(
            case.clean_source,
            input_shapes=case.input_shapes,
            filename=case.filename,
            soundness_mode="balanced",
        )
        assert result.verdict == "SAFE", (case.slug, [bug.message for bug in result.bugs])


def test_each_buggy_gallery_model_is_caught_by_tensorguard():
    for case in mg.gallery_cases():
        result = verify_architecture(
            case.buggy_source,
            input_shapes=case.input_shapes,
            filename=case.filename,
            soundness_mode="balanced",
        )
        messages = "\n".join(bug.message for bug in result.bugs)
        assert result.verdict == "UNSAFE", case.slug
        assert "Linear" in messages or "shape" in messages
