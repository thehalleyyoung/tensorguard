"""Pytest plugin that auto-verifies nn.Module fixtures with TensorGuard.

Register via entry point ``pytest11`` or load manually::

    # conftest.py
    pytest_plugins = ["src.integrations.pytest_plugin"]

Then mark tests::

    @pytest.mark.tensorguard_verify(input_shapes={"x": ("batch", 3, 224, 224)})
    def test_my_model(my_model_source):
        pass  # TensorGuard verification runs automatically
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, Optional

import pytest


# Marker name
MARKER_NAME = "tensorguard_verify"


class TensorGuardPlugin:
    """Pytest plugin that runs TensorGuard verification on marked tests."""

    @staticmethod
    def pytest_configure(config: Any) -> None:
        """Register the tensorguard_verify marker."""
        config.addinivalue_line(
            "markers",
            f"{MARKER_NAME}(input_shapes, high_confidence_only): "
            "Run TensorGuard verification on the test's source fixture.",
        )

    @staticmethod
    def pytest_collection_modifyitems(config: Any, items: list) -> None:
        """Tag marked items so we can identify them during execution."""
        for item in items:
            marker = item.get_closest_marker(MARKER_NAME)
            if marker is not None:
                item._tensorguard_marker = marker  # type: ignore[attr-defined]

    @staticmethod
    def pytest_runtest_call(item: Any) -> None:
        """If the test is marked, run TensorGuard verification."""
        marker = getattr(item, "_tensorguard_marker", None)
        if marker is None:
            return

        # Extract marker arguments
        input_shapes: Dict[str, tuple] = marker.kwargs.get("input_shapes", {})
        high_confidence_only: bool = marker.kwargs.get("high_confidence_only", False)

        # Look for a source fixture value on the item
        source = _extract_source(item)
        if source is None:
            pytest.skip("No source fixture found for TensorGuard verification")
            return

        from src.api import verify_architecture

        result = verify_architecture(
            source,
            input_shapes=input_shapes,
            high_confidence_only=high_confidence_only,
        )

        if result.bug_count > 0:
            bug_msgs = "\n".join(
                f"  [{b.severity}] {b.category.value} at line {b.location.line}: {b.message}"
                for b in result.bugs
            )
            pytest.fail(
                f"TensorGuard found {result.bug_count} bug(s):\n{bug_msgs}",
                pytrace=False,
            )


def _extract_source(item: Any) -> Optional[str]:
    """Try to get a source-code string from the test item's fixtures."""
    # Check common fixture names
    for fixture_name in ("model_source", "source", "nn_source", "module_source"):
        if fixture_name in getattr(item, "fixturenames", []):
            try:
                val = item.funcargs.get(fixture_name)
                if isinstance(val, str):
                    return val
            except Exception:
                pass

    # Check if the test function itself has a _tensorguard_source attribute
    func = getattr(item, "obj", None)
    if func is not None:
        src = getattr(func, "_tensorguard_source", None)
        if isinstance(src, str):
            return src

    return None


def pytest_configure(config: Any) -> None:
    """Entry point — register the plugin."""
    plugin = TensorGuardPlugin()
    config.pluginmanager.register(plugin, "tensorguard")
