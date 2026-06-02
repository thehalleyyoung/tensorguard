"""Tests for Step 115: the static evidence dashboard built from artifacts."""

from __future__ import annotations

import importlib
import json
import re

import pytest

dash = importlib.import_module("reproducibility.build_dashboard")


@pytest.fixture(scope="module")
def data():
    return dash.measure()


_VOLATILE = (
    "time",
    "elapsed",
    "timestamp",
    "wall",
    "clock",
    "_ms",
    "seconds",
    "duration",
    "date",
)


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_no_volatile_keys(data):
    for key in _walk_keys(data):
        low = str(key).lower()
        assert not any(tok in low for tok in _VOLATILE), key


def test_every_card_has_headline_and_metrics(data):
    assert data["n_cards"] == len(data["cards"]) >= 6
    seen = set()
    for c in data["cards"]:
        assert c["id"] not in seen
        seen.add(c["id"])
        assert c["headline"].strip()
        assert c["category"] in data["categories"]
        assert c["source_artifact"].endswith(".json")
        assert len(c["metrics"]) >= 2
        for m in c["metrics"]:
            assert m["label"] and str(m["value"]) != ""


def test_headlines_reflect_zero_soundness_gaps(data):
    by_id = {c["id"]: c for c in data["cards"]}
    assert "0 soundness violations" in by_id["differential"]["headline"]
    assert "0 false alarms" in by_id["differential"]["headline"]
    assert "0 soundness violations" in by_id["hypothesis"]["headline"]
    assert "100.0 percent" in by_id["mutation"]["headline"]


def test_source_artifacts_exist(data):
    for c in data["cards"]:
        assert (dash.REPRO / c["source_artifact"]).exists(), c["source_artifact"]


def test_html_is_self_contained_and_embeds_parsable_json(data):
    htmltext = dash.render_html(data)
    # No external resource loads (offline / reproducible).
    assert "http://" not in htmltext
    assert "https://" not in htmltext
    assert "src=" not in htmltext  # no external scripts
    # Embedded data round-trips.
    m = re.search(r'application/json">(.*?)</script>', htmltext, re.S)
    assert m
    raw = m.group(1).replace("<\\/", "</")
    parsed = json.loads(raw)
    assert parsed["n_cards"] == data["n_cards"]


def test_byte_determinism():
    assert dash.run(check=True) == 0
