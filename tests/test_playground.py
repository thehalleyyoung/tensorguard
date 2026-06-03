from __future__ import annotations

import json
import os
import re

from src.cli.main import ReftypeCliApp
from src.playground import (
    DEFAULT_EXAMPLES,
    PlaygroundExample,
    build_playground_manifest,
    render_playground_html,
    write_playground,
)


def test_manifest_runs_real_tensorguard_on_examples():
    manifest = build_playground_manifest(DEFAULT_EXAMPLES[:2])
    by_id = {ex["id"]: ex["result"] for ex in manifest["examples"]}
    assert by_id["clean-linear"]["verdict"] == "SAFE"
    assert by_id["shape-bug"]["verdict"] == "UNSAFE"
    assert by_id["shape-bug"]["bug_count"] >= 1
    assert any("Linear expects last dim=30, got 20" in bug["message"] for bug in by_id["shape-bug"]["bugs"])


def test_manifest_is_deterministic_and_omits_timing_paths():
    first = build_playground_manifest(DEFAULT_EXAMPLES[:2])
    second = build_playground_manifest(DEFAULT_EXAMPLES[:2])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    dumped = json.dumps(first, sort_keys=True)
    assert "duration_ms" not in dumped
    assert os.getcwd() not in dumped


def test_generated_html_has_no_network_or_active_code():
    html = render_playground_html(build_playground_manifest(DEFAULT_EXAMPLES[:2]))
    assert not re.search(r"https?://", html)
    assert not re.search(r"<\s*script\b", html, re.IGNORECASE)
    assert not re.search(r"<\s*form\b", html, re.IGNORECASE)
    assert not re.search(r"\baction\s*=", html, re.IGNORECASE)
    assert not re.search(r"fetch\s*\(", html)
    assert "XMLHttpRequest" not in html
    assert "No upload, no import, no execution" in html


def test_html_escapes_untrusted_source_and_messages():
    source = """\
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return x  # </textarea><script>alert(1)</script>
"""
    manifest = build_playground_manifest(
        [
            PlaygroundExample(
                id="escape",
                title="Escape",
                description="Escaping",
                source=source,
                input_shapes={"x": ("batch", 2)},
            )
        ]
    )
    html = render_playground_html(manifest)
    assert "</textarea><script>" not in html
    assert "&lt;/textarea&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_malicious_example_is_read_as_text_not_executed(tmp_path, monkeypatch):
    sentinel = tmp_path / "sentinel"
    monkeypatch.setenv("TENSORGUARD_PLAYGROUND_SENTINEL", str(sentinel))
    write_playground(tmp_path / "playground", DEFAULT_EXAMPLES)
    assert not sentinel.exists()


def test_write_playground_outputs_html_and_manifest(tmp_path):
    paths = write_playground(tmp_path / "playground", DEFAULT_EXAMPLES[:2])
    assert paths["html"].exists()
    assert paths["manifest"].exists()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["privacy"]["uploads_code"] is False
    assert manifest["privacy"]["executes_source"] is False


def test_cli_playground_writes_files(tmp_path, capsys):
    out = tmp_path / "site"
    code = ReftypeCliApp().run(["playground", "--output", str(out)])
    captured = capsys.readouterr()
    assert code == 0
    assert (out / "index.html").exists()
    assert (out / "manifest.json").exists()
    assert "no upload" in captured.out.lower()
