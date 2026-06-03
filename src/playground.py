"""Local static playground for TensorGuard examples.

The playground is intentionally a generated, self-contained HTML file: it runs
TensorGuard locally while generating the page, embeds only deterministic result
summaries, and ships no network code. Users can edit the examples in the page
as starting points, then copy them into ``tensorguard verify`` for a fresh local
analysis.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from src.safe_loader import verify_source_safely


InputShapes = Mapping[str, Tuple[Any, ...]]


@dataclass(frozen=True)
class PlaygroundExample:
    id: str
    title: str
    description: str
    source: str
    input_shapes: InputShapes
    expected_verdict: Optional[str] = None


GOOD_LINEAR = """\
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
"""


BAD_LINEAR = GOOD_LINEAR.replace("nn.Linear(20, 5)", "nn.Linear(30, 5)")


MALICIOUS_SENTINEL = """\
import os
from pathlib import Path

Path(os.environ["TENSORGUARD_PLAYGROUND_SENTINEL"]).write_text("executed")

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)
"""


DEFAULT_EXAMPLES: Tuple[PlaygroundExample, ...] = (
    PlaygroundExample(
        id="clean-linear",
        title="Clean two-layer MLP",
        description="A statically safe Linear -> Linear chain.",
        source=GOOD_LINEAR,
        input_shapes={"x": ("batch", 10)},
        expected_verdict="SAFE",
    ),
    PlaygroundExample(
        id="shape-bug",
        title="Caught shape bug",
        description="fc2 expects 30 features but receives the 20 features produced by fc1.",
        source=BAD_LINEAR,
        input_shapes={"x": ("batch", 10)},
        expected_verdict="UNSAFE",
    ),
    PlaygroundExample(
        id="malicious-top-level",
        title="Untrusted top-level code stays inert",
        description=(
            "This source would write a sentinel if imported; the playground reads "
            "and verifies it as text, so generation never executes it."
        ),
        source=MALICIOUS_SENTINEL,
        input_shapes={"x": ("batch", 4)},
    ),
)


def _stable_message(message: str) -> str:
    """Remove solver-generated numeric suffixes while preserving diagnostics."""

    headline = message.splitlines()[0]
    return re.sub(r"\b([A-Za-z][A-Za-z0-9]*)_\d+\b", r"\1_N", headline)


def _stable_result(example: PlaygroundExample) -> Dict[str, Any]:
    result = verify_source_safely(
        example.source,
        input_shapes=dict(example.input_shapes),
        soundness_mode="sound",
        filename=f"playground/{example.id}.py",
    )
    bugs = sorted(
        [
            {
                "category": bug.category.value if hasattr(bug.category, "value") else str(bug.category),
                "message": _stable_message(bug.message),
                "severity": bug.severity,
                "line": bug.location.line,
                "column": bug.location.column,
                "fix": bug.fix_suggestion or "",
            }
            for bug in result.bugs
        ],
        key=lambda bug: (
            str(bug["category"]),
            int(bug["line"]),
            int(bug["column"]),
            str(bug["message"]),
        ),
    )
    return {
        "id": example.id,
        "title": example.title,
        "description": example.description,
        "input_shapes": {
            name: list(shape) for name, shape in sorted(example.input_shapes.items())
        },
        "verdict": result.verdict,
        "bug_count": len(bugs),
        "bugs": bugs,
        "unknown_reasons": sorted(result.unknown_reasons),
        "expected_verdict": example.expected_verdict,
    }


def build_playground_manifest(
    examples: Iterable[PlaygroundExample] = DEFAULT_EXAMPLES,
) -> Dict[str, Any]:
    """Run TensorGuard locally and return a deterministic playground manifest."""

    rendered = []
    for example in examples:
        result = _stable_result(example)
        if example.expected_verdict and result["verdict"] != example.expected_verdict:
            raise AssertionError(
                f"{example.id} expected {example.expected_verdict}, got {result['verdict']}"
            )
        rendered.append(
            {
                "id": example.id,
                "title": example.title,
                "description": example.description,
                "source": example.source,
                "input_shapes": result["input_shapes"],
                "result": result,
            }
        )
    return {
        "schema": "tensorguard-local-playground-v1",
        "privacy": {
            "mode": "local-static",
            "uploads_code": False,
            "executes_source": False,
            "network_required": False,
            "note": "Generation reads examples as text and uses TensorGuard's AST-only safe loader.",
        },
        "examples": rendered,
    }


def _render_bug_list(result: Mapping[str, Any]) -> str:
    bugs = result["bugs"]
    if not bugs:
        return "<p class=\"ok\">No bugs reported.</p>"
    rows = []
    for bug in bugs:
        rows.append(
            "<li>"
            f"<strong>{html.escape(str(bug['category']))}</strong> "
            f"line {int(bug['line'])}: {html.escape(str(bug['message']))}"
            "</li>"
        )
    return "<ul>" + "".join(rows) + "</ul>"


def render_playground_html(manifest: Mapping[str, Any]) -> str:
    """Render a self-contained HTML playground from a deterministic manifest."""

    cards = []
    for example in manifest["examples"]:
        result = example["result"]
        verdict = str(result["verdict"])
        shapes = json.dumps(example["input_shapes"], sort_keys=True)
        cards.append(
            f"""
            <section class="card {html.escape(verdict.lower())}">
              <h2>{html.escape(example["title"])}</h2>
              <p>{html.escape(example["description"])}</p>
              <p><strong>Input shapes:</strong> <code>{html.escape(shapes)}</code></p>
              <textarea spellcheck="false" aria-label="{html.escape(example["title"])} source">{html.escape(example["source"])}</textarea>
              <div class="result">
                <strong>Precomputed local verdict:</strong>
                <span class="badge">{html.escape(verdict)}</span>
                {_render_bug_list(result)}
              </div>
            </section>
            """
        )

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<title>TensorGuard Local Playground</title>",
            "<style>",
            "body{font-family:ui-sans-serif,system-ui,sans-serif;margin:2rem;line-height:1.45;color:#18202f;background:#f8fafc}",
            "header,.card{max-width:1100px;margin:0 auto 1rem;background:white;border:1px solid #d9e2ef;border-radius:14px;padding:1.25rem;box-shadow:0 1px 3px #0001}",
            "textarea{width:100%;min-height:16rem;font:13px ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid #c9d4e5;border-radius:10px;padding:1rem;background:#0f172a;color:#e2e8f0}",
            ".badge{display:inline-block;border-radius:999px;padding:.2rem .7rem;font-weight:700;background:#e2e8f0}",
            ".safe .badge{background:#dcfce7;color:#166534}.unsafe .badge{background:#fee2e2;color:#991b1b}.unknown .badge{background:#fef3c7;color:#92400e}",
            ".privacy{background:#eef6ff;border-left:5px solid #2563eb}.ok{color:#166534;font-weight:600}",
            "code{background:#eef2f7;padding:.15rem .3rem;border-radius:5px}",
            "</style>",
            "</head>",
            "<body>",
            "<header>",
            "<h1>TensorGuard Local Static Playground</h1>",
            "<p class=\"privacy\"><strong>No upload, no import, no execution.</strong> "
            "This page is generated entirely on your machine. It embeds editable examples "
            "and deterministic TensorGuard verdict snapshots; to re-run after editing, copy "
            "the source into a local file and run <code>tensorguard verify path.py</code>.</p>",
            "</header>",
            *cards,
            "</body>",
            "</html>",
        ]
    )


def write_playground(
    output_dir: str | Path,
    examples: Iterable[PlaygroundExample] = DEFAULT_EXAMPLES,
) -> Dict[str, Path]:
    """Write ``index.html`` and ``manifest.json`` for the local playground."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_playground_manifest(examples)
    manifest_path = out / "manifest.json"
    html_path = out / "index.html"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(render_playground_html(manifest), encoding="utf-8")
    return {"html": html_path, "manifest": manifest_path}
