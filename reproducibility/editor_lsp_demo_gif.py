#!/usr/bin/env python3
"""Step 291 -- deterministic editor/LSP demo GIF from real diagnostics.

The GIF is generated, not screen-recorded.  It runs TensorGuard on a small
buggy ``nn.Module``, converts the result through the real LSP provider, and
renders the resulting squiggle, hover, and quick-fix into a stable GIF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from reproducibility import terminal_demo_gif as gif
from src.api import verify_architecture
from src.lsp_provider import build_lsp_report

OUT_GIF = REPO / "docs" / "launch" / "editor_lsp_demo.gif"
OUT_JSON = REPO / "reproducibility" / "editor_lsp_demo_gif.json"
OUT_MD = REPO / "reproducibility" / "editor_lsp_demo_gif.md"
OUTPUTS = (OUT_GIF, OUT_JSON, OUT_MD)

DEMO_URI = "file:///demo.py"
DEMO_SOURCE = """
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


def _verify_demo() -> Tuple[object, Dict[str, object]]:
    result = verify_architecture(DEMO_SOURCE, input_shapes={"x": ("batch", 10)})
    report = build_lsp_report(result, uri=DEMO_URI)
    diagnostics = report["diagnostics"]  # type: ignore[index]
    actions = report["codeActions"]  # type: ignore[index]
    hovers = report["hovers"]  # type: ignore[index]
    if getattr(result, "verdict", "") != "UNSAFE":
        raise ValueError("editor demo source must be UNSAFE")
    if not diagnostics or not actions or not hovers:
        raise ValueError("editor demo requires a diagnostic, quick-fix, and hover")
    message = str(diagnostics[0]["message"])
    if "expects input dimension 30" not in message or "receives (batch, 20)" not in message:
        raise ValueError("editor demo diagnostic is not the expected shape mismatch")
    if "in_features=20" not in str(actions[0]["title"]):
        raise ValueError("editor demo quick-fix is not tied to the expected repair")
    if not any("(batch, 20)" in str(hover["contents"]) for hover in hovers):
        raise ValueError("editor demo hover is missing the produced shape")
    return result, report


def build_manifest() -> Dict[str, object]:
    result, report = _verify_demo()
    diagnostics = report["diagnostics"]  # type: ignore[index]
    actions = report["codeActions"]  # type: ignore[index]
    hovers = report["hovers"]  # type: ignore[index]
    diag = diagnostics[0]
    action = actions[0]
    hover = next(h for h in hovers if "(batch, 20)" in str(h["contents"]))
    diag_line = int(diag["range"]["start"]["line"]) + 1
    transcript = [
        "OPEN DEMO.PY IN VS CODE",
        f"L{diag_line}: INLINE SHAPE SQUIGGLE",
        "FC2 EXPECTS 30 BUT GETS BATCH,20",
        f"L{hover['line']}: HOVER H : (BATCH,20)",
        "QUICK FIX: SET IN_FEATURES=20",
        "SOURCE: REAL LSP PROVIDER REPORT",
    ]
    return {
        "schema": "tensorguard.editor_lsp_demo_gif/v1",
        "step": 291,
        "source": "inline demo nn.Module verified by src.api.verify_architecture",
        "source_sha256": gif._sha256_bytes(DEMO_SOURCE.encode("utf-8")),
        "uri": DEMO_URI,
        "verdict": getattr(result, "verdict", ""),
        "bug_count": len(getattr(result, "bugs", []) or []),
        "lsp": {
            "diagnostic_count": len(diagnostics),
            "code_action_count": len(actions),
            "hover_count": len(hovers),
            "diagnostic_line_1indexed": diag_line,
            "diagnostic_message": str(diag["message"]).splitlines()[0],
            "quickfix_title": action["title"],
            "hover_line_1indexed": hover["line"],
            "hover_contents": hover["contents"],
        },
        "gif": {
            "path": "docs/launch/editor_lsp_demo.gif",
            "width": gif.WIDTH,
            "height": gif.HEIGHT,
            "frame_count": len(transcript),
            "duration_seconds": round(len(transcript) * 0.85, 2),
        },
        "transcript": transcript,
    }


def _base_frame() -> bytearray:
    frame = bytearray([0]) * (gif.WIDTH * gif.HEIGHT)
    gif._fill(frame, 10, 10, gif.WIDTH - 10, gif.HEIGHT - 10, 1)
    gif._fill(frame, 10, 42, 132, gif.HEIGHT - 10, 0)
    gif._fill(frame, 132, 42, gif.WIDTH - 10, gif.HEIGHT - 10, 0)
    gif._fill(frame, 10, 10, gif.WIDTH - 10, 42, 1)
    gif._draw_text(frame, 28, 24, "DEMO.PY - TENSORGUARD LSP", 2)
    for idx, line in enumerate(("6 SELF.FC2 = LINEAR(30,5)", "8 DEF FORWARD(SELF,X):", "9 H = SELF.FC1(X)", "10 RETURN SELF.FC2(H)")):
        color = 2 if idx >= 2 else 3
        gif._draw_text(frame, 152, 72 + idx * 34, line, color)
    gif._draw_text(frame, 28, 72, "EXPLORER", 3)
    gif._draw_text(frame, 28, 104, "DEMO.PY", 7)
    return frame


def _draw_diagnostic(frame: bytearray, message: str) -> None:
    gif._fill(frame, 152, 184, 430, 190, 6)
    gif._fill(frame, 340, 214, gif.WIDTH - 28, 292, 1)
    gif._fill(frame, 348, 222, gif.WIDTH - 36, 284, 0)
    gif._draw_text(frame, 360, 234, "TENSORGUARD ERROR", 6)
    gif._draw_text(frame, 360, 258, message[:35], 2)


def _draw_hover(frame: bytearray, text: str) -> None:
    gif._fill(frame, 246, 146, 476, 206, 1)
    gif._fill(frame, 254, 154, 468, 198, 0)
    gif._draw_text(frame, 266, 166, "HOVER", 7)
    gif._draw_text(frame, 266, 184, text[:28], 4)


def _draw_quickfix(frame: bytearray, title: str) -> None:
    gif._fill(frame, 210, 244, 560, 318, 1)
    gif._fill(frame, 218, 252, 552, 310, 0)
    gif._draw_text(frame, 230, 264, "QUICK FIX", 5)
    gif._draw_text(frame, 230, 288, title[:38], 4)


def render_frames(data: Mapping[str, object]) -> Sequence[bytes]:
    lsp = data["lsp"]  # type: ignore[index]
    frames = []
    for visible in range(1, int(data["gif"]["frame_count"]) + 1):  # type: ignore[index]
        frame = _base_frame()
        if visible >= 2:
            _draw_diagnostic(frame, "FC2 EXPECTS 30 GETS BATCH,20")
        if visible >= 4:
            _draw_hover(frame, str(lsp["hover_contents"]))
        if visible >= 5:
            _draw_quickfix(frame, str(lsp["quickfix_title"]))
        gif._draw_text(frame, 28, gif.HEIGHT - 34, str(data["transcript"][visible - 1])[:48], 7)  # type: ignore[index]
        frames.append(bytes(frame))
    return frames


def render_markdown(data: Mapping[str, object]) -> str:
    gif_data = data["gif"]  # type: ignore[index]
    lsp = data["lsp"]  # type: ignore[index]
    lines = [
        "# Editor/LSP inline diagnostics demo GIF",
        "",
        "This artifact is generated by `reproducibility/editor_lsp_demo_gif.py` from",
        "a real `verify_architecture` result rendered through `src.lsp_provider`, so",
        "the squiggle, hover, and quick-fix shown in the GIF are tied to the same",
        "payload used by editor clients.",
        "",
        f"- GIF: `{gif_data['path']}` ({gif_data['width']}x{gif_data['height']}, {gif_data['frame_count']} frames)",
        f"- source SHA-256: `{data['source_sha256']}`",
        f"- GIF SHA-256: `{gif_data['sha256']}`",
        f"- verdict: **{data['verdict']}** with **{data['bug_count']}** reported bugs",
        f"- diagnostic line: **{lsp['diagnostic_line_1indexed']}**",
        f"- quick-fix: `{lsp['quickfix_title']}`",
        f"- hover: line **{lsp['hover_line_1indexed']}**, `{lsp['hover_contents']}`",
        "",
        "## Transcript",
        "",
    ]
    for line in data["transcript"]:  # type: ignore[index]
        lines.append(f"- `{line}`")
    lines.append("")
    return "\n".join(lines)


def build_outputs() -> Tuple[bytes, Dict[str, object], str]:
    data = build_manifest()
    frames = render_frames(data)
    gif_bytes = gif.encode_gif(frames)
    data["gif"]["bytes"] = len(gif_bytes)  # type: ignore[index]
    data["gif"]["sha256"] = gif._sha256_bytes(gif_bytes)  # type: ignore[index]
    return gif_bytes, data, render_markdown(data)


def write_outputs() -> Dict[str, object]:
    gif_bytes, data, markdown = build_outputs()
    OUT_GIF.write_bytes(gif_bytes)
    OUT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown, encoding="utf-8")
    return data


def _snapshot(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if editor-demo artifacts are stale")
    args = parser.parse_args(argv)

    before = {path: _snapshot(path) for path in OUTPUTS} if args.check else {}
    data = write_outputs()
    if args.check:
        after = {path: _snapshot(path) for path in OUTPUTS}
        if before != after:
            print("editor LSP demo GIF artifacts are stale", file=sys.stderr)
            return 1
    print(f"editor LSP demo GIF: {data['gif']['path']}")  # type: ignore[index]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
