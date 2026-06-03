"""Step 291 -- editor/LSP demo GIF is generated from real LSP diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys

import reproducibility.editor_lsp_demo_gif as edg
import reproducibility.reproduce_all as ra


def _count_gif_frames(payload: bytes) -> int:
    if not payload.startswith(b"GIF89a"):
        raise AssertionError("not a GIF89a payload")
    packed = payload[10]
    global_table = 3 * (2 ** ((packed & 0x07) + 1)) if packed & 0x80 else 0
    i = 13 + global_table
    frames = 0
    while i < len(payload):
        introducer = payload[i]
        i += 1
        if introducer == 0x3B:
            return frames
        if introducer == 0x21:
            i += 1
            while True:
                block_len = payload[i]
                i += 1
                if block_len == 0:
                    break
                i += block_len
            continue
        if introducer == 0x2C:
            frames += 1
            packed = payload[i + 8]
            i += 9
            if packed & 0x80:
                i += 3 * (2 ** ((packed & 0x07) + 1))
            i += 1
            while True:
                block_len = payload[i]
                i += 1
                if block_len == 0:
                    break
                i += block_len
            continue
        raise AssertionError(f"unexpected GIF block: {introducer:#x}")
    raise AssertionError("unterminated GIF")


def test_manifest_is_tied_to_real_lsp_report():
    data = edg.build_manifest()
    assert data["schema"] == "tensorguard.editor_lsp_demo_gif/v1"
    assert data["step"] == 291
    assert data["verdict"] == "UNSAFE"
    assert data["bug_count"] >= 1
    assert data["lsp"]["diagnostic_count"] == 1
    assert data["lsp"]["code_action_count"] == 1
    assert data["lsp"]["hover_count"] >= 1
    assert data["lsp"]["diagnostic_line_1indexed"] == 10
    assert "expects input dimension 30" in data["lsp"]["diagnostic_message"]
    assert "in_features=20" in data["lsp"]["quickfix_title"]
    assert "(batch, 20)" in data["lsp"]["hover_contents"]


def test_gif_is_valid_animated_gif_with_stable_bytes():
    gif1, data1, _ = edg.build_outputs()
    gif2, data2, _ = edg.build_outputs()
    assert gif1 == gif2
    assert data1 == data2
    assert gif1.startswith(b"GIF89a")
    assert gif1.endswith(b"\x3b")
    assert int.from_bytes(gif1[6:8], "little") == 640
    assert int.from_bytes(gif1[8:10], "little") == 360
    assert _count_gif_frames(gif1) == data1["gif"]["frame_count"]
    assert data1["gif"]["sha256"] == edg.gif._sha256_bytes(gif1)
    assert edg.OUT_GIF.read_bytes() == gif1


def test_markdown_and_json_record_lsp_payload_and_gif_hash():
    gif_bytes, data, md = edg.build_outputs()
    committed = json.loads(edg.OUT_JSON.read_text(encoding="utf-8"))
    assert committed == data
    assert data["gif"]["bytes"] == len(gif_bytes)
    assert data["gif"]["sha256"] in md
    assert "real `verify_architecture` result" in md
    assert data["lsp"]["quickfix_title"] in md


def test_cli_check_passes_against_committed_artifacts():
    proc = subprocess.run(
        [sys.executable, "reproducibility/editor_lsp_demo_gif.py", "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_reproduce_all_and_makefile_own_editor_lsp_demo_gif():
    assert "docs/launch/editor_lsp_demo.gif" in ra.GENERATED_DETERMINISTIC
    assert "reproducibility/editor_lsp_demo_gif.json" in ra.GENERATED_DETERMINISTIC
    assert "reproducibility/editor_lsp_demo_gif.md" in ra.GENERATED_DETERMINISTIC
    assert any(step[1][-1] == "reproducibility/editor_lsp_demo_gif.py" for step in ra.STEPS)
    terminal_idx = next(i for i, step in enumerate(ra.STEPS) if step[1][-1] == "reproducibility/terminal_demo_gif.py")
    editor_idx = next(i for i, step in enumerate(ra.STEPS) if step[1][-1] == "reproducibility/editor_lsp_demo_gif.py")
    assert terminal_idx < editor_idx
    makefile = (edg.REPO / "Makefile").read_text(encoding="utf-8")
    assert "\neditor-lsp-demo-gif:" in makefile
    assert "reproducibility/editor_lsp_demo_gif.py --check" in makefile
