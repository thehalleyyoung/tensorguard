"""Step 290 -- quickstart terminal demo GIF is generated from real dry-run evidence."""

from __future__ import annotations

import json
import subprocess
import sys

import reproducibility.reproduce_all as ra
import reproducibility.terminal_demo_gif as tdg


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
            i += 1  # label
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
            i += 1  # LZW minimum code size
            while True:
                block_len = payload[i]
                i += 1
                if block_len == 0:
                    break
                i += block_len
            continue
        raise AssertionError(f"unexpected GIF block: {introducer:#x}")
    raise AssertionError("unterminated GIF")


def test_manifest_is_tied_to_launch_dry_run_quickstart_evidence():
    data = tdg.build_manifest()
    assert data["source"] == "reproducibility/launch_dry_run.json"
    assert data["source_quickstart_step"] == "run_quickstart_from_generated_demo"
    assert data["quickstart_verdict"] == "SAFE"
    assert data["quickstart_bug_count"] == 0
    assert any("examples/quickstart.py" in line for line in data["transcript"])


def test_gif_is_valid_animated_gif_with_stable_bytes():
    gif1, data1, _ = tdg.build_outputs()
    gif2, data2, _ = tdg.build_outputs()
    assert gif1 == gif2
    assert data1 == data2
    assert gif1.startswith(b"GIF89a")
    assert gif1.endswith(b"\x3b")
    assert int.from_bytes(gif1[6:8], "little") == tdg.WIDTH
    assert int.from_bytes(gif1[8:10], "little") == tdg.HEIGHT
    assert _count_gif_frames(gif1) == data1["gif"]["frame_count"]
    assert data1["gif"]["sha256"] == tdg._sha256_bytes(gif1)
    assert tdg.OUT_GIF.read_bytes() == gif1


def test_markdown_and_json_record_gif_hash_and_transcript():
    gif, data, md = tdg.build_outputs()
    committed = json.loads(tdg.OUT_JSON.read_text(encoding="utf-8"))
    assert committed == data
    assert f"`{data['gif']['sha256']}`" in md or data["gif"]["sha256"] in md
    assert "TensorGuard verdict: SAFE" in md
    assert data["gif"]["bytes"] == len(gif)


def test_cli_check_passes_against_committed_artifacts():
    proc = subprocess.run(
        [sys.executable, "reproducibility/terminal_demo_gif.py", "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_reproduce_all_and_makefile_own_terminal_demo_gif():
    assert "docs/launch/quickstart_terminal_demo.gif" in ra.GENERATED_DETERMINISTIC
    assert "reproducibility/terminal_demo_gif.json" in ra.GENERATED_DETERMINISTIC
    assert "reproducibility/terminal_demo_gif.md" in ra.GENERATED_DETERMINISTIC
    assert any(step[1][-1] == "reproducibility/terminal_demo_gif.py" for step in ra.STEPS)
    launch_idx = next(i for i, step in enumerate(ra.STEPS) if step[1][-1] == "reproducibility/launch_dry_run.py")
    gif_idx = next(i for i, step in enumerate(ra.STEPS) if step[1][-1] == "reproducibility/terminal_demo_gif.py")
    assert launch_idx < gif_idx
    makefile = (tdg.REPO / "Makefile").read_text(encoding="utf-8")
    assert "\nterminal-demo-gif:" in makefile
    assert "reproducibility/terminal_demo_gif.py --check" in makefile
