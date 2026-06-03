#!/usr/bin/env python3
"""Step 290 -- deterministic terminal GIF for the public quickstart path.

The GIF is not hand-recorded.  It is rendered from the committed Step 289
fresh-environment dry-run evidence, so the visual demo can only say the
quickstart is SAFE when the real launch dry-run proved that fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "reproducibility" / "launch_dry_run.json"
OUT_GIF = REPO / "docs" / "launch" / "quickstart_terminal_demo.gif"
OUT_JSON = REPO / "reproducibility" / "terminal_demo_gif.json"
OUT_MD = REPO / "reproducibility" / "terminal_demo_gif.md"
OUTPUTS = (OUT_GIF, OUT_JSON, OUT_MD)

WIDTH = 640
HEIGHT = 360
SCALE = 2
CHAR_W = 6 * SCALE
LINE_H = 10 * SCALE
LEFT = 24
TOP = 58

PALETTE: List[Tuple[int, int, int]] = [
    (10, 14, 28),    # 0 terminal background
    (22, 28, 48),    # 1 title bar
    (225, 231, 245), # 2 primary text
    (132, 151, 176), # 3 dim text
    (52, 211, 153),  # 4 safe green
    (251, 191, 36),  # 5 command yellow
    (248, 113, 113), # 6 unsafe red (unused here, kept for stable palette)
    (59, 130, 246),  # 7 accent blue
]

FONT: Dict[str, Tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "$": ("01110", "10100", "10100", "01110", "00101", "00101", "11110"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "&": ("01100", "10010", "10100", "01000", "10101", "10010", "01101"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    ",": ("00000", "00000", "00000", "00000", "00110", "00100", "01000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    ";": ("00000", "01100", "01100", "00000", "01100", "00100", "01000"),
    "<": ("00010", "00100", "01000", "10000", "01000", "00100", "00010"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    ">": ("01000", "00100", "00010", "00001", "00010", "00100", "01000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "[": ("01110", "01000", "01000", "01000", "01000", "01000", "01110"),
    "]": ("01110", "00010", "00010", "00010", "00010", "00010", "01110"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
}

FONT.update(
    {
        "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
        "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
        "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
        "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
        "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
        "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
        "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
        "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
        "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
        "9": ("01110", "10001", "10001", "01111", "00001", "00010", "11100"),
    }
)

FONT.update(
    {
        "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
        "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
        "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
        "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
        "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
        "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
        "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
        "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
        "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
        "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
        "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
        "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
        "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
        "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
        "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
        "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
        "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
        "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
        "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
        "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
        "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
        "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
        "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    }
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_source() -> Mapping[str, object]:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if data.get("schema") != "tensorguard.launch_dry_run/v1":
        raise ValueError("terminal demo requires launch_dry_run schema v1")
    return data


def _quickstart_step(data: Mapping[str, object]) -> Mapping[str, object]:
    for step in data["steps"]:  # type: ignore[index]
        if step["name"] == "run_quickstart_from_generated_demo":  # type: ignore[index]
            if step.get("passed") is not True or step.get("verdict") != "SAFE" or step.get("bug_count") != 0:
                raise ValueError("launch dry-run does not prove a clean SAFE quickstart")
            return step  # type: ignore[return-value]
    raise ValueError("launch dry-run is missing the quickstart step")


def _command(step: Mapping[str, object]) -> str:
    return " ".join(str(part) for part in step["command"])  # type: ignore[index]


def build_manifest() -> Dict[str, object]:
    data = _load_source()
    quick = _quickstart_step(data)
    transcript = [
        "$ python -m venv --system-site-packages .venv",
        "$ pip install --no-deps --editable .",
        "$ " + _command(quick).replace("$VENV/bin/", "").replace("$REPO/", ""),
        "TensorGuard verdict: SAFE",
        "Bugs: 0",
        "Fresh-env proof: reproducibility/launch_dry_run.md",
    ]
    return {
        "schema": "tensorguard.terminal_demo_gif/v1",
        "step": 290,
        "source": "reproducibility/launch_dry_run.json",
        "source_sha256": _sha256_bytes(SOURCE.read_bytes()),
        "source_quickstart_step": quick["name"],
        "quickstart_verdict": quick["verdict"],
        "quickstart_bug_count": quick["bug_count"],
        "gif": {
            "path": "docs/launch/quickstart_terminal_demo.gif",
            "width": WIDTH,
            "height": HEIGHT,
            "frame_count": len(transcript),
            "duration_seconds": round(len(transcript) * 0.85, 2),
        },
        "transcript": transcript,
    }


def _fill(frame: bytearray, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
    x0 = max(0, min(WIDTH, x0))
    x1 = max(0, min(WIDTH, x1))
    y0 = max(0, min(HEIGHT, y0))
    y1 = max(0, min(HEIGHT, y1))
    for y in range(y0, y1):
        row = y * WIDTH
        frame[row + x0 : row + x1] = bytes([color]) * (x1 - x0)


def _draw_text(frame: bytearray, x: int, y: int, text: str, color: int) -> None:
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    _fill(
                        frame,
                        cx + gx * SCALE,
                        y + gy * SCALE,
                        cx + (gx + 1) * SCALE,
                        y + (gy + 1) * SCALE,
                        color,
                    )
        cx += CHAR_W


def _base_frame() -> bytearray:
    frame = bytearray([0]) * (WIDTH * HEIGHT)
    _fill(frame, 12, 12, WIDTH - 12, HEIGHT - 12, 1)
    _fill(frame, 12, 42, WIDTH - 12, HEIGHT - 12, 0)
    _fill(frame, 28, 26, 36, 34, 6)
    _fill(frame, 44, 26, 52, 34, 5)
    _fill(frame, 60, 26, 68, 34, 4)
    _draw_text(frame, 86, 22, "TENSORGUARD QUICKSTART", 2)
    return frame


def render_frames(transcript: Sequence[str]) -> List[bytes]:
    frames: List[bytes] = []
    for visible in range(1, len(transcript) + 1):
        frame = _base_frame()
        for idx, line in enumerate(transcript[:visible]):
            color = 5 if line.startswith("$") else 4 if "SAFE" in line or "0" == line[-1:] else 3
            _draw_text(frame, LEFT, TOP + idx * LINE_H, line[:48], color)
        _draw_text(frame, LEFT, HEIGHT - 48, "REAL DRY-RUN SOURCE: LAUNCH_DRY_RUN.JSON", 7)
        frames.append(bytes(frame))
    return frames


def _lzw_encode(indices: bytes, min_code_size: int = 3) -> bytes:
    clear = 1 << min_code_size
    end = clear + 1
    next_code = end + 1
    code_size = min_code_size + 1
    dictionary = {(i,): i for i in range(clear)}
    codes = [clear]
    w = (indices[0],)
    for value in indices[1:]:
        wk = w + (value,)
        if wk in dictionary:
            w = wk
            continue
        codes.append(dictionary[w])
        if next_code < 4096:
            dictionary[wk] = next_code
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
        else:
            codes.append(clear)
            dictionary = {(i,): i for i in range(clear)}
            next_code = end + 1
            code_size = min_code_size + 1
        w = (value,)
    codes.append(dictionary[w])
    codes.append(end)

    out = bytearray()
    bit_buffer = 0
    bit_count = 0
    code_size = min_code_size + 1
    next_code = end + 1
    for code in codes:
        bit_buffer |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            out.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8
        if code == clear:
            code_size = min_code_size + 1
            next_code = end + 1
        elif code != end:
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
    if bit_count:
        out.append(bit_buffer & 0xFF)
    return bytes(out)


def _subblocks(payload: bytes) -> bytes:
    chunks = bytearray()
    for i in range(0, len(payload), 255):
        block = payload[i : i + 255]
        chunks.append(len(block))
        chunks.extend(block)
    chunks.append(0)
    return bytes(chunks)


def encode_gif(frames: Sequence[bytes], delay_cs: int = 85) -> bytes:
    palette_size = 8
    packed = 0x80 | 0x70 | 0x02
    out = bytearray(b"GIF89a")
    out.extend(WIDTH.to_bytes(2, "little"))
    out.extend(HEIGHT.to_bytes(2, "little"))
    out.extend(bytes([packed, 0, 0]))
    for rgb in PALETTE:
        out.extend(bytes(rgb))
    out.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")
    for frame in frames:
        if len(frame) != WIDTH * HEIGHT:
            raise ValueError("frame has unexpected dimensions")
        out.extend(b"\x21\xf9\x04")
        out.extend(bytes([0x04]))
        out.extend(delay_cs.to_bytes(2, "little"))
        out.extend(b"\x00\x00")
        out.extend(b"\x2c\x00\x00\x00\x00")
        out.extend(WIDTH.to_bytes(2, "little"))
        out.extend(HEIGHT.to_bytes(2, "little"))
        out.extend(b"\x00")
        out.append(3)
        out.extend(_subblocks(_lzw_encode(frame, 3)))
    out.extend(b"\x3b")
    assert palette_size == len(PALETTE)
    return bytes(out)


def render_markdown(data: Mapping[str, object]) -> str:
    gif = data["gif"]  # type: ignore[index]
    lines = [
        "# Quickstart terminal demo GIF",
        "",
        "This artifact is generated by `reproducibility/terminal_demo_gif.py` from",
        "`reproducibility/launch_dry_run.json`, so the recorded SAFE verdict is tied",
        "to the real fresh-environment quickstart proof.",
        "",
        f"- GIF: `{gif['path']}` ({gif['width']}x{gif['height']}, {gif['frame_count']} frames)",
        f"- source SHA-256: `{data['source_sha256']}`",
        f"- GIF SHA-256: `{gif['sha256']}`",
        f"- quickstart verdict: **{data['quickstart_verdict']}**",
        f"- quickstart bug count: **{data['quickstart_bug_count']}**",
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
    frames = render_frames(data["transcript"])  # type: ignore[arg-type]
    gif_bytes = encode_gif(frames)
    data["gif"]["bytes"] = len(gif_bytes)  # type: ignore[index]
    data["gif"]["sha256"] = _sha256_bytes(gif_bytes)  # type: ignore[index]
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
    parser.add_argument("--check", action="store_true", help="fail if terminal-demo artifacts are stale")
    args = parser.parse_args(argv)

    before = {path: _snapshot(path) for path in OUTPUTS} if args.check else {}
    data = write_outputs()
    if args.check:
        after = {path: _snapshot(path) for path in OUTPUTS}
        if before != after:
            print("terminal demo GIF artifacts are stale", file=sys.stderr)
            return 1
    print(f"terminal demo GIF: {data['gif']['path']}")  # type: ignore[index]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
