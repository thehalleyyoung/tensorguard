"""
Localization-quality experiment.

For each known-buggy nn.Module file, compare three tools on:
  Q1 (cause-line): does the tool's diagnostic point to the *cause* of
       the bug (the layer constructor with the wrong dim) rather than
       the symptom site (the line where the runtime error occurs)?
  Q2 (concrete-fix): does the diagnostic name a specific dim or arg
       value that would fix the bug (e.g. "expected 25088, got 147")?

Tools compared:
  TG  : TensorGuard (src/api.verify_architecture)
  FX  : torch.fx.symbolic_trace + ShapeProp on a meta-tensor
  FT  : torch._subclasses.fake_tensor.FakeTensorMode forward execution

Honest scoring is done with a deterministic rubric per tool:
  - cause_line_match: int (1/0) — analyzer reports a line ==
        the author-annotated cause line, OR within ±2 lines if the
        author allows a small window.
  - concrete_fix:     int (1/0) — diagnostic message contains a numeric
        token that matches one of the expected/got values.

Sources for buggy files:
  benchmarks/realcode_corpus/<NN>_*.py with `# bug:` annotation.
  Each gets `# cause_line: N` and `# expected: <int>` annotations
  (added once, deterministic).

Output: benchmarks/localization_quality_results.json
"""
from __future__ import annotations

import json
import re
import sys
import io
import contextlib
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture  # type: ignore

import torch
import torch.nn as nn
import torch.fx as fx


CAUSE_RE = re.compile(r"#\s*cause_line\s*:\s*(\d+)", re.IGNORECASE)
EXPECTED_RE = re.compile(r"#\s*expected\s*:\s*([\d, ]+)", re.IGNORECASE)
INPUT_RE = re.compile(r"#\s*input_shape\s*:\s*\(([^)]*)\)", re.IGNORECASE)
BUG_RE = re.compile(r"#\s*bug\s*:", re.IGNORECASE)


def parse_input_shape(src: str):
    m = INPUT_RE.search(src)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    return tuple(int(p) for p in parts if p.isdigit())


def parse_cause_line(src: str) -> Optional[int]:
    m = CAUSE_RE.search(src)
    return int(m.group(1)) if m else None


def parse_expected_values(src: str):
    m = EXPECTED_RE.search(src)
    if not m:
        return []
    return [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]


def load_module_from_src(src: str, name: str = "_bug_mod"):
    """Load a temporary module by exec-ing the source and return class M."""
    ns: dict = {}
    exec(compile(src, name + ".py", "exec"), ns)
    return ns.get("M")


def run_tg(src: str, in_shape) -> dict:
    try:
        result = verify_architecture(src, input_shapes={"x": in_shape})
        bugs = list(getattr(result, "bugs", []) or [])
        if not bugs:
            return {"diagnostic": "no bug reported", "lines": [], "values": []}
        lines = [getattr(b.location, "line", None) for b in bugs]
        msgs = [b.message for b in bugs]
        # Extract numeric tokens from the messages
        nums = []
        for m in msgs:
            nums.extend(int(t) for t in re.findall(r"\d+", m))
        return {
            "diagnostic": " || ".join(m[:200] for m in msgs[:3]),
            "lines": [l for l in lines if l is not None],
            "values": nums,
        }
    except Exception as e:
        return {"diagnostic": f"ERROR: {e}", "lines": [], "values": []}


def run_fx(src: str, in_shape) -> dict:
    """Try torch.fx.symbolic_trace + ShapeProp on meta tensor."""
    try:
        cls = load_module_from_src(src)
        if cls is None:
            return {"diagnostic": "no class M", "lines": [], "values": []}
        mod = cls()
        try:
            traced = fx.symbolic_trace(mod)
        except Exception as e:
            return {"diagnostic": f"trace-fail: {e}", "lines": [], "values": []}
        from torch.fx.passes.shape_prop import ShapeProp
        try:
            with torch.no_grad():
                ShapeProp(traced).propagate(torch.zeros(*in_shape, device="meta"))
            return {"diagnostic": "no error", "lines": [], "values": []}
        except Exception as e:
            tb = traceback.format_exc()
            # Pull line numbers from traceback that point into our source
            lines = [int(m.group(1))
                     for m in re.finditer(r"_bug_mod\.py.*line (\d+)", tb)]
            nums = [int(t) for t in re.findall(r"\d+", str(e))]
            return {"diagnostic": str(e)[:200], "lines": lines, "values": nums}
    except Exception as e:
        return {"diagnostic": f"ERROR: {e}", "lines": [], "values": []}


def run_fake_tensor(src: str, in_shape) -> dict:
    """Run forward under FakeTensorMode, instantiating the model
    inside the mode so parameters are also fake."""
    try:
        cls = load_module_from_src(src)
        if cls is None:
            return {"diagnostic": "no class M", "lines": [], "values": []}
        from torch._subclasses.fake_tensor import FakeTensorMode
        try:
            with FakeTensorMode(allow_non_fake_inputs=True) as mode:
                mod = cls()
                x = torch.zeros(*in_shape)
                _ = mod(x)
            return {"diagnostic": "no error", "lines": [], "values": []}
        except Exception as e:
            tb = traceback.format_exc()
            lines = [int(m.group(1))
                     for m in re.finditer(r"_bug_mod\.py.*line (\d+)", tb)]
            # Also try generic line number from the traceback frame for forward
            for m in re.finditer(r"line (\d+), in forward", tb):
                lines.append(int(m.group(1)))
            nums = [int(t) for t in re.findall(r"\d+", str(e))]
            return {"diagnostic": str(e)[:200], "lines": lines, "values": nums}
    except Exception as e:
        return {"diagnostic": f"ERROR: {e}", "lines": [], "values": []}


def score(record: dict, cause_line: Optional[int], expected_vals) -> dict:
    """Apply the deterministic rubric."""
    lines = record.get("lines") or []
    values = record.get("values") or []
    if cause_line is not None and lines:
        cause_match = int(any(abs(l - cause_line) <= 2 for l in lines))
    else:
        cause_match = 0
    if expected_vals and values:
        concrete = int(any(v in values for v in expected_vals))
    else:
        concrete = 0
    return {"cause_line_match": cause_match, "concrete_fix": concrete}


def main():
    corpus = ROOT / "benchmarks" / "realcode_corpus"
    annotated = ROOT / "benchmarks" / "localization_annotations.json"

    # Collect buggy files. Augment with explicit (cause_line, expected) annotations.
    bug_records = []
    for path in sorted(corpus.glob("*.py")):
        src = path.read_text()
        if not BUG_RE.search(src):
            continue
        in_shape = parse_input_shape(src)
        if in_shape is None:
            continue
        bug_records.append({"file": path.name, "src": src, "in_shape": in_shape})

    # Hard-coded annotations for the 2 corpus bugs (deterministic ground truth).
    ANNOTS = {
        "13_gan_discriminator.py": {"cause_line": 18, "expected": [16384, 4096, 256]},
        "27_segmentation_head.py": {"cause_line": 16, "expected": [32, 64]},
        "31_linear_dim_bug.py":    {"cause_line": 11, "expected": [32, 64]},
        "32_matmul_no_transpose_bug.py": {"cause_line": 14, "expected": [8, 16]},
        "33_cat_dim_bug.py":       {"cause_line": 17, "expected": [32, 64]},
    }

    rows = []
    for rec in bug_records:
        ann = ANNOTS.get(rec["file"], {})
        cl = ann.get("cause_line")
        ev = ann.get("expected", [])
        tg = run_tg(rec["src"], rec["in_shape"])
        fx_r = run_fx(rec["src"], rec["in_shape"])
        ft = run_fake_tensor(rec["src"], rec["in_shape"])
        rows.append({
            "file": rec["file"],
            "cause_line_gt": cl,
            "expected_values_gt": ev,
            "tg":   {**tg, **score(tg, cl, ev)},
            "fx":   {**fx_r, **score(fx_r, cl, ev)},
            "fake": {**ft, **score(ft, cl, ev)},
        })

    summary = {tool: {"cause": 0, "concrete": 0} for tool in ("tg", "fx", "fake")}
    for r in rows:
        for tool in ("tg", "fx", "fake"):
            summary[tool]["cause"] += r[tool]["cause_line_match"]
            summary[tool]["concrete"] += r[tool]["concrete_fix"]

    summary["n_files"] = len(rows)
    out = {"summary": summary, "rows": rows}
    out_path = ROOT / "benchmarks" / "localization_quality_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print(f"[localization] wrote {out_path}")


if __name__ == "__main__":
    main()
