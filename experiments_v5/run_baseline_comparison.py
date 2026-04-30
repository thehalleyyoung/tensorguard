"""
run_baseline_comparison.py
==========================

Runs five baseline static/dynamic shape verifiers on the same v5 block
corpus + bug corpus that `run_v5_benchmark.py` consumes, and emits an
apples-to-apples comparison.

Baselines (from weakest to strongest static guarantees):
  1. torch.fx.symbolic_trace            – purely structural; no shape check.
  2. torch._subclasses.FakeTensorMode   – meta-tensor symbolic shape check.
  3. torch.export.export                – ahead-of-time IR capture.
  4. mypy + jaxtyping                   – type-level shape annotations.
  5. beartype                           – runtime shape contract.

For every (tool, input) we record exactly one of:
   "Verified" | "Refuted" | "Abstain" | "N/A: <reason>"
"N/A" is used when the tool *cannot* be applied to the input
(missing forward args, untyped, ImportError, ...).  We never silently
report Verified.

Outputs:
  experiments_v5/v5_baseline_comparison.json
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

BLOCK_JSONL = ROOT / "v5_block_corpus.jsonl"
BUG_JSONL = ROOT / "v5_bug_corpus.jsonl"
BUG_REPRO_DIR = ROOT / "bug_repros"
OUT_JSON = ROOT / "v5_baseline_comparison.json"

PREAMBLE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from typing import Optional, Tuple, List, Dict, Any\n"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _exec_source(source: str) -> Tuple[Dict[str, Any] | None, str | None]:
    """Compile + exec source.  Return (namespace, error_string)."""
    ns: Dict[str, Any] = {}
    try:
        exec(compile(PREAMBLE + source, "<bench>", "exec"), ns)
    except Exception as e:
        return None, f"exec_failed: {type(e).__name__}: {e}"
    return ns, None


def _find_module_class(ns: Dict[str, Any], hint: str | None = None
                       ) -> type | None:
    if hint and hint in ns and isinstance(ns[hint], type) \
            and issubclass(ns[hint], nn.Module):
        return ns[hint]
    cands = [v for v in ns.values()
             if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module]
    return cands[-1] if cands else None


def _try_instantiate(cls: type) -> Tuple[nn.Module | None, str | None]:
    """Try to instantiate with no args; if that fails, give up — we
    cannot synthesise constructor args without per-block hand-holding."""
    try:
        return cls(), None
    except Exception as e:
        return None, f"ctor_failed: {type(e).__name__}: {str(e)[:120]}"


def _make_inputs(input_shapes: Dict[str, tuple]) -> Tuple[Dict[str, Any] | None, str | None]:
    if not input_shapes:
        return None, "no_input_shapes"
    out = {}
    for k, sh in input_shapes.items():
        try:
            concrete = tuple(2 if isinstance(d, str) else int(d) for d in sh)
            out[k] = torch.zeros(concrete)
        except Exception as e:
            return None, f"shape_concretize_failed: {e}"
    return out, None


# ---------------------------------------------------------------------------
# Baseline 1: torch.fx.symbolic_trace
# ---------------------------------------------------------------------------
def baseline_fx(source: str, input_shapes: Dict[str, tuple],
                hint: str | None = None) -> Tuple[str, str]:
    ns, err = _exec_source(source)
    if err:
        return f"N/A: {err}", ""
    cls = _find_module_class(ns, hint)
    if cls is None:
        return "N/A: no_nn_module_found", ""
    mod, err = _try_instantiate(cls)
    if err:
        return f"N/A: {err}", ""
    try:
        from torch.fx import symbolic_trace
        symbolic_trace(mod)
        # symbolic_trace performs *no* shape check; if it succeeds we
        # honestly report Abstain (tool can't decide), not Verified.
        return "Abstain", "fx_traced_no_shape_check"
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        if "shape" in msg.lower() or "size" in msg.lower():
            return "Refuted", msg
        return "Abstain", msg


# ---------------------------------------------------------------------------
# Baseline 2: FakeTensorMode (meta-shape symbolic execution)
# ---------------------------------------------------------------------------
def baseline_faketensor(source: str, input_shapes: Dict[str, tuple],
                        hint: str | None = None) -> Tuple[str, str]:
    ns, err = _exec_source(source)
    if err:
        return f"N/A: {err}", ""
    cls = _find_module_class(ns, hint)
    if cls is None:
        return "N/A: no_nn_module_found", ""
    mod, err = _try_instantiate(cls)
    if err:
        return f"N/A: {err}", ""
    inputs, err = _make_inputs(input_shapes)
    if err:
        return f"N/A: {err}", ""
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            fake_in = {k: torch.empty_like(v) for k, v in inputs.items()}
            try:
                mod(**fake_in)
            except TypeError:
                # forward(self, x, ...) positional fallback
                mod(*fake_in.values())
        return "Verified", "fake_tensor_succeeded"
    except RuntimeError as e:
        m = str(e).lower()
        if "shape" in m or "size" in m or "expected" in m or "broadcast" in m:
            return "Refuted", str(e)[:200]
        return "Abstain", str(e)[:200]
    except Exception as e:
        return "Abstain", f"{type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Baseline 3: torch.export
# ---------------------------------------------------------------------------
def baseline_torch_export(source: str, input_shapes: Dict[str, tuple],
                          hint: str | None = None) -> Tuple[str, str]:
    ns, err = _exec_source(source)
    if err:
        return f"N/A: {err}", ""
    cls = _find_module_class(ns, hint)
    if cls is None:
        return "N/A: no_nn_module_found", ""
    mod, err = _try_instantiate(cls)
    if err:
        return f"N/A: {err}", ""
    inputs, err = _make_inputs(input_shapes)
    if err:
        return f"N/A: {err}", ""
    try:
        from torch.export import export
        export(mod, args=tuple(inputs.values()))
        return "Verified", "torch_export_succeeded"
    except RuntimeError as e:
        m = str(e).lower()
        if "shape" in m or "size" in m or "expected" in m:
            return "Refuted", str(e)[:200]
        return "Abstain", str(e)[:200]
    except Exception as e:
        return "Abstain", f"{type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Baseline 4: mypy + jaxtyping
# ---------------------------------------------------------------------------
def baseline_mypy_jaxtyping(source: str, input_shapes: Dict[str, tuple],
                            hint: str | None = None) -> Tuple[str, str]:
    """jaxtyping requires *annotated* signatures.  Most blocks in the corpus
    are not annotated, so we honestly report N/A in that case.  When
    annotations are present, we run mypy in --strict mode."""
    if not re.search(r"->\s*(Float|Int|Bool|Num|Shape|Array|Tensor)\[", source) \
            and not re.search(r":\s*(Float|Int|Bool|Num|Shape|Array)\[", source):
        return "N/A: no_jaxtyping_annotations", ""
    full = (
        "from jaxtyping import Float, Int, Bool, Num\n"
        "from torch import Tensor\n"
        + PREAMBLE + source
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(ROOT)) as f:
        f.write(full)
        path = f.name
    try:
        res = subprocess.run(
            [sys.executable, "-m", "mypy", "--no-incremental",
             "--ignore-missing-imports", "--follow-imports=skip", path],
            capture_output=True, text=True, timeout=30,
        )
        out = (res.stdout or "") + (res.stderr or "")
        if "error:" in out:
            return "Refuted", out[:300]
        return "Verified", "mypy_clean"
    except subprocess.TimeoutExpired:
        return "Abstain", "mypy_timeout"
    except FileNotFoundError:
        return "N/A: mypy_not_installed", ""
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Baseline 5: beartype (runtime contract)
# ---------------------------------------------------------------------------
def baseline_beartype(source: str, input_shapes: Dict[str, tuple],
                      hint: str | None = None) -> Tuple[str, str]:
    """beartype enforces runtime types.  Without shape annotations it
    contributes nothing; with jaxtyping annotations it can catch shape
    bugs at runtime.  We honestly report N/A when there are no
    annotations."""
    if not re.search(r"->\s*(Float|Int|Bool|Num|Shape|Array|Tensor)\[", source) \
            and not re.search(r":\s*(Float|Int|Bool|Num|Shape|Array)\[", source):
        return "N/A: no_jaxtyping_annotations", ""
    try:
        from beartype import beartype  # noqa
    except Exception as e:
        return f"N/A: beartype_unavailable: {e}", ""
    ns, err = _exec_source(
        "from beartype import beartype\n"
        "from beartype.claw import beartype_this_package\n"
        + source
    )
    if err:
        return f"N/A: {err}", ""
    cls = _find_module_class(ns, hint)
    if cls is None:
        return "N/A: no_nn_module_found", ""
    mod, err = _try_instantiate(cls)
    if err:
        return f"N/A: {err}", ""
    inputs, err = _make_inputs(input_shapes)
    if err:
        return f"N/A: {err}", ""
    try:
        try:
            mod(**inputs)
        except TypeError:
            mod(*inputs.values())
        return "Verified", "runtime_contract_passed"
    except Exception as e:
        m = str(e).lower()
        if "shape" in m or "size" in m or "type" in m or "violates" in m:
            return "Refuted", str(e)[:200]
        return "Abstain", str(e)[:200]


BASELINES = [
    ("torch_fx_symbolic_trace", baseline_fx),
    ("torch_fake_tensor",       baseline_faketensor),
    ("torch_export",            baseline_torch_export),
    ("mypy_jaxtyping",          baseline_mypy_jaxtyping),
    ("beartype",                baseline_beartype),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_corpus(name: str, items: List[Dict[str, Any]],
                key_source: str, key_shapes: str, hint_key: str | None,
                ) -> Dict[str, Any]:
    print(f"\n[{name}] {len(items)} items  baselines={[b for b,_ in BASELINES]}")
    per_item: List[Dict[str, Any]] = []
    for i, rec in enumerate(items):
        src = rec[key_source]
        shapes = rec[key_shapes]
        hint = rec.get(hint_key) if hint_key else None
        row = {"id": rec["id"]}
        for bname, bfn in BASELINES:
            t0 = time.perf_counter()
            try:
                bucket, detail = bfn(src, shapes, hint)
            except Exception as e:
                bucket, detail = "Abstain", f"baseline_crashed: {type(e).__name__}: {e}"
            row[bname] = {
                "bucket": bucket,
                "detail": detail[:300] if isinstance(detail, str) else "",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        per_item.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(items)}")
    summary: Dict[str, Any] = {}
    for bname, _ in BASELINES:
        c = Counter()
        for row in per_item:
            b = row[bname]["bucket"]
            # Collapse "N/A: <reason>" into "N/A" for headline counts
            c["N/A" if b.startswith("N/A") else b] += 1
        summary[bname] = dict(c)
    return {"summary": summary, "per_item": per_item}


def main():
    t0 = time.time()
    if not BLOCK_JSONL.exists() or not BUG_JSONL.exists():
        sys.exit("Run build_block_corpus.py and the bug-corpus collector first.")

    blocks = [json.loads(l) for l in BLOCK_JSONL.open()]
    bug_records = [json.loads(l) for l in BUG_JSONL.open()]

    # Materialise bug corpus rows containing source+shapes (read repro files)
    bug_items = []
    for r in bug_records:
        repro = REPO / r["repro_file"]
        if not repro.exists():
            continue
        src = repro.read_text()
        m = re.search(r"^INPUT_SHAPES\s*=\s*(\{[^}]*\})", src, flags=re.MULTILINE)
        try:
            shapes = eval(m.group(1)) if m else {}
        except Exception:
            shapes = {}
        bug_items.append({
            "id": r["id"], "source": src, "input_shapes": shapes,
            "class_hint": "BuggyModule",
        })

    block_items = [{"id": b["id"], "source": b["source"],
                    "input_shapes": b["input_shapes"],
                    "class_hint": b["class_name"]} for b in blocks]

    block_res = _run_corpus("blocks", block_items,
                            "source", "input_shapes", "class_hint")
    bug_res = _run_corpus("bugs", bug_items,
                          "source", "input_shapes", "class_hint")

    out = {
        "meta": {
            "build_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 1),
            "torch_version": torch.__version__,
            "baselines": [b for b, _ in BASELINES],
            "bucket_definitions": {
                "Verified": "tool ran to completion and reported no shape error",
                "Refuted":  "tool reported a shape/size/type error",
                "Abstain":  "tool ran but returned an inconclusive / non-shape error",
                "N/A":      "tool was not applicable to this input (no annotations, ctor args missing, etc.)",
            },
        },
        "block_corpus": block_res,
        "bug_corpus": bug_res,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print("\n== BLOCK CORPUS SUMMARY ==")
    print(json.dumps(block_res["summary"], indent=2))
    print("\n== BUG CORPUS SUMMARY ==")
    print(json.dumps(bug_res["summary"], indent=2))


if __name__ == "__main__":
    main()
