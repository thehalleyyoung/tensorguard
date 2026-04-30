"""
hybrid_mode.py
==============

Implements TensorGuard's hybrid analysis: TG-static-first → FakeTensor fallback.

Public API: ``hybrid_check(source, input_shapes, qualified_name, filename) -> dict``
"""
from __future__ import annotations

import contextlib
import io
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture  # noqa: E402

# ── Mirrored from run_v5_benchmark.py ──────────────────────────────────
PREAMBLE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from typing import Optional, Tuple, List, Dict, Any\n"
)

# ── Helpers (mirrored from run_baseline_comparison.py) ─────────────────

def _exec_source(source: str):
    """Compile + exec source with PREAMBLE. Return (namespace, error_string)."""
    ns: Dict[str, Any] = {}
    try:
        exec(compile(PREAMBLE + source, "<hybrid>", "exec"), ns)
    except Exception as e:
        return None, f"exec_failed: {type(e).__name__}: {e}"
    return ns, None


def _find_module_class(ns: Dict[str, Any], hint: Optional[str] = None):
    if hint and hint in ns and isinstance(ns[hint], type) \
            and issubclass(ns[hint], nn.Module):
        return ns[hint]
    cands = [v for v in ns.values()
             if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module]
    return cands[-1] if cands else None


def _try_instantiate(cls: type):
    try:
        return cls(), None
    except Exception as e:
        return None, f"ctor_failed: {type(e).__name__}: {str(e)[:120]}"


def _make_inputs(input_shapes: Optional[Dict[str, tuple]]):
    """Build concrete zero tensors from input_shapes dict.
    Falls back to default shapes when input_shapes is None or empty."""
    if not input_shapes:
        return None, "no_input_shapes"
    out = {}
    for k, sh in input_shapes.items():
        try:
            concrete = tuple(2 if isinstance(d, str) else int(d) for d in sh)
            if k == "input_ids":
                out[k] = torch.zeros(concrete, dtype=torch.long)
            else:
                out[k] = torch.zeros(concrete)
        except Exception as e:
            return None, f"shape_concretize_failed: {e}"
    return out, None


def _decide_tg(res, err: Optional[str]) -> str:
    """Map TG AnalysisResult / exception → {Verified, Refuted, Abstain}.
    Mirrors _decide() in run_v5_benchmark.py exactly."""
    if err:
        return "Abstain"
    if res.abstained:
        return "Abstain"
    if res.bug_count > 0:
        return "Refuted"
    return "Verified"


# ── Timeout helper ──────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("FakeTensor forward timed out")


# ── FakeTensor fallback ─────────────────────────────────────────────────

def _run_fake_tensor(source: str, input_shapes: Optional[Dict[str, tuple]],
                     qualified_name: Optional[str] = None) -> Dict[str, Any]:
    """Run FakeTensorMode on the module.  Returns fallback dict."""
    ns, err = _exec_source(source)
    if err:
        return {"verdict": "Abstain", "error": err}

    # derive hint from qualified_name (last dotted segment)
    hint = qualified_name.split(".")[-1] if qualified_name else None
    cls = _find_module_class(ns, hint)
    if cls is None:
        return {"verdict": "Abstain", "error": "no_nn_module_found"}

    mod, err = _try_instantiate(cls)
    if err:
        return {"verdict": "Abstain", "error": err}

    # Build inputs; use defaults if missing
    if not input_shapes:
        # vision default or HF default based on hint
        if hint and "input_ids" in (input_shapes or {}):
            input_shapes = {"input_ids": (1, 128)}
        else:
            input_shapes = {"x": (1, 3, 224, 224)}

    inputs, err = _make_inputs(input_shapes)
    if err:
        # Try default fallback
        input_shapes = {"x": (1, 3, 224, 224)}
        inputs, err2 = _make_inputs(input_shapes)
        if err2:
            return {"verdict": "Abstain", "error": f"make_inputs: {err}"}

    # Install 30-second alarm (Unix only)
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(30)
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        # allow_non_fake_inputs=True: module params are real tensors created
        # outside the FakeTensorMode context; this flag permits mixing them
        # with fake input tensors during the forward pass.
        with FakeTensorMode(allow_non_fake_inputs=True):
            fake_in = {k: torch.empty(v.shape, dtype=v.dtype) for k, v in inputs.items()}
            try:
                mod(**fake_in)
            except TypeError:
                mod(*fake_in.values())
        return {"verdict": "Verified", "error": None}
    except _Timeout as e:
        return {"verdict": "Abstain", "error": f"timeout: {e}"}
    except RuntimeError as e:
        m = str(e).lower()
        if any(kw in m for kw in ("shape", "size", "expected", "broadcast",
                                   "dimension", "channel", "mismatch")):
            return {"verdict": "Refuted", "error": str(e)[:300]}
        return {"verdict": "Abstain", "error": str(e)[:300]}
    except Exception as e:
        return {"verdict": "Abstain", "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


# ── Main public function ────────────────────────────────────────────────

def hybrid_check(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    qualified_name: Optional[str] = None,
    filename: str = "<hybrid>",
) -> Dict[str, Any]:
    """Run TG-static first; fall back to FakeTensorMode on Abstain.

    Returns a dict with keys:
      verdict       – "Verified" | "Refuted" | "Abstain"
      source        – "tensorguard" | "fake_tensor"
      tg_bugs       – list of TG bug dicts (may be empty)
      fallback      – None (if not needed) or {"verdict": ..., "error": str|None}
    """
    full_source = PREAMBLE + source
    captured = io.StringIO()
    tg_res = None
    tg_err = None
    try:
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            tg_res = verify_architecture(
                full_source,
                input_shapes=input_shapes,
                max_cegar_iterations=3,
                filename=filename,
            )
    except Exception as e:
        tg_err = f"{type(e).__name__}: {e}"

    tg_verdict = _decide_tg(tg_res, tg_err)

    tg_bugs: List[Dict[str, Any]] = []
    if tg_res is not None:
        tg_bugs = [
            {
                "category": b.category.value,
                "severity": b.severity,
                "message": b.message[:300],
            }
            for b in tg_res.bugs[:10]
        ]

    # Step 2: TG gave a definitive answer → return immediately
    if tg_verdict in ("Verified", "Refuted"):
        return {
            "verdict": tg_verdict,
            "source": "tensorguard",
            "tg_bugs": tg_bugs,
            "fallback": None,
        }

    # Step 3: TG Abstained → try FakeTensor fallback
    fallback = _run_fake_tensor(source, input_shapes, qualified_name)
    hybrid_verdict = fallback["verdict"]  # may still be Abstain

    return {
        "verdict": hybrid_verdict,
        "source": "fake_tensor" if hybrid_verdict != "Abstain" else "tensorguard",
        "tg_bugs": tg_bugs,
        "fallback": fallback,
    }
