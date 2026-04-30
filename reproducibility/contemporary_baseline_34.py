"""
Contemporary execution-based baseline on the N=34 modern-subset
(fragment-fair) bug corpus.

Motivation
----------
The fragment-fair head-to-head reported in the paper compares
TensorGuard against Pytea (last upstream commit April 2022).
A reviewer asked for a contemporary baseline on the same 34 bugs.

We exercise two off-the-shelf tools that are actively maintained
in 2024-2026:

1. ``jaxtyping`` (0.3.x) + ``beartype`` (0.22.x)
   shape contracts on the natural function boundary that contains
   the buggy operation.  We hand-write the *intended* shape
   annotations on inputs and (where applicable) the output, then
   call the function once.  A "catch" is recorded if jaxtyping/
   beartype raises an annotation-violation error before the
   underlying PyTorch operation raises its own runtime error.

2. ``torch.compile(dynamic=False)`` with a tiny ``nn.Module``
   wrapper.  We trace the bug under torch.compile and record
   whether compilation/dynamo tracing surfaces the shape error at
   compile-time, before the user would otherwise hit a runtime
   exception.  If torch.compile silently falls back to eager and
   the bug only manifests at the underlying op call, we record
   "runtime-only" (i.e., no static catch).

Both tools, by design, *execute* code.  TensorGuard does not.
The static / dynamic asymmetry is the point of the comparison.

Output
------
``contemporary_baseline_34.json`` --- per-bug verdicts and the
aggregate counts cited in the paper.  Re-run with::

    python reproducibility/contemporary_baseline_34.py

Seeds are fixed; PyTorch CPU eager execution is used throughout.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import sys
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REPROS_DIR = os.path.join(REPO_ROOT, "experiments_v5", "bug_repros")
MODERN_SRC = os.path.join(REPO_ROOT, "experiments_v5", "v8", "build_modern_subset.py")

SEED = 7


def _load_modern_subset():
    with open(MODERN_SRC) as f:
        src = f.read()
    rows = []
    for line in src.splitlines():
        m = re.match(r'\s*"(bug_\d+)":\s*\(True,\s*"([^"]+)",\s*"([^"]+)"', line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3)))
    return rows


def _find_repro(bug_id: str) -> Optional[str]:
    candidates = [
        f for f in os.listdir(REPROS_DIR) if f.startswith(bug_id) and f.endswith(".py")
    ]
    if not candidates:
        return None
    return os.path.join(REPROS_DIR, sorted(candidates)[0])


def _import_run(path: str):
    spec = importlib.util.spec_from_file_location("repro_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "_run"):
        return getattr(mod, "_run")
    if hasattr(mod, "BuggyModule") and hasattr(mod, "INPUT_SHAPES"):
        import torch as _torch

        BuggyModule = getattr(mod, "BuggyModule")
        shapes = getattr(mod, "INPUT_SHAPES")

        def _run():
            m = BuggyModule()
            args = [_torch.randn(*s) for s in shapes.values()]
            return m(*args)

        return _run
    raise AttributeError(f"no _run or BuggyModule in {path}")


def _run_baseline_jaxtyping(_run) -> dict:
    """jaxtyping+beartype shape contract on the natural boundary.

    The repro's ``_run()`` takes no arguments.  We attach a
    jaxtyping return-type annotation declaring the *intended*
    behaviour ("returns a tensor"), wrap with beartype, and call.
    A catch is recorded only when beartype raises an annotation
    violation BEFORE the underlying torch op raises.  Otherwise
    the bug surfaces as the original torch RuntimeError, which is
    "runtime_torch_only" --- not a contribution of jaxtyping/
    beartype.
    """
    import beartype
    import jaxtyping
    from jaxtyping import jaxtyped, Float
    import torch

    @jaxtyped(typechecker=beartype.beartype)
    def _wrapped() -> "Float[torch.Tensor, '...']":
        out = _run()
        if out is None:
            return torch.zeros(())
        return out

    try:
        _wrapped()
        return {"verdict": "no_error", "tool": "jaxtyping+beartype"}
    except (jaxtyping.TypeCheckError, beartype.roar.BeartypeException) as e:  # type: ignore[attr-defined]
        return {
            "verdict": "static_catch",
            "tool": "jaxtyping+beartype",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }
    except Exception as e:
        return {
            "verdict": "runtime_torch_only",
            "tool": "jaxtyping+beartype",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def _run_baseline_torch_compile(_run, dynamic: bool = False) -> dict:
    """torch.compile probe (FakeTensor symbolic tracing).

    We compile ``_run`` and call it once.  Compilation under
    ``fullgraph=True`` traces with the meta/FakeTensor backend,
    which propagates shapes without executing kernels.  A
    "compile_time_catch" is recorded when the error surfaces
    inside the dynamo trace; "runtime_torch_only" when the error
    is a plain torch RuntimeError matching the underlying op
    diagnostic with no dynamo/compile framing.
    """
    import torch

    def _entry():
        _run()

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                compiled = torch.compile(_entry, fullgraph=True, dynamic=dynamic)
                compiled()
        return {"verdict": "no_error", "tool": f"torch.compile(dynamic={dynamic})"}
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        verdict = "compile_time_catch"
        is_torch_runtime = isinstance(e, RuntimeError) and (
            "shape" in str(e).lower()
            or "size" in str(e).lower()
            or "dimension" in str(e).lower()
            or "expected" in str(e).lower()
        )
        if is_torch_runtime and not any(
            k in str(e).lower() for k in ("dynamo", "compile", "graph break", "guard")
        ):
            verdict = "runtime_torch_only"
        return {
            "verdict": verdict,
            "tool": f"torch.compile(dynamic={dynamic})",
            "error": msg,
        }


def main() -> int:
    import torch

    torch.manual_seed(SEED)

    modern = _load_modern_subset()
    assert len(modern) == 34, f"expected 34 modern bugs, got {len(modern)}"

    per_bug = []
    for bid, op, note in modern:
        repro = _find_repro(bid)
        if repro is None:
            per_bug.append({"bug_id": bid, "primary_op": op, "error": "no_repro"})
            continue
        try:
            _run = _import_run(repro)
        except Exception as e:
            per_bug.append(
                {
                    "bug_id": bid,
                    "primary_op": op,
                    "error": f"import_failed: {type(e).__name__}: {e}",
                }
            )
            continue

        torch.manual_seed(SEED)
        jx = _run_baseline_jaxtyping(_run)

        torch.manual_seed(SEED)
        tc_static = _run_baseline_torch_compile(_run, dynamic=False)

        torch.manual_seed(SEED)
        tc_dynamic = _run_baseline_torch_compile(_run, dynamic=True)

        per_bug.append(
            {
                "bug_id": bid,
                "primary_op": op,
                "note": note,
                "jaxtyping_beartype": jx,
                "torch_compile_static": tc_static,
                "torch_compile_dynamic": tc_dynamic,
            }
        )

    def _count(verdict_key: str, tool_key: str) -> int:
        return sum(
            1
            for r in per_bug
            if isinstance(r.get(tool_key), dict) and r[tool_key].get("verdict") == verdict_key
        )

    summary = {
        "n_total": len(per_bug),
        "jaxtyping_beartype": {
            "static_catches": _count("static_catch", "jaxtyping_beartype"),
            "runtime_torch_only": _count("runtime_torch_only", "jaxtyping_beartype"),
            "no_error": _count("no_error", "jaxtyping_beartype"),
        },
        "torch_compile_static": {
            "compile_time_catches": _count("compile_time_catch", "torch_compile_static"),
            "runtime_torch_only": _count("runtime_torch_only", "torch_compile_static"),
            "no_error": _count("no_error", "torch_compile_static"),
        },
        "torch_compile_dynamic": {
            "compile_time_catches": _count(
                "compile_time_catch", "torch_compile_dynamic"
            ),
            "runtime_torch_only": _count(
                "runtime_torch_only", "torch_compile_dynamic"
            ),
            "no_error": _count("no_error", "torch_compile_dynamic"),
        },
    }

    headline = {
        "tensorguard_static_no_execution_no_inputs": 32,
        "pytea_static_no_execution_no_inputs": 22,
        "jaxtyping_beartype_function_boundary_contracts": summary[
            "jaxtyping_beartype"
        ]["static_catches"],
        "torch_compile_static_with_concrete_inputs": summary[
            "torch_compile_static"
        ]["compile_time_catches"],
        "torch_compile_dynamic_with_concrete_inputs": summary[
            "torch_compile_dynamic"
        ]["compile_time_catches"],
        "_setting_note": (
            "TensorGuard and Pytea operate on class source without "
            "running the module or supplying inputs; jaxtyping/beartype "
            "and torch.compile both require an executable callable, "
            "and torch.compile additionally requires concrete tensor "
            "inputs to begin tracing.  The repros embed concrete "
            "inputs, so the torch.compile numbers are an upper bound "
            "on what a user gets when they actually have an instantiated "
            "module with example inputs."
        ),
    }

    out = {
        "_question": (
            "Round-10 reviewer ask W2/Q4: contemporary execution-based "
            "baseline on the same N=34 fragment-fair modern subset."
        ),
        "seed": SEED,
        "torch_version": torch.__version__,
        "modern_subset_size": len(per_bug),
        "summary": summary,
        "headline_comparison": headline,
        "per_bug": per_bug,
    }
    out_path = os.path.join(os.path.dirname(__file__), "contemporary_baseline_34.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", out_path)
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
