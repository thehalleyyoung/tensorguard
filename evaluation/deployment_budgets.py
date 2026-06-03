"""Step 226 -- deployment latency/memory budgets for release gates.

This harness exercises TensorGuard in the places deployment teams actually need
it: immediately before export/compile, and again after the deployment frontend
has captured a graph.  It follows the repository's reproducibility pattern:

* the committed JSON/Markdown artifacts are deterministic budget manifests;
* `--gate` runs real TensorGuard frontends on real `nn.Module` instances and
  fails when a supported backend is unsafe, over latency budget, or over memory
  budget;
* optional compile/export backends are reported as explicit skips when the
  running Python/PyTorch build cannot support them.

The memory budget is the verifier-stage peak observed by Python's allocator
plus any process-RSS high-water increase visible through `resource`; this is a
release-gate budget for TensorGuard's deployment checks, not a model parameter
or CUDA allocator estimate.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

JSON_PATH = os.path.join(HERE, "deployment_budgets.json")
MD_PATH = os.path.join(HERE, "deployment_budgets.md")


class BackendUnavailable(RuntimeError):
    """Raised when an optional deployment backend cannot run in this process."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: str
    input_shapes: Dict[str, Tuple[int, ...]]
    parameter_count: int
    description: str


@dataclass(frozen=True)
class StageSpec:
    pipeline: str
    phase: str
    backend: str
    gate: str
    profile: str
    latency_budget_s: float
    memory_budget_mb: float
    description: str


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        name="tiny_mlp_classifier",
        tier="tabular-small",
        input_shapes={"x": (2, 16)},
        parameter_count=874,
        description="Two-layer ReLU MLP used as the fast interactive gate.",
    ),
    ModelSpec(
        name="tiny_vision_classifier",
        tier="vision-small",
        input_shapes={"x": (1, 3, 16, 16)},
        parameter_count=269,
        description="Conv/ReLU/pool/linear classifier with exportable image ops.",
    ),
)


STAGE_SPECS: Tuple[StageSpec, ...] = (
    StageSpec(
        pipeline="export",
        phase="before",
        backend="fx",
        gate="src.fx_extractor.verify_module(backend='fx')",
        profile="interactive",
        latency_budget_s=6.0,
        memory_budget_mb=96.0,
        description="TensorGuard FX gate before handing the module to export.",
    ),
    StageSpec(
        pipeline="export",
        phase="after",
        backend="torch.export",
        gate="src.export_extractor.verify_module_export",
        profile="balanced",
        latency_budget_s=18.0,
        memory_budget_mb=192.0,
        description="TensorGuard export frontend after torch.export graph capture.",
    ),
    StageSpec(
        pipeline="compile",
        phase="before",
        backend="fx",
        gate="src.fx_extractor.verify_module(backend='fx')",
        profile="memory_capped",
        latency_budget_s=8.0,
        memory_budget_mb=80.0,
        description="TensorGuard FX gate before compile-specialization begins.",
    ),
    StageSpec(
        pipeline="compile",
        phase="after",
        backend="torch.dynamo",
        gate="src.dynamo_extractor.verify_module_dynamo",
        profile="compile_cold_start",
        latency_budget_s=35.0,
        memory_budget_mb=320.0,
        description="TensorGuard Dynamo gate for the graph captured by compile.",
    ),
)


BACKEND_PROFILES: Dict[str, Tuple[Dict[str, object], ...]] = {
    "fx": (
        {
            "profile": "interactive",
            "latency_budget_s": 6.0,
            "memory_budget_mb": 96.0,
            "use": "pre-export local feedback",
        },
        {
            "profile": "memory_capped",
            "latency_budget_s": 8.0,
            "memory_budget_mb": 80.0,
            "use": "pre-compile constrained CI runner",
        },
    ),
    "torch.export": (
        {
            "profile": "fast_export",
            "latency_budget_s": 12.0,
            "memory_budget_mb": 240.0,
            "use": "release smoke on beefy runners",
        },
        {
            "profile": "balanced",
            "latency_budget_s": 18.0,
            "memory_budget_mb": 192.0,
            "use": "default post-export release gate",
        },
    ),
    "torch.dynamo": (
        {
            "profile": "fast_compile",
            "latency_budget_s": 24.0,
            "memory_budget_mb": 420.0,
            "use": "warm compile-capable host",
        },
        {
            "profile": "compile_cold_start",
            "latency_budget_s": 35.0,
            "memory_budget_mb": 320.0,
            "use": "default cold compile release gate",
        },
    ),
}


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _dominates(left: Dict[str, object], right: Dict[str, object]) -> bool:
    left_latency = float(left["latency_budget_s"])
    left_memory = float(left["memory_budget_mb"])
    right_latency = float(right["latency_budget_s"])
    right_memory = float(right["memory_budget_mb"])
    return (
        left_latency <= right_latency
        and left_memory <= right_memory
        and (left_latency < right_latency or left_memory < right_memory)
    )


def pareto_frontier(points: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return the non-dominated latency/memory budget frontier."""
    rows = [dict(point) for point in points]
    frontier = [
        row for row in rows
        if not any(_dominates(other, row) for other in rows if other is not row)
    ]
    return sorted(
        frontier,
        key=lambda row: (float(row["latency_budget_s"]), float(row["memory_budget_mb"])),
    )


def backend_pareto_curves() -> Dict[str, List[Dict[str, object]]]:
    return {
        backend: pareto_frontier(points)
        for backend, points in sorted(BACKEND_PROFILES.items())
    }


def budget_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in MODEL_SPECS:
        for stage in STAGE_SPECS:
            rows.append({
                "model": model.name,
                "tier": model.tier,
                "pipeline": stage.pipeline,
                "phase": stage.phase,
                "backend": stage.backend,
                "gate": stage.gate,
                "profile": stage.profile,
                "latency_budget_s": stage.latency_budget_s,
                "memory_budget_mb": stage.memory_budget_mb,
            })
    rows.sort(key=lambda row: (
        str(row["model"]), str(row["pipeline"]), str(row["phase"]),
        str(row["backend"]),
    ))
    return rows


def manifest() -> Dict[str, object]:
    return {
        "meta": {
            "generated_by": "evaluation/deployment_budgets.py",
            "command": "PYTHONPATH=. python3 evaluation/deployment_budgets.py",
            "gate_command": (
                "PYTHONPATH=. python3 evaluation/deployment_budgets.py --gate"
            ),
            "note": (
                "Manifest records deterministic deployment gate budgets only; "
                "wall-clock latency and verifier-stage memory are machine-"
                "dependent and checked live by --gate. Compile rows are explicit "
                "skips on Python/PyTorch builds where torch.compile/Dynamo is "
                "not supported."
            ),
        },
        "models": [
            {
                "model": model.name,
                "tier": model.tier,
                "input_shapes": {
                    name: list(shape) for name, shape in sorted(model.input_shapes.items())
                },
                "parameter_count": model.parameter_count,
                "description": model.description,
            }
            for model in MODEL_SPECS
        ],
        "budget_rows": budget_rows(),
        "backend_pareto_curves": backend_pareto_curves(),
    }


def render_markdown(man: Dict[str, object]) -> str:
    lines = [
        "# Deployment latency and memory budgets",
        "",
        (
            "Release gates run TensorGuard before and after export/compile. "
            "The committed artifact stores deterministic latency/memory budgets; "
            "`make deployment-budgets-gate` measures real wall-clock latency and "
            "verifier-stage memory live."
        ),
        "",
        "## Gate matrix",
        "",
        "| Model | Pipeline | Phase | Backend | Latency budget (s) | Memory budget (MB) |",
        "|-------|----------|-------|---------|--------------------|--------------------|",
    ]
    for row in man["budget_rows"]:
        lines.append(
            "| `{model}` | {pipeline} | {phase} | `{backend}` | {latency:.1f} | {memory:.1f} |".format(
                model=row["model"],
                pipeline=row["pipeline"],
                phase=row["phase"],
                backend=row["backend"],
                latency=row["latency_budget_s"],
                memory=row["memory_budget_mb"],
            )
        )
    lines.extend(["", "## Per-backend Pareto budget curves", ""])
    for backend, curve in man["backend_pareto_curves"].items():
        lines.append(f"### `{backend}`")
        lines.append("")
        lines.append("| Profile | Latency budget (s) | Memory budget (MB) | Use |")
        lines.append("|---------|--------------------|--------------------|-----|")
        for point in curve:
            lines.append(
                "| `{profile}` | {latency:.1f} | {memory:.1f} | {use} |".format(
                    profile=point["profile"],
                    latency=point["latency_budget_s"],
                    memory=point["memory_budget_mb"],
                    use=point["use"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def _shape_to_example(shape: Sequence[int]) -> Any:
    import torch

    return torch.randn(*shape)


def _build_model(model_name: str) -> Tuple[Any, Tuple[Any, ...]]:
    import torch
    import torch.nn as nn

    if model_name == "tiny_mlp_classifier":
        class TinyMLPClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 10)

            def forward(self, x: Any) -> Any:
                return self.fc2(torch.relu(self.fc1(x)))

        return TinyMLPClassifier().eval(), (_shape_to_example((2, 16)),)

    if model_name == "tiny_vision_classifier":
        class TinyVisionClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(8, 5)

            def forward(self, x: Any) -> Any:
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)

        return TinyVisionClassifier().eval(), (_shape_to_example((1, 3, 16, 16)),)

    raise ValueError(f"unknown deployment budget model: {model_name}")


def _normalise_result(result: Any) -> Tuple[bool, List[str]]:
    if result is None:
        return False, ["TensorGuard abstained without a verification result"]

    errors_attr = getattr(result, "errors", None)
    if callable(errors_attr):
        raw_errors = errors_attr()
    else:
        raw_errors = errors_attr or []
    errors = [
        getattr(err, "message", str(err)) for err in raw_errors
    ]

    safe_attr = getattr(result, "safe", None)
    if isinstance(safe_attr, bool):
        return safe_attr and not errors, errors
    if isinstance(safe_attr, str):
        return safe_attr.upper() == "SAFE" and not errors, errors

    verdict = getattr(result, "verdict", None)
    if verdict is not None:
        verdict_text = str(verdict).upper()
        bugs = getattr(result, "bugs", None) or []
        bug_messages = [getattr(bug, "message", str(bug)) for bug in bugs]
        return verdict_text == "SAFE" and not bug_messages, errors + bug_messages

    return False, [f"unrecognized TensorGuard result type: {type(result).__name__}"]


def _rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _run_timed(fn: Callable[[], Tuple[bool, List[str]]]) -> Dict[str, object]:
    rss_before = _rss_mb()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        safe, errors = fn()
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    elapsed = time.perf_counter() - t0
    rss_delta = max(0.0, _rss_mb() - rss_before)
    python_peak_mb = max(current, peak) / (1024.0 * 1024.0)
    return {
        "safe": safe,
        "errors": errors,
        "latency_s": round(elapsed, 3),
        "memory_mb": round(max(python_peak_mb, rss_delta), 3),
        "python_peak_mb": round(python_peak_mb, 3),
        "rss_delta_mb": round(rss_delta, 3),
    }


def _compile_supported(torch_mod: Any) -> Tuple[bool, Optional[str]]:
    if not hasattr(torch_mod, "compile"):
        return False, "torch.compile is unavailable"
    try:
        import torch._dynamo  # noqa: F401
    except Exception as exc:
        return False, f"TorchDynamo import failed: {exc}"
    return True, None


def _measure_stage(row: Dict[str, object], model: Any, examples: Tuple[Any, ...]) -> Dict[str, object]:
    model_spec = next(spec for spec in MODEL_SPECS if spec.name == row["model"])
    input_shapes = {
        name: tuple(shape) for name, shape in model_spec.input_shapes.items()
    }
    backend = str(row["backend"])
    pipeline = str(row["pipeline"])
    phase = str(row["phase"])

    def run_fx() -> Tuple[bool, List[str]]:
        from src.fx_extractor import verify_module

        return _normalise_result(
            verify_module(model, input_shapes=input_shapes, backend="fx")
        )

    def run_export() -> Tuple[bool, List[str]]:
        from src.export_extractor import HAS_EXPORT, verify_module_export

        if not HAS_EXPORT:
            raise BackendUnavailable("torch.export is unavailable")
        return _normalise_result(
            verify_module_export(
                model,
                input_shapes=input_shapes,
                example_inputs=examples,
            )
        )

    def run_dynamo() -> Tuple[bool, List[str]]:
        import torch
        from src.dynamo_extractor import HAS_DYNAMO, verify_module_dynamo

        ok, reason = _compile_supported(torch)
        if not ok:
            raise BackendUnavailable(reason or "torch.compile is unavailable")
        if not HAS_DYNAMO:
            raise BackendUnavailable("TorchDynamo is unavailable")
        compiled = torch.compile(model, backend="eager")
        with torch.no_grad():
            compiled(*examples)
        return _normalise_result(
            verify_module_dynamo(
                model,
                input_shapes=input_shapes,
                example_inputs=examples,
                fallback_to_fx=False,
            )
        )

    stage_fn: Callable[[], Tuple[bool, List[str]]]
    if backend == "fx" and phase == "before" and pipeline in {"export", "compile"}:
        stage_fn = run_fx
    elif backend == "torch.export" and pipeline == "export" and phase == "after":
        stage_fn = run_export
    elif backend == "torch.dynamo" and pipeline == "compile" and phase == "after":
        stage_fn = run_dynamo
    else:
        raise ValueError(f"unsupported deployment budget row: {row}")

    try:
        measured = _run_timed(stage_fn)
    except BackendUnavailable as exc:
        return {
            **row,
            "status": "skipped",
            "skip_reason": str(exc),
            "within_budget": True,
        }
    except Exception as exc:
        return {
            **row,
            "status": "failed",
            "safe": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "latency_s": 0.0,
            "memory_mb": 0.0,
            "within_budget": False,
        }

    latency_ok = float(measured["latency_s"]) <= float(row["latency_budget_s"])
    memory_ok = float(measured["memory_mb"]) <= float(row["memory_budget_mb"])
    safe = bool(measured["safe"])
    return {
        **row,
        **measured,
        "status": "passed" if safe and latency_ok and memory_ok else "failed",
        "within_budget": safe and latency_ok and memory_ok,
    }


def measure() -> List[Dict[str, object]]:
    """Run the live deployment budget gate rows; timings are not committed."""
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return [
            {
                **row,
                "status": "skipped",
                "skip_reason": f"PyTorch unavailable: {exc}",
                "within_budget": True,
            }
            for row in budget_rows()
        ]

    rows: List[Dict[str, object]] = []
    rows_by_model: Dict[str, List[Dict[str, object]]] = {}
    for row in budget_rows():
        rows_by_model.setdefault(str(row["model"]), []).append(row)

    for model_name, model_rows in sorted(rows_by_model.items()):
        model, examples = _build_model(model_name)
        for row in model_rows:
            rows.append(_measure_stage(row, model, examples))
    return rows


def gate() -> int:
    rows = measure()
    failures = [row for row in rows if row["status"] == "failed"]
    skipped = [row for row in rows if row["status"] == "skipped"]
    passed = [row for row in rows if row["status"] == "passed"]

    for row in rows:
        if row["status"] == "skipped":
            print(
                "  [skip] {model:24s} {pipeline}/{phase:<6s} {backend:12s} {reason}".format(
                    model=str(row["model"]),
                    pipeline=str(row["pipeline"]),
                    phase=str(row["phase"]),
                    backend=str(row["backend"]),
                    reason=str(row["skip_reason"]),
                )
            )
            continue
        flag = "ok" if row["status"] == "passed" else "FAIL"
        print(
            "  [{flag}] {model:24s} {pipeline}/{phase:<6s} {backend:12s} "
            "{latency:.3f}s/{latency_budget:.1f}s "
            "{memory:.1f}MB/{memory_budget:.1f}MB".format(
                flag=flag,
                model=str(row["model"]),
                pipeline=str(row["pipeline"]),
                phase=str(row["phase"]),
                backend=str(row["backend"]),
                latency=float(row.get("latency_s", 0.0)),
                latency_budget=float(row["latency_budget_s"]),
                memory=float(row.get("memory_mb", 0.0)),
                memory_budget=float(row["memory_budget_mb"]),
            )
        )

    if failures:
        print("DEPLOYMENT BUDGET GATE FAILED: %d row(s)" % len(failures))
        for row in failures:
            details = "; ".join(str(err) for err in row.get("errors", [])[:3])
            print(
                "  - {model} {pipeline}/{phase} {backend}: {details}".format(
                    model=row["model"],
                    pipeline=row["pipeline"],
                    phase=row["phase"],
                    backend=row["backend"],
                    details=details or "over budget",
                )
            )
        return 1
    print(
        "deployment budget gate PASS: %d checked, %d skipped optional backend row(s)"
        % (len(passed), len(skipped))
    )
    return 0


def run(check: bool = False, write: bool = True) -> int:
    man = manifest()
    text = _dumps(man)

    if check:
        if not os.path.exists(JSON_PATH):
            print("deployment_budgets.json missing; run the harness first")
            return 1
        if open(JSON_PATH).read() != text:
            print("deployment_budgets.json is stale; run `make deployment-budgets`")
            return 1
        md = render_markdown(man)
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            print("deployment_budgets.md is stale; run `make deployment-budgets`")
            return 1
        print("deployment budgets manifest up to date")
        return 0

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(render_markdown(man))
    print(
        "deployment budgets manifest written: %d gate rows, %d backend curves"
        % (len(man["budget_rows"]), len(man["backend_pareto_curves"]))
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed manifest is byte-identical.")
    ap.add_argument("--gate", action="store_true",
                    help="Run live deployment latency/memory budget checks.")
    args = ap.parse_args()
    if args.gate:
        return gate()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
