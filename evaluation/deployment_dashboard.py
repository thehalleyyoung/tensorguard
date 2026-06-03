"""Step 228 -- deployment release dashboard for backend gate regressions.

This dashboard is separate from the precision/recall dashboard: it tracks the
deployment release surfaces that can break independently of model-analysis
quality (quantization, export, compile, and distributed/sharded execution).

The committed JSON/Markdown artifacts are deterministic release manifests.  The
frozen baseline is a reviewed ratchet over supported backend outcomes: a
supported row cannot disappear, become unsupported, or move from ``passed`` to
``skipped``/``failed`` without a reviewable baseline diff.  ``--gate`` also runs
lightweight live smoke gates when the optional PyTorch backends are available;
runtime skips are reported explicitly but do not fail portability-only CI jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

JSON_PATH = os.path.join(HERE, "deployment_dashboard.json")
MD_PATH = os.path.join(HERE, "deployment_dashboard.md")
BASELINE_PATH = os.path.join(HERE, "deployment_dashboard_baseline.json")

CURRENT_RELEASE = "0.1.0-dev"
SUPPORTED_STATUSES = {"passed", "skipped", "failed"}


class BackendUnavailable(RuntimeError):
    """Raised when an optional deployment backend cannot run in this process."""


@dataclass(frozen=True)
class GateSpec:
    surface: str
    backend: str
    gate: str
    evidence: str
    status: str
    supported: bool
    required: bool
    description: str

    @property
    def key(self) -> str:
        return "|".join((CURRENT_RELEASE, self.surface, self.backend, self.gate))


GATE_SPECS: Tuple[GateSpec, ...] = (
    GateSpec(
        surface="quant",
        backend="torch.ao.quantization",
        gate="src.quantization_verify.verify_quantization_eager",
        evidence="live calibrated QuantStub -> Linear -> DeQuantStub prepared smoke",
        status="passed",
        supported=True,
        required=True,
        description=(
            "Prepared eager quantization modules keep calibrated observers, "
            "public float boundaries, and executable output shapes before a "
            "backend-specific conversion kernel is required."
        ),
    ),
    GateSpec(
        surface="export",
        backend="torch.export",
        gate="evaluation.deployment_gallery --gate post_export_torch_export",
        evidence="real-model gallery export gates",
        status="passed",
        supported=True,
        required=False,
        description=(
            "The deployment gallery verifies post-export graph capture for each "
            "real-model family when torch.export is available."
        ),
    ),
    GateSpec(
        surface="compile",
        backend="torch.dynamo",
        gate="evaluation.deployment_budgets --gate compile/after",
        evidence="deployment budget post-compile gates",
        status="passed",
        supported=True,
        required=False,
        description=(
            "Compile-capable hosts run TensorGuard after torch.compile/Dynamo "
            "capture and compare the result against latency/memory budgets."
        ),
    ),
    GateSpec(
        surface="distributed",
        backend="fsdp+dtensor-static",
        gate="src.distributed_verification.verify_distributed",
        evidence="FSDP world-size and parameter-sharding shape smoke",
        status="passed",
        supported=True,
        required=True,
        description=(
            "Distributed verification checks that sharded parameter layouts "
            "preserve the logical tensor shapes consumed by forward."
        ),
    ),
)


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _row(spec: GateSpec) -> Dict[str, object]:
    if spec.status not in SUPPORTED_STATUSES:
        raise ValueError(f"unsupported deployment dashboard status: {spec.status!r}")
    return {
        "key": spec.key,
        "release": CURRENT_RELEASE,
        "surface": spec.surface,
        "backend": spec.backend,
        "gate": spec.gate,
        "evidence": spec.evidence,
        "status": spec.status,
        "supported": spec.supported,
        "required": spec.required,
        "description": spec.description,
    }


def release_rows() -> List[Dict[str, object]]:
    rows = [_row(spec) for spec in GATE_SPECS]
    rows.sort(key=lambda row: (str(row["release"]), str(row["surface"]), str(row["backend"])))
    return rows


def summarize(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    materialized = list(rows)
    supported = [row for row in materialized if row["supported"]]
    by_surface: Dict[str, Dict[str, int]] = {}
    for row in materialized:
        bucket = by_surface.setdefault(
            str(row["surface"]),
            {"passed": 0, "skipped": 0, "failed": 0, "supported": 0},
        )
        bucket[str(row["status"])] += 1
        if row["supported"]:
            bucket["supported"] += 1
    return {
        "release": CURRENT_RELEASE,
        "total_rows": len(materialized),
        "supported_rows": len(supported),
        "supported_passed": sum(1 for row in supported if row["status"] == "passed"),
        "supported_failed": sum(1 for row in supported if row["status"] == "failed"),
        "supported_skipped": sum(1 for row in supported if row["status"] == "skipped"),
        "surfaces": by_surface,
    }


def manifest() -> Dict[str, object]:
    rows = release_rows()
    return {
        "meta": {
            "generated_by": "evaluation/deployment_dashboard.py",
            "command": "PYTHONPATH=. python3 evaluation/deployment_dashboard.py",
            "gate_command": "PYTHONPATH=. python3 evaluation/deployment_dashboard.py --gate",
            "note": (
                "Deterministic release dashboard for deployment backend outcomes. "
                "The baseline is a reviewed ratchet over supported rows; live "
                "smoke gates are environment-qualified and report optional skips."
            ),
        },
        "releases": [
            {
                "release": CURRENT_RELEASE,
                "rows": rows,
                "summary": summarize(rows),
            }
        ],
    }


def build_baseline(man: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    man = man or manifest()
    rows = _all_rows(man)
    return {
        "_doc": (
            "Frozen deployment release ratchet for evaluation/deployment_dashboard.py. "
            "Supported backend rows must not disappear, lose support, or regress "
            "from passed to skipped/failed. New supported rows require a reviewed "
            "baseline update."
        ),
        "rows": {
            str(row["key"]): {
                "release": row["release"],
                "surface": row["surface"],
                "backend": row["backend"],
                "gate": row["gate"],
                "status": row["status"],
                "supported": row["supported"],
                "required": row["required"],
                "evidence": row["evidence"],
            }
            for row in rows
        },
    }


def _all_rows(man: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for release in man.get("releases", []):  # type: ignore[union-attr]
        if not isinstance(release, dict):
            continue
        for row in release.get("rows", []):
            if isinstance(row, dict):
                rows.append(row)
    return rows


@dataclass(frozen=True)
class DashboardGateResult:
    ok: bool
    regressions: Tuple[str, ...]
    missing: Tuple[str, ...]
    unregistered_supported: Tuple[str, ...]


def _status_regressed(base_status: str, current_status: str) -> bool:
    if current_status not in SUPPORTED_STATUSES:
        return True
    if base_status == "passed":
        return current_status != "passed"
    if base_status == "skipped":
        return current_status == "failed"
    return False


def compare_to_baseline(
    current_manifest: Dict[str, object],
    baseline: Dict[str, object],
) -> DashboardGateResult:
    current = {str(row["key"]): row for row in _all_rows(current_manifest)}
    base = baseline.get("rows", {})
    if not isinstance(base, dict):
        raise ValueError("deployment dashboard baseline must contain a row map")

    missing: List[str] = []
    unregistered: List[str] = []
    regressions: List[str] = []

    for key, entry in sorted(base.items()):
        if not isinstance(entry, dict):
            regressions.append(f"{key}: malformed baseline row")
            continue
        if key not in current:
            missing.append(key)
            continue
        row = current[key]
        if entry.get("supported") and not row.get("supported"):
            regressions.append(f"{key}: supported -> unsupported")
        if entry.get("supported") and _status_regressed(
            str(entry.get("status")),
            str(row.get("status")),
        ):
            regressions.append(
                f"{key}: {entry.get('status')} -> {row.get('status')}"
            )

    for key, row in sorted(current.items()):
        if key not in base and row.get("supported"):
            unregistered.append(key)

    ok = not missing and not unregistered and not regressions
    return DashboardGateResult(
        ok=ok,
        regressions=tuple(regressions),
        missing=tuple(missing),
        unregistered_supported=tuple(unregistered),
    )


def load_baseline() -> Dict[str, object]:
    with open(BASELINE_PATH, "r") as fh:
        return json.load(fh)


def render_markdown(man: Dict[str, object], baseline: Optional[Dict[str, object]] = None) -> str:
    base_rows = (baseline or {}).get("rows", {})
    if not isinstance(base_rows, dict):
        base_rows = {}
    lines = [
        "# Deployment release dashboard",
        "",
        (
            "This tab tracks quantization, export, compile, and distributed gate "
            "outcomes per release. Supported rows are ratcheted by "
            "`deployment_dashboard_baseline.json`: a pass cannot silently become "
            "a skip/fail, and supported rows cannot disappear."
        ),
        "",
    ]
    for release in man["releases"]:  # type: ignore[index]
        lines.extend([
            f"## Release `{release['release']}`",
            "",
            "| Surface | Backend | Status | Supported | Required | Gate | Evidence |",
            "|---------|---------|--------|-----------|----------|------|----------|",
        ])
        for row in release["rows"]:
            base = base_rows.get(row["key"], {})
            status = row["status"]
            if base and base.get("status") != row["status"]:
                status = f"{base.get('status')} -> {row['status']}"
            lines.append(
                "| {surface} | `{backend}` | {status} | {supported} | {required} | `{gate}` | {evidence} |".format(
                    surface=row["surface"],
                    backend=row["backend"],
                    status=status,
                    supported="yes" if row["supported"] else "no",
                    required="yes" if row["required"] else "env-qualified",
                    gate=row["gate"],
                    evidence=row["evidence"],
                )
            )
        summary = release["summary"]
        lines.extend([
            "",
            (
                "**Summary.** {supported_passed}/{supported_rows} supported rows "
                "passed; failed={supported_failed}, skipped={supported_skipped}."
            ).format(**summary),
            "",
        ])
    return "\n".join(lines)


def _normalise_result(result: Any) -> Tuple[bool, List[str]]:
    if result is None:
        return False, ["TensorGuard returned no result"]
    if hasattr(result, "ok"):
        issues = getattr(result, "issues", ())
        messages = [getattr(issue, "message", str(issue)) for issue in issues]
        return bool(getattr(result, "ok")) and not messages, messages
    if hasattr(result, "safe"):
        errors = getattr(result, "errors", ())
        if callable(errors):
            errors = errors()
        messages = [getattr(err, "message", str(err)) for err in (errors or ())]
        return bool(getattr(result, "safe")) and not messages, messages
    return False, [f"unrecognized result type {type(result).__name__}"]


def _summarize_live_rows(rows: Sequence[Dict[str, object]]) -> Tuple[bool, List[str], str]:
    if not rows:
        raise BackendUnavailable("no gate rows were produced")
    failures = [row for row in rows if row.get("status") == "failed"]
    if failures:
        messages = []
        for row in failures:
            errors = row.get("errors", ())
            detail = "; ".join(str(err) for err in list(errors)[:3]) if isinstance(errors, list) else str(errors)
            messages.append(
                f"{row.get('model', row.get('backend'))} {row.get('phase')}: {detail or 'failed'}"
            )
        return False, messages, "failed"
    passed = [row for row in rows if row.get("status") == "passed"]
    if passed:
        return True, [], "passed"
    skipped = [row for row in rows if row.get("status") == "skipped"]
    if skipped:
        reasons = sorted({str(row.get("skip_reason", "backend unavailable")) for row in skipped})
        raise BackendUnavailable("; ".join(reasons))
    return False, ["unknown live gate status"], "failed"


def _run_quant_gate() -> Tuple[bool, List[str], str]:
    try:
        import torch
        import torch.nn as nn
        import torch.ao.quantization as tq
    except Exception as exc:
        raise BackendUnavailable(f"PyTorch quantization unavailable: {exc}") from exc

    from src.quantization_verify import verify_quantization_eager

    class QuantDashboardNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant = tq.QuantStub()
            self.fc = nn.Linear(4, 3)
            self.dequant = tq.DeQuantStub()

        def forward(self, x: Any) -> Any:
            return self.dequant(self.fc(self.quant(x)))

    engines = [engine for engine in torch.backends.quantized.supported_engines if engine != "none"]
    engine = "qnnpack" if "qnnpack" in engines else (engines[0] if engines else "x86")
    model = QuantDashboardNet().eval()
    model.qconfig = tq.get_default_qconfig(engine)
    prepared = tq.prepare(model, inplace=False)
    with torch.no_grad():
        prepared(torch.randn(2, 4))
    safe, errors = _normalise_result(verify_quantization_eager(prepared))
    with torch.no_grad():
        output = prepared(torch.randn(2, 4))
    if tuple(output.shape) != (2, 3):
        return False, [f"prepared quant output shape {tuple(output.shape)} != (2, 3)"], "failed"
    return safe, errors, "passed" if safe else "failed"


def _run_export_gate() -> Tuple[bool, List[str], str]:
    from evaluation import deployment_gallery

    rows = [
        row for row in deployment_gallery.measure()
        if row.get("backend") == "torch.export"
    ]
    return _summarize_live_rows(rows)


def _run_compile_gate() -> Tuple[bool, List[str], str]:
    from evaluation import deployment_budgets

    rows = [
        row for row in deployment_budgets.measure()
        if row.get("pipeline") == "compile"
        and row.get("phase") == "after"
        and row.get("backend") == "torch.dynamo"
    ]
    return _summarize_live_rows(rows)


def _run_distributed_gate() -> Tuple[bool, List[str], str]:
    from src.distributed_verification import FSDPConfig, verify_distributed

    source = """
import torch.nn as nn

class DashboardDistributedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
"""
    result = verify_distributed(
        source=source,
        input_shapes={"x": ("batch", 256)},
        fsdp_config=FSDPConfig(world_size=4),
    )
    errors: List[str] = []
    if result.fsdp_result is None:
        errors.append("FSDP result missing")
    elif not result.fsdp_result.safe:
        errors.extend(result.fsdp_result.violations)
    safe = bool(result.safe) and not errors
    return safe, errors, "passed" if safe else "failed"


def measure() -> List[Dict[str, object]]:
    runners = {
        "quant": _run_quant_gate,
        "export": _run_export_gate,
        "compile": _run_compile_gate,
        "distributed": _run_distributed_gate,
    }
    rows: List[Dict[str, object]] = []
    for spec in GATE_SPECS:
        row = _row(spec)
        try:
            safe, errors, live_status = runners[spec.surface]()
        except BackendUnavailable as exc:
            rows.append({
                **row,
                "live_status": "skipped",
                "safe": True,
                "errors": [],
                "skip_reason": str(exc),
            })
            continue
        except Exception as exc:
            rows.append({
                **row,
                "live_status": "failed",
                "safe": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            })
            continue
        rows.append({
            **row,
            "live_status": live_status,
            "safe": safe,
            "errors": errors,
        })
    return rows


def deployment_gate(run_live: bool = True) -> int:
    man = manifest()
    result = compare_to_baseline(man, load_baseline())
    problems: List[str] = []
    if result.regressions:
        problems.append("REGRESSIONS:\n  " + "\n  ".join(result.regressions))
    if result.missing:
        problems.append("baseline rows no longer produced: " + ", ".join(result.missing))
    if result.unregistered_supported:
        problems.append(
            "supported rows missing from baseline: "
            + ", ".join(result.unregistered_supported)
        )

    live_rows: List[Dict[str, object]] = []
    if run_live:
        live_rows = measure()
        live_failures = [
            row for row in live_rows
            if row.get("supported") and row.get("live_status") == "failed"
        ]
        for row in live_rows:
            flag = {
                "passed": "ok",
                "failed": "FAIL",
                "skipped": "skip",
            }.get(str(row.get("live_status")), str(row.get("live_status")))
            suffix = ""
            if row.get("live_status") == "skipped":
                suffix = f" ({row.get('skip_reason')})"
            elif row.get("errors"):
                suffix = ": " + "; ".join(str(err) for err in list(row["errors"])[:2])
            print(
                "  [{flag}] {surface:12s} {backend:24s}{suffix}".format(
                    flag=flag,
                    surface=str(row["surface"]),
                    backend=str(row["backend"]),
                    suffix=suffix,
                )
            )
        if live_failures:
            problems.append(
                "LIVE DEPLOYMENT FAILURES:\n  "
                + "\n  ".join(
                    f"{row['surface']} {row['backend']}: {row.get('errors')}"
                    for row in live_failures
                )
            )

    if problems:
        print("DEPLOYMENT DASHBOARD GATE FAILED")
        for problem in problems:
            print(problem)
        return 1
    print(
        "deployment dashboard gate PASS: %d supported release row(s)%s"
        % (
            int(man["releases"][0]["summary"]["supported_rows"]),  # type: ignore[index]
            ", live smoke checked" if run_live else "",
        )
    )
    return 0


def run(check: bool = False, update_baseline: bool = False, write: bool = True) -> int:
    man = manifest()
    text = _dumps(man)
    baseline = build_baseline(man)
    md = render_markdown(man, baseline)

    if update_baseline:
        if write:
            with open(JSON_PATH, "w") as fh:
                fh.write(text)
            with open(BASELINE_PATH, "w") as fh:
                fh.write(_dumps(baseline))
            with open(MD_PATH, "w") as fh:
                fh.write(md)
        print("deployment dashboard baseline updated: %d rows" % len(_all_rows(man)))
        return 0

    if check:
        problems = []
        if not os.path.exists(JSON_PATH) or open(JSON_PATH).read() != text:
            problems.append("deployment_dashboard.json is stale; run `make deployment-dashboard`")
        if not os.path.exists(BASELINE_PATH) or open(BASELINE_PATH).read() != _dumps(baseline):
            problems.append(
                "deployment_dashboard_baseline.json is stale; run `make deployment-dashboard`"
            )
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            problems.append("deployment_dashboard.md is stale; run `make deployment-dashboard`")
        gate_result = compare_to_baseline(man, baseline)
        if not gate_result.ok:
            problems.append("fresh deployment dashboard does not pass its own baseline")
        if problems:
            print("DEPLOYMENT DASHBOARD CHECK FAILED")
            for problem in problems:
                print(problem)
            return 1
        print("deployment dashboard artifacts up to date")
        return 0

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(md)
    print("deployment dashboard written: %d rows" % len(_all_rows(man)))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="assert committed dashboard artifacts are fresh")
    parser.add_argument("--update-baseline", action="store_true",
                        help="rewrite the reviewed deployment baseline")
    parser.add_argument("--gate", action="store_true",
                        help="compare against baseline and run live smoke gates")
    parser.add_argument("--no-live", action="store_true",
                        help="with --gate, skip environment-qualified live smoke")
    args = parser.parse_args(argv)
    if args.gate:
        return deployment_gate(run_live=not args.no_live)
    return run(check=args.check, update_baseline=args.update_baseline)


if __name__ == "__main__":
    raise SystemExit(main())
