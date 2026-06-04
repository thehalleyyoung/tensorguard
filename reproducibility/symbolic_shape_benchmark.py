#!/usr/bin/env python3
"""Symbolic input-shape benchmark for zero-example module verification.

The suite deliberately separates three outcomes:

* sound symbolic contracts for Conv/Norm-fronted models that must stay SAFE,
* downstream bugs made decidable by inferred symbolic ranks/channels, and
* rank-polymorphic Linear-first models where TensorGuard must abstain instead of
  guessing a concrete rank.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "reproducibility" / "symbolic_shape_benchmark.json"
OUT_MD = REPO / "reproducibility" / "symbolic_shape_benchmark.md"


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    source: str
    expected_verdict: str
    expected_inferred: Dict[str, Tuple]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    family: str
    expected_verdict: str
    verdict: str
    inferred_input_shapes: Dict[str, Tuple]
    passed: bool


def _conv_case(i: int, *, bug: bool) -> Case:
    channels = 1 + (i % 5)
    width = 4 + (i % 7)
    linear_in = width * (2 if bug else 1)
    source = f"""
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Conv2d({channels}, {width}, 3)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.head = nn.Linear({linear_in}, 3)
            def forward(self, x):
                x = self.stem(x)
                x = self.pool(x)
                x = x.flatten(1)
                return self.head(x)
    """
    suffix = "bug" if bug else "safe"
    return Case(
        case_id=f"conv2d_{suffix}_{i:02d}",
        family=f"conv2d_{suffix}",
        source=textwrap.dedent(source),
        expected_verdict="UNSAFE" if bug else "SAFE",
        expected_inferred={"x": ("batch", channels, "height", "width")},
    )


def _annotated_linear_case(i: int) -> Case:
    features = 4 + (i % 9)
    hidden = 6 + (i % 5)
    source = f"""
        import torch.nn as nn
        from jaxtyping import Float
        from torch import Tensor
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear({features}, {hidden})
                self.fc2 = nn.Linear({hidden + 2}, 2)
            def forward(self, x: Float[Tensor, "batch {features}"]):
                return self.fc2(self.fc1(x))
    """
    return Case(
        case_id=f"annotated_linear_bug_{i:02d}",
        family="annotated_linear_bug",
        source=textwrap.dedent(source),
        expected_verdict="UNSAFE",
        expected_inferred={"x": ("batch", features)},
    )


def _docstring_linear_case(i: int) -> Case:
    features = 5 + (i % 8)
    hidden = 7 + (i % 6)
    source = f'''
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear({features}, {hidden})
                self.fc2 = nn.Linear({hidden + 3}, 2)
            def forward(self, x):
                """Run the classifier.

                Args:
                    x: shape (batch, {features})
                """
                return self.fc2(self.fc1(x))
    '''
    return Case(
        case_id=f"docstring_linear_bug_{i:02d}",
        family="docstring_linear_bug",
        source=textwrap.dedent(source),
        expected_verdict="UNSAFE",
        expected_inferred={"x": ("batch", features)},
    )


def _ambiguous_linear_case(i: int) -> Case:
    features = 8 + (i % 11)
    source = f"""
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear({features}, 2)
            def forward(self, x):
                return self.fc(x)
    """
    return Case(
        case_id=f"linear_abstain_{i:02d}",
        family="linear_abstain",
        source=textwrap.dedent(source),
        expected_verdict="SAFE",
        expected_inferred={},
    )


def cases() -> List[Case]:
    out: List[Case] = []
    out.extend(_conv_case(i, bug=False) for i in range(30))
    out.extend(_conv_case(i, bug=True) for i in range(30))
    out.extend(_annotated_linear_case(i) for i in range(20))
    out.extend(_docstring_linear_case(i) for i in range(10))
    out.extend(_ambiguous_linear_case(i) for i in range(10))
    assert len(out) == 100
    return out


def _verdict(result: object) -> str:
    return "SAFE" if getattr(result, "safe", False) else "UNSAFE"


def run_cases(items: Iterable[Case]) -> List[CaseResult]:
    from src.model_checker import verify_model

    results: List[CaseResult] = []
    for case in items:
        result = verify_model(case.source)
        verdict = _verdict(result)
        inferred = dict(getattr(result, "inferred_input_shapes", {}) or {})
        passed = verdict == case.expected_verdict and inferred == case.expected_inferred
        results.append(
            CaseResult(
                case_id=case.case_id,
                family=case.family,
                expected_verdict=case.expected_verdict,
                verdict=verdict,
                inferred_input_shapes=inferred,
                passed=passed,
            )
        )
    return results


def _summary(results: List[CaseResult]) -> Dict[str, object]:
    families: Dict[str, Dict[str, int]] = {}
    for result in results:
        bucket = families.setdefault(result.family, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)
    return {
        "total": len(results),
        "passed": sum(int(r.passed) for r in results),
        "families": families,
    }


def _render_json(results: List[CaseResult]) -> str:
    payload = {
        "benchmark": "symbolic_shape_benchmark",
        "summary": _summary(results),
        "cases": [asdict(r) for r in results],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _render_md(results: List[CaseResult]) -> str:
    summary = _summary(results)
    lines = [
        "# Symbolic shape benchmark",
        "",
        f"Total cases: **{summary['passed']} / {summary['total']}** passed.",
        "",
        "| Family | Passed | Total |",
        "| --- | ---: | ---: |",
    ]
    for family, stats in sorted(summary["families"].items()):
        lines.append(f"| `{family}` | {stats['passed']} | {stats['total']} |")
    lines.extend(
        [
            "",
            "This benchmark verifies modules without concrete `input_shapes`: Conv2d",
            "front-ends infer symbolic `(batch, channels, height, width)` contracts,",
            "shape-annotated and docstring-documented Linear models use symbolic API",
            "contracts, and ambiguous Linear-first modules prove TensorGuard abstains",
            "instead of guessing rank.",
            "",
        ]
    )
    return "\n".join(lines)


def run(*, check: bool = False) -> int:
    results = run_cases(cases())
    js = _render_json(results)
    md = _render_md(results)
    if any(not result.passed for result in results):
        for result in results:
            if not result.passed:
                print(f"FAILED: {result.case_id}", file=sys.stderr)
        return 1
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}", file=sys.stderr)
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}", file=sys.stderr)
            ok = False
        if ok:
            print("symbolic_shape_benchmark: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"symbolic_shape_benchmark: {_summary(results)['passed']} / {len(results)} passed")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
