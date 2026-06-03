#!/usr/bin/env python3
"""Step 266: negative controls where runtime value checks should beat TensorGuard.

TensorGuard's public contract is about tensor-structure facts: shape, device,
dtype, phase, stride/permutation, and gradient-flow.  It is not a value-domain
runtime monitor.  This harness deliberately builds value-dependent PyTorch
failures that are structurally well-formed, then reports the honest result:
TensorGuard should not catch them, a plain runtime smoke test may not catch
silent NaN/Inf results either, and an explicit finite-output runtime checker
does catch them on crafted concrete inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
from typing import Any, Dict, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_JSON = os.path.join(THIS_DIR, "negative_controls.json")
OUT_MD = os.path.join(THIS_DIR, "negative_controls.md")

_HEADER = "import torch\nimport torch.nn as nn\n\n\n"


def _case(
    case_id: str,
    family: str,
    forward: str,
    trigger_values: List[List[float]],
    *,
    expected_runtime_signal: str,
) -> Dict[str, Any]:
    source = _HEADER + (
        "class ValueBugModule(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n\n"
        "    def forward(self, x):\n"
        f"{forward}\n"
    )
    return {
        "id": case_id,
        "family": family,
        "source": source,
        "input_shapes": {"x": [len(trigger_values), len(trigger_values[0])]},
        "trigger_values": trigger_values,
        "expected_runtime_signal": expected_runtime_signal,
        "out_of_contract_dimension": "value_semantics",
    }


def build_corpus() -> List[Dict[str, Any]]:
    return [
        _case(
            "log_negative_nan",
            "nonfinite_value",
            "        return torch.log(x)\n",
            [[-1.0, -0.5], [2.0, 3.0]],
            expected_runtime_signal="nonfinite_output",
        ),
        _case(
            "sqrt_negative_nan",
            "nonfinite_value",
            "        return torch.sqrt(x)\n",
            [[4.0, -9.0], [16.0, 25.0]],
            expected_runtime_signal="nonfinite_output",
        ),
        _case(
            "divide_by_zero_inf",
            "nonfinite_value",
            "        return x / (x - x)\n",
            [[1.0, 2.0], [3.0, 4.0]],
            expected_runtime_signal="nonfinite_output",
        ),
        _case(
            "reciprocal_zero_inf",
            "nonfinite_value",
            "        return torch.reciprocal(x)\n",
            [[0.0, 2.0], [3.0, 4.0]],
            expected_runtime_signal="nonfinite_output",
        ),
        _case(
            "assert_positive_min",
            "value_assertion",
            "        assert torch.all(x > 0), 'expected positive activations'\n"
            "        return x\n",
            [[1.0, -1.0], [2.0, 3.0]],
            expected_runtime_signal="exception:AssertionError",
        ),
        _case(
            "assert_probability_range",
            "value_assertion",
            "        assert torch.all((x >= 0) & (x <= 1)), 'expected probabilities'\n"
            "        return x\n",
            [[0.25, 1.25], [0.5, 0.75]],
            expected_runtime_signal="exception:AssertionError",
        ),
    ]


def _load_module(source: str):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(source)
    tmp.close()
    try:
        spec = importlib.util.spec_from_file_location("tg_negative_control", tmp.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        os.unlink(tmp.name)


def trigger_tensor(case: Dict[str, Any]):
    import torch

    return torch.tensor(case["trigger_values"], dtype=torch.float32)


def runtime_smoke(case: Dict[str, Any]) -> Tuple[bool, str]:
    """Plain eager forward: catches exceptions, not silent non-finite outputs."""
    import torch

    mod = _load_module(case["source"])
    model = mod.ValueBugModule()
    try:
        with torch.no_grad():
            model(trigger_tensor(case))
    except Exception as exc:
        return True, f"exception:{type(exc).__name__}"
    return False, "ran_ok"


def runtime_finite_output_check(case: Dict[str, Any]) -> Tuple[bool, str]:
    """Runtime checker with an explicit finite-output assertion."""
    import torch

    mod = _load_module(case["source"])
    model = mod.ValueBugModule()
    try:
        with torch.no_grad():
            out = model(trigger_tensor(case))
    except Exception as exc:
        return True, f"exception:{type(exc).__name__}"
    tensors = [out] if isinstance(out, torch.Tensor) else list(out)
    if any(not torch.isfinite(t).all().item() for t in tensors if isinstance(t, torch.Tensor)):
        return True, "nonfinite_output"
    return False, "finite_output"


def tensorguard_result(case: Dict[str, Any]) -> Dict[str, Any]:
    from src.api import verify_architecture

    shapes = {k: tuple(v) for k, v in case["input_shapes"].items()}
    result = verify_architecture(
        case["source"],
        input_shapes=shapes,
        check_devices=True,
        check_gradients=True,
        max_cegar_iterations=0,
        soundness_mode="sound",
    )
    return {
        "caught": result.bug_count > 0,
        "bug_count": result.bug_count,
        "verdict": result.verdict,
        "unknown_reasons": list(result.unknown_reasons),
    }


def is_genuine_negative_control(case: Dict[str, Any]) -> Tuple[bool, str]:
    finite_buggy, finite_detail = runtime_finite_output_check(case)
    if not finite_buggy:
        return False, f"finite_checker_missed:{finite_detail}"
    expected = case["expected_runtime_signal"]
    if expected == finite_detail or (
        expected.startswith("exception:") and finite_detail == expected
    ):
        return True, finite_detail
    return False, f"expected_{expected}_got_{finite_detail}"


def _rate(caught: int, total: int) -> float:
    return round(caught / total, 4) if total else math.nan


def run(check: bool = False) -> Dict[str, Any]:
    corpus = build_corpus()
    per_case: List[Dict[str, Any]] = []
    tg_caught = smoke_caught = finite_caught = 0

    for case in corpus:
        genuine, gdetail = is_genuine_negative_control(case)
        if not genuine:
            raise AssertionError(f"{case['id']} is not a genuine control: {gdetail}")
        smoke_buggy, smoke_detail = runtime_smoke(case)
        finite_buggy, finite_detail = runtime_finite_output_check(case)
        tg = tensorguard_result(case)

        tg_caught += int(tg["caught"])
        smoke_caught += int(smoke_buggy)
        finite_caught += int(finite_buggy)

        per_case.append({
            "id": case["id"],
            "family": case["family"],
            "out_of_contract_dimension": case["out_of_contract_dimension"],
            "genuine_runtime_signal": gdetail,
            "tensorguard": tg,
            "runtime_smoke": {"caught": smoke_buggy, "detail": smoke_detail},
            "runtime_finite_output_check": {
                "caught": finite_buggy,
                "detail": finite_detail,
            },
        })

    total = len(corpus)
    by_family: Dict[str, Dict[str, int]] = {}
    for row in per_case:
        fam = by_family.setdefault(
            row["family"],
            {"total": 0, "tensorguard_caught": 0, "runtime_smoke_caught": 0,
             "runtime_finite_output_check_caught": 0},
        )
        fam["total"] += 1
        fam["tensorguard_caught"] += int(row["tensorguard"]["caught"])
        fam["runtime_smoke_caught"] += int(row["runtime_smoke"]["caught"])
        fam["runtime_finite_output_check_caught"] += int(
            row["runtime_finite_output_check"]["caught"]
        )

    artifact = {
        "meta": {
            "step": 266,
            "generated_by": "evaluation/negative_controls.py",
            "command": "python3 evaluation/negative_controls.py",
            "design": (
                "value-dependent negative controls with crafted concrete inputs; "
                "TensorGuard is expected not to beat runtime value checks because "
                "value semantics are outside its declared tensor-structure contract"
            ),
            "tensorguard_scope": (
                "shape/device/dtype/phase/stride/permutation/gradient-flow; "
                "not arbitrary value-domain assertions or finite-output monitoring"
            ),
            "headline_method": "runtime_finite_output_check",
        },
        "summary": {
            "n_cases": total,
            "tensorguard_caught": tg_caught,
            "tensorguard_recall": _rate(tg_caught, total),
            "runtime_smoke_caught": smoke_caught,
            "runtime_smoke_recall": _rate(smoke_caught, total),
            "runtime_finite_output_check_caught": finite_caught,
            "runtime_finite_output_check_recall": _rate(finite_caught, total),
            "honest_outcome": "TensorGuard loss to explicit runtime finite-output checking",
        },
        "by_family": by_family,
        "per_case": per_case,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit(f"missing {OUT_JSON}; run without --check first")
        with open(OUT_JSON, encoding="utf-8") as fh:
            if fh.read() != text:
                raise SystemExit("negative_controls.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(a: Dict[str, Any]) -> str:
    s = a["summary"]
    lines = [
        "# Step 266 -- negative controls: where runtime value checks should win",
        "",
        "This suite deliberately uses value-dependent failures outside TensorGuard's "
        "declared tensor-structure contract.  The inputs are crafted concrete "
        "tensors, not random smoke-test draws, because the point is to show the "
        "boundary honestly.",
        "",
        "## Recall on value-domain controls",
        "",
        "| Detector | Caught | Recall | Interpretation |",
        "| --- | ---: | ---: | --- |",
        "| TensorGuard (`sound` mode) | %d / %d | %.3f | expected loss: value semantics are out of contract |"
        % (s["tensorguard_caught"], s["n_cases"], s["tensorguard_recall"]),
        "| Runtime smoke test | %d / %d | %.3f | catches assertions, misses silent NaN/Inf outputs |"
        % (s["runtime_smoke_caught"], s["n_cases"], s["runtime_smoke_recall"]),
        "| Runtime finite-output check | %d / %d | %.3f | explicit value monitor on crafted inputs |"
        % (
            s["runtime_finite_output_check_caught"],
            s["n_cases"],
            s["runtime_finite_output_check_recall"],
        ),
        "",
        "## By family",
        "",
        "| Family | Cases | TensorGuard | Smoke test | Finite-output check |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for fam in sorted(a["by_family"]):
        row = a["by_family"][fam]
        lines.append(
            "| `%s` | %d | %d | %d | %d |"
            % (
                fam,
                row["total"],
                row["tensorguard_caught"],
                row["runtime_smoke_caught"],
                row["runtime_finite_output_check_caught"],
            )
        )
    lines.extend([
        "",
        "## Per-case signals",
        "",
        "| Case | Family | Runtime signal | TG verdict |",
        "| --- | --- | --- | --- |",
    ])
    for row in a["per_case"]:
        lines.append(
            "| `%s` | `%s` | `%s` | `%s`, bugs=%d |"
            % (
                row["id"],
                row["family"],
                row["genuine_runtime_signal"],
                row["tensorguard"]["verdict"],
                row["tensorguard"]["bug_count"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    artifact = run(check=args.check)
    if args.check:
        print("negative_controls.json is up to date")
        return
    s = artifact["summary"]
    print("Wrote %s and %s" % (os.path.relpath(OUT_JSON, REPO_ROOT),
                               os.path.relpath(OUT_MD, REPO_ROOT)))
    print(
        "  controls: %d | TG: %d | smoke: %d | finite-output: %d"
        % (
            s["n_cases"],
            s["tensorguard_caught"],
            s["runtime_smoke_caught"],
            s["runtime_finite_output_check_caught"],
        )
    )


if __name__ == "__main__":
    main()
