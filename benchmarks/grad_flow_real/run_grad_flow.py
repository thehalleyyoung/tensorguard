"""Gradient-flow real corpus runner.

Runs verify_architecture(check_gradients=True) on each Python snippet in
this directory, assigns a verdict of REFUTED-PROOF when the analyser flags
at least one error-level bug, and writes grad_flow_results.json.

Exit 0 iff >= 5 of the 6 snippets receive verdict == "REFUTED-PROOF".

Usage::

    python3 benchmarks/grad_flow_real/run_grad_flow.py

Artifact written::

    benchmarks/grad_flow_real/grad_flow_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture  # noqa: E402

HERE    = Path(__file__).resolve().parent
OUT     = HERE / "grad_flow_results.json"
EXPECTED = json.loads((HERE / "expected.json").read_text())

# Fixed input shapes used for each snippet (B=2, seq/feature as written).
INPUT_SHAPES: dict[str, dict] = {
    "snippet_01_detach_before_head.py":               {"x": (2, 16)},
    "snippet_02_gradient_checkpoint.py":              {"x": (2, 16)},
    "snippet_03_double_detach_residual.py":           {"x": (2, 8)},
    "snippet_04_attention_keys_detach.py":            {"x": (2, 4, 8)},
    "snippet_05_finetune_detach_before_trainable.py": {"x": (2, 16)},
    "snippet_06_checkpoint_deep_net.py":              {"x": (2, 16)},
}

PASS_THRESHOLD = 5   # at least 5/6 must be REFUTED-PROOF


def _run_snippet(path: Path) -> dict:
    src    = path.read_text()
    shapes = INPUT_SHAPES.get(path.name, {"x": (2, 8)})
    expected = EXPECTED.get(path.name, {}).get("expected_verdict", "REFUTED-PROOF")

    t0 = time.perf_counter()
    try:
        result = verify_architecture(
            src,
            input_shapes=shapes,
            check_gradients=True,
            check_devices=False,
            check_phases=False,
            high_confidence_only=False,
            max_cegar_iterations=1,
        )
        has_error = any(b.severity == "error" for b in result.bugs)
        verdict   = "REFUTED-PROOF" if has_error else "VERIFIED"
        error_msgs = [b.message[:120] for b in result.bugs if b.severity == "error"]
    except Exception as exc:
        verdict    = "TOOL-ERROR"
        error_msgs = [f"{type(exc).__name__}: {exc}"]

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    ok = verdict == expected

    return {
        "file":             path.name,
        "verdict":          verdict,
        "expected_verdict": expected,
        "match":            ok,
        "error_messages":   error_msgs,
        "duration_ms":      duration_ms,
    }


def main() -> int:
    snippets = sorted(HERE.glob("snippet_*.py"))
    if not snippets:
        print("ERROR: no snippet_*.py files found in", HERE)
        return 1

    results = []
    for path in snippets:
        rec = _run_snippet(path)
        verdict_mark = "✓" if rec["match"] else "✗"
        print(
            f"  {verdict_mark} {rec['file']:<52s}  "
            f"verdict={rec['verdict']:<14s}  ({rec['duration_ms']:.0f} ms)"
        )
        if rec["error_messages"]:
            for msg in rec["error_messages"][:2]:
                print(f"      {msg}")
        results.append(rec)

    proof_count = sum(1 for r in results if r["verdict"] == "REFUTED-PROOF")
    total       = len(results)
    passed      = proof_count >= PASS_THRESHOLD

    summary = {
        "total":              total,
        "refuted_proof":      proof_count,
        "pass_threshold":     PASS_THRESHOLD,
        "suite_pass":         passed,
        "entries":            results,
    }

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")
    print(f"Result: {proof_count}/{total} REFUTED-PROOF  "
          f"(need >= {PASS_THRESHOLD}) — {'PASS' if passed else 'FAIL'}")

    # Top-level entries list mirrors the success criterion's check:
    # grad_flow_results.json must have >= 6 entries, >= 5 with verdict == "REFUTED-PROOF"
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
