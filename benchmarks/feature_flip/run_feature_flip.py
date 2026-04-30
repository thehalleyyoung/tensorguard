"""Feature-flip benchmark: verify that check_devices, check_phases, and
check_gradients each flip the verdict from VERIFIED to REFUTED-PROOF on a
real-source example.

Exit 0 iff all three flags satisfy:
  verdict_off != verdict_on AND verdict_on == "REFUTED-PROOF"

Output artifact: benchmarks/feature_flip/feature_flip_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture  # noqa: E402

HERE = Path(__file__).resolve().parent

EXAMPLES = [
    {
        "flag": "check_devices",
        "path": HERE / "device_mismatch_residual.py",
        "input_shapes": {"x": (2, 8)},
    },
    {
        "flag": "check_phases",
        "path": HERE / "phase_dependent_head.py",
        "input_shapes": {"x": (2, 8)},
    },
    {
        "flag": "check_gradients",
        "path": HERE / "grad_checkpoint_block.py",
        "input_shapes": {"x": (2, 8)},
    },
]

_ALL_FLAGS = ("check_devices", "check_phases", "check_gradients")


def _run(source: str, input_shapes: dict, **flags) -> str:
    res = verify_architecture(
        source,
        input_shapes=input_shapes,
        high_confidence_only=False,
        max_cegar_iterations=0,
        **flags,
    )
    return "REFUTED-PROOF" if any(b.severity == "error" for b in res.bugs) else "VERIFIED"


def main() -> int:
    results = []
    all_pass = True

    for ex in EXAMPLES:
        src = ex["path"].read_text()
        flag = ex["flag"]
        shapes = ex["input_shapes"]

        flags_on  = {f: (f == flag) for f in _ALL_FLAGS}
        flags_off = {f: False for f in _ALL_FLAGS}

        verdict_on  = _run(src, shapes, **flags_on)
        verdict_off = _run(src, shapes, **flags_off)

        ok = (verdict_off != verdict_on) and (verdict_on == "REFUTED-PROOF")
        all_pass = all_pass and ok

        results.append({
            "flag": flag,
            "verdict_on": verdict_on,
            "verdict_off": verdict_off,
            "flip_ok": ok,
        })
        print(f"  {flag}: off={verdict_off!r}  on={verdict_on!r}  {'PASS' if ok else 'FAIL'}")

    out_path = HERE / "feature_flip_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
