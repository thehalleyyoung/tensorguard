"""Live end-to-end demonstration that ``check_devices``, ``check_phases``
and ``check_gradients`` flip verdicts on real-source examples.

Runs three small but realistic modules under all 8 combinations of the
three secondary-check flags and emits a JSON artifact recording, for
every (example, flag-combo) pair, the full bug list and the resulting
overall verdict (REFUTED if any error-severity bug remains, otherwise
VERIFIED/OK).

Output: ``reproducibility/check_flag_demo.json``.

This is the artifact the third-round reviewer asked for: it converts
the 5-theory product domain from a documented no-op on real corpora to
a demonstrated contribution.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture  # noqa: E402

EXAMPLES_DIR = ROOT / "examples" / "check_flag_demo"
OUT_PATH = ROOT / "reproducibility" / "check_flag_demo.json"

EXAMPLES = [
    {
        "name": "device_mismatch_residual",
        "path": EXAMPLES_DIR / "device_mismatch_residual.py",
        "input_shapes": {"x": (2, 8)},
        "primary_flag": "check_devices",
        "expected_with_flag":    "REFUTED",
        "expected_without_flag": "VERIFIED",
    },
    {
        "name": "phase_dependent_head",
        "path": EXAMPLES_DIR / "phase_dependent_head.py",
        "input_shapes": {"x": (2, 8)},
        "primary_flag": "check_phases",
        "expected_with_flag":    "REFUTED",
        "expected_without_flag": "VERIFIED",
    },
    {
        "name": "grad_checkpoint_block",
        "path": EXAMPLES_DIR / "grad_checkpoint_block.py",
        "input_shapes": {"x": (2, 8)},
        "primary_flag": "check_gradients",
        "expected_with_flag":    "REFUTED",
        "expected_without_flag": "VERIFIED",
    },
]


def run_one(source: str, input_shapes: dict, flags: dict) -> dict:
    t0 = time.perf_counter()
    res = verify_architecture(
        source,
        input_shapes=input_shapes,
        check_devices=flags["check_devices"],
        check_phases=flags["check_phases"],
        check_gradients=flags["check_gradients"],
        high_confidence_only=False,
        max_cegar_iterations=0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    err_bugs = [b for b in res.bugs if b.severity == "error"]
    verdict = "REFUTED" if err_bugs else "VERIFIED"
    return {
        "verdict": verdict,
        "n_error_bugs": len(err_bugs),
        "n_total_bugs": len(res.bugs),
        "first_error_msg": (err_bugs[0].message if err_bugs else None),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def main() -> int:
    flag_names = ("check_devices", "check_phases", "check_gradients")
    combos = list(itertools.product([True, False], repeat=3))

    out = {
        "meta": {
            "purpose": (
                "Live end-to-end demonstration that the three secondary-"
                "check flags flip verdicts on committed real-source "
                "examples."
            ),
            "schema_version": 1,
            "n_examples": len(EXAMPLES),
            "n_flag_combos": len(combos),
        },
        "examples": [],
    }

    all_pass = True
    for ex in EXAMPLES:
        src = ex["path"].read_text()
        rows = []
        for combo in combos:
            flags = dict(zip(flag_names, combo))
            row = {**flags, **run_one(src, ex["input_shapes"], flags)}
            rows.append(row)
        # Verdict-flip check on the primary flag, holding the other two
        # at False so we isolate the contribution.
        flip_on = next(
            r for r in rows
            if r[ex["primary_flag"]] is True
            and all(not r[f] for f in flag_names if f != ex["primary_flag"])
        )
        flip_off = next(
            r for r in rows
            if not any(r[f] for f in flag_names)
        )
        flipped = flip_on["verdict"] != flip_off["verdict"]
        ex_record = {
            "name": ex["name"],
            "path": str(ex["path"].relative_to(ROOT)),
            "primary_flag": ex["primary_flag"],
            "expected_with_flag": ex["expected_with_flag"],
            "expected_without_flag": ex["expected_without_flag"],
            "flip_on_verdict": flip_on["verdict"],
            "flip_off_verdict": flip_off["verdict"],
            "flag_flips_verdict": flipped,
            "expectation_met": (
                flip_on["verdict"] == ex["expected_with_flag"]
                and flip_off["verdict"] == ex["expected_without_flag"]
            ),
            "rows": rows,
        }
        out["examples"].append(ex_record)
        all_pass = all_pass and ex_record["expectation_met"]

    out["meta"]["all_examples_flip_verdict"] = all_pass
    out["meta"]["n_examples_flip_verdict"] = sum(
        1 for e in out["examples"] if e["flag_flips_verdict"]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(
        f"Examples that flip verdict on primary flag: "
        f"{out['meta']['n_examples_flip_verdict']}/{len(EXAMPLES)}"
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
