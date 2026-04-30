"""Verify rb_001..rb_010 *upstream-faithful* repros against TG.

This harness runs TG on the upstream-faithful versions in
``real_bugs_upstream/`` (real ``__init__`` with ``nn.Linear``/etc. and a
multi-step ``forward`` driven by config-attribute arithmetic), in
contrast to the original ``real_bugs/rb_*.py`` one-line-view distillates
called out by reviewer W1.

Output: writes a JSON line per repro to
``reproducibility/real_bugs_upstream.json`` and prints a summary.
"""
import json
import os
import re
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.api import verify_architecture  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "real_bugs_upstream")
OUT  = os.path.join(ROOT, "reproducibility", "real_bugs_upstream.json")

records = []
pass_count = 0
fail_count = 0

for fname in sorted(os.listdir(BASE)):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(BASE, fname)
    with open(fpath) as f:
        src = f.read()

    ns: dict = {"__name__": "__rb_loader__"}
    try:
        exec(compile(src, fpath, "exec"), ns)
    except Exception as e:
        print(f"LOAD_ERR          {fname}: {e}")
        continue
    input_shapes = ns.get("INPUT_SHAPES")
    if input_shapes is None:
        continue

    rec = {
        "id": fname.split("_", 1)[0] + "_" + fname.split("_", 1)[1].split("_")[0],
        "file": os.path.relpath(fpath, ROOT),
        "input_shapes": {k: list(v) for k, v in input_shapes.items()},
    }
    try:
        result = verify_architecture(src, input_shapes=input_shapes)
        bugs = list(result.bugs)
        max_conf = max((b.confidence for b in bugs), default=0.0)
        rec["n_bugs"] = len(bugs)
        rec["max_confidence"] = max_conf
        rec["abstained"] = bool(getattr(result, "abstained", False))
        rec["status"] = (
            "RP_0.99" if max_conf >= 0.99 else
            "RP_lt_0.99" if bugs else
            ("Abstain" if rec["abstained"] else "Verified_or_Silent")
        )
        rec["first_bug"] = bugs[0].message[:200] if bugs else None
        if max_conf >= 0.99:
            pass_count += 1
        else:
            fail_count += 1
        print(f"{rec['status']:18s}  {fname}  conf={max_conf:.2f}  "
              f"bugs={len(bugs)}  abstain={rec['abstained']}")
    except Exception as e:
        rec["status"] = "ERROR"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        fail_count += 1
        print(f"ERROR             {fname}: {e}")
    records.append(rec)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump({
        "summary": {
            "n_total": len(records),
            "rp_at_0.99": pass_count,
            "not_rp_at_0.99": fail_count,
        },
        "records": records,
    }, f, indent=2)

print(f"\nResult: {pass_count}/{len(records)} RP@0.99   "
      f"({fail_count} not RP@0.99)\nWrote {OUT}")
