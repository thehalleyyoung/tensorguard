"""Verify all 10 real bug repros produce RP verdict at ≥0.99 confidence."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from src.api import verify_architecture  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "real_bugs")

pass_count = 0
fail_count = 0

for fname in sorted(os.listdir(BASE)):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(BASE, fname)
    with open(fpath) as f:
        src = f.read()

    m = re.search(r"INPUT_SHAPES\s*=\s*(\{[^}]+\})", src)
    if not m:
        print(f"SKIP  {fname}: no INPUT_SHAPES")
        continue

    input_shapes = eval(m.group(1))

    try:
        result = verify_architecture(src, input_shapes=input_shapes)
        max_conf = max((b.confidence for b in result.bugs), default=0.0)
        if max_conf >= 0.99:
            print(f"PASS  {fname}: max_conf={max_conf:.2f}")
            pass_count += 1
        else:
            msg = result.bugs[0].message[:50] if result.bugs else "no bugs"
            print(f"FAIL  {fname}: max_conf={max_conf:.2f} — {msg}")
            fail_count += 1
    except Exception as e:
        print(f"ERROR {fname}: {e}")
        fail_count += 1

print(f"\nResult: {pass_count}/10 passed, {fail_count} failed")
sys.exit(0 if fail_count == 0 else 1)
