#!/usr/bin/env python3
"""Step 16 -- negative fuzzing (false-negative hunting).

The dual of Step 15. Instead of asserting TensorGuard stays silent on clean
code, here we **inject a fault** into an otherwise-valid random `nn.Module` and
assert TensorGuard *catches* it. This hunts for false negatives: bugs a static
verifier should reject but lets through.

The valid base models come from the Step 15 random-architecture fuzzer
(`evaluation/diff_fuzz.py`). Into each we inject a fault from a catalogue:

* ``linear_in``   -- corrupt an `nn.Linear`'s ``in_features`` so the incoming
  activation no longer matches;
* ``conv_in``     -- corrupt an `nn.Conv2d`'s ``in_channels``;
* ``linear_out``  -- corrupt an `nn.Linear`'s ``out_features`` so a downstream
  layer mismatches;
* ``bad_reshape`` -- splice an explicit `reshape` to an incompatible size.

Every injected fault is **proven genuine** by running the mutated model once in
eager PyTorch and observing a real `RuntimeError`; a candidate that happens to
still execute (e.g. a reshape that coincidentally fits) is *not admitted*. We
then run TensorGuard on each genuine fault and measure **recall** (the share it
catches), tagging any fault it misses with a root cause so residual
false negatives are documented rather than hidden.

Deterministic: base models are seeded and injectors are pure, so the corpus,
verdicts, and committed artifact regenerate byte-for-byte.

Usage
-----
    cd tensorguard && PYTHONPATH=. python3 evaluation/neg_fuzz.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/neg_fuzz.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation import diff_fuzz  # noqa: E402

OUT_JSON = os.path.join(THIS_DIR, "neg_fuzz.json")
OUT_MD = os.path.join(THIS_DIR, "neg_fuzz.md")

N_BASE = 120  # base random seeds to draw valid models from

# Root-cause tags consulted only when TensorGuard misses an injected fault.
MISS_ROOT_CAUSE: Dict[str, str] = {}


# --------------------------------------------------------------------------
# Fault injectors. Each maps a valid source to a mutated source, or None when
# the injector does not apply to that model.
# --------------------------------------------------------------------------
def inject_linear_in(src: str) -> Optional[str]:
    m = re.search(r"nn\.Linear\((\d+), (\d+)\)", src)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return src[:m.start()] + "nn.Linear(%d, %d)" % (a + 5, b) + src[m.end():]


def inject_linear_out(src: str) -> Optional[str]:
    # Corrupt the FIRST Linear's out_features; only bites if a later layer
    # consumes it (genuineness check rejects the no-op case).
    m = re.search(r"nn\.Linear\((\d+), (\d+)\)", src)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return src[:m.start()] + "nn.Linear(%d, %d)" % (a, b + 7) + src[m.end():]


def inject_conv_in(src: str) -> Optional[str]:
    m = re.search(r"nn\.Conv2d\((\d+), (\d+),", src)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return src[:m.start()] + "nn.Conv2d(%d, %d," % (a + 3, b) + src[m.end():]


def inject_bad_reshape(src: str) -> Optional[str]:
    marker = "        return h"
    if marker not in src:
        return None
    # Splice an explicit reshape to a fixed prime width before the return;
    # genuineness check keeps only the cases where it really mismatches.
    spliced = "        h = h.reshape(h.size(0), 13)\n" + marker
    return src.replace(marker, spliced, 1)


INJECTORS = [
    ("linear_in", inject_linear_in),
    ("linear_out", inject_linear_out),
    ("conv_in", inject_conv_in),
    ("bad_reshape", inject_bad_reshape),
]


# --------------------------------------------------------------------------
# Genuineness + TensorGuard verdict
# --------------------------------------------------------------------------
def fault_is_genuine(source: str, input_shapes: Dict[str, tuple]) -> bool:
    """A genuine injected fault makes the model raise at runtime."""
    return not diff_fuzz.runtime_runs_clean(source, input_shapes)


def tensorguard_catches(source: str, input_shapes: Dict[str, tuple]) -> Tuple[bool, int]:
    from src.api import verify_architecture
    shapes = {k: tuple(v) for k, v in input_shapes.items()}
    result = verify_architecture(
        source, input_shapes=shapes, max_cegar_iterations=0,
        soundness_mode="balanced",
    )
    return (result.bug_count > 0), result.bug_count


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(check: bool = False, n_base: int = N_BASE,
        write: bool = True) -> Dict[str, Any]:
    by_family: Dict[str, Dict[str, int]] = {
        name: {"injected": 0, "genuine": 0, "caught": 0}
        for name, _ in INJECTORS
    }
    misses: List[Dict[str, Any]] = []
    total_genuine = total_caught = 0

    for seed in range(n_base):
        base_src, shapes = diff_fuzz.build_model(seed)
        # Only inject into models that are themselves valid + executing.
        if not diff_fuzz.runtime_runs_clean(base_src, shapes):
            continue
        for name, injector in INJECTORS:
            mutated = injector(base_src)
            if mutated is None:
                continue
            by_family[name]["injected"] += 1
            if not fault_is_genuine(mutated, shapes):
                continue  # injection did not actually break the model
            by_family[name]["genuine"] += 1
            total_genuine += 1
            caught, bug_count = tensorguard_catches(mutated, shapes)
            if caught:
                by_family[name]["caught"] += 1
                total_caught += 1
            else:
                tag = MISS_ROOT_CAUSE.get(name, "uncategorised false negative")
                misses.append({"seed": seed, "family": name,
                               "root_cause": tag,
                               "input_shapes": {k: list(v) for k, v in shapes.items()},
                               "source": mutated})

    recall = round(total_caught / total_genuine, 4) if total_genuine else 0.0
    per_family_recall = {
        name: round(d["caught"] / d["genuine"], 4) if d["genuine"] else None
        for name, d in by_family.items()
    }

    artifact = {
        "meta": {
            "generated_by": "evaluation/neg_fuzz.py",
            "command": "python3 evaluation/neg_fuzz.py",
            "n_base_seeds": n_base,
            "injector_families": [n for n, _ in INJECTORS],
            "design": (
                "inject a fault into a valid random nn.Module, prove the fault "
                "genuine by observing a real eager-PyTorch RuntimeError, then "
                "assert TensorGuard catches it (false-negative hunt); residual "
                "misses are root-cause tagged"
            ),
        },
        "summary": {
            "genuine_faults": total_genuine,
            "caught": total_caught,
            "recall": recall,
            "false_negatives": len(misses),
        },
        "by_family": by_family,
        "per_family_recall": per_family_recall,
        "false_negatives": misses,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            if fh.read() != text:
                raise SystemExit("neg_fuzz.json is stale; regenerate it")
        return artifact

    if not write:
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(a: Dict[str, Any]) -> str:
    s = a["summary"]
    lines = [
        "# Step 16 -- negative fuzzing (false-negative hunting)",
        "",
        "Faults are injected into valid random `nn.Module`s (from the Step 15 "
        "fuzzer): corrupting a `Linear`'s `in_features` or `out_features`, a "
        "`Conv2d`'s `in_channels`, or splicing an incompatible `reshape`. Every "
        "injected fault is **proven genuine** by observing a real eager-PyTorch "
        "`RuntimeError`; TensorGuard is then required to catch it. Generated by "
        "`evaluation/neg_fuzz.py`.",
        "",
        "## Recall",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| Genuine injected faults | %d |" % s["genuine_faults"],
        "| Caught by TensorGuard | %d |" % s["caught"],
        "| **False negatives** | **%d** |" % s["false_negatives"],
        "| Recall | %.3f |" % s["recall"],
        "",
        "## By injector family",
        "",
        "| Family | Genuine | Caught | Recall |",
        "|---|---|---|---|",
    ]
    for name in [n for n, _ in INJECTORS]:
        d = a["by_family"][name]
        r = a["per_family_recall"][name]
        rstr = "n/a" if r is None else "%.3f" % r
        lines.append("| `%s` | %d | %d | %s |" % (name, d["genuine"], d["caught"], rstr))
    lines.append("")
    lines.append("## False negatives (root-cause tagged)")
    lines.append("")
    if not a["false_negatives"]:
        lines.append("None.")
    else:
        lines.append("| Seed | Family | Root cause |")
        lines.append("|---|---|---|")
        for m in a["false_negatives"]:
            lines.append("| %d | `%s` | %s |" % (m["seed"], m["family"], m["root_cause"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=N_BASE)
    args = ap.parse_args()
    a = run(check=args.check, n_base=args.n)
    if args.check:
        print("neg_fuzz.json is up to date")
        return
    s = a["summary"]
    print("Wrote %s and %s" % (os.path.relpath(OUT_JSON, REPO_ROOT),
                               os.path.relpath(OUT_MD, REPO_ROOT)))
    print("  genuine faults: %d | caught: %d | recall: %.3f | false negatives: %d"
          % (s["genuine_faults"], s["caught"], s["recall"], s["false_negatives"]))
    if a["false_negatives"]:
        print("  MISSES:", [(m["seed"], m["family"]) for m in a["false_negatives"]])


if __name__ == "__main__":
    main()
