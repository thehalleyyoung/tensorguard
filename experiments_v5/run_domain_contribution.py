"""
run_domain_contribution.py
==========================

Per-domain marginal-contribution table for TensorGuard (100_STEPS.md Step 3).

Motivation
----------
The earlier `feature_ablation.json` ladder was "flat": rows L1-L5 reported
identical verdict counts because `check_devices` / `check_phases` /
`check_gradients` were accepted by the API but never forwarded to the solver
(documented as a no-op). Steps 1-2 fixed that — CEGAR-discovered
unsatisfiable contracts now emit Bugs, and the per-domain flags now gate the
solver. This script demonstrates the consequence: each VERIFICATION domain
contributes at least one real bug that the base shape view misses, and any
domain that never contributes is recorded as DIAGNOSTIC-ONLY.

Method
------
We run a curated per-domain corpus (experiments_v5/domain_corpus/, described
in experiments_v5/domain_corpus_manifest.json) under an ablation ladder:

  base    : shape view only  (all domain flags off)
  +device : base + check_devices=True
  +phase  : base + check_phases=True
  +grad   : base + check_gradients=True

For each (entry, level) we record whether the buggy module is REFUTED. The
MARGINAL CONTRIBUTION of a domain is the set of buggy entries that the base
view leaves SAFE (a silent miss) but that the domain flips to REFUTED.

Outputs
-------
experiments_v5/domain_contribution.json  — structured results + summary
A Markdown table printed to stdout.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402

try:
    import torch
    TORCH_VERSION = torch.__version__
except Exception:
    TORCH_VERSION = "unknown"

MANIFEST = ROOT / "domain_corpus_manifest.json"
OUT_JSON = ROOT / "domain_contribution.json"

# Ablation ladder: a base shape-only level plus one level per domain flag.
LEVELS: List[Dict[str, Any]] = [
    {"level": "base", "kwargs": {
        "check_devices": False, "check_phases": False, "check_gradients": False}},
    {"level": "+device", "kwargs": {
        "check_devices": True, "check_phases": False, "check_gradients": False}},
    {"level": "+phase", "kwargs": {
        "check_devices": False, "check_phases": True, "check_gradients": False}},
    {"level": "+grad", "kwargs": {
        "check_devices": False, "check_phases": False, "check_gradients": True}},
]

# Map a ladder level to the domain it isolates (base isolates the shape view).
LEVEL_DOMAIN = {
    "base": "shape", "+device": "device", "+phase": "phase", "+grad": "gradient",
}


def _read_shapes(src: str) -> Dict[str, tuple]:
    m = re.search(r"^INPUT_SHAPES\s*=\s*(\{.*?\})", src, flags=re.MULTILINE | re.DOTALL)
    if not m:
        return {}
    try:
        return eval(m.group(1), {"__builtins__": {}}, {})
    except Exception:
        return {}


def _refuted(src: str, shapes: Dict[str, tuple], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    captured = io.StringIO()
    t0 = time.perf_counter()
    err = None
    res = None
    try:
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            res = verify_architecture(
                src,
                input_shapes=shapes,
                filename="<domain>",
                high_confidence_only=False,
                max_cegar_iterations=0,
                **kwargs,
            )
    except Exception as e:  # pragma: no cover - defensive
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    out: Dict[str, Any] = {
        "elapsed_ms": round(elapsed_ms, 1),
        "exception": err,
        "refuted": False,
        "bug_count": 0,
    }
    if res is not None:
        out["bug_count"] = int(res.bug_count)
        out["refuted"] = bool(res.bug_count > 0)
    return out


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["entries"]

    # results[entry_id][level] = {refuted, bug_count, ...}
    results: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        eid = entry["id"]
        src_path = REPO / entry["repro_file"]
        src = src_path.read_text()
        shapes = _read_shapes(src)
        results[eid] = {}
        for level_def in LEVELS:
            lvl = level_def["level"]
            results[eid][lvl] = _refuted(src, shapes, level_def["kwargs"])

    # Marginal contribution per domain: buggy entries the base view misses
    # but the domain level refutes.
    marginal: Dict[str, List[str]] = {}
    for level_def in LEVELS:
        lvl = level_def["level"]
        domain = LEVEL_DOMAIN[lvl]
        if lvl == "base":
            # The shape view's own contribution: entries it refutes directly.
            marginal[domain] = [
                e["id"] for e in entries
                if e.get("is_buggy") and results[e["id"]]["base"]["refuted"]
            ]
            continue
        contrib = []
        for e in entries:
            if not e.get("is_buggy"):
                continue
            eid = e["id"]
            base_ref = results[eid]["base"]["refuted"]
            lvl_ref = results[eid][lvl]["refuted"]
            if lvl_ref and not base_ref:
                contrib.append(eid)
        marginal[domain] = contrib

    classification = {}
    for domain, contrib in marginal.items():
        classification[domain] = "verification" if contrib else "diagnostic-only"

    summary = {
        "marginal_contribution": marginal,
        "classification": classification,
    }

    out = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch_version": TORCH_VERSION,
            "python_version": sys.version.split()[0],
            "method": (
                "Each domain's marginal contribution = buggy modules the base "
                "shape view leaves SAFE but the domain flag flips to REFUTED."
            ),
        },
        "levels": [d["level"] for d in LEVELS],
        "results": results,
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON}\n")

    # Human-readable table.
    print("=" * 78)
    print("PER-DOMAIN CONTRIBUTION")
    print("=" * 78)
    header = f"{'entry':<12} {'domain':<9} " + " ".join(f"{d['level']:<8}" for d in LEVELS)
    print(header)
    print("-" * 78)
    for e in entries:
        eid = e["id"]
        row = f"{eid:<12} {e['domain']:<9} "
        for level_def in LEVELS:
            cell = "R" if results[eid][level_def["level"]]["refuted"] else "."
            row += f"{cell:<8} "
        print(row)
    print("-" * 78)
    print("R = refuted (bug found), . = SAFE")
    print()
    print("Marginal contribution (bugs the base shape view misses):")
    for domain, contrib in marginal.items():
        tag = classification[domain]
        if domain == "shape":
            print(f"  shape    [{tag}]: refutes {contrib} directly (base view)")
        else:
            print(f"  {domain:<8} [{tag}]: +{len(contrib)} bug(s) {contrib}")
    print()
    print("Markdown:")
    print("| domain | classification | marginal bugs caught (missed by shape view) |")
    print("|--------|----------------|---------------------------------------------|")
    for domain, contrib in marginal.items():
        tag = classification[domain]
        if domain == "shape":
            cells = f"base view refutes {contrib}"
        else:
            cells = ", ".join(contrib) if contrib else "(none — diagnostic-only)"
        print(f"| {domain} | {tag} | {cells} |")


if __name__ == "__main__":
    main()
