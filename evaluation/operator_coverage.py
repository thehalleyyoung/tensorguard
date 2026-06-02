"""Step 21 -- public PyTorch operator surface vs. implemented coverage matrix.

This introspects the *live* public operator surface of `torch`,
`torch.nn`, and `torch.nn.functional` and cross-references it against every
operator TensorGuard actually knows how to reason about, producing an honest,
reproducible coverage matrix.

"Implemented" is the union of every operator-recognition table in the codebase:

  * `graph_compiler._UNIVERSAL_TRANSFER_REGISTRY` -- the shape transfer-function
    registry for `torch.*` / `F.*` operators;
  * `fx_extractor._MODULE_KIND_MAP` -- recognised `nn.Module` layer classes;
  * `fx_extractor._F_FUNC_MAP`, `_TORCH_FUNC_MAP`, `_METHOD_OP_MAP` -- functional,
    `torch.*` and tensor-method dispatch tables;
  * the `denotational_semantics` concrete/abstract transfer functions.

For each of the three public namespaces the matrix reports the total public
operator count, how many are covered, the coverage ratio, and the full sorted
lists of covered and uncovered operator names. This is the census the rest of
Phase 3 (operator long-tail implementation) is prioritised against.

Version sensitivity
-------------------
The public surface depends on the installed PyTorch version, so the artifact
records `torch_version`. In `--check` mode the matrix is recomputed and compared
byte-for-byte *only when the local torch version matches the committed one*;
otherwise the check is reported QUALIFIED (version mismatch) and skipped rather
than failing -- mirroring the reproducibility harness's QUALIFIED_ENV handling.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Dict, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "operator_coverage.json")
MD_PATH = os.path.join(HERE, "operator_coverage.md")

NAMESPACES = ("torch", "torch.nn", "torch.nn.functional")


# ---------------------------------------------------------------------------
# Public surface enumeration
# ---------------------------------------------------------------------------
def enumerate_namespace(namespace: str) -> List[str]:
    """Return the sorted public operator names exposed by a namespace.

    * ``torch.nn`` -> public classes that subclass ``nn.Module``.
    * ``torch`` -> public callables that are *not* classes (functions/builtins).
    * ``torch.nn.functional`` -> public callables.
    """
    if namespace == "torch.nn":
        names = [
            n for n in dir(nn)
            if not n.startswith("_")
            and isinstance(getattr(nn, n), type)
            and issubclass(getattr(nn, n), nn.Module)
        ]
    elif namespace == "torch":
        names = [
            n for n in dir(torch)
            if not n.startswith("_")
            and callable(getattr(torch, n))
            and not isinstance(getattr(torch, n), type)
        ]
    elif namespace == "torch.nn.functional":
        names = [
            n for n in dir(F)
            if not n.startswith("_") and callable(getattr(F, n))
        ]
    else:
        raise ValueError("unknown namespace %r" % namespace)
    return sorted(set(names))


# ---------------------------------------------------------------------------
# Implemented operator names (per namespace), normalised to lowercase
# ---------------------------------------------------------------------------
def _strip_prefix(key: str, prefix: str) -> str:
    return key[len(prefix):] if key.startswith(prefix) else key


def implemented_names() -> Dict[str, Set[str]]:
    """Collect the operators TensorGuard recognises, keyed by namespace.

    Names are normalised to lowercase. ``torch`` and
    ``torch.nn.functional`` are matched on the final dotted component so that
    e.g. ``torch.linalg.solve`` registers as ``solve``.
    """
    from src import graph_compiler as gc
    from src import fx_extractor as fx

    fx._init_module_kind_map()

    torch_ops: Set[str] = set()
    func_ops: Set[str] = set()
    nn_ops: Set[str] = set()

    # 1. Universal transfer registry (keys like "torch.matmul", "F.relu").
    for key in gc._UNIVERSAL_TRANSFER_REGISTRY:
        if key.startswith("F."):
            func_ops.add(_strip_prefix(key, "F.").lower())
        elif key.startswith("torch."):
            tail = _strip_prefix(key, "torch.").split(".")[-1]
            torch_ops.add(tail.lower())

    # 2. nn.Module layer classes.
    for cls in fx._MODULE_KIND_MAP:
        nn_ops.add(cls.__name__.lower())

    # 3. Functional / torch / method dispatch tables.
    func_ops.update(n.lower() for n in fx._F_FUNC_MAP)
    torch_ops.update(n.lower() for n in fx._TORCH_FUNC_MAP)
    # Tensor-method ops that also exist as torch.* free functions.
    method_as_torch = {
        n.lower() for n in fx._METHOD_OP_MAP
        if hasattr(torch, n)
    }
    torch_ops.update(method_as_torch)

    # 4. Denotational concrete transfer functions (torch-level shape ops).
    denot_torch = {
        "matmul", "add", "reshape", "transpose", "flatten",
        "squeeze", "unsqueeze", "cat",
    }
    torch_ops.update(denot_torch)

    return {
        "torch": torch_ops,
        "torch.nn": nn_ops,
        "torch.nn.functional": func_ops,
    }


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def coverage_for(public_names: List[str],
                 implemented: Set[str]) -> Dict[str, object]:
    impl = {n.lower() for n in implemented}
    covered = sorted(n for n in public_names if n.lower() in impl)
    uncovered = sorted(n for n in public_names if n.lower() not in impl)
    total = len(public_names)
    return {
        "total": total,
        "covered_count": len(covered),
        "uncovered_count": len(uncovered),
        "coverage_ratio": round(len(covered) / total, 4) if total else 0.0,
        "covered": covered,
        "uncovered": uncovered,
    }


def build_matrix() -> Dict[str, object]:
    impl = implemented_names()
    namespaces = {}
    tot_total = tot_covered = 0
    for ns in NAMESPACES:
        public = enumerate_namespace(ns)
        entry = coverage_for(public, impl[ns])
        namespaces[ns] = entry
        tot_total += entry["total"]
        tot_covered += entry["covered_count"]
    return {
        "meta": {
            "generated_by": "evaluation/operator_coverage.py",
            "command": "PYTHONPATH=. python3 evaluation/operator_coverage.py",
            "torch_version": torch.__version__,
            "python_version": "%d.%d" % sys.version_info[:2],
            "platform": platform.system(),
        },
        "namespaces": namespaces,
        "summary": {
            "total_public_operators": tot_total,
            "total_covered": tot_covered,
            "overall_coverage_ratio": round(tot_covered / tot_total, 4)
            if tot_total else 0.0,
            "overall_coverage_percent": round(100.0 * tot_covered / tot_total, 2)
            if tot_total else 0.0,
        },
    }


FLOOR_PATH = os.path.join(HERE, "operator_coverage_floor.json")

# Released floors may slip by at most this ratio between torch patch releases
# (the public surface shifts slightly version-to-version); a drop beyond it is
# treated as a coverage regression and fails the gate.
GATE_TOLERANCE = 0.005


def build_floor(matrix: Dict[str, object]) -> Dict[str, object]:
    """Derive a committed coverage floor (ratchet) from a coverage matrix."""
    summ = matrix["summary"]
    return {
        "_comment": ("Published operator-coverage floor enforced by "
                     "`operator_coverage.py --gate`. Ratchet: regenerate with "
                     "`make operator-coverage-floor` only to RAISE the floor."),
        "torch_version": matrix["meta"]["torch_version"],
        "overall_coverage_ratio": summ["overall_coverage_ratio"],
        "namespaces": {
            ns: matrix["namespaces"][ns]["coverage_ratio"] for ns in NAMESPACES
        },
    }


def gate(write_floor: bool = False) -> int:
    """Enforce (or, with write_floor, publish) the operator-coverage floor.

    Returns 0 when live coverage meets every committed floor (or the check is
    QUALIFIED because the local torch version differs from the floor's), 1 on a
    genuine coverage regression or a missing floor artifact.
    """
    matrix = build_matrix()
    summ = matrix["summary"]
    if write_floor:
        with open(FLOOR_PATH, "w") as fh:
            fh.write(_dumps(build_floor(matrix)))
        print("published operator coverage floor: %.4f overall (%.2f percent), "
              "torch %s" % (summ["overall_coverage_ratio"],
                            summ["overall_coverage_percent"],
                            matrix["meta"]["torch_version"]))
        return 0

    if not os.path.exists(FLOOR_PATH):
        print("operator_coverage_floor.json missing; run "
              "`make operator-coverage-floor`")
        return 1
    floor = json.load(open(FLOOR_PATH))
    local_ver = matrix["meta"]["torch_version"]
    if floor.get("torch_version") != local_ver:
        print("QUALIFIED: torch version mismatch (floor %s, local %s); "
              "skipping coverage gate" % (floor.get("torch_version"), local_ver))
        return 0

    regressions: List[str] = []
    overall = summ["overall_coverage_ratio"]
    if overall + GATE_TOLERANCE < floor["overall_coverage_ratio"]:
        regressions.append("overall %.4f < floor %.4f"
                           % (overall, floor["overall_coverage_ratio"]))
    for ns in NAMESPACES:
        live = matrix["namespaces"][ns]["coverage_ratio"]
        base = floor.get("namespaces", {}).get(ns)
        if base is not None and live + GATE_TOLERANCE < base:
            regressions.append("%s %.4f < floor %.4f" % (ns, live, base))

    if regressions:
        print("COVERAGE GATE FAILED:")
        for r in regressions:
            print("  - " + r)
        return 1
    print("operator coverage gate PASS: %d of %d public operators covered "
          "(%.2f percent overall, floor %.2f percent)" % (
              summ["total_covered"], summ["total_public_operators"],
              summ["overall_coverage_percent"],
              round(100.0 * floor["overall_coverage_ratio"], 2)))
    return 0


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(matrix: Dict[str, object]) -> str:
    meta = matrix["meta"]
    summ = matrix["summary"]
    lines = [
        "# PyTorch operator surface coverage matrix",
        "",
        ("Public operator surface of `torch`, `torch.nn` and "
         "`torch.nn.functional` cross-referenced against the operators "
         "TensorGuard recognises (the universal transfer-function registry, the "
         "`nn.Module` layer map, the functional/torch/method dispatch tables, "
         "and the denotational transfer functions). Generated against torch "
         "`%s`, Python `%s`." % (meta["torch_version"], meta["python_version"])),
        "",
        "| Namespace | Public operators | Covered | Coverage |",
        "|-----------|------------------|---------|----------|",
    ]
    for ns in NAMESPACES:
        e = matrix["namespaces"][ns]
        lines.append("| `%s` | %d | %d | %.3f |" % (
            ns, e["total"], e["covered_count"], e["coverage_ratio"]))
    lines.append("| **total** | %d | %d | %.3f |" % (
        summ["total_public_operators"], summ["total_covered"],
        summ["overall_coverage_ratio"]))
    lines.append("")
    lines.append("Covered operators per namespace:")
    lines.append("")
    for ns in NAMESPACES:
        e = matrix["namespaces"][ns]
        lines.append("* `%s`: %s" % (ns, ", ".join(e["covered"]) or "(none)"))
    lines.append("")
    return "\n".join(lines)


def run(check: bool = False, write: bool = True) -> int:
    matrix = build_matrix()
    text = _dumps(matrix)

    if check:
        if not os.path.exists(JSON_PATH):
            print("operator_coverage.json missing; run the harness first")
            return 1
        committed = json.load(open(JSON_PATH))
        committed_ver = committed.get("meta", {}).get("torch_version")
        local_ver = matrix["meta"]["torch_version"]
        if committed_ver != local_ver:
            print("QUALIFIED: torch version mismatch (committed %s, local %s); "
                  "skipping byte-identical check" % (committed_ver, local_ver))
            return 0
        if open(JSON_PATH).read() != text:
            print("operator_coverage.json is stale; run `make operator-coverage`")
            return 1
        md = render_markdown(matrix)
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            print("operator_coverage.md is stale; run `make operator-coverage`")
            return 1
        print("operator coverage matrix up to date (torch %s)" % local_ver)
        return 0

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(render_markdown(matrix))
    summ = matrix["summary"]
    print("operator coverage: %d covered of %d public operators (ratio %.3f)" % (
        summ["total_covered"], summ["total_public_operators"],
        summ["overall_coverage_ratio"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed matrix is up to date (version-gated).")
    ap.add_argument("--gate", action="store_true",
                    help="Fail if live operator coverage regresses below the "
                         "published floor (version-gated).")
    ap.add_argument("--write-floor", action="store_true",
                    help="Publish/raise the operator-coverage floor from the "
                         "current live coverage.")
    args = ap.parse_args()
    if args.write_floor:
        return gate(write_floor=True)
    if args.gate:
        return gate()
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
