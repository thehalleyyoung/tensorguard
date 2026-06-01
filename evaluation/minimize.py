#!/usr/bin/env python3
"""Step 17 -- shrink a TensorGuard-vs-runtime disagreement to a minimal module.

A delta-debugging *minimizer* for `nn.Module`s. Given any model that satisfies a
**predicate**, it produces a much smaller model that still satisfies it -- the
classic "minimal reproducer". The minimizer is generic over the predicate; the
built-in predicates capture the three interesting TensorGuard/runtime relations:

* ``false_positive``  -- TensorGuard refutes, but the model runs clean
  (a real disagreement: the verifier is wrong);
* ``false_negative``  -- TensorGuard accepts, but the model raises at runtime
  (a real disagreement: the verifier missed a bug);
* ``agreement_bug``   -- both flag the bug (TensorGuard refutes *and* runtime
  raises): a genuine caught bug, shrunk to its smallest reproducer.

Prior steps found **no** real false positives or false negatives, so there is no
disagreement corpus to draw from yet; the live demo therefore minimizes an
``agreement_bug`` (a genuine caught shape fault) down to a tiny reproducer. The
``false_positive``/``false_negative`` paths are exercised by unit tests with a
synthetic oracle, so the disagreement-minimization machinery is covered even
without a natural disagreement.

**Contract.** The minimizer preserves the *truth* of the predicate, not its
*mechanism*: a smaller model may satisfy the predicate for a slightly different
internal reason. The result is *locally* (1-)minimal under op removal and
coordinate-wise dim shrinking, not provably globally minimal.

Normal layers are shape-consistent by construction (each `Linear` derives its
`in_features` from the running shape); an intentional fault is represented by an
explicit ``in_override`` so removing or shrinking other ops cannot accidentally
create or erase a mismatch.

Usage
-----
    cd tensorguard && PYTHONPATH=. python3 evaluation/minimize.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/minimize.py --check
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_JSON = os.path.join(THIS_DIR, "minimize_demo.json")
OUT_MD = os.path.join(THIS_DIR, "minimize_demo.md")

Model = Dict[str, Any]
Predicate = Callable[[Model], bool]


# --------------------------------------------------------------------------
# Structured layer-chain IR -> source
# --------------------------------------------------------------------------
def emit_source(model: Model) -> str:
    """Render a model dict to nn.Module source. Normal Linears take their
    in_features from the running shape; a faulted Linear uses in_override."""
    init: List[str] = []
    fwd: List[str] = ["        h = x"]
    running = model["in_features"]
    idx = 0
    for op in model["ops"]:
        idx += 1
        if op["k"] == "linear":
            in_f = op.get("in_override", running)
            init.append("        self.l%d = nn.Linear(%d, %d)" % (idx, in_f, op["out"]))
            fwd.append("        h = self.l%d(h)" % idx)
            running = op["out"]
        elif op["k"] == "layernorm":
            init.append("        self.n%d = nn.LayerNorm(%d)" % (idx, running))
            fwd.append("        h = self.n%d(h)" % idx)
        elif op["k"] == "act":
            fwd.append("        h = torch.%s(h)" % op.get("fn", "relu"))
        else:
            raise ValueError("unknown op %r" % op["k"])
    fwd.append("        return h")
    init_block = "\n".join(init) if init else "        pass"
    return (
        "import torch\n"
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n%s\n\n"
        "    def forward(self, x):\n%s\n" % (init_block, "\n".join(fwd))
    )


def _input_shape(model: Model) -> Tuple[int, int]:
    return (model["batch"], model["in_features"])


# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------
def runtime_raises(model: Model) -> bool:
    import torch
    torch.manual_seed(0)
    src = emit_source(model)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(src)
    tmp.close()
    try:
        spec = importlib.util.spec_from_file_location("min_mod", tmp.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        inst = mod.M()
        inst.eval()
        with torch.no_grad():
            inst(torch.rand(*_input_shape(model)))
        return False
    except Exception:
        return True
    finally:
        os.unlink(tmp.name)


def tensorguard_refutes(model: Model) -> bool:
    from src.api import verify_architecture
    try:
        result = verify_architecture(
            emit_source(model), input_shapes={"x": _input_shape(model)},
            max_cegar_iterations=0, soundness_mode="balanced",
        )
        return result.bug_count > 0
    except Exception:
        return False


# Built-in predicates over the TensorGuard/runtime relation.
def pred_false_positive(model: Model) -> bool:
    return tensorguard_refutes(model) and not runtime_raises(model)


def pred_false_negative(model: Model) -> bool:
    return (not tensorguard_refutes(model)) and runtime_raises(model)


def pred_agreement_bug(model: Model) -> bool:
    return tensorguard_refutes(model) and runtime_raises(model)


PREDICATES: Dict[str, Predicate] = {
    "false_positive": pred_false_positive,
    "false_negative": pred_false_negative,
    "agreement_bug": pred_agreement_bug,
}


# --------------------------------------------------------------------------
# Minimization: ddmin over ops, then coordinate-descent dim shrink
# --------------------------------------------------------------------------
def _with_ops(model: Model, ops: List[Dict[str, Any]]) -> Model:
    m = copy.deepcopy(model)
    m["ops"] = copy.deepcopy(ops)
    return m


def ddmin_ops(model: Model, predicate: Predicate) -> Model:
    """Delta-debug the op list to a 1-minimal subsequence (Zeller ddmin)."""
    ops = list(model["ops"])
    assert predicate(_with_ops(model, ops)), "predicate must hold initially"
    n = 2
    while len(ops) >= 2:
        chunk = max(1, len(ops) // n)
        chunks = [ops[i:i + chunk] for i in range(0, len(ops), chunk)]
        reduced = False
        for i in range(len(chunks)):
            complement = [op for j, c in enumerate(chunks) if j != i for op in c]
            if complement and predicate(_with_ops(model, complement)):
                ops = complement
                n = max(n - 1, 2)
                reduced = True
                break
        if not reduced:
            if n >= len(ops):
                break
            n = min(len(ops), n * 2)
    return _with_ops(model, ops)


def shrink_dims(model: Model, predicate: Predicate) -> Model:
    """Coordinate-descent: lower each integer field to the smallest value that
    keeps the predicate true; iterate to a fixed point."""
    m = copy.deepcopy(model)

    def try_field(get, set_) -> bool:
        cur = get(m)
        for cand in range(1, cur):
            old = get(m)
            set_(m, cand)
            if predicate(m):
                return True  # accepted a smaller value
            set_(m, old)
        return False

    changed = True
    while changed:
        changed = False
        # batch
        changed |= try_field(lambda mm: mm["batch"],
                             lambda mm, v: mm.__setitem__("batch", v))
        # input features
        changed |= try_field(lambda mm: mm["in_features"],
                             lambda mm, v: mm.__setitem__("in_features", v))
        # per-op integer fields
        for op in m["ops"]:
            if op["k"] == "linear":
                changed |= try_field(
                    lambda mm, o=op: o["out"],
                    lambda mm, v, o=op: o.__setitem__("out", v))
                if "in_override" in op:
                    changed |= try_field(
                        lambda mm, o=op: o["in_override"],
                        lambda mm, v, o=op: o.__setitem__("in_override", v))
    return m


def minimize(model: Model, predicate: Predicate) -> Model:
    assert predicate(model), "the seed model must satisfy the predicate"
    reduced = ddmin_ops(model, predicate)
    reduced = shrink_dims(reduced, predicate)
    assert predicate(reduced), "minimized model must still satisfy the predicate"
    return reduced


def _size(model: Model) -> int:
    """A simple size metric: number of ops + total of integer parameters."""
    total = model["batch"] + model["in_features"]
    for op in model["ops"]:
        total += int(op.get("out", 0)) + int(op.get("in_override", 0))
    return len(model["ops"]) * 1000 + total  # op count dominates


# --------------------------------------------------------------------------
# Live demo: minimize a genuine caught shape fault (agreement_bug)
# --------------------------------------------------------------------------
def demo_seed_model() -> Model:
    """A deliberately large clean-ish chain with ONE injected shape fault
    (a Linear whose in_features (in_override) cannot match the running shape)."""
    ops: List[Dict[str, Any]] = [
        {"k": "linear", "out": 64},
        {"k": "act", "fn": "relu"},
        {"k": "linear", "out": 64},
        {"k": "layernorm"},
        {"k": "act", "fn": "tanh"},
        {"k": "linear", "out": 48},
        {"k": "act", "fn": "relu"},
        {"k": "linear", "out": 32, "in_override": 9973},  # <-- the fault
        {"k": "act", "fn": "relu"},
        {"k": "linear", "out": 16},
        {"k": "layernorm"},
        {"k": "linear", "out": 8},
    ]
    return {"batch": 8, "in_features": 32, "ops": ops}


def run(check: bool = False) -> Dict[str, Any]:
    predicate = PREDICATES["agreement_bug"]
    seed = demo_seed_model()
    minimal = minimize(seed, predicate)

    artifact = {
        "meta": {
            "generated_by": "evaluation/minimize.py",
            "command": "python3 evaluation/minimize.py",
            "predicate": "agreement_bug",
            "contract": (
                "preserves predicate truth, not mechanism; locally 1-minimal "
                "under op removal and coordinate-wise dim shrink"
            ),
        },
        "seed": {
            "op_count": len(seed["ops"]),
            "input_shape": list(_input_shape(seed)),
            "predicate_holds": True,
        },
        "minimal": {
            "op_count": len(minimal["ops"]),
            "input_shape": list(_input_shape(minimal)),
            "ops": minimal["ops"],
            "source": emit_source(minimal),
            "predicate_holds": True,
            "runtime_raises": True,
            "tensorguard_refutes": True,
        },
        "reduction": {
            "ops_before": len(seed["ops"]),
            "ops_after": len(minimal["ops"]),
            "ops_removed": len(seed["ops"]) - len(minimal["ops"]),
        },
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            if fh.read() != text:
                raise SystemExit("minimize_demo.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(a: Dict[str, Any]) -> str:
    return "\n".join([
        "# Step 17 -- minimal-reproducer shrinker",
        "",
        "A delta-debugging minimizer for `nn.Module`s, generic over a predicate "
        "capturing the TensorGuard/runtime relation (`false_positive`, "
        "`false_negative`, or `agreement_bug`). It ddmin-removes ops and then "
        "coordinate-shrinks every integer dimension while preserving the "
        "predicate. Generated by `evaluation/minimize.py`.",
        "",
        "## Demo: shrinking a genuine caught shape fault (`agreement_bug`)",
        "",
        "| | Ops | Input shape |",
        "|---|---|---|",
        "| Seed model | %d | %s |" % (a["seed"]["op_count"], a["seed"]["input_shape"]),
        "| Minimal reproducer | %d | %s |" % (
            a["minimal"]["op_count"], a["minimal"]["input_shape"]),
        "",
        "The minimizer removed **%d** ops while preserving the predicate "
        "(TensorGuard still refutes *and* the model still raises at runtime). "
        "The minimal reproducer is:" % a["reduction"]["ops_removed"],
        "",
        "```python",
        a["minimal"]["source"].rstrip("\n"),
        "```",
        "",
        "**Contract:** %s" % a["meta"]["contract"],
        "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    a = run(check=args.check)
    if args.check:
        print("minimize_demo.json is up to date")
        return
    print("Wrote %s and %s" % (os.path.relpath(OUT_JSON, REPO_ROOT),
                               os.path.relpath(OUT_MD, REPO_ROOT)))
    print("  seed ops: %d -> minimal ops: %d (removed %d)"
          % (a["reduction"]["ops_before"], a["reduction"]["ops_after"],
             a["reduction"]["ops_removed"]))


if __name__ == "__main__":
    main()
