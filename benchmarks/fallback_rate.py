#!/usr/bin/env python3
"""Unsupported-op fallback rate on the real-code corpus.

Walks every ``.py`` file under ``real_benchmarks/data/pytea_tests``,
parses it into an AST, and for every call of the form
``torch.<f>(...)``, ``F.<f>(...)``, ``functional.<f>(...)``,
``np.<f>(...)``, or ``numpy.<f>(...)`` -- i.e. *namespaced* framework
calls -- classifies the attribute name against ``TORCH_SHAPE_OPS``
(merged with ``MODERN_TORCH_SHAPE_OPS``) and ``NUMPY_SHAPE_OPS``.

We deliberately restrict to namespaced calls because:

* Calls of the form ``self.<sub>(x)`` dispatch to user-defined
  ``nn.Module`` sub-modules and are handled by recursive graph
  extraction, not by the per-op shape table.
* Tensor-method calls of the form ``x.view(...)``, ``x.transpose(...)``
  are method calls whose attribute names are already covered by the
  shape table when the receiver type is inferred.  Including them
  would double count.
* Framework utility calls (``torch.no_grad``, ``torch.manual_seed``,
  ``torch.save``) and tensor-metadata accessors (``x.size``, ``x.shape``,
  ``x.device``, ``x.numel``) are not shape transfer functions; we
  exclude them via ``NON_SHAPE_ATTRS`` to avoid inflating the
  denominator.

The resulting fallback rate is the fraction of namespaced framework
calls whose attribute is *not* known to TensorGuard.  This is the
number reported in the paper's threats-to-validity section.

Output: ``benchmarks/fallback_rate.json``.

Reproduce::

    python3.11 benchmarks/fallback_rate.py
"""
from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tensor_shapes import TORCH_SHAPE_OPS, NUMPY_SHAPE_OPS  # noqa: E402

CORPUS_ROOT = REPO_ROOT / "real_benchmarks" / "data" / "pytea_tests"

# Restrict the denominator to *namespaced* framework calls.
TORCH_NAMESPACES = {"torch", "nn", "F", "functional"}
NUMPY_NAMESPACES = {"np", "numpy"}
ALL_NAMESPACES = TORCH_NAMESPACES | NUMPY_NAMESPACES

# (legacy) commented out: tensor-method receivers used by the prior
# version of the metric; superseded by the namespaced scheme above.
TENSOR_RECEIVERS = {
    "torch", "nn", "F", "functional", "np", "numpy",
    "x", "y", "z", "h", "out", "output", "feat", "feats",
    "input", "inputs", "logits", "hidden", "h_t", "c_t",
    "q", "k", "v", "embed", "emb", "attn", "scores",
    "self",
}

# Attribute names we explicitly *exclude* from both numerator and
# denominator because they are not shape transfer functions in any
# verifier (they are framework utilities or tensor-metadata accessors
# that have no shape-changing semantics).  Keeping them in would
# falsely inflate the fallback rate.
NON_SHAPE_ATTRS = {
    # array constructors / type predicates / device probes
    "array", "asarray", "as_array", "fromnumpy", "is_tensor",
    "is_floating_point", "is_complex", "is_grad_enabled",
    # framework utilities / IO / RNG / device probes
    "no_grad", "enable_grad", "manual_seed", "save", "load",
    "is_available", "device_count", "set_device",
    "set_default_dtype", "set_grad_enabled", "set_default_tensor_type",
    "set_num_threads", "compile", "jit",
    # tensor metadata accessors (no shape effect)
    "size", "shape", "device", "dtype", "numel", "dim", "ndimension",
    "item", "tolist", "to", "cuda", "cpu", "detach", "clone",
    "requires_grad_", "zero_", "fill_", "copy_", "data",
    "parameters", "named_parameters", "modules", "named_modules",
    "children", "named_children", "state_dict", "load_state_dict",
    "buffers", "named_buffers", "register_buffer", "register_parameter",
    "register_forward_hook", "apply", "eval", "train", "zero_grad",
    "step", "backward", "grad", "requires_grad",
    "type", "type_as", "as_tensor", "tensor", "Tensor",
    "DataLoader", "Dataset", "DataParallel", "device_of",
    # printing / iteration
    "print", "format", "join", "split", "strip", "startswith",
    "append", "extend", "items", "keys", "values", "update",
    "Generator", "default_generator",
}


def _is_constructor_name(name: str) -> bool:
    """Heuristically identify ``nn.<Module>``-style constructors.

    nn.Module subclasses (Linear, Conv2d, BatchNorm2d, LSTM, GRU, ...) are
    handled by a separate ``__init__``-time graph extractor and are not
    part of the per-call shape transfer table; we exclude them from this
    metric.  We treat any attribute beginning with an uppercase letter
    as a constructor.
    """
    return bool(name) and name[0].isupper()


def is_tensor_call(node: ast.Call) -> tuple[bool, str | None, str | None]:
    """Return (is_namespaced, namespace, attr) for an AST Call.

    Only direct ``ns.attr(...)`` calls where ``ns`` is a Name in
    ``ALL_NAMESPACES`` are considered.  Anything else returns
    ``(False, None, None)`` and is ignored entirely.
    """
    func = node.func
    if not isinstance(func, ast.Attribute):
        return (False, None, None)
    base = func.value
    if not isinstance(base, ast.Name):
        return (False, None, None)
    if base.id not in ALL_NAMESPACES:
        return (False, None, None)
    return (True, base.id, func.attr)


def scan_file(path: Path) -> tuple[int, int, Counter, int]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return (0, 0, Counter())
    n_total = 0
    n_unsupported = 0
    n_excluded = 0
    unsupported = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        ok, ns, attr = is_tensor_call(node)
        if not ok or attr is None:
            continue
        if attr in NON_SHAPE_ATTRS or _is_constructor_name(attr):
            n_excluded += 1
            continue
        n_total += 1
        if ns in TORCH_NAMESPACES:
            covered = attr in TORCH_SHAPE_OPS
        else:
            covered = attr in NUMPY_SHAPE_OPS
        if not covered:
            n_unsupported += 1
            unsupported[f"{ns}.{attr}"] += 1
    return (n_total, n_unsupported, unsupported, n_excluded)


def main() -> int:
    files = sorted(CORPUS_ROOT.rglob("*.py"))
    rows = []
    total = 0
    total_unsupported = 0
    total_excluded = 0
    global_unknown: Counter = Counter()
    for f in files:
        n, u, uc, ex = scan_file(f)
        total_excluded += ex
        if n == 0:
            continue
        rows.append({
            "file": str(f.relative_to(REPO_ROOT)),
            "n_shape_calls": n,
            "n_unsupported": u,
            "fallback_rate": round(u / n, 3) if n else 0.0,
        })
        total += n
        total_unsupported += u
        global_unknown.update(uc)
    out = {
        "corpus": "real_benchmarks/data/pytea_tests",
        "n_files_with_shape_calls": len(rows),
        "n_files_scanned": len(files),
        "n_shape_calls": total,
        "n_unsupported": total_unsupported,
        "n_excluded_metadata_or_constructor": total_excluded,
        "fallback_rate": (
            round(total_unsupported / total, 4) if total else 0.0
        ),
        "top_unknown_attrs": global_unknown.most_common(15),
        "per_file": rows,
    }
    p = REPO_ROOT / "benchmarks" / "fallback_rate.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    print(
        f"corpus: {out['n_files_with_shape_calls']} files, "
        f"{out['n_shape_calls']} shape-relevant calls, "
        f"{out['n_unsupported']} unsupported "
        f"({out['fallback_rate']*100:.2f}%); "
        f"excluded {out['n_excluded_metadata_or_constructor']} "
        f"metadata/constructor calls"
    )
    print("top unknown attrs:", out["top_unknown_attrs"][:10])
    return 0


if __name__ == "__main__":
    sys.exit(main())
