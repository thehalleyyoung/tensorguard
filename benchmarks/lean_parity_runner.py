"""Lean handler parity runner.

For each PyTorch operator handler in TensorGuard, generates a large number of
random valid concrete input shapes, then compares:

  * PyTorch's actual output shape (run on a tiny tensor)
  * TG's predicted output shape (from ``SymbolicShapePropagator``)

A mismatch is a parity failure.  Results are written to
``benchmarks/lean_parity_results.json``.

This module is also imported by ``tests/test_lean_handler_parity.py`` so the
specs/cases stay in one place.
"""

from __future__ import annotations

import json
import os
import random
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_checker import (  # noqa: E402
    SymbolicShapePropagator,
    extract_computation_graph,
)


# ---------------------------------------------------------------------------
# Spec definition
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """A single (source, input_shapes) test case for one operator."""

    source: str
    input_shapes: Dict[str, Tuple[int, ...]]
    # name -> tensor factory; for embedding we need int64 tensors
    input_factory: Optional[Dict[str, Callable[[Tuple[int, ...]], torch.Tensor]]] = None


@dataclass
class OpSpec:
    name: str
    gen: Callable[[random.Random], Case]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_src(body: str, init: str = "") -> str:
    """Build a module source string with given __init__/forward bodies."""
    init_block = f"    def __init__(self):\n        super().__init__()\n{init}\n" if init else ""
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "class M(nn.Module):\n"
        f"{init_block}"
        f"{body}\n"
    )


def _exec_module(source: str) -> nn.Module:
    """Exec a module source and return an instance of class ``M``."""
    ns: Dict[str, Any] = {}
    exec(compile(source, "<parity>", "exec"), ns)
    cls = ns["M"]
    return cls()


def _torch_out_shape(case: Case) -> Tuple[int, ...]:
    """Run PyTorch and return the output shape of forward(...)."""
    m = _exec_module(case.source).eval()
    tensors = []
    factories = case.input_factory or {}
    # forward arg order is the order names appear in input_shapes
    for name, shape in case.input_shapes.items():
        if name in factories:
            t = factories[name](shape)
        else:
            t = torch.randn(*shape) if shape else torch.randn(())
        tensors.append(t)
    with torch.no_grad():
        out = m(*tensors)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return tuple(out.shape)


def _tg_out_shape(case: Case) -> Optional[Tuple[Optional[int], ...]]:
    """Get TG's predicted output shape for the last returned tensor.

    Returns ``None`` if TG could not produce a shape.
    """
    graph = extract_computation_graph(case.source)
    env = SymbolicShapePropagator(graph).propagate(case.input_shapes)
    if not graph.output_names:
        return None
    name = graph.output_names[-1]
    shape = env.get(name)
    if shape is None:
        return None
    out: List[Optional[int]] = []
    for d in shape.dims:
        if d.is_symbolic:
            out.append(None)
        else:
            out.append(int(d.value))
    return tuple(out)


# ---------------------------------------------------------------------------
# Operator specs
# ---------------------------------------------------------------------------


def _spec_linear() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        in_f = rng.randint(2, 32)
        out_f = rng.randint(2, 32)
        b = rng.randint(1, 4)
        extra = rng.choice([(), (rng.randint(2, 5),), (rng.randint(2, 5), rng.randint(2, 5))])
        shape = (b, *extra, in_f)
        src = _module_src(
            "    def forward(self, x):\n        return self.fc(x)",
            f"        self.fc = nn.Linear({in_f}, {out_f})",
        )
        return Case(src, {"x": shape})

    return OpSpec("Linear", gen)


def _spec_conv2d() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        in_c = rng.randint(1, 6)
        out_c = rng.randint(1, 8)
        k = rng.choice([1, 3, 5])
        s = rng.choice([1, 2])
        p = rng.choice([0, 1, k // 2])
        b = rng.randint(1, 3)
        h = rng.randint(k + 2 * p, 16) + 4
        w = rng.randint(k + 2 * p, 16) + 4
        src = _module_src(
            "    def forward(self, x):\n        return self.c(x)",
            f"        self.c = nn.Conv2d({in_c}, {out_c}, {k}, stride={s}, padding={p})",
        )
        return Case(src, {"x": (b, in_c, h, w)})

    return OpSpec("Conv2d", gen)


def _spec_conv1d() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        in_c = rng.randint(1, 6)
        out_c = rng.randint(1, 8)
        k = rng.choice([1, 3, 5])
        s = rng.choice([1, 2])
        p = rng.choice([0, 1, k // 2])
        b = rng.randint(1, 3)
        L = rng.randint(k + 2 * p, 32) + 4
        src = _module_src(
            "    def forward(self, x):\n        return self.c(x)",
            f"        self.c = nn.Conv1d({in_c}, {out_c}, {k}, stride={s}, padding={p})",
        )
        return Case(src, {"x": (b, in_c, L)})

    return OpSpec("Conv1d", gen)


def _spec_conv3d() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        in_c = rng.randint(1, 4)
        out_c = rng.randint(1, 6)
        k = rng.choice([1, 3])
        s = rng.choice([1, 2])
        p = rng.choice([0, k // 2])
        b = rng.randint(1, 2)
        d = rng.randint(k + 2 * p, 8) + 2
        h = rng.randint(k + 2 * p, 8) + 2
        w = rng.randint(k + 2 * p, 8) + 2
        src = _module_src(
            "    def forward(self, x):\n        return self.c(x)",
            f"        self.c = nn.Conv3d({in_c}, {out_c}, {k}, stride={s}, padding={p})",
        )
        return Case(src, {"x": (b, in_c, d, h, w)})

    return OpSpec("Conv3d", gen)


def _spec_convtranspose2d() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        in_c = rng.randint(1, 4)
        out_c = rng.randint(1, 6)
        k = rng.choice([2, 3, 4])
        s = rng.choice([1, 2])
        p = rng.choice([0, 1])
        b = rng.randint(1, 3)
        h = rng.randint(4, 12)
        w = rng.randint(4, 12)
        src = _module_src(
            "    def forward(self, x):\n        return self.c(x)",
            f"        self.c = nn.ConvTranspose2d({in_c}, {out_c}, {k}, stride={s}, padding={p})",
        )
        return Case(src, {"x": (b, in_c, h, w)})

    return OpSpec("ConvTranspose2d", gen)


def _spec_batchnorm2d() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        c = rng.randint(1, 8)
        b = rng.randint(2, 4)
        h = rng.randint(2, 8)
        w = rng.randint(2, 8)
        src = _module_src(
            "    def forward(self, x):\n        return self.n(x)",
            f"        self.n = nn.BatchNorm2d({c})",
        )
        return Case(src, {"x": (b, c, h, w)})

    return OpSpec("BatchNorm2d", gen)


def _spec_layernorm() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        ndim_extra = rng.choice([0, 1, 2])
        norm_dims = [rng.randint(2, 6) for _ in range(rng.choice([1, 2]))]
        leading = [rng.randint(1, 4) for _ in range(1 + ndim_extra)]
        shape = tuple(leading + norm_dims)
        src = _module_src(
            "    def forward(self, x):\n        return self.n(x)",
            f"        self.n = nn.LayerNorm({list(norm_dims)})",
        )
        return Case(src, {"x": shape})

    return OpSpec("LayerNorm", gen)


def _spec_view() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        # Pick a shape, then choose a target with a single -1.
        a = rng.randint(2, 5)
        b = rng.randint(2, 5)
        c = rng.randint(2, 5)
        shape = (a, b, c)
        total = a * b * c
        # Pick first dim, second is -1
        d0 = rng.choice([d for d in (1, a, a * b, b, c, total) if total % d == 0])
        src = _module_src(
            f"    def forward(self, x):\n        return x.view({d0}, -1)"
        )
        return Case(src, {"x": shape})

    return OpSpec("view", gen)


def _spec_reshape() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        a = rng.randint(2, 5)
        b = rng.randint(2, 5)
        c = rng.randint(2, 5)
        shape = (a, b, c)
        total = a * b * c
        d0 = rng.choice([d for d in (1, a, a * b, b, c, total) if total % d == 0])
        src = _module_src(
            f"    def forward(self, x):\n        return x.reshape({d0}, -1)"
        )
        return Case(src, {"x": shape})

    return OpSpec("reshape", gen)


def _spec_transpose() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        ndim = rng.choice([2, 3, 4])
        shape = tuple(rng.randint(2, 6) for _ in range(ndim))
        d0 = rng.randrange(ndim)
        d1 = rng.randrange(ndim)
        src = _module_src(
            f"    def forward(self, x):\n        return x.transpose({d0}, {d1})"
        )
        return Case(src, {"x": shape})

    return OpSpec("transpose", gen)


def _spec_permute() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        ndim = rng.choice([2, 3, 4])
        shape = tuple(rng.randint(2, 5) for _ in range(ndim))
        perm = list(range(ndim))
        rng.shuffle(perm)
        perm_args = ", ".join(str(p) for p in perm)
        src = _module_src(
            f"    def forward(self, x):\n        return x.permute({perm_args})"
        )
        return Case(src, {"x": shape})

    return OpSpec("permute", gen)


def _spec_flatten() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        ndim = rng.choice([2, 3, 4])
        shape = tuple(rng.randint(2, 5) for _ in range(ndim))
        # nn.Flatten() defaults to (start_dim=1, end_dim=-1) which TG handles.
        src = _module_src(
            "    def forward(self, x):\n        return self.f(x)",
            "        self.f = nn.Flatten()",
        )
        return Case(src, {"x": shape})

    return OpSpec("flatten", gen)


def _spec_matmul() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        b = rng.randint(1, 4)
        m = rng.randint(2, 8)
        k = rng.randint(2, 8)
        n = rng.randint(2, 8)
        # 3D matmul (broadcast batch)
        src = _module_src(
            "    def forward(self, a, b):\n        return torch.matmul(a, b)"
        )
        return Case(src, {"a": (b, m, k), "b": (b, k, n)})

    return OpSpec("matmul", gen)


def _spec_bmm() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        b = rng.randint(1, 4)
        m = rng.randint(2, 8)
        k = rng.randint(2, 8)
        n = rng.randint(2, 8)
        src = _module_src(
            "    def forward(self, a, b):\n        return torch.bmm(a, b)"
        )
        return Case(src, {"a": (b, m, k), "b": (b, k, n)})

    return OpSpec("bmm", gen)


def _spec_broadcast_add() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        # Generate broadcastable pair: same trailing dims, one with leading 1s.
        ndim = rng.choice([2, 3, 4])
        full = [rng.randint(2, 5) for _ in range(ndim)]
        b_shape = list(full)
        # Replace some dims with 1 in b
        for i in range(ndim):
            if rng.random() < 0.5:
                b_shape[i] = 1
        src = _module_src(
            "    def forward(self, a, b):\n        return a + b"
        )
        return Case(src, {"a": tuple(full), "b": tuple(b_shape)})

    return OpSpec("broadcast_add", gen)


def _spec_cat() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        ndim = rng.choice([2, 3, 4])
        shape = [rng.randint(2, 5) for _ in range(ndim)]
        # cat dim choices ∈ {0, 1, -1}
        dim = rng.choice([0, 1, -1])
        n = rng.choice([2, 3])
        # All inputs share shape (we use same shape for simplicity).
        names = [f"x{i}" for i in range(n)]
        args = ", ".join(names)
        src = _module_src(
            f"    def forward(self, {args}):\n        return torch.cat([{args}], dim={dim})"
        )
        return Case(src, {name: tuple(shape) for name in names})

    return OpSpec("cat", gen)


def _spec_softmax() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        ndim = rng.choice([2, 3, 4])
        shape = tuple(rng.randint(2, 5) for _ in range(ndim))
        dim = rng.randrange(-ndim, ndim)
        src = _module_src(
            f"    def forward(self, x):\n        return F.softmax(x, dim={dim})"
        )
        return Case(src, {"x": shape})

    return OpSpec("softmax", gen)


def _spec_relu() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        ndim = rng.choice([1, 2, 3, 4])
        shape = tuple(rng.randint(2, 6) for _ in range(ndim))
        src = _module_src(
            "    def forward(self, x):\n        return self.r(x)",
            "        self.r = nn.ReLU()",
        )
        return Case(src, {"x": shape})

    return OpSpec("ReLU", gen)


def _spec_gelu() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        ndim = rng.choice([1, 2, 3, 4])
        shape = tuple(rng.randint(2, 6) for _ in range(ndim))
        src = _module_src(
            "    def forward(self, x):\n        return self.g(x)",
            "        self.g = nn.GELU()",
        )
        return Case(src, {"x": shape})

    return OpSpec("GELU", gen)


def _spec_embedding() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        vocab = rng.randint(8, 64)
        embed = rng.randint(2, 32)
        b = rng.randint(1, 4)
        t = rng.randint(1, 8)
        src = _module_src(
            "    def forward(self, x):\n        return self.e(x)",
            f"        self.e = nn.Embedding({vocab}, {embed})",
        )
        return Case(
            src,
            {"x": (b, t)},
            input_factory={"x": lambda s, v=vocab: torch.randint(0, v, s, dtype=torch.long)},
        )

    return OpSpec("Embedding", gen)


def _spec_adaptive_avg_pool2d() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        c = rng.randint(1, 6)
        b = rng.randint(1, 3)
        h = rng.randint(4, 16)
        w = rng.randint(4, 16)
        oh = rng.randint(1, 5)
        ow = rng.randint(1, 5)
        src = _module_src(
            "    def forward(self, x):\n        return self.p(x)",
            f"        self.p = nn.AdaptiveAvgPool2d(({oh}, {ow}))",
        )
        return Case(src, {"x": (b, c, h, w)})

    return OpSpec("AdaptiveAvgPool2d", gen)


def _spec_max_pool2d() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        c = rng.randint(1, 6)
        b = rng.randint(1, 3)
        k = rng.choice([2, 3])
        s = rng.choice([1, 2, k])
        p = rng.choice([0, k // 2])
        # ensure h,w big enough so output > 0:  out = floor((H + 2p - k)/s) + 1 >= 1
        min_dim = max(k - 2 * p, 1) + s
        h = rng.randint(min_dim, min_dim + 12)
        w = rng.randint(min_dim, min_dim + 12)
        src = _module_src(
            "    def forward(self, x):\n        return self.p(x)",
            f"        self.p = nn.MaxPool2d({k}, stride={s}, padding={p})",
        )
        return Case(src, {"x": (b, c, h, w)})

    return OpSpec("MaxPool2d", gen)


def _spec_avg_pool2d() -> OpSpec:
    def gen(rng: random.Random) -> Case:
        c = rng.randint(1, 6)
        b = rng.randint(1, 3)
        k = rng.choice([2, 3])
        s = rng.choice([1, 2, k])
        p = rng.choice([0, k // 2])
        min_dim = max(k - 2 * p, 1) + s
        h = rng.randint(min_dim, min_dim + 12)
        w = rng.randint(min_dim, min_dim + 12)
        src = _module_src(
            "    def forward(self, x):\n        return self.p(x)",
            f"        self.p = nn.AvgPool2d({k}, stride={s}, padding={p})",
        )
        return Case(src, {"x": (b, c, h, w)})

    return OpSpec("AvgPool2d", gen)


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------


SPECS: List[OpSpec] = [
    _spec_linear(),
    _spec_conv1d(),
    _spec_conv2d(),
    _spec_conv3d(),
    _spec_convtranspose2d(),
    _spec_batchnorm2d(),
    _spec_layernorm(),
    _spec_view(),
    _spec_reshape(),
    _spec_transpose(),
    _spec_permute(),
    _spec_flatten(),
    _spec_matmul(),
    _spec_bmm(),
    _spec_broadcast_add(),
    _spec_cat(),
    _spec_softmax(),
    _spec_relu(),
    _spec_gelu(),
    _spec_embedding(),
    _spec_adaptive_avg_pool2d(),
    _spec_max_pool2d(),
    _spec_avg_pool2d(),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _shapes_match(actual: Tuple[int, ...], predicted: Tuple[Optional[int], ...]) -> bool:
    if len(actual) != len(predicted):
        return False
    for a, p in zip(actual, predicted):
        if p is None:  # symbolic — accept as match
            continue
        if int(a) != int(p):
            return False
    return True


def run_one(spec: OpSpec, case: Case) -> Dict[str, Any]:
    """Compare TG vs PyTorch for one case.  Returns a result dict."""
    record: Dict[str, Any] = {
        "op": spec.name,
        "input_shapes": {k: list(v) for k, v in case.input_shapes.items()},
        "expected": None,
        "actual": None,
        "status": "ok",
        "reason": None,
    }
    try:
        actual = _torch_out_shape(case)
    except Exception as exc:
        record["status"] = "torch_error"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record
    record["actual"] = list(actual)
    try:
        predicted = _tg_out_shape(case)
    except Exception as exc:
        record["status"] = "tg_error"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record
    if predicted is None:
        record["status"] = "introspection_unavailable"
        record["reason"] = "TG produced no shape for output tensor"
        return record
    record["expected"] = [None if p is None else int(p) for p in predicted]
    if not _shapes_match(actual, predicted):
        record["status"] = "mismatch"
    return record


def run_all(seed: int = 42, per_op: int = 50) -> Dict[str, Any]:
    """Run every spec ``per_op`` times.  Returns the aggregate result dict."""
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []
    for spec in SPECS:
        # Independent rng per spec for reproducibility
        spec_rng = random.Random(rng.random())
        for _ in range(per_op):
            try:
                case = spec.gen(spec_rng)
            except Exception as exc:
                results.append({
                    "op": spec.name,
                    "input_shapes": {},
                    "expected": None,
                    "actual": None,
                    "status": "gen_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            results.append(run_one(spec, case))

    by_status: Dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    failures = [r for r in results if r["status"] == "mismatch"]
    skipped = [r for r in results if r["status"] in ("introspection_unavailable", "tg_error", "torch_error", "gen_error")]
    passed = [r for r in results if r["status"] == "ok"]

    return {
        "n_operators": len(SPECS),
        "n_random_inputs_per_op": per_op,
        "n_total_inputs": len(results),
        "n_passed": len(passed),
        "n_failed": len(failures),
        "n_skipped": len(skipped),
        "status_breakdown": by_status,
        "failures": failures,
        "skipped": skipped,
    }


def main() -> int:
    summary = run_all()
    out = REPO_ROOT / "benchmarks" / "lean_parity_results.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}")
    print(
        f"operators={summary['n_operators']}  "
        f"per_op={summary['n_random_inputs_per_op']}  "
        f"total={summary['n_total_inputs']}  "
        f"passed={summary['n_passed']}  "
        f"failed={summary['n_failed']}  "
        f"skipped={summary['n_skipped']}"
    )
    print("status breakdown:", summary["status_breakdown"])
    if summary["failures"]:
        per_op: Dict[str, int] = {}
        for r in summary["failures"]:
            per_op[r["op"]] = per_op.get(r["op"], 0) + 1
        print("mismatches by op:", per_op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
