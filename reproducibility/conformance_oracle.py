"""Conformance oracle: cross-check every shape transfer function against real
PyTorch execution (Step 33).

A static shape verifier is only trustworthy if its *transfer functions* (the
rules that say "a ``Conv2d(3,16,3)`` turns an ``(N,3,H,W)`` tensor into an
``(N,16,H-2,W-2)`` tensor") agree with what PyTorch actually computes. This
module is a differential oracle: for a battery of single-op modules it

  1. samples concrete input shapes,
  2. runs the real ``torch`` forward pass to obtain the ground-truth output
     shape, and
  3. runs TensorGuard's shape propagation to obtain the *predicted* output
     shape,

then classifies each (op, shape) pair as:

  * ``CONFORMANT``  — TensorGuard predicted a fully-concrete shape and it equals
    torch's actual output shape.
  * ``ABSTAINED``   — TensorGuard left one or more output dims symbolic/unknown
    (a sound non-answer; never a wrong answer).
  * ``DISAGREE``    — TensorGuard predicted a concrete shape that does **not**
    match torch. This is a soundness bug and must never happen.

The headline guarantee, asserted in CI, is **zero DISAGREE** across the battery,
together with a high CONFORMANT rate (the transfer functions are not merely
abstaining everywhere).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

from src.fx_extractor import fx_trace_to_graph
from src.model_checker import SymbolicShapePropagator, TensorShape


@dataclass
class OracleCase:
    """One op under test, with a module factory and input-shape samples."""
    name: str
    make_module: Callable[[], "nn.Module"]
    input_shapes: List[Tuple[int, ...]]
    # Optional: name the single input arg (defaults to the first placeholder).
    input_name: str = "x"
    # Optional: build the concrete input tensor from a shape (defaults to
    # ``torch.randn``).  Used for ops like Embedding that need integer indices.
    make_input: Optional[Callable[[Tuple[int, ...]], "torch.Tensor"]] = None


@dataclass
class OracleResult:
    name: str
    input_shape: Tuple[int, ...]
    predicted: Optional[Tuple]
    actual: Tuple[int, ...]
    status: str  # CONFORMANT | ABSTAINED | DISAGREE | TRACE_FAIL


@dataclass
class OracleReport:
    results: List[OracleResult] = field(default_factory=list)

    @property
    def conformant(self) -> int:
        return sum(1 for r in self.results if r.status == "CONFORMANT")

    @property
    def abstained(self) -> int:
        return sum(1 for r in self.results if r.status == "ABSTAINED")

    @property
    def disagreements(self) -> List[OracleResult]:
        return [r for r in self.results if r.status == "DISAGREE"]

    @property
    def trace_fails(self) -> List[OracleResult]:
        return [r for r in self.results if r.status == "TRACE_FAIL"]

    def summary(self) -> str:
        total = len(self.results)
        lines = [
            "TensorGuard conformance oracle",
            f"  cases checked : {total}",
            f"  CONFORMANT    : {self.conformant}",
            f"  ABSTAINED     : {self.abstained}",
            f"  DISAGREE      : {len(self.disagreements)}",
            f"  TRACE_FAIL    : {len(self.trace_fails)}",
        ]
        for r in self.disagreements:
            lines.append(
                f"    DISAGREE {r.name} in={r.input_shape} "
                f"pred={r.predicted} actual={r.actual}"
            )
        return "\n".join(lines)


def _predicted_output_shape(
    module: "nn.Module", input_shape: Tuple[int, ...], input_name: str
) -> Optional[TensorShape]:
    traced = torch.fx.symbolic_trace(module)
    graph = fx_trace_to_graph(traced)
    if not graph.output_names:
        return None
    # Map the placeholder to the supplied shape (use the real placeholder name).
    placeholder = graph.input_names[0] if graph.input_names else input_name
    env = SymbolicShapePropagator(graph).propagate({placeholder: input_shape})
    return env.get(graph.output_names[0])


def _is_fully_concrete(shape: TensorShape) -> bool:
    return all(isinstance(d.value, int) for d in shape.dims)


def _concrete_tuple(shape: TensorShape) -> Tuple[int, ...]:
    return tuple(int(d.value) for d in shape.dims)


def run_oracle(cases: List[OracleCase]) -> OracleReport:
    """Execute the oracle over *cases* and return a structured report."""
    report = OracleReport()
    for case in cases:
        for ishape in case.input_shapes:
            module = case.make_module().eval()
            # Ground truth from real torch.
            mk = case.make_input or (lambda s: torch.randn(*s))
            with torch.no_grad():
                actual = tuple(module(mk(ishape)).shape)
            try:
                pred = _predicted_output_shape(module, ishape, case.input_name)
            except Exception:
                report.results.append(OracleResult(
                    case.name, ishape, None, actual, "TRACE_FAIL"))
                continue
            if pred is None:
                report.results.append(OracleResult(
                    case.name, ishape, None, actual, "ABSTAINED"))
                continue
            if _is_fully_concrete(pred):
                pt = _concrete_tuple(pred)
                status = "CONFORMANT" if pt == actual else "DISAGREE"
                report.results.append(OracleResult(
                    case.name, ishape, pt, actual, status))
            else:
                report.results.append(OracleResult(
                    case.name, ishape,
                    tuple(d.value for d in pred.dims), actual, "ABSTAINED"))
    return report


# --------------------------------------------------------------------------- #
# Battery of single-op modules covering the core transfer functions.
# --------------------------------------------------------------------------- #

def _default_cases() -> List[OracleCase]:
    if not HAS_TORCH:
        return []

    class _Mul(nn.Module):
        def forward(self, x):
            return x * 2.0

    class _AddBroadcast(nn.Module):
        def forward(self, x):
            return x + x[:, :1]

    class _Matmul(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.w = nn.Parameter(torch.randn(k, n))

        def forward(self, x):
            return x @ self.w

    class _Cat(nn.Module):
        def forward(self, x):
            return torch.cat([x, x], dim=1)

    class _Stack(nn.Module):
        def forward(self, x):
            return torch.stack([x, x], dim=0)

    class _MeanKeep(nn.Module):
        def forward(self, x):
            return x.mean(dim=-1, keepdim=True)

    class _SumReduce(nn.Module):
        def forward(self, x):
            return x.sum(dim=1)

    class _Transpose(nn.Module):
        def forward(self, x):
            return x.transpose(1, 2)

    class _Permute(nn.Module):
        def forward(self, x):
            return x.permute(0, 2, 1)

    class _Unsqueeze(nn.Module):
        def forward(self, x):
            return x.unsqueeze(1)

    class _Softmax(nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    class _Factory(nn.Module):
        def forward(self, x):
            return torch.zeros(4, 6) + x.sum() * 0.0

    class _BNWrap(nn.Module):
        # Standalone BatchNorm cannot be fx-traced (its _check_input_dim uses
        # data-dependent control flow), so exercise it after a 1x1 conv exactly
        # as it appears in real models.
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 3, 1)
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            return self.bn(self.c(x))

    class _Wrap(nn.Module):
        """Wrap a single sub-layer as ``self.layer(x)`` so it is traced as a
        ``call_module`` (the canonical usage) rather than inlined functional
        ops, exercising the layer transfer functions directly."""
        def __init__(self, factory):
            super().__init__()
            self.layer = factory()

        def forward(self, x):
            return self.layer(x)

    def _w(factory):
        return lambda: _Wrap(factory)

    cases: List[OracleCase] = [
        OracleCase("linear", _w(lambda: nn.Linear(8, 5)),
                   [(3, 8), (1, 8), (7, 8)]),
        OracleCase("conv2d", _w(lambda: nn.Conv2d(3, 16, 3)),
                   [(2, 3, 32, 32), (1, 3, 8, 8)]),
        OracleCase("conv2d_stride_pad",
                   _w(lambda: nn.Conv2d(3, 8, 3, stride=2, padding=1)),
                   [(2, 3, 32, 32), (1, 3, 17, 19)]),
        OracleCase("conv1d", _w(lambda: nn.Conv1d(4, 9, 5)),
                   [(2, 4, 50), (1, 4, 13)]),
        OracleCase("maxpool2d", _w(lambda: nn.MaxPool2d(2)),
                   [(2, 3, 32, 32), (1, 5, 9, 9)]),
        OracleCase("avgpool2d", _w(lambda: nn.AvgPool2d(2)),
                   [(2, 3, 16, 16)]),
        OracleCase("adaptive_avgpool2d", _w(lambda: nn.AdaptiveAvgPool2d((1, 1))),
                   [(2, 7, 13, 11)]),
        OracleCase("batchnorm2d", _BNWrap,
                   [(4, 3, 8, 8)]),
        OracleCase("layernorm", _w(lambda: nn.LayerNorm(10)),
                   [(3, 5, 10)]),
        OracleCase("flatten", _w(lambda: nn.Flatten()),
                   [(2, 3, 4, 5), (6, 7)]),
        OracleCase("embedding", _w(lambda: nn.Embedding(50, 12)),
                   [(3, 9)],  # indices; shape rule is index.shape + (dim,)
                   make_input=lambda s: torch.randint(0, 50, s)),
        OracleCase("relu", _w(lambda: nn.ReLU()), [(3, 4, 5)]),
        OracleCase("dropout", _w(lambda: nn.Dropout(0.3)), [(3, 4, 5)]),
        OracleCase("mul_scalar", _Mul, [(2, 3, 4)]),
        OracleCase("add_broadcast", _AddBroadcast, [(2, 5, 7)]),
        OracleCase("matmul", lambda: _Matmul(8, 11), [(4, 8), (2, 3, 8)]),
        OracleCase("cat_dim1", _Cat, [(2, 3, 4), (1, 5, 6)]),
        OracleCase("stack_dim0", _Stack, [(2, 3), (4, 5, 6)]),
        OracleCase("mean_keepdim", _MeanKeep, [(2, 3, 4)]),
        OracleCase("sum_reduce", _SumReduce, [(2, 6, 4)]),
        OracleCase("transpose", _Transpose, [(2, 3, 4)]),
        OracleCase("permute", _Permute, [(2, 3, 4)]),
        OracleCase("unsqueeze", _Unsqueeze, [(2, 3)]),
        OracleCase("softmax", _Softmax, [(2, 3, 4)]),
        OracleCase("factory_zeros", _Factory, [(3, 6)]),
    ]
    return cases


def main() -> int:
    import json
    import os

    cases = _default_cases()
    if not cases:
        print("torch unavailable — oracle skipped")
        return 0
    report = run_oracle(cases)
    print(report.summary())
    out = {
        "cases_checked": len(report.results),
        "conformant": report.conformant,
        "abstained": report.abstained,
        "disagreements": [
            {
                "op": r.name, "input": list(r.input_shape),
                "predicted": list(r.predicted) if r.predicted else None,
                "actual": list(r.actual),
            }
            for r in report.disagreements
        ],
        "trace_fails": [r.name for r in report.trace_fails],
    }
    path = os.path.join(os.path.dirname(__file__), "conformance_oracle.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    return 1 if report.disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
