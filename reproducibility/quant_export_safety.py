"""Deterministic harness: quantization & export safety checks vs real PyTorch.

For each case we record the *static* verdict of
:mod:`src.quant_export_checks` and the *live* behavior of the matching real
PyTorch operation:

* **export cases** -- we run ``torch.export.export`` on the actual module and
  record whether it traced cleanly (``live_exports_clean``).  A case is
  *consistent* when the static analyzer flags an export hazard **iff** real
  ``torch.export`` fails.

* **quant cases** -- for the arithmetic hazard we directly build quantized
  tensors and attempt the bare ``+`` / ``*`` op, recording whether real
  PyTorch raised (``live_raises``).  Structural boundary hazards
  (missing de/quant stub) have no single runtime op to trip, so their
  ``live_raises`` is recorded as ``null``.

Only booleans / verdict strings are written to the artifact (no timings, no
floats) so the JSON is byte-identical across machines and runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.quant_export_checks import (  # noqa: E402
    analyze_export_safety,
    analyze_quantization,
)

OUT_JSON = REPO / "reproducibility" / "quant_export_safety.json"
OUT_MD = REPO / "reproducibility" / "quant_export_safety.md"


# --- export cases: (name, source, builder) -------------------------------
def _clean_linear():
    import torch.nn as nn

    class Clean(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(4, 4)

        def forward(self, x):
            return self.l(x)

    return Clean(), (4,), (2, 4)


def _data_dependent_branch():
    import torch.nn as nn

    class DD(nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x * 2
            return x

    return DD(), (4,), (4,)


def _tensor_to_scalar_item():
    import torch.nn as nn

    class ItemMod(nn.Module):
        def forward(self, x):
            k = int(x.sum().item())
            return x + k

    return ItemMod(), (4,), (4,)


def _data_dependent_loop():
    import torch.nn as nn

    class LoopMod(nn.Module):
        def forward(self, x):
            n = int(x.sum().item())
            for _ in range(n):
                x = x + 1
            return x

    return LoopMod(), (4,), (4,)


EXPORT_CASES = [
    (
        "clean_linear",
        '''
import torch.nn as nn
class Clean(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)
    def forward(self, x):
        return self.l(x)
''',
        _clean_linear,
    ),
    (
        "data_dependent_branch",
        '''
import torch.nn as nn
class DD(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        return x
''',
        _data_dependent_branch,
    ),
    (
        "tensor_to_scalar_item",
        '''
import torch.nn as nn
class ItemMod(nn.Module):
    def forward(self, x):
        k = int(x.sum().item())
        return x + k
''',
        _tensor_to_scalar_item,
    ),
    (
        "data_dependent_loop",
        '''
import torch.nn as nn
class LoopMod(nn.Module):
    def forward(self, x):
        n = int(x.sum().item())
        for _ in range(n):
            x = x + 1
        return x
''',
        _data_dependent_loop,
    ),
]


# --- quant cases: (name, source, runtime_kind) ---------------------------
# runtime_kind in {"arith_add", "arith_mul", None}
QUANT_CASES = [
    (
        "quant_add_no_floatfunctional",
        '''
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, a, b):
        a = self.quant(a)
        b = self.quant(b)
        c = a + b
        return self.dequant(c)
''',
        "arith_add",
    ),
    (
        "quant_add_with_floatfunctional",
        '''
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.nn.quantized import FloatFunctional
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.ff = FloatFunctional()
    def forward(self, a, b):
        a = self.quant(a)
        b = self.quant(b)
        c = self.ff.add(a, b)
        return self.dequant(c)
''',
        None,
    ),
    (
        "missing_dequantstub",
        '''
import torch.nn as nn
from torch.ao.quantization import QuantStub
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.l = nn.Linear(4, 4)
    def forward(self, x):
        return self.l(self.quant(x))
''',
        None,
    ),
    (
        "plain_float_model",
        '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)
    def forward(self, x):
        return self.l(x) + self.l(x)
''',
        None,
    ),
]


def _live_export_clean(builder) -> bool:
    import torch

    module, in_shape, call_shape = builder()
    example = torch.randn(*call_shape)
    try:
        torch.export.export(module, (example,))
        return True
    except Exception:
        return False


def _live_quant_arith_raises(kind: str) -> bool:
    import torch

    q1 = torch.quantize_per_tensor(torch.randn(4), 0.1, 0, torch.quint8)
    q2 = torch.quantize_per_tensor(torch.randn(4), 0.1, 0, torch.quint8)
    try:
        if kind == "arith_add":
            _ = q1 + q2
        elif kind == "arith_mul":
            _ = q1 * q2
        else:
            return False
        return False
    except (NotImplementedError, RuntimeError):
        return True


def measure() -> dict:
    export_rows = []
    for name, source, builder in EXPORT_CASES:
        hazards = analyze_export_safety(source)
        static_flagged = len(hazards) > 0
        live_clean = _live_export_clean(builder)
        export_rows.append(
            {
                "name": name,
                "static_export_hazard": static_flagged,
                "static_kinds": sorted(h.kind.value for h in hazards),
                "live_exports_clean": live_clean,
                # consistent: flagged a hazard <=> export actually failed
                "consistent": static_flagged == (not live_clean),
            }
        )

    quant_rows = []
    for name, source, runtime_kind in QUANT_CASES:
        hazards = analyze_quantization(source)
        static_flagged = len(hazards) > 0
        if runtime_kind is None:
            live_raises = None
        else:
            live_raises = _live_quant_arith_raises(runtime_kind)
        # consistency only assertable for cases with a concrete runtime op
        if live_raises is None:
            consistent = True
        else:
            consistent = static_flagged == live_raises
        quant_rows.append(
            {
                "name": name,
                "static_quant_hazard": static_flagged,
                "static_kinds": sorted(h.kind.value for h in hazards),
                "live_raises": live_raises,
                "consistent": consistent,
            }
        )

    return {
        "export_cases": export_rows,
        "quant_cases": quant_rows,
        "n_export_cases": len(export_rows),
        "n_quant_cases": len(quant_rows),
        "all_export_consistent": all(r["consistent"] for r in export_rows),
        "all_quant_consistent": all(r["consistent"] for r in quant_rows),
    }


def render_markdown(data: dict) -> str:
    lines = [
        "# Quantization & Export Safety: static verdict vs real PyTorch",
        "",
        "TensorGuard's `src/quant_export_checks.py` flags two deployment-time "
        "hazard classes that the float shape/device/dtype verifier does not "
        "target. Each row is cross-checked against the **live** behavior of "
        "real PyTorch (`torch.export` tracing; quantized-tensor arithmetic).",
        "",
        "## Export safety (`analyze_export_safety` vs `torch.export.export`)",
        "",
        "| case | static export hazard | live exports clean | consistent |",
        "| --- | --- | --- | --- |",
    ]
    for r in data["export_cases"]:
        lines.append(
            f"| `{r['name']}` | {r['static_export_hazard']} | "
            f"{r['live_exports_clean']} | {r['consistent']} |"
        )
    lines += [
        "",
        "A case is **consistent** when the static analyzer flags an export "
        "hazard if and only if real `torch.export` fails to trace.",
        "",
        "## Quantization placement "
        "(`analyze_quantization` vs quantized-tensor ops)",
        "",
        "| case | static quant hazard | live raises | consistent |",
        "| --- | --- | --- | --- |",
    ]
    for r in data["quant_cases"]:
        lr = "null" if r["live_raises"] is None else str(r["live_raises"])
        lines.append(
            f"| `{r['name']}` | {r['static_quant_hazard']} | {lr} | "
            f"{r['consistent']} |"
        )
    lines += [
        "",
        f"All export cases consistent: **{data['all_export_consistent']}**. "
        f"All quant cases consistent: **{data['all_quant_consistent']}**.",
        "",
        "`live raises = null` marks structural boundary hazards (e.g. a missing "
        "`DeQuantStub`) that have no single runtime op to trip; they are "
        "verified statically only.",
        "",
    ]
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        old_json = OUT_JSON.read_text() if OUT_JSON.exists() else ""
        old_md = OUT_MD.read_text() if OUT_MD.exists() else ""
        if old_json != new_json or old_md != new_md:
            print("MISMATCH: quant_export_safety artifacts differ from committed")
            return 1
        print("OK: quant_export_safety artifacts byte-identical")
        return 0
    OUT_JSON.write_text(new_json)
    OUT_MD.write_text(new_md)
    print(f"Wrote {OUT_JSON.name} and {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(run(check=args.check))
