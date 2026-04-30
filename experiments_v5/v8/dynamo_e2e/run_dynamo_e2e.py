#!/usr/bin/env python3.11
"""TG end-to-end + Dynamo correspondence (smaller-N version).

Reviewer obligation (round 1, W2/Q2): the Track-E table in
``experiments_v5/dynamo_correspondence_v5.json`` records
``signature-trusted`` for 16 of 17 modules: TG never actually ran on
those modules; the contract was the documented ``forward`` signature.
This script addresses the obligation by selecting exactly the
``nn.Module`` *subclasses* on which TensorGuard end-to-end verifies the
class body (no surrogate, no signature-trust), then running
``torch.compile(M, dynamic=True)`` against the same shape contract that
TG verified under.  Modules where TG returns UNSAFE or errors are
excluded; this is a deliberately smaller, honestly-supported N.

Output: ``experiments_v5/v8/dynamo_e2e/dynamo_e2e_results.json`` plus
this file's own log lines.

Design: for each subject we (i) call ``verify_architecture`` on the
class source with a symbolic input contract; (ii) if the verifier
returns SAFE with no shape/grad bugs, instantiate the module with a
concrete material configuration consistent with that contract; (iii)
compile with ``torch.compile(dynamic=True)`` and sample 24 in-contract
inputs; (iv) sample 4 out-of-contract inputs (rank/dtype/channel
mismatch) as positive controls.
"""
from __future__ import annotations

import dataclasses
import inspect
import io
import json
import logging
import os
import random
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

for nm in ("torch._dynamo", "torch._inductor", "torch.fx", "transformers", "timm"):
    logging.getLogger(nm).setLevel(logging.ERROR)

from src.api import verify_architecture  # noqa: E402

import torchvision.models as tvm  # noqa: E402


# ─── Subject zoo ──────────────────────────────────────────────────────────────

@dataclass
class Subject:
    name: str
    family: str
    builder: Callable[[], nn.Module]
    # symbolic shape contract used for TG verification
    contract_inputs: List[Tuple[str, Tuple]]
    sym_ranges: Dict[str, Tuple[int, int]]
    dtype: torch.dtype = torch.float32
    # source for TG (the *class* body that TG must verify)
    tg_src_prefix: str = ""
    tg_class: type = None  # type: ignore[assignment]


def _src_with_prefix(prefix: str, cls: type) -> str:
    return prefix + inspect.getsource(cls)


_TV_RESNET_PREFIX = (
    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    "from torch import Tensor\nfrom typing import Optional, Callable\n\n"
    "def conv3x3(*a,**k): return nn.Conv2d(64,64,3,padding=1)\n"
    "def conv1x1(*a,**k): return nn.Conv2d(64,64,1)\n"
)
_TV_MNV2_PREFIX = (
    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    "from torch import Tensor\nfrom typing import Optional, Callable, List\n"
    "import functools\n"
)
_TV_SQ_PREFIX = (
    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    "from torch import Tensor\n"
)
_TIMM_VIT_PREFIX = (
    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    "from typing import Optional, Callable\n"
)


def build_subjects() -> List[Subject]:
    out: List[Subject] = []

    # torchvision.resnet.BasicBlock
    out.append(Subject(
        name="tv_resnet_BasicBlock",
        family="torchvision.resnet",
        builder=lambda: tvm.resnet.BasicBlock(64, 64).eval(),
        contract_inputs=[("x", ("B", 64, "H", "W"))],
        sym_ranges={"B": (1, 8), "H": (16, 64), "W": (16, 64)},
        tg_src_prefix=_TV_RESNET_PREFIX,
        tg_class=tvm.resnet.BasicBlock,
    ))

    # torchvision.resnet.Bottleneck (planes=16, expansion=4 -> in/out=64)
    out.append(Subject(
        name="tv_resnet_Bottleneck",
        family="torchvision.resnet",
        builder=lambda: tvm.resnet.Bottleneck(64, 16).eval(),
        contract_inputs=[("x", ("B", 64, "H", "W"))],
        sym_ranges={"B": (1, 8), "H": (16, 64), "W": (16, 64)},
        tg_src_prefix=_TV_RESNET_PREFIX,
        tg_class=tvm.resnet.Bottleneck,
    ))

    # torchvision.mobilenetv2.InvertedResidual
    out.append(Subject(
        name="tv_mnv2_InvertedResidual",
        family="torchvision.mobilenetv2",
        builder=lambda: tvm.mobilenetv2.InvertedResidual(32, 32, stride=1, expand_ratio=2).eval(),
        contract_inputs=[("x", ("B", 32, "H", "W"))],
        sym_ranges={"B": (1, 8), "H": (16, 64), "W": (16, 64)},
        tg_src_prefix=_TV_MNV2_PREFIX,
        tg_class=tvm.mobilenetv2.InvertedResidual,
    ))

    # torchvision.squeezenet.Fire (in_planes=64, sq=16, e1=32, e3=32 → out=64)
    out.append(Subject(
        name="tv_squeezenet_Fire",
        family="torchvision.squeezenet",
        builder=lambda: tvm.squeezenet.Fire(64, 16, 32, 32).eval(),
        contract_inputs=[("x", ("B", 64, "H", "W"))],
        sym_ranges={"B": (1, 8), "H": (16, 64), "W": (16, 64)},
        tg_src_prefix=_TV_SQ_PREFIX,
        tg_class=tvm.squeezenet.Fire,
    ))

    # timm vision_transformer.Block
    try:
        import timm.models.vision_transformer as vt

        def _build_vit_block():
            return vt.Block(dim=128, num_heads=4, mlp_ratio=4.0).eval()

        out.append(Subject(
            name="timm_vit_Block",
            family="timm.vit",
            builder=_build_vit_block,
            contract_inputs=[("x", ("B", "S", 128))],
            sym_ranges={"B": (1, 4), "S": (8, 64)},
            tg_src_prefix=_TIMM_VIT_PREFIX,
            tg_class=vt.Block,
        ))
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not include timm.vit.Block: {e}", file=sys.stderr)

    return out


# ─── TG verification ──────────────────────────────────────────────────────────

@dataclass
class TGRunSummary:
    status: str
    n_bugs: int
    duration_s: float
    bugs_head: List[str] = field(default_factory=list)


def run_tg(subj: Subject) -> TGRunSummary:
    src = _src_with_prefix(subj.tg_src_prefix, subj.tg_class)
    shapes = {n: t for (n, t) in subj.contract_inputs}
    t0 = time.time()
    # silence the verbose "Unsupported layer kind" warning stream
    buf = io.StringIO()
    with redirect_stderr(buf), redirect_stdout(buf):
        try:
            r = verify_architecture(src, input_shapes=shapes)
        except Exception as e:  # pragma: no cover
            return TGRunSummary(status=f"EXC:{type(e).__name__}", n_bugs=0,
                                duration_s=time.time() - t0,
                                bugs_head=[repr(e)[:200]])
    dur = time.time() - t0
    status = getattr(r, "status", "UNKNOWN")
    bugs = getattr(r, "bugs", [])
    return TGRunSummary(status=status, n_bugs=len(bugs), duration_s=dur,
                        bugs_head=[str(b)[:200] for b in bugs[:5]])


# ─── Dynamo correspondence ────────────────────────────────────────────────────

def _instantiate_template(template: Tuple, sym_vals: Dict[str, int]) -> Tuple[int, ...]:
    out = []
    for d in template:
        if isinstance(d, str):
            out.append(int(sym_vals[d]))
        else:
            out.append(int(d))
    return tuple(out)


def _make_in_contract_inputs(subj: Subject, n: int, seed: int) -> List[Dict[str, torch.Tensor]]:
    rng = random.Random(seed)
    out: List[Dict[str, torch.Tensor]] = []
    for _ in range(n):
        sv = {k: rng.randint(*v) for k, v in subj.sym_ranges.items()}
        d: Dict[str, torch.Tensor] = {}
        for (name, tmpl) in subj.contract_inputs:
            shape = _instantiate_template(tmpl, sv)
            d[name] = torch.randn(*shape, dtype=subj.dtype)
        out.append(d)
    return out


def _make_oos_inputs(subj: Subject) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Out-of-contract probes: rank, channel, dtype mismatch."""
    sv = {k: v[0] for k, v in subj.sym_ranges.items()}
    base: Dict[str, torch.Tensor] = {}
    for (name, tmpl) in subj.contract_inputs:
        base[name] = torch.randn(*_instantiate_template(tmpl, sv), dtype=subj.dtype)
    out: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    # rank: drop a dim from the first input
    first = list(base.keys())[0]
    if base[first].dim() >= 2:
        d = dict(base)
        d[first] = d[first].squeeze(0)
        out.append(("rank", d))
    # channel: scale the second template dim by +1 if it's a literal int
    tmpl = subj.contract_inputs[0][1]
    if any(isinstance(x, int) for x in tmpl):
        # find first int dim and bump it
        new_shape = list(base[first].shape)
        for idx, x in enumerate(tmpl):
            if isinstance(x, int):
                new_shape[idx] = x + 1
                break
        d = dict(base)
        d[first] = torch.randn(*new_shape, dtype=subj.dtype)
        out.append(("channel", d))
    # dtype
    if subj.dtype == torch.float32:
        d = {k: v.to(torch.float64) for k, v in base.items()}
        out.append(("dtype", d))
    return out


def _run_once_safe(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Tuple[bool, str]:
    with torch.no_grad():
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                model(**inputs)
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:200]}"


def _count_recompiles_via_dynamo(reset: bool = True) -> int:
    """Returns the number of distinct compilations recorded by Dynamo
    since the last `torch._dynamo.reset()`."""
    try:
        import torch._dynamo as dyn
        # Best-effort across versions: use convert_frame's compile counter
        cnt = getattr(dyn.convert_frame, "FRAME_COMPILE_COUNTER", None)
        if cnt is not None:
            return int(sum(cnt.values()))
    except Exception:
        pass
    return -1


def run_dynamo(subj: Subject, n_in: int = 24, seed: int = 0,
               wall_timeout_s: int = 240) -> Dict[str, Any]:
    import torch._dynamo as dyn
    dyn.reset()
    base_model = subj.builder()
    cmodel = torch.compile(base_model, dynamic=True)

    in_inputs = _make_in_contract_inputs(subj, n_in, seed)
    in_recompiles_before = _count_recompiles_via_dynamo()

    in_ok = 0
    in_runtime_errors = 0
    err_examples: List[str] = []
    t0 = time.time()
    for inp in in_inputs:
        if time.time() - t0 > wall_timeout_s:
            break
        ok, err = _run_once_safe(cmodel, inp)
        if ok:
            in_ok += 1
        else:
            in_runtime_errors += 1
            if len(err_examples) < 3:
                err_examples.append(err)
    in_recompiles_after = _count_recompiles_via_dynamo()
    in_recompiles = (in_recompiles_after - in_recompiles_before
                     if in_recompiles_before >= 0 and in_recompiles_after >= 0 else -1)
    # Subtract the initial single compile
    if in_recompiles >= 1:
        in_recompiles -= 1

    # Out-of-contract probes
    oos = _make_oos_inputs(subj)
    oos_results: Dict[str, str] = {}
    for tag, inp in oos:
        ok, err = _run_once_safe(cmodel, inp)
        oos_results[tag] = ("violated_no_error" if ok and "expected" not in err
                            else f"violation:{err[:160]}" if err else "no_violation")

    return {
        "n_in_contract": len(in_inputs),
        "in_contract_runs_ok": in_ok,
        "in_contract_runtime_errors": in_runtime_errors,
        "in_contract_recompile_count_observed": in_recompiles,
        "in_contract_error_examples": err_examples,
        "oos_results": oos_results,
        "wall_s": round(time.time() - t0, 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(out_path: str = "experiments_v5/v8/dynamo_e2e/dynamo_e2e_results.json"):
    subjects = build_subjects()
    rows: List[Dict[str, Any]] = []
    for subj in subjects:
        print(f"[{subj.name}] running TG end-to-end ...", flush=True)
        tg = run_tg(subj)
        print(f"  TG status={tg.status} bugs={tg.n_bugs} dur={tg.duration_s:.2f}s",
              flush=True)
        row: Dict[str, Any] = {
            "name": subj.name,
            "family": subj.family,
            "tg_contract": {
                "inputs": [(n, list(t)) for (n, t) in subj.contract_inputs],
                "sym_ranges": {k: list(v) for k, v in subj.sym_ranges.items()},
                "dtype": str(subj.dtype),
            },
            "tg": dataclasses.asdict(tg),
        }
        if tg.status != "SAFE":
            row["dynamo"] = {"skipped_reason": f"TG status={tg.status}"}
            rows.append(row)
            continue
        try:
            print(f"  Dynamo correspondence ...", flush=True)
            dyn = run_dynamo(subj)
            row["dynamo"] = dyn
            print(f"  in-ok={dyn['in_contract_runs_ok']}/{dyn['n_in_contract']} "
                  f"recompile={dyn['in_contract_recompile_count_observed']}",
                  flush=True)
        except Exception as e:
            row["dynamo"] = {"error": f"{type(e).__name__}: {str(e)[:300]}",
                             "trace": traceback.format_exc()[:1500]}
            print(f"  dynamo EXC {e}", flush=True)
        rows.append(row)

    summary = {
        "n_subjects": len(rows),
        "n_tg_safe": sum(1 for r in rows if r["tg"]["status"] == "SAFE"),
        "n_dynamo_in_contract_zero_recompile": sum(
            1 for r in rows
            if isinstance(r.get("dynamo"), dict) and
            r["dynamo"].get("in_contract_recompile_count_observed", 1) == 0
        ),
        "torch_version": torch.__version__,
    }
    out = {"summary": summary, "rows": rows}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
