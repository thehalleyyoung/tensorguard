#!/usr/bin/env python3.11
"""TG end-to-end + Dynamo correspondence for HuggingFace Transformers subjects.

Companion to run_dynamo_e2e.py (which covers torchvision and timm subjects).
Adds subjects from the HuggingFace Transformers library (T5 and BERT families)
to extend the domain-breadth of the necessary-direction Dynamo audit.

Design follows run_dynamo_e2e.py exactly:
  (i)  verify_architecture on the class source with a symbolic input contract;
  (ii) torch.compile(dynamic=True) on >=16 in-contract inputs;
  (iii) 3 out-of-contract probes (rank/dtype/channel mismatch).

Output: experiments_v5/v8/dynamo_e2e/dynamo_e2e_hf_results.json
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


# ─── Subject zoo ──────────────────────────────────────────────────────────────

@dataclass
class Subject:
    name: str
    family: str
    builder: Callable[[], nn.Module]
    contract_inputs: List[Tuple[str, Tuple]]
    sym_ranges: Dict[str, Tuple[int, int]]
    dtype: torch.dtype = torch.float32
    tg_src_prefix: str = ""
    tg_class: type = None  # type: ignore[assignment]


def _src_with_prefix(prefix: str, cls: type) -> str:
    return prefix + inspect.getsource(cls)


_HF_BASE_PREFIX = (
    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    "from torch import Tensor\nfrom typing import Optional, Tuple, Union\n\n"
)


def build_subjects() -> List[Subject]:
    out: List[Subject] = []

    # ── T5 family ────────────────────────────────────────────────────────────
    try:
        from transformers.models.t5.modeling_t5 import T5LayerNorm, T5DenseActDense, T5Config

        # Subject 1: T5LayerNorm
        # Simple scale-only layer norm (no mean subtraction).
        # Contract: hidden_states of shape (B, S, 128)
        out.append(Subject(
            name="hf_t5_T5LayerNorm",
            family="huggingface.t5",
            builder=lambda: T5LayerNorm(128).eval(),
            contract_inputs=[("hidden_states", ("B", "S", 128))],
            sym_ranges={"B": (1, 8), "S": (8, 64)},
            tg_src_prefix=_HF_BASE_PREFIX,
            tg_class=T5LayerNorm,
        ))

        # Subject 2: T5DenseActDense
        # Two-layer dense block (wi, wo) with activation and dropout=0.
        # Contract: hidden_states of shape (B, S, 128)
        def _build_t5_dense():
            cfg = T5Config(d_model=128, d_ff=512, dropout_rate=0.0)
            return T5DenseActDense(cfg).eval()

        out.append(Subject(
            name="hf_t5_T5DenseActDense",
            family="huggingface.t5",
            builder=_build_t5_dense,
            contract_inputs=[("hidden_states", ("B", "S", 128))],
            sym_ranges={"B": (1, 8), "S": (8, 64)},
            tg_src_prefix=_HF_BASE_PREFIX,
            tg_class=T5DenseActDense,
        ))

    except Exception as e:
        print(f"[warn] could not include T5 subjects: {e}", file=sys.stderr)

    # ── BERT family ──────────────────────────────────────────────────────────
    try:
        from transformers.models.bert.modeling_bert import BertIntermediate, BertConfig

        # Subject 3: BertIntermediate
        # Projects hidden_size→intermediate_size and applies a GELU activation.
        # Contract: hidden_states of shape (B, S, 128)
        def _build_bert_intermediate():
            cfg = BertConfig(hidden_size=128, intermediate_size=512,
                             num_attention_heads=4, hidden_act="gelu")
            return BertIntermediate(cfg).eval()

        out.append(Subject(
            name="hf_bert_BertIntermediate",
            family="huggingface.bert",
            builder=_build_bert_intermediate,
            contract_inputs=[("hidden_states", ("B", "S", 128))],
            sym_ranges={"B": (1, 8), "S": (8, 64)},
            tg_src_prefix=_HF_BASE_PREFIX,
            tg_class=BertIntermediate,
        ))

    except Exception as e:
        print(f"[warn] could not include BERT subjects: {e}", file=sys.stderr)

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
    buf = io.StringIO()
    with redirect_stderr(buf), redirect_stdout(buf):
        try:
            r = verify_architecture(src, input_shapes=shapes)
        except Exception as e:
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
    first = list(base.keys())[0]
    if base[first].dim() >= 2:
        d = dict(base)
        d[first] = d[first].squeeze(0)
        out.append(("rank", d))
    tmpl = subj.contract_inputs[0][1]
    if any(isinstance(x, int) for x in tmpl):
        new_shape = list(base[first].shape)
        for idx, x in enumerate(tmpl):
            if isinstance(x, int):
                new_shape[idx] = x + 1
                break
        d = dict(base)
        d[first] = torch.randn(*new_shape, dtype=subj.dtype)
        out.append(("channel", d))
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


def _count_recompiles_via_dynamo() -> int:
    try:
        import torch._dynamo as dyn
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
    if in_recompiles >= 1:
        in_recompiles -= 1

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

def main(out_path: str = "experiments_v5/v8/dynamo_e2e/dynamo_e2e_hf_results.json"):
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
            dyn_result = run_dynamo(subj)
            row["dynamo"] = dyn_result
            print(f"  in-ok={dyn_result['in_contract_runs_ok']}/{dyn_result['n_in_contract']} "
                  f"recompile={dyn_result['in_contract_recompile_count_observed']}",
                  flush=True)
        except Exception as e:
            row["dynamo"] = {"error": f"{type(e).__name__}: {str(e)[:300]}",
                             "trace": traceback.format_exc()[:1500]}
            print(f"  dynamo EXC {e}", flush=True)
        rows.append(row)

    n_safe = sum(1 for r in rows if r["tg"]["status"] == "SAFE")
    summary = {
        "n_subjects": len(rows),
        "n_tg_safe": n_safe,
        "n_dynamo_in_contract_zero_recompile": sum(
            1 for r in rows
            if isinstance(r.get("dynamo"), dict) and
            r["dynamo"].get("in_contract_recompile_count_observed", 1) == 0
        ),
        "torch_version": torch.__version__,
        "families": sorted(set(r["family"] for r in rows)),
    }
    out_data = {"summary": summary, "rows": rows}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
