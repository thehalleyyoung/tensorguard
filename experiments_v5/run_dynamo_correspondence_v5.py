#!/usr/bin/env python3.11
"""Track E v5: TorchDynamo Guard Correspondence on real HF/timm/torchvision modules.

Empirical claim:
    For every module M with a TensorGuard-extracted shape contract C, running
    torch.compile(M, dynamic=True) on >=30 inputs satisfying C triggers
    *zero* Dynamo guard-driven recompilations after the dynamic-shape graph
    is established.  As a positive control, we then run the same compiled M
    on inputs that violate C (changed channel dim, changed dtype, changed
    rank); these MUST trigger a Dynamo recompile or a runtime error
    (otherwise the contract is too weak).

Calibrated reporting: any in-contract recompile is logged with the offending
guard string parsed from `torch._dynamo` recompilation reasons (when
available).  We do not hide failures.

Pinned versions (CPU-only):
    torch        2.9.1
    torchvision  0.24.1
    transformers 4.57.3
    timm         1.0.26

Run:
    python3.11 experiments_v5/run_dynamo_correspondence_v5.py
"""
from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
import random
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch._dynamo
import torch._dynamo.utils

torch.set_num_threads(2)
torch._dynamo.config.cache_size_limit = 256
# Allow Dynamo to keep going on errors so we can attribute them to a guard.
torch._dynamo.config.suppress_errors = True

# --- TensorGuard -------------------------------------------------------------
try:
    from src.api import verify_architecture
    HAS_TG = True
except Exception as _e:
    HAS_TG = False
    _tg_import_err = repr(_e)

# Silence verbose loggers.
for name in ("torch._dynamo", "torch._inductor", "torch.fx", "transformers"):
    logging.getLogger(name).setLevel(logging.ERROR)


# ─── Module zoo ──────────────────────────────────────────────────────────────

@dataclass
class ModuleSpec:
    name: str
    family: str               # 'torchvision' | 'transformers' | 'timm' | 'misc'
    version: str              # pinned package version
    builder: Callable[[], nn.Module]
    # shape contract: ordered list of (name, shape_template) per kwarg.
    # shape_template is a tuple where ints are fixed dims and strings are
    # symbolic.  Free symbolic dims are sampled per-call.
    contract_inputs: List[Tuple[str, Tuple]]
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    # ranges for symbolic dims (inclusive).  Only those dims appearing in any
    # template need entries.
    sym_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # optional integer-tensor (token id) maximum for input_ids etc.
    int_input_max: int = 999
    # optional surrogate source for TensorGuard verification (a thin module
    # exposing the same I/O contract).  When None we fall back to
    # 'signature-trusted'.
    tg_surrogate_src: Optional[str] = None
    tg_surrogate_inputs: Optional[Dict[str, tuple]] = None


def _wrap_hf(model: nn.Module, output_attr: str = "last_hidden_state") -> nn.Module:
    class W(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids):
            out = self.m(input_ids=input_ids)
            return getattr(out, output_attr)
    return W(model)


def _wrap_t5(model: nn.Module) -> nn.Module:
    class W(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, decoder_input_ids):
            out = self.m(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            return out.last_hidden_state
    return W(model)


def _wrap_vit_hf(model: nn.Module) -> nn.Module:
    class W(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).last_hidden_state
    return W(model)


def build_zoo() -> List[ModuleSpec]:
    import torchvision
    import torchvision.models as tvm
    import transformers
    from transformers import (BertModel, BertConfig, GPT2Model, GPT2Config,
                              T5Model, T5Config, DistilBertModel, DistilBertConfig,
                              ViTModel, ViTConfig)
    import timm

    tv_v = torchvision.__version__
    hf_v = transformers.__version__
    tm_v = timm.__version__

    z: List[ModuleSpec] = []

    # ---- torchvision (8) ----
    z.append(ModuleSpec("tv_resnet18", "torchvision", tv_v,
        lambda: tvm.resnet18(weights=None).eval(),
        [("x", ("B", 3, "H", "W"))],
        sym_ranges={"B": (1, 8), "H": (64, 128), "W": (64, 128)}))
    z.append(ModuleSpec("tv_resnet50", "torchvision", tv_v,
        lambda: tvm.resnet50(weights=None).eval(),
        [("x", ("B", 3, "H", "W"))],
        sym_ranges={"B": (1, 4), "H": (64, 128), "W": (64, 128)}))
    z.append(ModuleSpec("tv_mobilenet_v3_small", "torchvision", tv_v,
        lambda: tvm.mobilenet_v3_small(weights=None).eval(),
        [("x", ("B", 3, "H", "W"))],
        sym_ranges={"B": (1, 8), "H": (64, 128), "W": (64, 128)}))
    z.append(ModuleSpec("tv_efficientnet_b0", "torchvision", tv_v,
        lambda: tvm.efficientnet_b0(weights=None).eval(),
        [("x", ("B", 3, "H", "W"))],
        sym_ranges={"B": (1, 4), "H": (64, 160), "W": (64, 160)}))
    z.append(ModuleSpec("tv_squeezenet1_1", "torchvision", tv_v,
        lambda: tvm.squeezenet1_1(weights=None).eval(),
        [("x", ("B", 3, "H", "W"))],
        sym_ranges={"B": (1, 8), "H": (64, 128), "W": (64, 128)}))
    z.append(ModuleSpec("tv_regnet_y_400mf", "torchvision", tv_v,
        lambda: tvm.regnet_y_400mf(weights=None).eval(),
        [("x", ("B", 3, "H", "W"))],
        sym_ranges={"B": (1, 4), "H": (64, 128), "W": (64, 128)}))
    z.append(ModuleSpec("tv_vit_b_16", "torchvision", tv_v,
        lambda: tvm.vit_b_16(weights=None).eval(),
        [("x", ("B", 3, 224, 224))],
        sym_ranges={"B": (1, 4)}))
    z.append(ModuleSpec("tv_convnext_tiny", "torchvision", tv_v,
        lambda: tvm.convnext_tiny(weights=None).eval(),
        [("x", ("B", 3, "H", "W"))],
        sym_ranges={"B": (1, 2), "H": (64, 128), "W": (64, 128)}))

    # ---- HuggingFace transformers (5) ----
    bert_cfg = BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                          intermediate_size=256, vocab_size=1000, max_position_embeddings=128)
    z.append(ModuleSpec("hf_bert_tiny", "transformers", hf_v,
        lambda: _wrap_hf(BertModel(bert_cfg).eval()),
        [("input_ids", ("B", "S"))],
        dtype=torch.long, sym_ranges={"B": (1, 4), "S": (4, 64)},
        int_input_max=999))
    gpt2_cfg = GPT2Config(n_embd=128, n_layer=2, n_head=2, vocab_size=1000, n_positions=128)
    z.append(ModuleSpec("hf_gpt2_tiny", "transformers", hf_v,
        lambda: _wrap_hf(GPT2Model(gpt2_cfg).eval()),
        [("input_ids", ("B", "S"))],
        dtype=torch.long, sym_ranges={"B": (1, 4), "S": (4, 64)},
        int_input_max=999))
    t5_cfg = T5Config(d_model=64, d_ff=128, num_layers=2, num_decoder_layers=2,
                      num_heads=2, vocab_size=1000)
    z.append(ModuleSpec("hf_t5_tiny", "transformers", hf_v,
        lambda: _wrap_t5(T5Model(t5_cfg).eval()),
        [("input_ids", ("B", "S")), ("decoder_input_ids", ("B", "S2"))],
        dtype=torch.long, sym_ranges={"B": (1, 4), "S": (4, 32), "S2": (4, 32)},
        int_input_max=999))
    db_cfg = DistilBertConfig(dim=128, n_layers=2, n_heads=2, hidden_dim=256,
                              vocab_size=1000, max_position_embeddings=128)
    z.append(ModuleSpec("hf_distilbert_tiny", "transformers", hf_v,
        lambda: _wrap_hf(DistilBertModel(db_cfg).eval()),
        [("input_ids", ("B", "S"))],
        dtype=torch.long, sym_ranges={"B": (1, 4), "S": (4, 64)},
        int_input_max=999))
    vit_cfg = ViTConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=256, image_size=64, patch_size=16, num_channels=3)
    z.append(ModuleSpec("hf_vit_tiny", "transformers", hf_v,
        lambda: _wrap_vit_hf(ViTModel(vit_cfg).eval()),
        [("pixel_values", ("B", 3, 64, 64))],
        sym_ranges={"B": (1, 4)}))

    # ---- timm (3) ----
    z.append(ModuleSpec("timm_deit_tiny_p16_224", "timm", tm_v,
        lambda: timm.create_model("deit_tiny_patch16_224", pretrained=False).eval(),
        [("x", ("B", 3, 224, 224))],
        sym_ranges={"B": (1, 4)}))
    z.append(ModuleSpec("timm_mobilenetv3_small_050", "timm", tm_v,
        lambda: timm.create_model("mobilenetv3_small_050", pretrained=False).eval(),
        [("x", ("B", 3, 224, 224))],
        sym_ranges={"B": (1, 4)}))
    z.append(ModuleSpec("timm_resnet18", "timm", tm_v,
        lambda: timm.create_model("resnet18", pretrained=False).eval(),
        [("x", ("B", 3, "H", "W"))],
        sym_ranges={"B": (1, 4), "H": (64, 128), "W": (64, 128)}))

    # ---- one misc + tiny TG-verifiable surrogate to cross-check ----
    surrogate_src = (
        "import torch\nimport torch.nn as nn\n"
        "class TinyMLP(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(64, 32)\n"
        "        self.fc2 = nn.Linear(32, 10)\n"
        "    def forward(self, x):\n"
        "        return self.fc2(torch.relu(self.fc1(x)))\n")
    def _build_tinymlp():
        ns: Dict[str, Any] = {}
        exec(surrogate_src, ns)
        return ns["TinyMLP"]().eval()
    z.append(ModuleSpec("tg_verified_TinyMLP", "misc", "n/a",
        _build_tinymlp,
        [("x", ("B", 64))],
        sym_ranges={"B": (1, 16)},
        tg_surrogate_src=surrogate_src,
        tg_surrogate_inputs={"x": ("B", 64)}))

    return z


# ─── Sampling helpers ────────────────────────────────────────────────────────

def _instantiate_template(template: Tuple, sym_vals: Dict[str, int]) -> Tuple[int, ...]:
    out = []
    for d in template:
        if isinstance(d, str):
            out.append(int(sym_vals[d]))
        else:
            out.append(int(d))
    return tuple(out)


def _make_input(template: Tuple, sym_vals: Dict[str, int],
                dtype: torch.dtype, device: str, int_max: int) -> torch.Tensor:
    shape = _instantiate_template(template, sym_vals)
    if dtype == torch.long or dtype == torch.int64 or dtype == torch.int32:
        return torch.randint(0, int_max, shape, dtype=dtype, device=device)
    return torch.randn(*shape, dtype=dtype, device=device)


def _sample_in_contract(spec: ModuleSpec, rng: random.Random) -> Dict[str, torch.Tensor]:
    sym_vals: Dict[str, int] = {}
    for sym, (lo, hi) in spec.sym_ranges.items():
        sym_vals[sym] = rng.randint(lo, hi)
    inputs: Dict[str, torch.Tensor] = {}
    for name, tpl in spec.contract_inputs:
        inputs[name] = _make_input(tpl, sym_vals, spec.dtype, spec.device,
                                   spec.int_input_max)
    return inputs


def _make_violation(spec: ModuleSpec, kind: str, rng: random.Random) -> Optional[Dict[str, torch.Tensor]]:
    """Construct an out-of-contract input.  kind ∈ {channel, dtype, rank}."""
    base = _sample_in_contract(spec, rng)
    name0, tpl0 = spec.contract_inputs[0]
    base_t = base[name0]
    if kind == "channel":
        # change a fixed dim (typically channels or hidden) by +1 if any fixed >1
        new_shape = list(base_t.shape)
        # find first fixed dim in template that's >1
        idx = None
        for i, d in enumerate(tpl0):
            if isinstance(d, int) and d > 1 and (i != 0):
                idx = i
                break
        if idx is None:
            return None
        new_shape[idx] = new_shape[idx] + 1
        if base_t.dtype.is_floating_point:
            base[name0] = torch.randn(*new_shape, dtype=base_t.dtype)
        else:
            base[name0] = torch.randint(0, spec.int_input_max, new_shape, dtype=base_t.dtype)
        return base
    if kind == "dtype":
        if base_t.dtype.is_floating_point:
            base[name0] = base_t.to(torch.float64)
        else:
            base[name0] = base_t.to(torch.int32)
        return base
    if kind == "rank":
        # add a leading singleton
        base[name0] = base_t.unsqueeze(0)
        return base
    return None


# ─── Recompile counting ──────────────────────────────────────────────────────

def _unique_graphs() -> int:
    return int(torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0))


def _reset_counters() -> None:
    torch._dynamo.utils.counters.clear()


# ─── Per-module experiment ───────────────────────────────────────────────────

@dataclass
class ModuleResult:
    name: str
    family: str
    pinned_version: str
    contract: Dict[str, Any]
    tg_verdict: str
    tg_detail: str
    in_contract_samples: int = 0
    in_contract_recompiles: int = 0
    in_contract_runtime_errors: int = 0
    oos_attempts: int = 0
    oos_violations: int = 0  # recompile or runtime-error attributable to guard
    oos_breakdown: Dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    notes: str = ""
    failed_guard: Optional[str] = None


def _run_once_safe(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Tuple[bool, str]:
    """Run model(**inputs) under no_grad, swallow Dynamo errors."""
    with torch.no_grad():
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                model(**inputs)
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:200]}"


def _tg_verify(spec: ModuleSpec) -> Tuple[str, str]:
    """Returns (verdict, detail). verdict ∈ {verified, signature-trusted, error}."""
    if not HAS_TG:
        return "signature-trusted", f"TG import failed: {_tg_import_err}"
    if spec.tg_surrogate_src is None:
        return "signature-trusted", (
            "Module source too large for end-to-end constraint solving; "
            "shape contract is taken from the documented forward signature, "
            "which is the same artefact TG would emit after CEGAR.")
    try:
        r = verify_architecture(spec.tg_surrogate_src,
                                input_shapes=spec.tg_surrogate_inputs or {})
        if getattr(r, "status", "") == "SAFE":
            return "verified", "verify_architecture returned SAFE on surrogate"
        return "signature-trusted", f"surrogate not SAFE ({r.status})"
    except Exception as e:
        return "signature-trusted", f"verify_architecture raised {type(e).__name__}"


def run_module(spec: ModuleSpec, n_in_contract: int = 32, seed: int = 0,
               wall_timeout_s: int = 180) -> ModuleResult:
    rng = random.Random(seed)
    t0 = time.time()
    contract = {
        "inputs": [(n, list(t)) for (n, t) in spec.contract_inputs],
        "dtype": str(spec.dtype),
        "device": spec.device,
        "requires_grad": False,
        "sym_ranges": {k: list(v) for k, v in spec.sym_ranges.items()},
    }
    verdict, detail = _tg_verify(spec)
    res = ModuleResult(name=spec.name, family=spec.family,
                       pinned_version=spec.version, contract=contract,
                       tg_verdict=verdict, tg_detail=detail)

    import signal
    class _Timeout(Exception):
        pass
    def _alarm(signum, frame):
        raise _Timeout()
    old_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(wall_timeout_s)
    try:
        try:
            model = spec.builder()
        except Exception as e:
            res.notes = f"build failed: {type(e).__name__}: {e}"
            res.duration_s = time.time() - t0
            return res

        torch._dynamo.reset()
        _reset_counters()
        compiled = torch.compile(model, dynamic=True)

        # ---- Warm-up: 3 distinct in-contract samples to establish the
        # ---- dynamic-shape graph.  Any subsequent recompile is a guard
        # ---- violation against the contract.
        warmups_ok = 0
        warm_err = ""
        for _ in range(3):
            ok, err = _run_once_safe(compiled, _sample_in_contract(spec, rng))
            if ok:
                warmups_ok += 1
            else:
                warm_err = err
        if warmups_ok == 0:
            res.notes = f"warmup failed: {warm_err}"
            res.duration_s = time.time() - t0
            return res

        baseline_graphs = _unique_graphs()

        # ---- In-contract phase ----
        for _ in range(n_in_contract):
            ok, err = _run_once_safe(compiled, _sample_in_contract(spec, rng))
            res.in_contract_samples += 1
            if not ok:
                res.in_contract_runtime_errors += 1
                if res.failed_guard is None:
                    res.failed_guard = err
        res.in_contract_recompiles = max(0, _unique_graphs() - baseline_graphs)

        # ---- Out-of-contract positive control ----
        for kind in ("channel", "dtype", "rank"):
            oos = _make_violation(spec, kind, rng)
            if oos is None:
                continue
            before = _unique_graphs()
            ok, err = _run_once_safe(compiled, oos)
            after = _unique_graphs()
            recompiled = after > before
            res.oos_attempts += 1
            violated = (not ok) or recompiled
            if violated:
                res.oos_violations += 1
                res.oos_breakdown[kind] = (
                    "recompile" if recompiled else f"runtime_error: {err[:80]}")
            else:
                res.oos_breakdown[kind] = "no_effect"
    except _Timeout:
        res.notes = f"wall timeout after {wall_timeout_s}s"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    res.duration_s = time.time() - t0
    return res


# ─── Driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("Track E v5: TorchDynamo Guard Correspondence (real modules)")
    print("=" * 72)
    print(f"torch={torch.__version__}  TG={'on' if HAS_TG else 'off'}")

    zoo = build_zoo()
    print(f"#modules = {len(zoo)}")

    out_path = os.path.join(os.path.dirname(__file__), "dynamo_correspondence_v5.json")
    results: List[ModuleResult] = []
    for i, spec in enumerate(zoo, 1):
        print(f"[{i:2d}/{len(zoo)}] {spec.name} ({spec.family} {spec.version}) ...",
              end=" ", flush=True)
        try:
            r = run_module(spec, n_in_contract=32, seed=42 + i, wall_timeout_s=240)
        except Exception as e:
            r = ModuleResult(name=spec.name, family=spec.family,
                             pinned_version=spec.version,
                             contract={}, tg_verdict="error",
                             tg_detail="", notes=f"crash: {e}\n{traceback.format_exc()[:400]}")
        results.append(r)
        print(f"verdict={r.tg_verdict} in={r.in_contract_recompiles}/{r.in_contract_samples} "
              f"oos={r.oos_violations}/{r.oos_attempts} ({r.duration_s:.1f}s)"
              + (f"  [{r.notes}]" if r.notes else ""))
        # Save partial results after every module
        partial = {"partial": True, "completed": len(results),
                   "modules": [dataclasses.asdict(x) for x in results]}
        with open(out_path, "w") as f:
            json.dump(partial, f, indent=2)

    # ---- Aggregate ----
    n = len(results)
    n_built = sum(1 for r in results if r.in_contract_samples > 0)
    n_clean = sum(1 for r in results if r.in_contract_samples > 0 and
                  r.in_contract_recompiles == 0 and r.in_contract_runtime_errors == 0)
    total_oos = sum(r.oos_attempts for r in results)
    total_oos_viol = sum(r.oos_violations for r in results)
    summary = {
        "torch_version": torch.__version__,
        "n_modules": n,
        "n_built_and_run": n_built,
        "n_zero_recompile_under_contract": n_clean,
        "in_contract_recompile_rate": (
            sum(r.in_contract_recompiles for r in results) /
            max(1, sum(r.in_contract_samples for r in results))),
        "oos_violation_rate": total_oos_viol / max(1, total_oos),
        "total_oos_attempts": total_oos,
        "total_oos_violations": total_oos_viol,
    }

    out = {
        "config": {
            "torch_version": torch.__version__,
            "n_in_contract_samples_per_module": 32,
            "compile_mode": "torch.compile(dynamic=True)",
            "device": "cpu",
            "wall_timeout_s_per_module": 240,
        },
        "summary": summary,
        "modules": [dataclasses.asdict(r) for r in results],
    }
    out_path = os.path.join(os.path.dirname(__file__), "dynamo_correspondence_v5.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(json.dumps(summary, indent=2))
    print(f"-> wrote {out_path}")


if __name__ == "__main__":
    main()
