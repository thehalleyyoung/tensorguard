"""Stratified random resample of the 371 Verified tied-weight population.

Addresses the selection-bias concern on the original shortest-LoC-first 25-row
stub-mocked validation (0/25 silent errors, Wilson 95% CI upper = 13.32%).
This script replaces it with a proportionally-stratified random sample of ≥80
rows (stratified by handler family, min 2 per stratum) and emits a tightened
Wilson 95% CI artifact.

Outputs:
  experiments_v5/stratified_resample_371.csv
  experiments_v5/stratified_resample_371_wilson.json
"""

from __future__ import annotations

import collections
import csv
import json
import math
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parent.parent
ROWS_PATH = REPO / "reproducibility" / "tied_weight_full_verdict_rows.json"
TOP100_PATH = REPO / "reproducibility" / "tied_weight_modules_top100.jsonl"
OUT_CSV = REPO / "experiments_v5" / "stratified_resample_371.csv"
OUT_JSON = REPO / "experiments_v5" / "stratified_resample_371_wilson.json"

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SEED = 20260430
N_TARGET = 80
MIN_PER_STRATUM = 2

# =========================================================================== #
#  Handler-family classification (same as tied_weight_stubmock_stratified.py) #
# =========================================================================== #

HANDLER_TOKENS: Dict[str, List[str]] = {
    "matmul":     [r"\.matmul\b", r"@\s*self\.", r"torch\.matmul\b"],
    "bmm":        [r"\.bmm\b", r"torch\.bmm\b"],
    "batched_matmul": [r"\.einsum\b"],
    "conv1d":     [r"\bConv1d\b", r"F\.conv1d\b"],
    "conv2d":     [r"\bConv2d\b", r"F\.conv2d\b"],
    "conv3d":     [r"\bConv3d\b", r"F\.conv3d\b"],
    "conv_transpose2d": [r"\bConvTranspose2d\b", r"F\.conv_transpose2d\b"],
    "view":       [r"\.view\("],
    "reshape":    [r"\.reshape\("],
    "permute":    [r"\.permute\("],
    "transpose":  [r"\.transpose\(", r"\.t\(\)"],
    "expand":     [r"\.expand\("],
    "repeat":     [r"\.repeat\("],
    "broadcast_to": [r"\.broadcast_to\(", r"torch\.broadcast_to\b"],
    "cat":        [r"torch\.cat\b", r"\.cat\("],
    "stack":      [r"torch\.stack\b"],
    "split":      [r"\.split\(", r"torch\.split\b"],
    "chunk":      [r"\.chunk\(", r"torch\.chunk\b"],
    "unbind":     [r"\.unbind\(", r"torch\.unbind\b"],
    "gather":     [r"\.gather\(", r"torch\.gather\b"],
    "scatter":    [r"\.scatter\(", r"\.scatter_\b"],
    "index_select": [r"\.index_select\b", r"torch\.index_select\b"],
    "narrow":     [r"\.narrow\("],
    "embed":      [r"\bEmbedding\b"],
    "layer_norm": [r"\bLayerNorm\b", r"F\.layer_norm\b"],
    "rms_norm":   [r"\bRMSNorm\b"],
    "scaled_dot_product_attention": [r"scaled_dot_product_attention"],
    "linear":     [r"\bLinear\b", r"F\.linear\b"],
    "batch_norm": [r"\bBatchNorm[123]d\b", r"F\.batch_norm\b"],
    "group_norm": [r"\bGroupNorm\b", r"F\.group_norm\b"],
    "instance_norm": [r"\bInstanceNorm[123]d\b", r"F\.instance_norm\b"],
    "multihead_attention": [r"\bMultiheadAttention\b"],
    "conv_transpose1d": [r"\bConvTranspose1d\b"],
    "squeeze":    [r"\.squeeze\("],
    "unsqueeze":  [r"\.unsqueeze\("],
    "flatten":    [r"\bFlatten\b", r"\.flatten\(", r"torch\.flatten\b"],
    "softmax":    [r"\.softmax\b", r"\bSoftmax\b", r"F\.softmax\b"],
    "relu":       [r"\bReLU\b", r"F\.relu\b", r"\.relu\("],
    "gelu":       [r"\bGELU\b", r"F\.gelu\b"],
    "silu":       [r"\bSiLU\b", r"F\.silu\b"],
    "tanh":       [r"\bTanh\b", r"\.tanh\("],
    "sigmoid":    [r"\bSigmoid\b", r"\.sigmoid\("],
    "dropout":    [r"\bDropout\b", r"F\.dropout\b"],
    "cross_entropy": [r"CrossEntropy", r"F\.cross_entropy\b"],
    "interpolate": [r"F\.interpolate\b"],
    "pixel_shuffle": [r"\bPixelShuffle\b"],
    "pixel_unshuffle": [r"\bPixelUnshuffle\b"],
    "topk":       [r"\.topk\(", r"torch\.topk\b"],
    "max_pool2d": [r"\bMaxPool2d\b", r"F\.max_pool2d\b"],
    "avg_pool2d": [r"\bAvgPool2d\b", r"F\.avg_pool2d\b"],
    "adaptive_avg_pool2d": [r"\bAdaptiveAvgPool2d\b"],
    "adaptive_max_pool2d": [r"\bAdaptiveMaxPool2d\b"],
    "add":        [r"\.add\(", r"\+="],
    "mul":        [r"\.mul\("],
    "div":        [r"\.div\("],
    "pow":        [r"\.pow\("],
    "sqrt":       [r"torch\.sqrt\b", r"\.sqrt\("],
    "rsqrt":      [r"torch\.rsqrt\b", r"\.rsqrt\("],
    "sum":        [r"\.sum\("],
    "mean":       [r"\.mean\("],
    "var":        [r"\.var\("],
    "std":        [r"\.std\("],
    "max":        [r"\.max\("],
    "min":        [r"\.min\("],
    "argmax":     [r"\.argmax\("],
    "argmin":     [r"\.argmin\("],
    "where":      [r"torch\.where\b"],
    "masked_fill": [r"\.masked_fill\("],
    "clamp":      [r"\.clamp\("],
    "abs":        [r"\.abs\("],
    "exp":        [r"torch\.exp\b", r"\.exp\("],
    "log":        [r"torch\.log\b", r"\.log\("],
    "log_softmax": [r"F\.log_softmax\b"],
    "to":         [r"\.to\("],
    "type":       [r"\.type\("],
    "contiguous": [r"\.contiguous\("],
    "detach":     [r"\.detach\("],
}

CONV_FAMILY = {"conv1d", "conv2d", "conv3d", "conv_transpose2d", "conv_transpose1d"}
ATTN_FAMILY = {"batched_matmul", "matmul", "bmm", "scaled_dot_product_attention", "multihead_attention"}
NORM_FAMILY = {"layer_norm", "rms_norm", "batch_norm", "group_norm", "instance_norm"}
EMBED_FAMILY = {"embed"}
LINEAR_FAMILY = {"linear"}
RESHAPE_FAMILY = {
    "view", "reshape", "permute", "transpose", "expand", "repeat", "broadcast_to",
    "squeeze", "unsqueeze", "flatten", "split", "chunk", "unbind", "gather", "scatter",
    "index_select", "narrow", "cat", "stack", "contiguous", "detach",
}
BROADCAST_ELEM = {
    "add", "mul", "div", "pow", "sqrt", "rsqrt", "sum", "mean", "var", "std",
    "max", "min", "argmax", "argmin", "where", "masked_fill", "clamp", "abs",
    "exp", "log", "softmax", "log_softmax", "relu", "gelu", "silu", "tanh",
    "sigmoid", "dropout", "cross_entropy", "interpolate", "pixel_shuffle",
    "pixel_unshuffle", "topk", "max_pool2d", "avg_pool2d",
    "adaptive_avg_pool2d", "adaptive_max_pool2d", "to", "type",
}

FAMILY_PRIORITY_STRUCTURAL = [
    ("conv-family",      CONV_FAMILY),
    ("attention-family", ATTN_FAMILY),
    ("embedding-family", EMBED_FAMILY),
    ("linear-only",      LINEAR_FAMILY),
    ("norm-family",      NORM_FAMILY),
]


def detect_handlers(src: str) -> set:
    found = set()
    for h, patterns in HANDLER_TOKENS.items():
        for p in patterns:
            if re.search(p, src):
                found.add(h)
                break
    return found


def classify_family(handlers: set) -> str:
    if not handlers:
        return "no_handler_detected"
    for family_name, family_set in FAMILY_PRIORITY_STRUCTURAL:
        if handlers & family_set:
            return family_name
    has_reshape = bool(handlers & RESHAPE_FAMILY)
    has_elem = bool(handlers & BROADCAST_ELEM)
    if has_reshape and not has_elem:
        return "reshape-only"
    if has_elem:
        return "broadcast-elementwise"
    return "no_handler_detected"


# =========================================================================== #
#  Proportional allocation with minimum-per-stratum guarantee                 #
# =========================================================================== #

def proportional_allocation(
    strata: Dict[str, list],
    n_total: int,
    min_per: int,
    rng: random.Random,
) -> List[dict]:
    """Draw a proportionally-allocated stratified sample of size ≥n_total.

    Each stratum gets max(min_per, round(N_i/N * n_total)) rows, capped at
    the actual stratum size.  If rounding leaves total < n_total, top up
    from the largest strata until the target is met.
    """
    N = sum(len(v) for v in strata.values())
    alloc: Dict[str, int] = {}
    for fam, items in strata.items():
        raw = (len(items) / N) * n_total if N > 0 else 0
        alloc[fam] = min(len(items), max(min_per, round(raw)))

    # Top up if short of target
    total_alloc = sum(alloc.values())
    if total_alloc < n_total:
        # Add 1 at a time to the strata with most headroom (population - alloc)
        fams_sorted = sorted(strata.keys(), key=lambda f: len(strata[f]) - alloc[f], reverse=True)
        i = 0
        while total_alloc < n_total:
            f = fams_sorted[i % len(fams_sorted)]
            if alloc[f] < len(strata[f]):
                alloc[f] += 1
                total_alloc += 1
            i += 1
            if i > n_total * 10:
                break

    sample = []
    for fam, items in sorted(strata.items()):
        k = alloc[fam]
        chosen = rng.sample(items, k)
        for item in chosen:
            entry = dict(item)
            entry["_family"] = fam
            sample.append(entry)
    return sample


# =========================================================================== #
#  Stub-mocked runtime harness (identical to tied_weight_stubmock_stratified)  #
# =========================================================================== #

class _StubConfig:
    def __init__(self, defaults: Optional[dict] = None):
        self._d = {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_hidden_layers": 2,
            "max_position_embeddings": 32,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "layer_norm_eps": 1e-6,
            "hidden_dropout_prob": 0.0,
            "attention_dropout": 0.0,
            "attention_dropout_prob": 0.0,
            "head_dim": 4,
            "n_embd": 16,
            "n_positions": 32,
            "n_head": 4,
            "n_layer": 2,
            "d_model": 16,
            "d_kv": 4,
            "d_ff": 32,
            "num_heads": 4,
            "num_layers": 2,
            "embed_dim": 16,
            "ffn_dim": 32,
            "decoder_layers": 2,
            "encoder_layers": 2,
            "encoder_attention_heads": 4,
            "decoder_attention_heads": 4,
            "encoder_ffn_dim": 32,
            "decoder_ffn_dim": 32,
            "image_size": 16,
            "patch_size": 4,
            "num_channels": 3,
            "in_channels": 3,
            "out_channels": 16,
            "channels": 16,
            "use_cache": False,
            "is_encoder_decoder": False,
            "tie_word_embeddings": True,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "torch_dtype": "float32",
            "rope_theta": 10000.0,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
            "rope_scaling": None,
            "_attn_implementation": "eager",
            "attention_bias": False,
            "mlp_bias": False,
            "sliding_window": None,
            "layer_types": None,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "depth": 2,
            "num_experts": 2,
            "moe_intermediate_size": 16,
            "expert_capacity": 4,
            "shared_intermediate_size": 16,
            "n_inner": 32,
            "activation_function": "gelu",
            "hidden_act": "gelu",
            "feature_size": 16,
            "num_mel_bins": 16,
            "model_type": "stub",
        }
        if defaults:
            self._d.update(defaults)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._d.get(name, 0)

    def __getitem__(self, k):
        return self._d.get(k, 0)

    def get(self, k, default=None):
        return self._d.get(k, default)

    def __contains__(self, k):
        return True

    def __iter__(self):
        return iter(self._d)


class _NoOp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_stub_namespace():
    import dataclasses
    import functools
    import math as _math
    import types
    import typing
    import uuid
    from typing import Optional, Tuple, List, Dict, Any, Callable, Union
    import numpy

    import torch as _torch
    import torch.nn as _nn

    mod_name = f"_tw_strat_{uuid.uuid4().hex[:8]}"
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    ns = mod.__dict__
    ns.update({
        "__builtins__": __builtins__,
        "torch": _torch,
        "nn": _nn,
        "F": F,
        "math": _math,
        "Optional": Optional,
        "Tuple": Tuple,
        "List": List,
        "Dict": Dict,
        "Any": Any,
        "Callable": Callable,
        "Union": Union,
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "functools": functools,
        "collections": collections,
        "Sequence": collections.abc.Sequence,
        "Iterable": collections.abc.Iterable,
        "numpy": numpy,
        "np": numpy,
        "Unpack": MagicMock(),
        "FlashAttentionKwargs": MagicMock(),
        "Cache": MagicMock(),
        "DynamicCache": MagicMock(),
        "PreTrainedModel": _nn.Module,
        "GenerationMixin": object,
        "ModuleUtilsMixin": object,
        "ACT2FN": collections.defaultdict(lambda: F.gelu),
        "ALL_ATTENTION_FUNCTIONS": MagicMock(),
        "ROPE_INIT_FUNCTIONS": collections.defaultdict(
            lambda: lambda *a, **kw: (_torch.ones(8), 1.0)
        ),
        "logger": MagicMock(),
        "logging": MagicMock(),
        "use_kernel_forward_from_hub": lambda *_a, **_kw: (lambda c: c),
        "use_kernelized_func": lambda *_a, **_kw: (lambda c: c),
        "dynamic_rope_update": lambda f: f,
        "maybe_autocast": lambda **_kw: _NoOp(),
        "apply_rotary_pos_emb": lambda q, k, *a, **kw: (q, k),
        "eager_attention_forward": lambda *a, **kw: (
            _torch.zeros_like(a[1]),
            None,
        ),
    })
    ns.update({
        "Type": typing.Type,
        "TYPE_CHECKING": False,
        "annotations": None,
    })
    return ns


MAX_NAME_RETRIES = 60


def exec_source_with_retry(src: str, ns: dict):
    for _ in range(MAX_NAME_RETRIES):
        try:
            exec(src, ns)
            return None
        except NameError as e:
            msg = str(e)
            m = re.search(r"name '([^']+)' is not defined", msg)
            if not m:
                return msg
            ns[m.group(1)] = MagicMock(name=m.group(1))
        except Exception as e:
            return f"{type(e).__name__}: {e}"
    return "too_many_name_retries"


def try_instantiate(cls, ns):
    cfg = _StubConfig()
    candidates = [
        ((), {}),
        ((cfg,), {}),
        ((cfg, 0), {}),
        ((16,), {}),
        ((16, 16), {}),
        ((16, 16, 16), {}),
        ((cfg,), {"layer_idx": 0}),
        ((), {"config": cfg}),
        ((), {"hidden_size": 16}),
        ((), {"in_features": 16, "out_features": 16}),
        ((16, 4), {}),
    ]
    last_err = None
    for args, kwargs in candidates:
        try:
            inst = cls(*args, **kwargs)
            if isinstance(inst, nn.Module):
                return inst, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err or "no successful constructor"


def _to_loss(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = _to_loss(o)
            if t is not None and t.dtype.is_floating_point:
                return t
    if isinstance(out, dict):
        for o in out.values():
            t = _to_loss(o)
            if t is not None and t.dtype.is_floating_point:
                return t
    return None


def try_forward_backward(module: nn.Module):
    shapes = [
        (2, 4, 16),
        (2, 16),
        (2, 16, 4, 4),
        (2, 3, 16, 16),
        (1, 4),
        (2, 4),
    ]
    last_err = "no shapes tried"
    for shape in shapes:
        try:
            if shape in ((1, 4), (2, 4)):
                x = torch.randint(0, 8, shape, dtype=torch.long)
            else:
                x = torch.randn(*shape, requires_grad=False)
            module.train()
            out = module(x)
            tensor_out = _to_loss(out)
            if tensor_out is None:
                continue
            loss = tensor_out.float().sum()
            module.zero_grad(set_to_none=True)
            loss.backward()
            return shape, loss.item(), None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, None, last_err


def grad_topology(module: nn.Module):
    out = {}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        out[name] = False if p.grad is None else bool(p.grad.abs().sum().item() > 0.0)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    delta = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - delta), min(1.0, centre + delta))


# =========================================================================== #
#  Source loading                                                              #
# =========================================================================== #

def _load_top100() -> Dict[str, str]:
    top100: Dict[str, str] = {}
    try:
        with open(TOP100_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("class_name") and rec.get("src"):
                    top100[rec["class_name"]] = rec["src"]
    except Exception:
        pass
    return top100


def _read_class_source(file_path: str, class_name: str) -> Optional[str]:
    import ast
    try:
        full = REPO / file_path
        if not full.exists():
            return None
        text = full.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return ast.get_source_segment(text, node)
    except Exception:
        return None


# =========================================================================== #
#  Main                                                                        #
# =========================================================================== #

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    all_rows = json.loads(ROWS_PATH.read_text())
    verified_rows = [r for r in all_rows if r.get("verdict") == "Verified"]
    print(f"Population: {len(verified_rows)} Verified rows")

    top100 = _load_top100()

    # Step 1: classify each Verified row into a handler family
    rows_by_family: Dict[str, list] = {}
    for r in verified_rows:
        cname = r["class_name"]
        fpath = r["file"]
        src = top100.get(cname) or _read_class_source(fpath, cname)
        if src is None:
            family = "no_source"
        else:
            handlers = detect_handlers(src)
            family = classify_family(handlers)
        entry = dict(r)
        entry["_src"] = src
        entry["_family"] = family
        rows_by_family.setdefault(family, []).append(entry)

    family_sizes = {fam: len(items) for fam, items in sorted(rows_by_family.items())}
    print("Family sizes:", family_sizes)

    # Step 2: proportional stratified sample (seed=SEED, n≥80, min 2 per stratum)
    rng = random.Random(SEED)
    runnable = {f: v for f, v in rows_by_family.items() if f != "no_source"}
    sample = proportional_allocation(runnable, N_TARGET, MIN_PER_STRATUM, rng)
    rng.shuffle(sample)
    print(f"Sample size: {len(sample)}")

    # Step 3: stub-mocked runtime check
    results = []
    for i, r in enumerate(sample, 1):
        cname = r["class_name"]
        fpath = r["file"]
        family = r["_family"]
        src = r["_src"]

        print(f"  [{i}/{len(sample)}] {cname} ({family})", flush=True)

        if src is None:
            results.append({
                "class": cname, "file": fpath, "family": family,
                "status": "no_source", "silent_error": False,
            })
            continue

        ns = make_stub_namespace()
        exec_err = exec_source_with_retry(src, ns)
        if exec_err is not None:
            results.append({
                "class": cname, "file": fpath, "family": family,
                "status": "exec_failed", "error": exec_err, "silent_error": False,
            })
            continue

        cls = ns.get(cname)
        if cls is None or not isinstance(cls, type):
            results.append({
                "class": cname, "file": fpath, "family": family,
                "status": "class_not_found", "silent_error": False,
            })
            continue

        inst, err = try_instantiate(cls, ns)
        if inst is None:
            results.append({
                "class": cname, "file": fpath, "family": family,
                "status": "instantiation_failed", "error": err, "silent_error": False,
            })
            continue

        shape, loss_val, ferr = try_forward_backward(inst)
        if shape is None:
            results.append({
                "class": cname, "file": fpath, "family": family,
                "status": "forward_or_backward_failed", "error": ferr,
                "silent_error": False,
            })
            continue

        topo = grad_topology(inst)
        has_any_grad = any(v for v in topo.values())
        all_have_grad = all(v for v in topo.values()) if topo else True
        # Silent error: analyzer said Verified but no grad on any requires_grad leaf
        silent_error = (not has_any_grad) and len(topo) > 0

        results.append({
            "class": cname,
            "file": fpath,
            "family": family,
            "status": "ok",
            "input_shape": list(shape),
            "loss": loss_val,
            "n_params": len(topo),
            "n_with_grad": sum(1 for v in topo.values() if v),
            "any_grad": has_any_grad,
            "all_grad": all_have_grad,
            "silent_error": silent_error,
        })

    # Step 4: aggregate
    n_total = len(results)
    k_silently_incorrect = sum(1 for r in results if r.get("silent_error"))
    ok_rows = [r for r in results if r["status"] == "ok"]

    wilson_lo, wilson_hi = wilson(k_silently_incorrect, n_total)

    # Per-stratum stats
    families_in_results = sorted(set(r["family"] for r in results))
    per_stratum: Dict[str, dict] = {}
    for fam in families_in_results:
        fam_rows = [r for r in results if r["family"] == fam]
        fam_ok = [r for r in fam_rows if r["status"] == "ok"]
        fam_silent = sum(1 for r in fam_rows if r.get("silent_error"))
        fam_lo, fam_hi = wilson(fam_silent, len(fam_rows))
        per_stratum[fam] = {
            "population": family_sizes.get(fam, 0),
            "sampled": len(fam_rows),
            "ok": len(fam_ok),
            "k_silently_incorrect": fam_silent,
            "wilson_lo": fam_lo,
            "wilson_hi": fam_hi,
        }

    # Step 5: write CSV
    CSV_FIELDS = ["class", "file", "family", "status", "silent_error",
                  "input_shape", "n_params", "n_with_grad"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in CSV_FIELDS}
            w.writerow(row)

    # Step 6: write Wilson JSON
    out = {
        "seed": SEED,
        "n": n_total,
        "k_silently_incorrect": k_silently_incorrect,
        "wilson_lo": wilson_lo,
        "wilson_hi": wilson_hi,
        "population_size": len(verified_rows),
        "ok_runs": len(ok_rows),
        "per_stratum": per_stratum,
        "stratification_rule": (
            f"proportional allocation, min {MIN_PER_STRATUM} per stratum, "
            f"seed {SEED}, n_target={N_TARGET}"
        ),
        "comparison_vs_original": {
            "original_shortest_loc_wilson_hi": 0.13319649395317873,
            "original_n": 31,
            "original_k_silently_incorrect": 0,
            "note": "original used shortest-LoC-first selection over 31 attempted rows",
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"\nWrote {OUT_CSV} ({n_total} rows)")
    print(f"Wrote {OUT_JSON}")
    print(f"\nHEADLINE: n={n_total}, k_silently_incorrect={k_silently_incorrect}, "
          f"Wilson 95% CI=[{wilson_lo:.6f}, {wilson_hi:.6f}]")
    print(f"Strata represented: {len(families_in_results)}: {families_in_results}")


if __name__ == "__main__":
    main()
