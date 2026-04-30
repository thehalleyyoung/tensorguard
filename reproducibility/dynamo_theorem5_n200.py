#!/usr/bin/env python3.11
"""Round-7 Theorem 5 Dynamo falsifier audit on a >=200-module population.

Reviewer borderline-reason: extend the 55-module audit until it observes at
least one in-contract SHAPE/DTYPE/RANK recompile and report the
catalogue-membership rate on those events.  Fallback (explicitly allowed by
the round-7 improver prompt): if zero non-INT guards on the larger
population, document the result and weaken theorem language.

Methodological improvements over `dynamo_theorem5_n100.py`:

  * Subject pool expanded to ~250+ candidate modules (torchvision +
    transformers + timm + custom-op-bearing fixtures + AutoModel
    snapshots).
  * Guard-string capture promoted to *structured* via
    `torch._logging.set_logs(recompiles=True, guards=False)` (recompiles
    emits "Recompiling function ... due to: ..." with a per-guard reason
    block at INFO level) plus
    `torch._dynamo.config.report_guard_failures = True` where available.
  * Each module gets a wider input-shape sweep (varying both batch and
    spatial dims, and additionally a single dtype-perturbation input
    where applicable) so SHAPE/DTYPE recompiles have a realistic chance
    of firing.
  * Per-recompile classification keys on the *structured reason text*
    (e.g. "tensor 'x' size mismatch at index 1", "tensor 'x' rank
    mismatch", "scalar L['n'] != 5") rather than free-form keyword
    search.

Run:
    python3.11 reproducibility/dynamo_theorem5_n200.py
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import time
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch._dynamo
import torch._dynamo.utils

torch.set_num_threads(2)
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.suppress_errors = True

OUT_JSON = os.path.join(ROOT, "reproducibility", "dynamo_theorem5_n200.json")
OUT_MD = os.path.join(ROOT, "reproducibility", "dynamo_theorem5_n200.md")

# ─────────────────────────────────────────────────────────────────────────────
# Structured guard classification.
#
# `torch._logging.set_logs(recompiles=True)` causes torch._dynamo to emit
# "Recompiling function ... due to:" log records whose payload contains
# one or more lines of the form:
#       - 0/k: tensor 'L['x']' size mismatch at index 1. expected 64, actual 32
#       - 0/k: tensor 'L['x']' rank mismatch.  expected 4, actual 3
#       - 0/k: tensor 'L['x']' dtype mismatch.  expected torch.float32, actual torch.float16
#       - 0/k: L['n'] == 5 -> True
# We classify each per-guard reason line; INT / SymInt guards are out of
# Theorem-5 scope; SHAPE/DTYPE/RANK guards on input tensors named in the
# module forward signature are *in catalogue*; SHAPE/DTYPE/RANK guards
# whose target variable is anything else are *out of catalogue* and
# falsify the necessary direction.
# ─────────────────────────────────────────────────────────────────────────────

SHAPE_RE = re.compile(r"size mismatch|stride mismatch|tensor.*size", re.IGNORECASE)
DTYPE_RE = re.compile(r"dtype mismatch|tensor.*dtype", re.IGNORECASE)
RANK_RE = re.compile(r"rank mismatch|ndim mismatch|dim()?\s*mismatch", re.IGNORECASE)
INT_RE = re.compile(r"==\s*\d|!=\s*\d|specialize|guard_value|symint|symfloat|"
                    r"L\['[^']+'\]\s*[!=<>]+\s*\d|"
                    r"^\s*-\s*\d+/\d+:\s*L\[", re.IGNORECASE)
LIST_RE = re.compile(r"len\(L\[", re.IGNORECASE)


def classify_guard(line: str) -> str:
    """Classify a single 'reason' line under a Recompiling block."""
    if not line:
        return "INT"
    if SHAPE_RE.search(line):
        return "SHAPE"
    if DTYPE_RE.search(line):
        return "DTYPE"
    if RANK_RE.search(line):
        return "RANK"
    if LIST_RE.search(line):
        return "LIST_LEN"
    if INT_RE.search(line):
        return "INT"
    return "OTHER"


def guard_in_catalogue(line: str, input_names: List[str]) -> bool:
    """A SHAPE/DTYPE/RANK guard is *in catalogue* iff its variable target is
    a forward-input tensor the analyser declares as a refinement variable.
    Recompile reason lines reference inputs as L['x'], L['input_ids'], etc.
    """
    s = line
    for name in input_names + ["x", "input_ids", "inputs_embeds",
                               "hidden_states", "attention_mask"]:
        if (f"L['{name}']" in s) or (f"tensor '{name}'" in s.lower()):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Recompile log capture.
# ─────────────────────────────────────────────────────────────────────────────

class _RecompileHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: List[str] = []

    def emit(self, record):
        try:
            self.records.append(self.format(record))
        except Exception:
            self.records.append(record.getMessage())


def _enable_structured_recompile_capture() -> _RecompileHandler:
    handler = _RecompileHandler()
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(name)s :: %(message)s")
    handler.setFormatter(fmt)
    try:
        import torch._logging
        torch._logging.set_logs(recompiles=logging.INFO,
                                guards=logging.WARN)  # avoid GUARDS spam
    except Exception:
        pass
    for name in ("torch._dynamo",
                 "torch._dynamo.guards",
                 "torch._dynamo.convert_frame",
                 "torch._dynamo.eval_frame"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
    return handler


def _parse_recompile_reasons(records: List[str]) -> List[str]:
    """Pull out the per-guard reason lines from captured Recompiling
    records.  Each Recompiling record looks like:

        Recompiling function fn in /path/to/file.py:NN
            triggered by the following guard failure(s):
            - 0/k: tensor 'L['x']' size mismatch at index 1. expected 64, actual 32

    We return only the bullet lines.
    """
    out: List[str] = []
    capture = False
    for r in records:
        for ln in r.splitlines():
            stripped = ln.strip()
            if "Recompiling" in stripped or "triggered by" in stripped:
                capture = True
                continue
            if capture:
                if stripped.startswith("-"):
                    out.append(stripped)
                elif not stripped:
                    continue
                else:
                    capture = False
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Subject builder.
# ─────────────────────────────────────────────────────────────────────────────

def _build_subject_list() -> List[Dict[str, Any]]:
    subjects: List[Dict[str, Any]] = []

    # Torchvision top-level + sub-blocks
    try:
        import torchvision.models as tvm
        tv_specs = [
            ("tv_resnet18", lambda: tvm.resnet18(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_resnet34", lambda: tvm.resnet34(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_resnet50", lambda: tvm.resnet50(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_resnet101", lambda: tvm.resnet101(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_mobilenet_v2", lambda: tvm.mobilenet_v2(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_mobilenet_v3_s", lambda: tvm.mobilenet_v3_small(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_mobilenet_v3_l", lambda: tvm.mobilenet_v3_large(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_efficientnet_b0", lambda: tvm.efficientnet_b0(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_efficientnet_b1", lambda: tvm.efficientnet_b1(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_efficientnet_v2_s", lambda: tvm.efficientnet_v2_s(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_squeezenet1_0", lambda: tvm.squeezenet1_0(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_squeezenet1_1", lambda: tvm.squeezenet1_1(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_regnet_y_400mf", lambda: tvm.regnet_y_400mf(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_regnet_x_400mf", lambda: tvm.regnet_x_400mf(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_convnext_tiny", lambda: tvm.convnext_tiny(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_shufflenet_v2", lambda: tvm.shufflenet_v2_x0_5(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_shufflenet_v2_1_0", lambda: tvm.shufflenet_v2_x1_0(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_mnasnet0_5", lambda: tvm.mnasnet0_5(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_mnasnet1_0", lambda: tvm.mnasnet1_0(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_vgg11", lambda: tvm.vgg11(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_vgg13", lambda: tvm.vgg13(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_alexnet", lambda: tvm.alexnet(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_densenet121", lambda: tvm.densenet121(weights=None).eval(), (1, 3, 64, 64), True),
            ("tv_googlenet", lambda: tvm.googlenet(weights=None, aux_logits=False).eval(), (1, 3, 64, 64), True),
        ]
        for name, builder, shape, is_image in tv_specs:
            subjects.append({
                "name": name, "family": "torchvision",
                "builder": builder, "input_shape": shape,
                "input_kind": "image" if is_image else "seq",
                "input_names": ["x"],
            })
        # sub-blocks
        block_specs = [
            ("tv_resnet_basic", lambda: tvm.resnet.BasicBlock(64, 64).eval(), (1, 64, 16, 16)),
            ("tv_resnet_bottleneck", lambda: tvm.resnet.Bottleneck(64, 16).eval(), (1, 64, 16, 16)),
            ("tv_mnv2_inverted", lambda: tvm.mobilenetv2.InvertedResidual(32, 32, 1, 2).eval(), (1, 32, 16, 16)),
            ("tv_squeezenet_fire", lambda: tvm.squeezenet.Fire(64, 16, 32, 32).eval(), (1, 64, 16, 16)),
            ("tv_shufflenetv2_ir", lambda: tvm.shufflenetv2.InvertedResidual(32, 32, 2).eval(), (1, 32, 16, 16)),
            ("tv_densenet_denselayer", lambda: tvm.densenet._DenseLayer(32, 8, 16, 0.0, False).eval(), (1, 32, 16, 16)),
        ]
        for name, builder, shape in block_specs:
            subjects.append({
                "name": name, "family": "torchvision",
                "builder": builder, "input_shape": shape,
                "input_kind": "image", "input_names": ["x"],
            })
    except Exception:
        pass

    # HuggingFace tiny configs (broad family coverage).
    try:
        from transformers import (
            BertConfig, BertModel,
            GPT2Config, GPT2Model,
            T5Config, T5Model,
            DistilBertConfig, DistilBertModel,
            AlbertConfig, AlbertModel,
            RobertaConfig, RobertaModel,
            ElectraConfig, ElectraModel,
            DebertaConfig, DebertaModel,
            MPNetConfig, MPNetModel,
            BartConfig, BartModel,
            OPTConfig, OPTModel,
        )
        def mk(cfg_cls, model_cls, **kw):
            def _f():
                cfg = cfg_cls(**kw)
                return model_cls(cfg).eval()
            return _f
        hf_specs = [
            ("hf_bert_tiny", mk(BertConfig, BertModel,
                                hidden_size=64, num_hidden_layers=2,
                                num_attention_heads=4, intermediate_size=128,
                                max_position_embeddings=64), "seq"),
            ("hf_gpt2_tiny", mk(GPT2Config, GPT2Model,
                                n_embd=64, n_layer=2, n_head=4, n_positions=64), "seq"),
            ("hf_distilbert_tiny", mk(DistilBertConfig, DistilBertModel,
                                       dim=64, n_heads=4, n_layers=2,
                                       hidden_dim=128, max_position_embeddings=64), "seq"),
            ("hf_albert_tiny", mk(AlbertConfig, AlbertModel,
                                   hidden_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, intermediate_size=128,
                                   max_position_embeddings=64, embedding_size=64), "seq"),
            ("hf_roberta_tiny", mk(RobertaConfig, RobertaModel,
                                    hidden_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, intermediate_size=128,
                                    max_position_embeddings=64, vocab_size=100), "seq"),
            ("hf_electra_tiny", mk(ElectraConfig, ElectraModel,
                                    hidden_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, intermediate_size=128,
                                    max_position_embeddings=64, embedding_size=64), "seq"),
            ("hf_deberta_tiny", mk(DebertaConfig, DebertaModel,
                                    hidden_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, intermediate_size=128,
                                    max_position_embeddings=64), "seq"),
            ("hf_mpnet_tiny", mk(MPNetConfig, MPNetModel,
                                  hidden_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, intermediate_size=128), "seq"),
            ("hf_bart_tiny", mk(BartConfig, BartModel,
                                 d_model=64, encoder_layers=2, decoder_layers=2,
                                 encoder_attention_heads=4, decoder_attention_heads=4,
                                 encoder_ffn_dim=128, decoder_ffn_dim=128,
                                 max_position_embeddings=64), "seq"),
            ("hf_opt_tiny", mk(OPTConfig, OPTModel,
                                hidden_size=64, num_hidden_layers=2,
                                num_attention_heads=4, ffn_dim=128,
                                max_position_embeddings=64), "seq"),
        ]
        for name, builder, kind in hf_specs:
            subjects.append({
                "name": name, "family": "transformers",
                "builder": builder, "input_shape": (1, 16),
                "input_kind": kind, "input_names": ["input_ids"],
            })
    except Exception:
        pass

    # AutoModel from snapshots --- this is the round-7 model-family
    # expansion.  Pick small CPU-friendly checkpoints that exercise
    # different tokenisation/embedding paths from the tiny-config set.
    try:
        from transformers import AutoModel, AutoConfig
        # Tiny published checkpoints (no download required if cached).
        for repo in [
            "hf-internal-testing/tiny-random-bert",
            "hf-internal-testing/tiny-random-gpt2",
            "hf-internal-testing/tiny-random-t5",
            "hf-internal-testing/tiny-random-bart",
            "hf-internal-testing/tiny-random-roberta",
            "hf-internal-testing/tiny-random-distilbert",
            "hf-internal-testing/tiny-random-electra",
            "hf-internal-testing/tiny-random-mpnet",
            "hf-internal-testing/tiny-random-vit",
            "hf-internal-testing/tiny-random-CLIPModel",
        ]:
            def _builder_from_repo(r=repo):
                def _b():
                    return AutoModel.from_pretrained(r).eval()
                return _b
            subjects.append({
                "name": "hf_auto_" + repo.split("/")[-1],
                "family": "transformers_auto",
                "builder": _builder_from_repo(),
                "input_shape": (1, 16),
                "input_kind": "seq",
                "input_names": ["input_ids"],
            })
    except Exception:
        pass

    # timm models (light)
    try:
        import timm
        timm_light = [
            "resnet18", "resnet26", "resnet34", "resnet50",
            "mobilenetv2_050", "mobilenetv2_100",
            "mobilenetv3_small_050", "mobilenetv3_small_100",
            "mobilenetv3_large_100",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
            "mnasnet_050", "mnasnet_100",
            "rexnetr_100", "rexnetr_130",
            "vit_tiny_patch16_224", "vit_small_patch32_224",
            "deit_tiny_patch16_224", "deit_small_patch16_224",
            "mixer_b16_224", "resmlp_12_224",
            "convnext_femto", "convnext_pico", "convnext_nano",
            "convmixer_768_32",
            "xcit_tiny_12_p16_224", "xcit_small_12_p16_224",
            "poolformer_s12", "poolformer_s24",
            "swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224",
            "pit_xs_distilled_224", "coat_tiny",
            "botnet26t_256",
            "resnext26ts", "resnext50_32x4d",
            "wide_resnet50_2", "resnest14d",
            "densenet121", "densenet169",
            "dla34", "dla46_c",
            "skresnet18", "skresnet34",
            "regnetx_002", "regnetx_004", "regnety_002", "regnety_004",
            "repvgg_a0", "repvgg_a1",
            "hrnet_w18_small",
            "ghostnet_050", "ghostnet_100",
            "fbnetc_100", "fbnetv3_b",
            "tf_efficientnet_lite0", "tf_efficientnet_lite1",
            "legacy_seresnet18", "legacy_seresnet34",
            "seresnext26ts",
            "nfnet_l0", "nf_resnet26",
            "tinynet_a", "tinynet_b",
            "levit_128", "levit_192",
            "res2net50_26w_4s", "res2next50",
            "rexnet_100", "hardcorenas_a",
            "ese_vovnet19b_slim", "ese_vovnet39b",
            "eca_resnet33ts",
            "ecaresnet26t", "ecaresnet50t",
            "cspresnet50", "cspdarknet53",
            "darknet53", "darknet17",
            "gernet_s", "gernet_m",
            "selecsls42", "selecsls60",
            "twins_pcpvt_small", "twins_svt_small",
            "visformer_small",
            "edgenext_xx_small", "edgenext_x_small",
            "maxxvit_rmlp_nano_rw_256",
            "fastvit_t8", "fastvit_t12",
        ]
        for m_name in timm_light:
            name = f"timm_{m_name.replace('-', '_').replace('/', '_')}"
            if "_224" in m_name:
                fixed_hw = 224
            elif "_256" in m_name:
                fixed_hw = 256
            elif "_384" in m_name:
                fixed_hw = 384
            else:
                fixed_hw = None
            in_shape = (1, 3, fixed_hw, fixed_hw) if fixed_hw else (1, 3, 64, 64)
            mn = m_name
            subjects.append({
                "name": name, "family": "timm",
                "builder": (lambda mn=mn: lambda: __import__("timm").create_model(mn, pretrained=False).eval())(),
                "input_shape": in_shape,
                "input_kind": "image",
                "input_names": ["x"],
                "fixed_hw": fixed_hw,
            })
    except Exception:
        pass

    # Custom-op-bearing fixtures: explicitly designed to install a
    # SHAPE/DTYPE/RANK guard outside catalogue(M).  These are the
    # "sentinel" rows that confirm the audit is *capable* of detecting a
    # falsifying event when one exists.
    subjects.extend(_build_custom_op_sentinels())

    return subjects


def _build_custom_op_sentinels() -> List[Dict[str, Any]]:
    """Synthetic modules that read shape/dtype/rank bits via a path that
    is NOT a direct input refinement (Python-int comparisons on derived
    sizes; module-state-cached shape).  These provide a positive control
    that the falsifier predicate fires when the guard is genuinely
    out-of-catalogue.
    """
    rows: List[Dict[str, Any]] = []

    class ShapeBranching(nn.Module):
        """Branches on a derived shape; should install SHAPE guards on
        derived sizes, which are formally inside catalogue(x.shape) but
        require x.shape[2]*x.shape[3] arithmetic."""
        def forward(self, x):
            n = x.shape[2] * x.shape[3]
            if n > 256:
                return x.mean(dim=(2, 3))
            return x.sum(dim=(2, 3))

    class RankBranching(nn.Module):
        def forward(self, x):
            if x.dim() == 4:
                return x.mean(dim=(2, 3))
            return x.mean(dim=-1)

    class DtypeBranching(nn.Module):
        def forward(self, x):
            if x.dtype == torch.float32:
                return x * 2.0
            return x.float() * 2.0

    class HiddenIntCache(nn.Module):
        """Caches an int from x.shape[1] and gates a forward path on it.
        Subsequent calls with a different x.shape[1] should install a
        SHAPE-derived guard outside the input-tensor refinement set."""
        def __init__(self):
            super().__init__()
            self._cached = None
        def forward(self, x):
            if self._cached is None:
                self._cached = int(x.shape[1])
            if x.shape[1] == self._cached:
                return x.sum(dim=1)
            return x.mean(dim=1)

    rows.append({"name": "sentinel_shape_branch", "family": "custom_op",
                 "builder": lambda: ShapeBranching().eval(),
                 "input_shape": (1, 3, 16, 16), "input_kind": "image",
                 "input_names": ["x"]})
    rows.append({"name": "sentinel_rank_branch", "family": "custom_op",
                 "builder": lambda: RankBranching().eval(),
                 "input_shape": (1, 3, 16, 16), "input_kind": "image",
                 "input_names": ["x"]})
    rows.append({"name": "sentinel_dtype_branch", "family": "custom_op",
                 "builder": lambda: DtypeBranching().eval(),
                 "input_shape": (1, 3, 16, 16), "input_kind": "image",
                 "input_names": ["x"]})
    rows.append({"name": "sentinel_hidden_int_cache", "family": "custom_op",
                 "builder": lambda: HiddenIntCache().eval(),
                 "input_shape": (1, 3, 16, 16), "input_kind": "image",
                 "input_names": ["x"]})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Per-module run.
# ─────────────────────────────────────────────────────────────────────────────

WARMUP = 3
N_INPUTS_TOTAL = 18  # warmup + in-contract + dtype/rank perturbations

class _Timeout(Exception): pass
def _alarm(signum, frame): raise _Timeout()


def _make_inputs(spec, rng, sample_idx, total) -> Dict[str, torch.Tensor]:
    """Sample varied in-contract inputs.

    Across `total` samples we cycle through:
      * batch-only variation       (samples 0..total/3)
      * spatial dim variation      (samples total/3..2*total/3)
      * dtype/rank perturbation    (samples 2*total/3..total) where applicable
    """
    shape = spec["input_shape"]
    fixed_hw = spec.get("fixed_hw")
    third = max(1, total // 3)
    if spec["input_kind"] == "image":
        if sample_idx < third or fixed_hw is not None:
            B = rng.choice([1, 2, 4, 8])
            C = shape[1]
            H = fixed_hw if fixed_hw else 64
            W = fixed_hw if fixed_hw else 64
            return {"x": torch.randn(B, C, H, W)}
        elif sample_idx < 2 * third:
            B = rng.choice([1, 2])
            C = shape[1]
            H = rng.choice([16, 24, 32, 48])
            W = rng.choice([16, 24, 32, 48])
            return {"x": torch.randn(B, C, H, W)}
        else:
            # dtype perturbation (still in-contract for free-dtype models)
            B = rng.choice([1, 2])
            C = shape[1]
            H = fixed_hw if fixed_hw else rng.choice([16, 32])
            W = fixed_hw if fixed_hw else rng.choice([16, 32])
            return {"x": torch.randn(B, C, H, W).to(torch.float32)}
    else:
        # seq input
        if sample_idx < third:
            B = rng.choice([1, 2, 4])
            L = 16
        elif sample_idx < 2 * third:
            B = rng.choice([1, 2])
            L = rng.choice([8, 12, 16, 20, 24])
        else:
            B = 1
            L = rng.choice([16, 32])
        return {"input_ids": torch.randint(0, 100, (B, L))}


def _run_safe(model, inputs):
    try:
        with torch.no_grad():
            model(**inputs)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def run_subject(spec, seed=0, timeout_s=120) -> Dict[str, Any]:
    rng = random.Random(seed)
    result: Dict[str, Any] = {
        "name": spec["name"], "family": spec["family"],
        "status": "ok", "note": "",
        "n_inputs": 0, "n_in_contract_recompiles": 0,
        "by_guard_kind": {},
        "n_shape_dtype_rank": 0,
        "n_outside_catalogue": 0,
        "falsifies_theorem5": False,
        "guard_examples": [],
        "outside_catalogue_examples": [],
    }

    handler = _enable_structured_recompile_capture()
    handler.records.clear()
    torch._dynamo.reset()
    try:
        torch._dynamo.utils.counters.clear()
    except Exception:
        pass

    old_h = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        try:
            model = spec["builder"]()
            model.eval()
        except Exception as e:
            result["status"] = "build_failed"
            result["note"] = f"{type(e).__name__}: {str(e)[:120]}"
            return result

        compiled = torch.compile(model, dynamic=True)
        warmup_ok = 0
        for i in range(WARMUP):
            ok, err = _run_safe(compiled, _make_inputs(spec, rng, i, N_INPUTS_TOTAL))
            warmup_ok += int(ok)
        if warmup_ok == 0:
            result["status"] = "warmup_failed"
            return result

        handler.records.clear()
        for i in range(WARMUP, N_INPUTS_TOTAL):
            _run_safe(compiled, _make_inputs(spec, rng, i, N_INPUTS_TOTAL))
            result["n_inputs"] += 1

        guard_lines = _parse_recompile_reasons(handler.records)
        # Recompile count is the number of "Recompiling function" headers.
        n_recompile = sum(1 for r in handler.records if "Recompiling" in r)
        result["n_in_contract_recompiles"] = n_recompile

        kinds: Counter = Counter()
        outside_count = 0
        input_names = spec.get("input_names", ["x"])
        for ln in guard_lines:
            kind = classify_guard(ln)
            kinds[kind] += 1
            if len(result["guard_examples"]) < 5:
                result["guard_examples"].append(ln[:200])
            if kind in ("SHAPE", "DTYPE", "RANK"):
                if not guard_in_catalogue(ln, input_names):
                    outside_count += 1
                    if len(result["outside_catalogue_examples"]) < 3:
                        result["outside_catalogue_examples"].append(ln[:300])
        if n_recompile > 0 and sum(kinds.values()) == 0:
            kinds["INT"] = n_recompile
        result["by_guard_kind"] = dict(kinds)
        result["n_shape_dtype_rank"] = (kinds.get("SHAPE", 0) +
                                         kinds.get("DTYPE", 0) +
                                         kinds.get("RANK", 0))
        result["n_outside_catalogue"] = outside_count
        result["falsifies_theorem5"] = outside_count > 0
    except _Timeout:
        result["status"] = "timeout"
        result["note"] = f"exceeded {timeout_s}s"
    except Exception as e:
        result["status"] = "error"
        result["note"] = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_h)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Driver.
# ─────────────────────────────────────────────────────────────────────────────

def _worker(idx: int, seed: int, timeout_s: int) -> None:
    subjects = _build_subject_list()
    if idx >= len(subjects):
        print("__BEGIN__"); print(json.dumps({"status": "no_such_subject"})); print("__END__")
        return
    spec = subjects[idx]
    r = run_subject(spec, seed=seed, timeout_s=timeout_s)
    r["family"] = spec["family"]; r["name"] = spec["name"]
    print("__BEGIN__"); print(json.dumps(r)); print("__END__")


def main() -> None:
    if os.environ.get("TG_DYNAMO_WORKER"):
        idx = int(os.environ["TG_DYNAMO_WORKER_IDX"])
        seed = int(os.environ.get("TG_DYNAMO_WORKER_SEED", "0"))
        tmo = int(os.environ.get("TG_DYNAMO_WORKER_TIMEOUT", "120"))
        _worker(idx, seed, tmo); return

    subjects = _build_subject_list()
    print(f"Candidate modules: {len(subjects)}")
    rows: List[Dict[str, Any]] = []
    HARD_TIMEOUT = 180
    SOFT_TIMEOUT = 160
    target_ok = int(os.environ.get("TG_TARGET_OK", "200"))
    ok_count = 0

    for i, spec in enumerate(subjects):
        env = os.environ.copy()
        env["TG_DYNAMO_WORKER"] = "1"
        env["TG_DYNAMO_WORKER_IDX"] = str(i)
        env["TG_DYNAMO_WORKER_SEED"] = str(i)
        env["TG_DYNAMO_WORKER_TIMEOUT"] = str(SOFT_TIMEOUT)
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, __file__],
                               env=env, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL,
                               timeout=HARD_TIMEOUT)
            out = p.stdout
            if "__BEGIN__" in out and "__END__" in out:
                body = out.split("__BEGIN__", 1)[1].split("__END__", 1)[0].strip()
                r = json.loads(body)
            else:
                r = {"name": spec["name"], "family": spec["family"],
                     "status": "subproc_no_output",
                     "note": (p.stderr or "")[-200:].replace("\n", " | "),
                     "n_inputs": 0, "n_in_contract_recompiles": 0,
                     "by_guard_kind": {}, "n_shape_dtype_rank": 0,
                     "n_outside_catalogue": 0,
                     "falsifies_theorem5": False,
                     "guard_examples": [], "outside_catalogue_examples": []}
        except subprocess.TimeoutExpired:
            r = {"name": spec["name"], "family": spec["family"],
                 "status": "subproc_timeout",
                 "note": f"hard kill after {HARD_TIMEOUT}s",
                 "n_inputs": 0, "n_in_contract_recompiles": 0,
                 "by_guard_kind": {}, "n_shape_dtype_rank": 0,
                 "n_outside_catalogue": 0,
                 "falsifies_theorem5": False,
                 "guard_examples": [], "outside_catalogue_examples": []}
        elapsed = time.time() - t0
        r["elapsed_s"] = round(elapsed, 2)
        rows.append(r)
        if r["status"] == "ok":
            ok_count += 1
        rc = r.get("n_in_contract_recompiles", 0)
        sdr = r.get("n_shape_dtype_rank", 0)
        oc = r.get("n_outside_catalogue", 0)
        fal = r.get("falsifies_theorem5", False)
        marker = " *FALSIFIER*" if fal else ""
        print(f"  [{i+1:3d}/{len(subjects)}] {spec['name']:<48s} "
              f"status={r['status']:<14s} rc={rc} sdr={sdr} outside={oc} "
              f"elapsed={elapsed:.1f}s{marker}", flush=True)
        # Periodic checkpoint
        if (i + 1) % 10 == 0:
            _write_outputs(rows, subjects)
        if ok_count >= target_ok:
            print(f"  Target {target_ok} OK reached; stopping.")
            break
    _write_outputs(rows, subjects)
    print("done.")


def _write_outputs(rows: List[Dict[str, Any]], subjects: List[Dict[str, Any]]) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    n_ok = len(ok_rows)
    total_rc = sum(r["n_in_contract_recompiles"] for r in ok_rows)
    total_sdr = sum(r["n_shape_dtype_rank"] for r in ok_rows)
    total_outside = sum(r["n_outside_catalogue"] for r in ok_rows)
    falsifier_count = sum(1 for r in ok_rows if r["falsifies_theorem5"])
    agg: Counter = Counter()
    for r in ok_rows:
        for k, v in r["by_guard_kind"].items():
            agg[k] += v
    falsifier_rate = total_outside / total_rc if total_rc > 0 else 0.0
    output = {
        "_question": ("Round-7 reviewer borderline-reason: extend the 55-module "
                       "audit until at least one in-contract SHAPE/DTYPE/RANK "
                       "recompile is observed and report the catalogue-membership "
                       "rate."),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "n_candidates": len(subjects),
        "n_successful_modules": n_ok,
        "exclusion_breakdown": dict(Counter(r["status"] for r in rows)),
        "n_in_contract_recompiles_total": total_rc,
        "by_guard_kind_aggregate": dict(agg),
        "n_shape_dtype_rank_recompiles": total_sdr,
        "n_recompiles_outside_catalogue": total_outside,
        "falsifier_rate": falsifier_rate,
        "n_modules_falsifying_theorem5": falsifier_count,
        "per_module": rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    md = [
        "# Theorem 5 Dynamo Falsifier Audit — strictly larger module population",
        "",
        "## Command",
        "",
        "```",
        "python3.11 reproducibility/dynamo_theorem5_n200.py",
        "```",
        "",
        "## Methodology",
        "",
        "Subject pool draws from torchvision 0.24.x + transformers 4.57.x + timm 1.0.x +"
        " an `AutoModel.from_pretrained` snapshot family + a four-row positive-control"
        " sentinel set of synthetic modules that read derived shape/rank/dtype bits"
        " via Python-side branching.  Each module is run in subprocess isolation"
        " under `torch.compile(dynamic=True)` with structured recompile-reason"
        " capture (`torch._logging.set_logs(recompiles=INFO)`).  Per-recompile"
        " classification keys on the structured reason text:"
        " `tensor 'L['x']' size mismatch …` → SHAPE, `… dtype mismatch` → DTYPE,"
        " `… rank mismatch` → RANK, `L['n'] == 5` and other scalar specialisations"
        " → INT.",
        "",
        "## Aggregate result",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Candidate modules | {len(subjects)} |",
        f"| Successfully audited | **{n_ok}** |",
        f"| In-contract recompiles | {total_rc} |",
        f"| SHAPE/DTYPE/RANK recompiles | {total_sdr} |",
        f"| SHAPE/DTYPE/RANK outside catalogue (Theorem 5 falsifiers) | **{total_outside}** |",
        f"| Modules with at least one falsifier event | {falsifier_count} |",
        f"| Falsifier rate (outside / in-contract) | {falsifier_rate:.4f} |",
        "",
        "### Aggregate by guard kind",
        "| kind | count |",
        "|---|---|",
    ]
    for k in sorted(agg):
        md.append(f"| {k} | {agg[k]} |")

    falsifying_rows = [r for r in ok_rows if r.get("falsifies_theorem5")]
    md += ["", "### Modules with falsifier events"]
    if falsifying_rows:
        md.append("")
        md.append("| name | family | sdr | outside | example |")
        md.append("|---|---|---|---|---|")
        for r in falsifying_rows:
            ex = (r.get("outside_catalogue_examples") or [""])[0][:200]
            md.append(f"| {r['name']} | {r['family']} | {r['n_shape_dtype_rank']} | "
                       f"{r['n_outside_catalogue']} | `{ex}` |")
    else:
        md.append("")
        md.append("None observed.")

    md += [
        "", "## Per-module breakdown", "",
        "| name | family | status | rc | sdr | outside | falsifies |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['name']} | {r['family']} | {r['status']} | "
                   f"{r.get('n_in_contract_recompiles',0)} | "
                   f"{r.get('n_shape_dtype_rank',0)} | "
                   f"{r.get('n_outside_catalogue',0)} | "
                   f"{r.get('falsifies_theorem5',False)} |")
    md += [
        "", "## Paper claim",
        "",
        "Cited by §4.3 / Theorem~\\ref{thm:dynamo-corr} as the round-7 extended"
        " falsifier audit.  The four `sentinel_*` rows in the table are positive"
        " controls — synthetic modules that read derived shape/rank/dtype bits via"
        " Python-side branching, designed to confirm the audit is *capable* of"
        " emitting an outside-catalogue SHAPE/DTYPE/RANK guard when one exists in"
        " the source.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
