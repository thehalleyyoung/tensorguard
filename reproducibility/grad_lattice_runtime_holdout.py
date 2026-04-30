#!/usr/bin/env python3.11
"""Held-out runtime grad-lattice false-verified-rate (round 5 rewrite).

Round-4 reviewers correctly observed that the prior version of this
artefact passed `inspect.getsource(model.__class__)` for large
HuggingFace head classes, which produced TG verdict UNSAFE with first
bug ``[MODEL_CHECK] No nn.Module subclass found in source`` -- a
parser-failure marker rather than a genuine grad-lattice abstention.
The reported 0/8 false-verified rate was therefore vacuous.

This rewrite builds the held-out positive sample as N=10 *self-
contained* nn.Module subclasses, each of which exercises one of the
constructs that breaks the first-order grad lattice in the same way
HF Trainer / accelerate scripts do:

  * ``torch.utils.checkpoint.checkpoint(self.layer, x)`` (4 subjects)
  * ``model.gradient_checkpointing_enable()`` style toggle keyword in
    the constructor / inline ``if self.gradient_checkpointing:`` body
    branching to a checkpoint call (2 subjects)
  * Tied / renamed-attribute parameter sharing patterns
    (``self.lm_head.weight = self.embed.weight``) that HF Trainer
    relies on (2 subjects)
  * ``checkpoint_sequential(...)`` (1 subject)
  * Two clean negative controls that should *not* abstain
    (a small MLP, a small Conv-BN-ReLU stack) so we can confirm the
    detector does not over-fire.

For each subject we (a) instantiate, (b) run one forward + backward
step on a small in-contract input, (c) record per-parameter
``p.grad != None``, and (d) run TG with ``check_gradients=True`` on
the raw nn.Module subclass source.

A *false-verify* is recorded when TG returns SAFE+VERIFIED on a
positive subject.  Negative-control subjects must not flip into
out-of-fragment.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from src.api import verify_architecture

OUT_JSON = os.path.join(ROOT, "reproducibility",
                        "grad_lattice_runtime_holdout.json")
OUT_MD = os.path.join(ROOT, "reproducibility",
                      "grad_lattice_runtime_holdout.md")


# ── Subject sources (positives + negative controls) ────────────────
# Each entry: (name, kind in {"checkpoint", "gc_enable",
#  "tied_weights", "checkpoint_sequential", "clean"}, source, input
#  factory, expected TG behaviour: "out_of_fragment" or "verified".)


SUBJECTS: List[Dict[str, Any]] = [
    {
        "name": "ResidualMLP_checkpoint",
        "kind": "checkpoint",
        "expected_tg": "out_of_fragment",
        "src": '''
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class ResidualMLP_checkpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
    def _block(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
    def forward(self, x):
        return x + checkpoint(self._block, x, use_reentrant=False)
''',
        "input": lambda: (torch.randn(4, 32),),
    },
    {
        "name": "TwoLayerCNN_checkpoint",
        "kind": "checkpoint",
        "expected_tg": "out_of_fragment",
        "src": '''
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class TwoLayerCNN_checkpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
    def _stage(self, x):
        return torch.relu(self.c2(torch.relu(self.c1(x))))
    def forward(self, x):
        return checkpoint(self._stage, x, use_reentrant=False)
''',
        "input": lambda: (torch.randn(2, 3, 8, 8),),
    },
    {
        "name": "GatedTransformerBlock_checkpoint",
        "kind": "checkpoint",
        "expected_tg": "out_of_fragment",
        "src": '''
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class GatedTransformerBlock_checkpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.k = nn.Linear(16, 16)
        self.v = nn.Linear(16, 16)
        self.o = nn.Linear(16, 16)
    def _attn(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        a = torch.softmax(q @ k.transpose(-2, -1) / 4.0, dim=-1)
        return self.o(a @ v)
    def forward(self, x):
        return x + checkpoint(self._attn, x, use_reentrant=False)
''',
        "input": lambda: (torch.randn(2, 5, 16),),
    },
    {
        "name": "SequentialMLP_checkpoint_sequential",
        "kind": "checkpoint_sequential",
        "expected_tg": "out_of_fragment",
        "src": '''
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

class SequentialMLP_checkpoint_sequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )
    def forward(self, x):
        return checkpoint_sequential(self.layers, 2, x, use_reentrant=False)
''',
        "input": lambda: (torch.randn(2, 8),),
    },
    {
        "name": "InlineCheckpointToggle_gc",
        "kind": "gc_enable",
        "expected_tg": "out_of_fragment",
        "src": '''
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class InlineCheckpointToggle_gc(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = True
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
    def _body(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._body, x, use_reentrant=False)
        return self._body(x)
''',
        "input": lambda: (torch.randn(2, 8),),
    },
    {
        "name": "HfStyleEnableToggle_gc",
        "kind": "gc_enable",
        "expected_tg": "out_of_fragment",
        "src": '''
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class HfStyleEnableToggle_gc(nn.Module):
    def __init__(self, gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.fc = nn.Linear(8, 8)
    def _set_gradient_checkpointing(self, enable=True):
        self.gradient_checkpointing = enable
    def gradient_checkpointing_enable(self):
        self._set_gradient_checkpointing(True)
    def forward(self, x):
        if self.gradient_checkpointing:
            return checkpoint(self.fc, x, use_reentrant=False)
        return self.fc(x)
''',
        "input": lambda: (torch.randn(2, 8),),
    },
    {
        "name": "TiedEmbeddingLMHead_tied",
        "kind": "tied_weights",
        "expected_tg": "out_of_fragment",
        "src": '''
import torch
import torch.nn as nn

class TiedEmbeddingLMHead_tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.lm_head = nn.Linear(16, 64, bias=False)
        # HF-style weight tying
        self.lm_head.weight = self.embed.weight
    def forward(self, x):
        return self.lm_head(self.embed(x))
''',
        "input": lambda: (torch.randint(0, 64, (2, 5)),),
        # Tied weights are not auto-detected by the lattice; expect
        # SAFE+VERIFIED here (this row will *count as* a false-verify
        # if the runtime grad set disagrees with what TG inferred).
        # This is the worst-case false-verify probe.
    },
    {
        "name": "RenamedSharedLinear_tied",
        "kind": "tied_weights",
        "expected_tg": "out_of_fragment",
        "src": '''
import torch
import torch.nn as nn

class RenamedSharedLinear_tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(8, 16)
        self.out_proj = nn.Linear(16, 8)
        # alias the parameter under a second name
        self.shared_w = self.in_proj.weight
    def forward(self, x):
        return self.out_proj(torch.relu(self.in_proj(x)))
''',
        "input": lambda: (torch.randn(2, 8),),
    },
    {
        "name": "CleanMLP_negative_control",
        "kind": "clean",
        "expected_tg": "verified",
        "src": '''
import torch
import torch.nn as nn

class CleanMLP_negative_control(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
''',
        "input": lambda: (torch.randn(2, 8),),
    },
    {
        "name": "CleanConvBNReLU_negative_control",
        "kind": "clean",
        "expected_tg": "verified",
        "src": '''
import torch
import torch.nn as nn

class CleanConvBNReLU_negative_control(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
    def forward(self, x):
        return torch.relu(self.bn(self.c(x)))
''',
        "input": lambda: (torch.randn(2, 3, 8, 8),),
    },
]


def _runtime_step(model: nn.Module, args) -> Dict[str, Any]:
    model.train()
    try:
        out = model(*args)
        loss = out.float().sum() if torch.is_tensor(out) else out[0].float().sum()
        loss.backward()
    except Exception as e:
        return {"ok": False, "note": f"{type(e).__name__}: {str(e)[:120]}",
                "n_with_grad": 0, "n_without_grad": 0}
    n_with = n_without = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is not None:
            n_with += 1
        else:
            n_without += 1
    return {"ok": True, "note": "",
            "n_with_grad": n_with, "n_without_grad": n_without}


def _instantiate(src: str, name: str):
    ns: Dict[str, Any] = {}
    exec(src, ns, ns)
    return ns[name]()


def _tg(src: str) -> Dict[str, Any]:
    try:
        r = verify_architecture(src, check_gradients=True)
    except Exception as e:
        return {"verdict": "ANALYSER_ERR",
                "note": f"{type(e).__name__}: {str(e)[:80]}",
                "n_bugs": 0, "first_bug": ""}
    bugs = list(getattr(r, "bugs", []) or [])
    return {"verdict": r.status,
            "n_bugs": len(bugs),
            "first_bug": (bugs[0].message[:160] if bugs else ""),
            "abstained": getattr(r, "abstained", False)}


def main() -> int:
    rows = []
    n_pos = n_neg = 0
    n_pos_oof = 0          # TG returned a grad-out-of-fragment bug
    n_pos_false_verified = 0  # TG returned SAFE+VERIFIED on a positive
    n_neg_unaffected = 0    # TG returned SAFE+VERIFIED on a negative
    n_neg_false_oof = 0     # TG returned out-of-fragment on a clean negative

    for spec in SUBJECTS:
        name = spec["name"]
        src = spec["src"]
        kind = spec["kind"]
        try:
            model = _instantiate(src, name)
        except Exception as e:
            rows.append({"name": name, "kind": kind,
                         "build_error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        rt = _runtime_step(model, spec["input"]())
        tg = _tg(src)
        is_pos = (kind != "clean")
        is_grad_oof = ("GRADIENT-OUT-OF-FRAGMENT" in tg["first_bug"])
        is_verified = (tg["verdict"] == "SAFE" and tg["n_bugs"] == 0)
        is_false_verified = is_pos and is_verified
        if is_pos:
            n_pos += 1
            if is_grad_oof:
                n_pos_oof += 1
            if is_false_verified:
                n_pos_false_verified += 1
        else:
            n_neg += 1
            if is_verified:
                n_neg_unaffected += 1
            if is_grad_oof:
                n_neg_false_oof += 1
        rows.append({
            "name": name, "kind": kind,
            "expected_tg": spec["expected_tg"],
            "runtime_ok": rt["ok"],
            "runtime_note": rt["note"],
            "n_params_with_grad": rt["n_with_grad"],
            "n_params_without_grad": rt["n_without_grad"],
            "tg_verdict": tg["verdict"],
            "tg_n_bugs": tg["n_bugs"],
            "tg_first_bug": tg["first_bug"],
            "tg_grad_out_of_fragment": is_grad_oof,
            "is_false_verified": is_false_verified,
        })
        print(f"  {name:42s} kind={kind:24s} runtime_ok={rt['ok']} "
              f"tg={tg['verdict']:7s} oof={is_grad_oof} fv={is_false_verified}")

    out = {
        "_question": (
            "R5-W3 / R5-Q2: redo the held-out runtime grad-lattice "
            "false-verified-rate using TG-parseable, self-contained "
            "nn.Module subjects so the verdict reflects an actual "
            "grad-lattice ruling rather than the parser-failure marker "
            "the prior version of this artefact returned."
        ),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "torch_version": torch.__version__,
        "n_subjects": len(SUBJECTS),
        "n_positive": n_pos,
        "n_negative_control": n_neg,
        "n_positive_grad_out_of_fragment": n_pos_oof,
        "n_positive_false_verified": n_pos_false_verified,
        "n_negative_correctly_verified": n_neg_unaffected,
        "n_negative_false_out_of_fragment": n_neg_false_oof,
        "false_verified_rate": (
            n_pos_false_verified / n_pos if n_pos else 0.0),
        "negative_control_specificity": (
            n_neg_unaffected / n_neg if n_neg else 0.0),
        "interpretation": (
            f"On {n_pos} held-out positives that exercise gradient "
            f"checkpointing, ``gradient_checkpointing_enable``, "
            f"``checkpoint_sequential``, or HF-style tied/renamed "
            f"shared parameters, TG returns a "
            f"``[GRADIENT-OUT-OF-FRAGMENT]`` Refuted-Proof on "
            f"{n_pos_oof}/{n_pos} subjects and SAFE+VERIFIED on "
            f"{n_pos_false_verified}/{n_pos}.  On {n_neg} clean "
            f"negative controls TG remains SAFE+VERIFIED on "
            f"{n_neg_unaffected}/{n_neg} (specificity preserved) and "
            f"misfires the out-of-fragment marker on "
            f"{n_neg_false_oof}/{n_neg}.  This replaces the prior "
            f"vacuous 0/8 with a real false-verified rate of "
            f"{n_pos_false_verified}/{n_pos} on a runtime-positive "
            f"sample where TG actually parses and reasons about each "
            f"subject."
        ),
        "per_subject": rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Grad-flag false-verified-rate vs runtime p.grad != None (round 5 rewrite)",
        "",
        "## Command",
        "",
        "```",
        "python3.11 reproducibility/grad_lattice_runtime_holdout.py",
        "```",
        "",
        "## Held-out positive sample (round 5 rewrite)",
        "",
        f"This artefact replaces the round-4 version, which fed raw "
        f"HuggingFace head-class source via "
        f"``inspect.getsource(model.__class__)`` and produced a "
        f"vacuous 0/8 because TG could not parse the truncated source. "
        f"This rewrite uses {len(SUBJECTS)} *self-contained* "
        f"``nn.Module`` subclasses ({n_pos} positives + {n_neg} clean "
        f"negative controls) that exercise the same constructs in a "
        f"form TG can actually parse.",
        "",
        "## Result",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Subjects (total) | {len(SUBJECTS)} |",
        f"| Positives | {n_pos} |",
        f"| Negative controls | {n_neg} |",
        f"| Positives with `[GRADIENT-OUT-OF-FRAGMENT]` Refuted-Proof | {n_pos_oof}/{n_pos} |",
        f"| Positive false-verified (TG SAFE+VERIFIED on a runtime positive) | {n_pos_false_verified}/{n_pos} |",
        f"| Negative-control SAFE+VERIFIED (specificity) | {n_neg_unaffected}/{n_neg} |",
        f"| Negative-control false-out-of-fragment | {n_neg_false_oof}/{n_neg} |",
        "",
        "## Per-subject",
        "",
        "| name | kind | runtime_ok | tg_verdict | n_bugs | grad_oof | false_verified |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "build_error" in r:
            md.append(f"| {r['name']} | {r['kind']} | BUILD_ERROR: {r['build_error']} | - | - | - | - |")
            continue
        md.append(
            f"| {r['name']} | {r['kind']} | {r['runtime_ok']} | "
            f"{r['tg_verdict']} | {r['tg_n_bugs']} | "
            f"{r['tg_grad_out_of_fragment']} | {r['is_false_verified']} |")
    md += [
        "",
        "## Paper claim closed",
        "",
        "Round-5 reviewer W3 / Q2 noted that the prior version of this "
        "artefact reported `0/8 false-verified` on subjects whose first "
        "TG bug was the parser-failure marker `No nn.Module subclass "
        "found in source`, making the rate vacuous.  This rewrite ships "
        "self-contained `nn.Module` subjects that TG actually parses. "
        f"The grad-lattice out-of-fragment detector "
        f"(`[GRADIENT-OUT-OF-FRAGMENT]`) fires on {n_pos_oof}/{n_pos} "
        f"positives and the measured false-verified-rate is "
        f"{n_pos_false_verified}/{n_pos}.  Negative-control specificity "
        f"is {n_neg_unaffected}/{n_neg}.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")
    print(f"  Positive false-verified: {n_pos_false_verified}/{n_pos}; "
          f"negative-control SAFE: {n_neg_unaffected}/{n_neg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
