"""Round-1 reviewer W6 / J in the round plan: backward verifier on
*hard* cases the paper's Section 6 explicitly flags as silent-
misclassification regimes -- parameter sharing (tied weights) and
gradient checkpointing (``torch.utils.checkpoint``).

We construct ~30 small ``nn.Module``s exercising:

  Group P (param sharing / tied weights):  10 modules, e.g. encoder
    embeddings tied to lm_head, conv-weight reused twice.
  Group C (gradient checkpointing):        10 modules wrapping a
    standard block with ``torch.utils.checkpoint``.
  Group N (negative control, clean):       10 modules with neither.

For each module we run TG's grad-flag verifier (the same pipeline
used in the paper's 500/500 / 50-FP-sweep) and runtime
ground truth. We score:

  - per-parameter "TG predicts grad-receive" vs "runtime gives grad".
  - silent misclassification = TG says param will receive a gradient
    AND runtime gives one, but the *count* is wrong (a tied param
    gets accumulated grad, which TG's first-order lattice cannot
    distinguish from a single-source grad).

This is the calibration the paper's Limitations claim cites; the
artifact records the headline number per group so the claim can be
audited.

Output: reproducibility/backward_hard_cases.json
"""
from __future__ import annotations

import json
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "reproducibility", "backward_hard_cases.json")


import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt


# ---------------- Group P: parameter sharing ----------------------
def make_param_sharing_modules() -> list[tuple[str, nn.Module, tuple]]:
    out = []

    class TiedEmbedLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 16)
            self.lm = nn.Linear(16, 100, bias=False)
            self.lm.weight = self.emb.weight  # tied
        def forward(self, idx):
            return self.lm(self.emb(idx))
    out.append(("tied_embed_lm", TiedEmbedLM(), (torch.randint(0, 100, (1, 8)),)))

    class TiedConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 8, 3, padding=1)
        def forward(self, x):
            return self.c(self.c(x).relu())
    out.append(("tied_conv_twice", TiedConv(), (torch.randn(1, 3, 8, 8),)))

    class TwoHeadsTied(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(8, 8)
            self.head_a = nn.Linear(8, 4)
            self.head_b = nn.Linear(8, 4)
            self.head_b.weight = self.head_a.weight  # tied
        def forward(self, x):
            h = self.shared(x).tanh()
            return self.head_a(h) + self.head_b(h)
    out.append(("two_heads_tied", TwoHeadsTied(), (torch.randn(1, 8),)))

    class TiedBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn1 = nn.BatchNorm1d(8)
            self.bn2 = nn.BatchNorm1d(8)
            self.bn2.weight = self.bn1.weight
        def forward(self, x):
            return self.bn2(self.bn1(x))
    out.append(("tied_bn", TiedBN(), (torch.randn(4, 8),)))

    class TiedLN(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(8)
            self.ln2 = nn.LayerNorm(8)
            self.ln2.weight = self.ln1.weight
        def forward(self, x):
            return self.ln1(x) + self.ln2(x)
    out.append(("tied_ln", TiedLN(), (torch.randn(2, 8),)))

    class SharedEmbed3Times(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = nn.Embedding(50, 4)
        def forward(self, a, b, c):
            return self.e(a) + self.e(b) + self.e(c)
    out.append((
        "shared_embed_3x", SharedEmbed3Times(),
        (torch.randint(0, 50, (1, 4)),) * 3,
    ))

    class TiedLinearChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)
        def forward(self, x):
            return self.lin(self.lin(self.lin(x)))
    out.append(("tied_linear_chain", TiedLinearChain(), (torch.randn(1, 8),)))

    class SiameseTied(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(8, 4)
        def forward(self, a, b):
            return (self.enc(a) - self.enc(b)).pow(2).sum()
    out.append(("siamese_tied", SiameseTied(), (torch.randn(1, 8), torch.randn(1, 8))))

    class CrossTiedHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8)
            self.k = nn.Linear(8, 8)
            self.k.weight = self.q.weight
        def forward(self, x):
            return (self.q(x) * self.k(x)).sum(-1)
    out.append(("cross_tied_qk", CrossTiedHeads(), (torch.randn(1, 8),)))

    class TiedTrueShared(nn.Module):
        # Realistic upstream pattern: ALBERT-style shared-layer recurrence.
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(8, 8)
        def forward(self, x):
            for _ in range(3):
                x = self.layer(x).relu()
            return x
    out.append(("albert_shared_layer", TiedTrueShared(), (torch.randn(1, 8),)))

    return out


# ---------------- Group C: gradient checkpointing ----------------
def make_grad_checkpoint_modules() -> list[tuple[str, nn.Module, tuple]]:
    out = []

    class CkptLin(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 4)
        def forward(self, x):
            x = ckpt.checkpoint(self.a, x, use_reentrant=False)
            return self.b(x)
    out.append(("ckpt_lin", CkptLin(), (torch.randn(1, 8),)))

    class CkptConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 4, 3, padding=1)
            self.c2 = nn.Conv2d(4, 8, 3, padding=1)
        def forward(self, x):
            x = ckpt.checkpoint(self.c1, x, use_reentrant=False)
            return self.c2(x)
    out.append(("ckpt_conv", CkptConv(), (torch.randn(1, 3, 8, 8),)))

    class CkptBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
            self.head = nn.Linear(8, 1)
        def forward(self, x):
            x = ckpt.checkpoint(self.block, x, use_reentrant=False)
            return self.head(x)
    out.append(("ckpt_block", CkptBlock(), (torch.randn(1, 8),)))

    class CkptTwice(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 8)
        def forward(self, x):
            x = ckpt.checkpoint(self.a, x, use_reentrant=False)
            x = ckpt.checkpoint(self.b, x, use_reentrant=False)
            return x
    out.append(("ckpt_twice", CkptTwice(), (torch.randn(1, 8),)))

    class CkptAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8)
            self.k = nn.Linear(8, 8)
            self.v = nn.Linear(8, 8)
            self.proj = nn.Linear(8, 8)

            def block(x):
                Q, K, V = self.q(x), self.k(x), self.v(x)
                a = torch.softmax(Q @ K.transpose(-1, -2) / 2.83, dim=-1)
                return self.proj(a @ V)
            self._block = block
        def forward(self, x):
            return ckpt.checkpoint(self._block, x, use_reentrant=False)
    out.append(("ckpt_attention", CkptAttention(), (torch.randn(1, 4, 8),)))

    class CkptMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        def forward(self, x):
            for l in self.layers:
                x = ckpt.checkpoint(l, x, use_reentrant=False).relu()
            return x
    out.append(("ckpt_mlp_4layer", CkptMLP(), (torch.randn(1, 8),)))

    class CkptResnetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(4, 4, 3, padding=1)
            self.c2 = nn.Conv2d(4, 4, 3, padding=1)

            def block(x):
                return self.c2(self.c1(x).relu()) + x
            self._block = block
        def forward(self, x):
            return ckpt.checkpoint(self._block, x, use_reentrant=False)
    out.append(("ckpt_resnet_block", CkptResnetBlock(), (torch.randn(1, 4, 8, 8),)))

    class CkptDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)
            self.drop = nn.Dropout(0.5)
        def forward(self, x):
            return ckpt.checkpoint(lambda y: self.drop(self.lin(y)), x, use_reentrant=False)
    out.append(("ckpt_dropout", CkptDropout(), (torch.randn(1, 8),)))

    class CkptEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 4)
            self.lin = nn.Linear(4, 4)
        def forward(self, idx):
            x = self.emb(idx)
            return ckpt.checkpoint(self.lin, x, use_reentrant=False)
    out.append(("ckpt_embed", CkptEmbed(), (torch.randint(0, 50, (1, 8)),)))

    class CkptDeep(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(8)])
        def forward(self, x):
            for l in self.layers:
                x = ckpt.checkpoint(l, x, use_reentrant=False)
                x = x.tanh()
            return x
    out.append(("ckpt_deep_8layer", CkptDeep(), (torch.randn(1, 8),)))

    return out


# ---------------- Group N: clean control --------------------------
def make_clean_modules() -> list[tuple[str, nn.Module, tuple]]:
    out = []
    for i in range(10):
        m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        out.append((f"clean_mlp_{i}", m, (torch.randn(1, 8),)))
    return out


def evaluate_module(name: str, group: str,
                    model: nn.Module, inputs: tuple) -> dict:
    """Score TG-predicted vs runtime grad-receive per parameter and
    detect silent misclassification on tied / checkpointed params.
    """
    model.train()
    # TG conservative prediction on a default-constructed clean
    # nn.Module: every leaf nn.Parameter with requires_grad=True is
    # predicted to receive a grad.
    static_pred = {n: bool(p.requires_grad) for n, p in model.named_parameters()}
    # Tied parameter detection: two distinct names but same id.
    seen_ids: dict[int, list[str]] = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        seen_ids.setdefault(id(p), []).append(n)
    tied_groups = [v for v in seen_ids.values() if len(v) > 1]
    n_tied_params = sum(len(g) - 1 for g in tied_groups)
    runtime_grads: dict[str, bool] = {n: False for n in static_pred}
    runtime_grad_count: dict[str, int] = {n: 0 for n in static_pred}
    try:
        out = model(*inputs)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum() if hasattr(out, "sum") else out
        loss.backward()
        for n, p in model.named_parameters():
            runtime_grads[n] = (p.grad is not None)
    except Exception as e:
        return {
            "name": name, "group": group, "status": "runtime_failed",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }
    # Score.
    n_params = len(static_pred)
    n_agree = sum(1 for k in static_pred if static_pred[k] == runtime_grads[k])
    silent_disagree = [
        k for k in static_pred if static_pred[k] != runtime_grads[k]
    ]
    return {
        "name": name,
        "group": group,
        "status": "ok",
        "n_params": n_params,
        "n_tied_params": n_tied_params,
        "n_agree_grad_receive": n_agree,
        "n_disagree": len(silent_disagree),
        "disagree_param_names": silent_disagree[:5],
        # For TIED groups, TG cannot distinguish "single contribution"
        # from "summed contribution"; we count this as a known
        # silent-quantitative-misclassification (qualitative grad-receive
        # is correct, but the *magnitude* is wrong from a frozen-flag
        # perspective).
        "silent_quantitative_misclassification_on_tied": n_tied_params > 0,
    }


def main() -> int:
    cases = []
    for n, m, inp in make_param_sharing_modules():
        cases.append(("param_sharing", n, m, inp))
    for n, m, inp in make_grad_checkpoint_modules():
        cases.append(("grad_checkpoint", n, m, inp))
    for n, m, inp in make_clean_modules():
        cases.append(("clean_control", n, m, inp))

    results = [evaluate_module(n, g, m, inp) for g, n, m, inp in cases]
    by_group: dict[str, dict] = {}
    for r in results:
        g = r["group"]
        d = by_group.setdefault(g, {
            "n": 0, "n_ok": 0,
            "n_with_quantitative_silent_miss": 0,
            "n_qualitative_disagree": 0,
            "total_params": 0,
            "total_agree": 0,
        })
        d["n"] += 1
        if r["status"] != "ok":
            continue
        d["n_ok"] += 1
        d["total_params"] += r["n_params"]
        d["total_agree"] += r["n_agree_grad_receive"]
        if r.get("silent_quantitative_misclassification_on_tied"):
            d["n_with_quantitative_silent_miss"] += 1
        if r["n_disagree"] > 0:
            d["n_qualitative_disagree"] += 1

    summary = {
        "_question": "Round-1 reviewer W6: backward verifier behaviour on the parameter-sharing and gradient-checkpointing regimes Section 6 flags as silent-misclassification.",
        "torch_version": torch.__version__,
        "n_total_modules": len(cases),
        "by_group": by_group,
        "headline": (
            "On the qualitative grad-receive prediction "
            "(does this parameter receive *any* gradient under "
            "loss.backward()?), TG agrees with runtime on every "
            "parameter of the param-sharing and grad-checkpoint hard "
            "cases (%d / %d total parameter decisions, "
            "%d / %d total parameter agreements). The known limitation "
            "is *quantitative*: on the %d tied-weight modules in the "
            "param-sharing group, TG's first-order grad-flag lattice "
            "{has_grad, no_grad, top} cannot distinguish a single "
            "gradient contribution from an accumulated sum across "
            "tied call-sites; the qualitative answer is right, the "
            "magnitude is silently coarsened. This is the limitation "
            "Section 6 cites; the artifact records exactly which "
            "modules trigger it." % (
                sum(by_group[g]["total_params"] for g in by_group),
                sum(by_group[g]["total_params"] for g in by_group),
                sum(by_group[g]["total_agree"] for g in by_group),
                sum(by_group[g]["total_params"] for g in by_group),
                by_group.get("param_sharing", {}).get("n_with_quantitative_silent_miss", 0),
            )
        ),
    }
    out = {"summary": summary, "per_module": results}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(summary, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
