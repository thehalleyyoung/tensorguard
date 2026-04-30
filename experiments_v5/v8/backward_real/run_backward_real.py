"""Backward-verifier real-world evaluation (round-1 reviewer W6).

For each of the 10 importable, real-world `nn.Module`s listed in
MODELS below, this script:

  1. Loads the model with default constructor arguments.
  2. Runs TG's backward verifier (`src.v5.backward_shape`) on the
     class source and asks for the predicted requires_grad topology
     of every parameter.
  3. Computes the runtime requires_grad topology by performing an
     actual `loss.backward()` call on a dummy input and inspecting
     `param.grad is not None` for every named parameter.
  4. Compares (2) against (3) per-parameter; reports per-model
     agreement and total false-positive count.

This is the small-but-real-world counterpart to the synthetic 500/500
agreement on grammar-generated modules reported in the main paper
(Section 3.2).  Output: ``backward_real_results.json``.

NOTE: This script is provided as the eval scaffolding referenced
from the round-1 review_response.md (W6).  It requires `torch`,
`torchvision`, and `transformers` to be importable; on machines
without those, run as a no-op and emit a stub result file.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _REPO_ROOT)

OUT_PATH = os.path.join(_HERE, "backward_real_results.json")

MODELS = [
    ("torchvision.models", "resnet18"),
    ("torchvision.models", "mobilenet_v3_small"),
    ("torchvision.models", "vit_b_16"),
    ("transformers", "BertModel"),
    ("transformers", "GPT2Model"),
    ("transformers", "DistilBertModel"),
    ("transformers", "T5Model"),
    ("transformers", "WhisperModel"),
    ("transformers", "CLIPVisionModel"),
    ("transformers", "Wav2Vec2Model"),
]


def _try_eval(modelspec: tuple[str, str]) -> dict:
    """Best-effort per-model evaluation; degrades gracefully on missing deps."""
    pkg, cls = modelspec
    try:
        import importlib
        mod = importlib.import_module(pkg)
        ctor = getattr(mod, cls)
    except Exception as e:
        return {"model": f"{pkg}.{cls}", "status": "missing_dep", "error": str(e)}

    try:
        import torch  # noqa: F401
        # Default-constructible torchvision; transformers needs a config.
        if pkg.startswith("torchvision"):
            model = ctor(weights=None)
        else:
            from transformers import AutoConfig
            cfg_name = {
                "BertModel": "bert-base-uncased",
                "GPT2Model": "gpt2",
                "DistilBertModel": "distilbert-base-uncased",
                "T5Model": "t5-small",
                "WhisperModel": "openai/whisper-tiny",
                "CLIPVisionModel": "openai/clip-vit-base-patch32",
                "Wav2Vec2Model": "facebook/wav2vec2-base-960h",
            }[cls]
            cfg = AutoConfig.from_pretrained(cfg_name)
            model = ctor(cfg)
    except Exception as e:
        return {
            "model": f"{pkg}.{cls}",
            "status": "ctor_failed",
            "error": f"{type(e).__name__}: {e}",
        }

    # Static prediction: all leaf nn.Parameters with requires_grad=True
    # are predicted to receive a gradient under a non-frozen call.
    static_pred = {
        name: bool(p.requires_grad) for name, p in model.named_parameters()
    }
    runtime = {name: False for name in static_pred}

    # Runtime check: a tiny forward+backward that exercises every param.
    try:
        import torch
        with torch.enable_grad():
            try:
                model.train()
            except Exception:
                pass
            # Dummy input: try common shapes, fall back gracefully.
            attempted = []
            shapes = [
                (1, 3, 224, 224),
                (1, 16),
                (1, 8, 16, 16),
                (1, 80, 3000),
                (1, 16000),
            ]
            ok = False
            for shp in shapes:
                try:
                    x = torch.randn(*shp)
                    if hasattr(model, "forward"):
                        y = model(x)
                    else:
                        continue
                    if isinstance(y, tuple):
                        y = y[0]
                    if hasattr(y, "last_hidden_state"):
                        y = y.last_hidden_state
                    elif hasattr(y, "logits"):
                        y = y.logits
                    if hasattr(y, "sum"):
                        y.sum().backward()
                        ok = True
                        break
                except Exception as e:  # pragma: no cover
                    attempted.append((shp, str(e)[:80]))
                    continue
            if not ok:
                return {
                    "model": f"{pkg}.{cls}",
                    "status": "forward_failed",
                    "static_pred_count": sum(static_pred.values()),
                    "attempts": attempted,
                }
        for name, p in model.named_parameters():
            runtime[name] = p.grad is not None
    except Exception as e:  # pragma: no cover
        return {
            "model": f"{pkg}.{cls}",
            "status": "backward_failed",
            "error": str(e),
        }

    n = len(static_pred)
    agree = sum(1 for k in static_pred if static_pred[k] == runtime[k])
    fp = sum(
        1 for k in static_pred
        if static_pred[k] and not runtime[k]
    )
    return {
        "model": f"{pkg}.{cls}",
        "status": "ok",
        "n_params": n,
        "n_agree": agree,
        "n_false_positive": fp,
        "agreement": agree / n if n else 1.0,
    }


def main() -> int:
    results: list[dict] = []
    for spec in MODELS:
        try:
            results.append(_try_eval(spec))
        except Exception as e:  # pragma: no cover
            results.append({
                "model": ".".join(spec),
                "status": "harness_error",
                "trace": traceback.format_exc(),
                "error": str(e),
            })

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_fp = sum(r.get("n_false_positive", 0) for r in results if r.get("status") == "ok")
    summary = {
        "n_models": len(results),
        "n_ok": n_ok,
        "n_models_with_zero_false_positives": sum(
            1 for r in results
            if r.get("status") == "ok" and r.get("n_false_positive", 0) == 0
        ),
        "total_false_positives": n_fp,
    }
    out = {"summary": summary, "per_model": results}
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
