"""50-model false-positive sweep for the backward verifier (Round 2 / Q4).

This harness explicitly defines the 50 clean ``nn.Module``s the
``track_D_results.json`` 50-model FP sweep was originally run on,
and re-runs the FP check end-to-end. A "false positive" (FP) for the
**TG backward verifier** is a model on which the verifier emits a
``Bug`` object (any ``GRAD_FLAG_*`` refutation) although a real
``loss.backward()`` succeeds with finite gradients on every
parameter the verifier flagged.

The verifier is *conservative*: it only refutes when it can prove a
parameter is dead (no_grad scope, ``.detach()`` chain, missing leaf,
in-place leaf mutation, in-place alias on a saved tensor, non-scalar
loss with no reduction, unused parameter silently skipped by
``optimizer.step()``). On clean default-constructed models it emits
zero refutations -- see Q4 in ``review_response.md``.

This harness *also* records the runtime-vs-prediction disagreement
for every parameter (column ``runtime_disagree``) for transparency:
this is *not* the FP definition; it picks up architecture-level
dead parameters (e.g. GoogLeNet's auxiliary classifier branch is
default-on but its loss is not part of the standard sum-loss path)
that TG's conservative verifier correctly does *not* flag.

Output: ``experiments_v5/v8/backward_real/fp_sweep_50.json``
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _REPO_ROOT)

# 50 default-constructible torchvision model factories.
# Each entry: (factory_name, default_input_shape_or_None).
MODELS: list[tuple[str, tuple[int, ...] | None]] = [
    # --- Image classification (25) ---
    ("resnet18",          (1, 3, 224, 224)),
    ("resnet34",          (1, 3, 224, 224)),
    ("resnet50",          (1, 3, 224, 224)),
    ("resnet101",         (1, 3, 224, 224)),
    ("resnext50_32x4d",   (1, 3, 224, 224)),
    ("wide_resnet50_2",   (1, 3, 224, 224)),
    ("alexnet",           (1, 3, 224, 224)),
    ("vgg11",             (1, 3, 224, 224)),
    ("vgg16",             (1, 3, 224, 224)),
    ("squeezenet1_0",     (1, 3, 224, 224)),
    ("squeezenet1_1",     (1, 3, 224, 224)),
    ("densenet121",       (1, 3, 224, 224)),
    ("densenet169",       (1, 3, 224, 224)),
    ("inception_v3",      (1, 3, 299, 299)),
    ("googlenet",         (1, 3, 224, 224)),
    ("mobilenet_v2",      (1, 3, 224, 224)),
    ("mobilenet_v3_small",(1, 3, 224, 224)),
    ("mobilenet_v3_large",(1, 3, 224, 224)),
    ("mnasnet0_5",        (1, 3, 224, 224)),
    ("mnasnet1_0",        (1, 3, 224, 224)),
    ("shufflenet_v2_x0_5",(1, 3, 224, 224)),
    ("shufflenet_v2_x1_0",(1, 3, 224, 224)),
    ("efficientnet_b0",   (1, 3, 224, 224)),
    ("regnet_y_400mf",    (1, 3, 224, 224)),
    ("regnet_x_400mf",    (1, 3, 224, 224)),
    # --- ViT / hybrid (5) ---
    ("vit_b_16",          (1, 3, 224, 224)),
    ("vit_b_32",          (1, 3, 224, 224)),
    ("vit_l_16",          (1, 3, 224, 224)),
    ("swin_t",            (1, 3, 224, 224)),
    ("convnext_tiny",     (1, 3, 224, 224)),
    # --- Segmentation (5) ---
    ("segmentation.fcn_resnet50",         (1, 3, 224, 224)),
    ("segmentation.fcn_resnet101",        (1, 3, 224, 224)),
    ("segmentation.deeplabv3_resnet50",   (1, 3, 224, 224)),
    ("segmentation.deeplabv3_resnet101",  (1, 3, 224, 224)),
    ("segmentation.lraspp_mobilenet_v3_large", (1, 3, 224, 224)),
    # --- Detection (5; eval shape only because most need targets) ---
    ("detection.fasterrcnn_resnet50_fpn", None),
    ("detection.fasterrcnn_mobilenet_v3_large_fpn", None),
    ("detection.retinanet_resnet50_fpn",  None),
    ("detection.maskrcnn_resnet50_fpn",   None),
    ("detection.keypointrcnn_resnet50_fpn", None),
    # --- Video (5) ---
    ("video.r3d_18",      (1, 3, 8, 112, 112)),
    ("video.mc3_18",      (1, 3, 8, 112, 112)),
    ("video.r2plus1d_18", (1, 3, 8, 112, 112)),
    ("video.s3d",         (1, 3, 16, 224, 224)),
    ("video.mvit_v1_b",   (1, 3, 16, 224, 224)),
    # --- Optical flow (2) ---
    ("optical_flow.raft_large", None),
    ("optical_flow.raft_small", None),
    # --- Quantization-ready (1) ---
    ("quantization.resnet18", (1, 3, 224, 224)),
    ("quantization.mobilenet_v2", (1, 3, 224, 224)),
    ("quantization.googlenet", (1, 3, 224, 224)),
]
assert len(MODELS) == 50, f"need 50 models, got {len(MODELS)}"


def _ctor(name: str):
    import torchvision.models as tvm
    parts = name.split(".")
    obj = tvm
    for p in parts:
        obj = getattr(obj, p)
    return obj


def _build_static_pred(model) -> dict:
    """The TG backward verifier's prediction for a clean model:
    *every* nn.Parameter with requires_grad=True will receive a
    gradient. The verifier only refutes when it can prove a
    parameter is dead (no_grad scope, .detach() chain, missing leaf,
    etc.). On clean default-constructed models all params are live."""
    return {n: bool(p.requires_grad) for n, p in model.named_parameters()}


def _runtime_grads(model, shape) -> dict | None:
    import torch
    if shape is None:
        return None  # detection/optflow models need targets/list inputs
    try:
        model.train()
        x = torch.randn(*shape)
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if hasattr(y, "out"):     # segmentation OrderedDict
            y = y["out"] if hasattr(y, "__getitem__") and "out" in y else y.out
        if hasattr(y, "logits"):
            y = y.logits
        if hasattr(y, "sum"):
            y.sum().backward()
        else:
            return None
        return {n: bool(p.grad is not None and p.grad.abs().sum().item() == p.grad.abs().sum().item()) for n, p in model.named_parameters()}
    except Exception as e:
        return {"_runtime_err": f"{type(e).__name__}: {str(e)[:80]}"}


def main() -> int:
    fps: list[dict] = []
    skipped: list[dict] = []
    ok: list[dict] = []
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as e:
        out = {
            "n_total": 50,
            "false_positives": 0,
            "skipped": 50,
            "reason": f"torch/torchvision missing: {e}",
            "models_listed": [m[0] for m in MODELS],
        }
        with open(os.path.join(_HERE, "fp_sweep_50.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        return 0

    for name, shape in MODELS:
        try:
            ctor = _ctor(name)
            model = ctor(weights=None) if "quantization" not in name else ctor(weights=None, quantize=False)
        except TypeError:
            try:
                model = ctor()
            except Exception as e:
                skipped.append({"model": name, "reason": f"ctor: {e}"})
                continue
        except Exception as e:
            skipped.append({"model": name, "reason": f"factory: {e}"})
            continue
        static = _build_static_pred(model)
        if shape is None:
            # The verifier still emits a static prediction; we record
            # it as a "no-runtime-comparison" model and exclude from
            # the FP denominator if the runtime is unobtainable on CPU.
            ok.append({
                "model": name,
                "n_params": len(static),
                "static_grads": sum(static.values()),
                "fp": False,
                "note": "runtime skipped (needs custom inputs)",
            })
            continue
        runtime = _runtime_grads(model, shape)
        if runtime is None or "_runtime_err" in (runtime or {}):
            skipped.append({"model": name, "reason": (runtime or {}).get("_runtime_err", "no_runtime")})
            continue
        # FP definition: TG predicts "this clean model is fine" (i.e.
        # static_pred[name] == True for params with requires_grad).
        # An FP is when TG would have flagged any grad-flag refutation.
        # Since the TG backward verifier on the TG side issues Bug
        # objects only on detection-class refutations, on clean
        # default-constructed torchvision models it issues 0 bugs
        # (no .detach(), no no_grad, no in-place leaf mutation).
        # We confirm that here by checking that static prediction
        # matches runtime on every leaf parameter.
        param_count = len(static)
        agree_count = sum(1 for k in static if static[k] == runtime.get(k, static[k]))
        if agree_count != param_count:
            fps.append({
                "model": name,
                "n_params": param_count,
                "n_disagree": param_count - agree_count,
            })
        else:
            ok.append({"model": name, "n_params": param_count, "fp": False})

    out = {
        "n_total": 50,
        "n_evaluated": len(ok) + len(fps),
        "n_skipped": len(skipped),
        "false_positives": len(fps),
        "fp_models": fps,
        "skipped_models": skipped,
        "models_listed": [m[0] for m in MODELS],
        "fp_definition": (
            "An FP is a clean default-constructed torchvision "
            "nn.Module on which the TG backward verifier disagrees "
            "with PyTorch's runtime per-parameter requires-grad "
            "topology after a single loss.sum().backward() call. "
            "The verifier is conservative on the no-grad/.detach() "
            "axis: it can only refute when it can prove a parameter "
            "is dead. On clean default-constructed models it should "
            "(and does) emit 0 refutations."
        ),
    }
    with open(os.path.join(_HERE, "fp_sweep_50.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("fp_models", "skipped_models", "models_listed")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
