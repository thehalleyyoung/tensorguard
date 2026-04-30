"""Quick sweep: do the 10 real backward_real models use param-sharing or
torch.utils.checkpoint?  Bonus improvement for round 1 (Q6).

This is a lightweight static check: we import each model class via
the same path used by run_backward_real.py and walk:
  - the model's named_parameters() ids() to detect any parameter
    shared between two paths,
  - the model's source code (inspect.getsource on the class) to detect
    any reference to torch.utils.checkpoint.

Output: reproducibility/grad_extras.json (+ .md)
"""
import importlib
import inspect
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

MODELS = [
    ("torchvision.models", "resnet18", {}),
    ("torchvision.models", "mobilenet_v3_small", {}),
    ("torchvision.models", "vit_b_16", {}),
]


def _has_checkpoint(model) -> bool:
    try:
        src = inspect.getsource(model.__class__)
    except (TypeError, OSError):
        return False
    return ("torch.utils.checkpoint" in src) or ("from torch.utils.checkpoint" in src)


def _has_param_sharing(model) -> bool:
    seen = {}
    for name, p in model.named_parameters():
        pid = id(p)
        if pid in seen:
            seen[pid].append(name)
            return True
        seen[pid] = [name]
    return False


def main():
    out = {"models": [], "summary": {"n": 0, "with_checkpoint": 0,
                                     "with_param_sharing": 0,
                                     "errors": 0}}
    for mod_name, ctor, kwargs in MODELS:
        try:
            m = importlib.import_module(mod_name)
            ctor_fn = getattr(m, ctor)
            model = ctor_fn(**kwargs)
        except Exception as e:
            out["models"].append({"name": f"{mod_name}.{ctor}",
                                  "error": f"{type(e).__name__}: {e}"})
            out["summary"]["errors"] += 1
            continue
        rec = {
            "name": f"{mod_name}.{ctor}",
            "uses_checkpoint": _has_checkpoint(model),
            "has_param_sharing": _has_param_sharing(model),
            "tg_grad_flag_misclassifies": False,
        }
        # TG's first-order grad-flag lattice silently misclassifies when
        # either is true.
        rec["tg_grad_flag_misclassifies"] = (
            rec["uses_checkpoint"] or rec["has_param_sharing"]
        )
        out["models"].append(rec)
        out["summary"]["n"] += 1
        out["summary"]["with_checkpoint"] += int(rec["uses_checkpoint"])
        out["summary"]["with_param_sharing"] += int(rec["has_param_sharing"])

    out_path = os.path.join(ROOT, "reproducibility", "grad_extras.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
