"""Subprocess runner: import a buggy source string, instantiate a target
nn.Module class with given ctor kwargs, and run one of {fx, fakettensor}.

Reads JSON from stdin with keys:
  source: str
  target_class: str
  ctor_kwargs: dict (literal-typed values only)
  input_shape: list[int]
  input_dtype: "float" | "long"
  tool: "fx" | "fakettensor"
  module_name: str (a unique name for the temporary module)

Writes JSON to stdout with keys:
  verdict: "detected" | "missed" | "tool-error" | "import-failed"
  message: str
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
import importlib
import importlib.util
import os

import torch

torch.manual_seed(0)


SHAPE_KEYS = (
    "shape", "size mismatch", "mat1 and mat2", "size of tensor", "must match",
    "expected", "channels", "out_features", "in_features", "RuntimeError",
    "dimension", "dim ", "tensors must have the same",
)


def looks_like_shape_error(msg: str) -> bool:
    if not msg:
        return False
    low = msg.lower()
    return any(k.lower() in low for k in SHAPE_KEYS)


def main() -> None:
    spec = json.load(sys.stdin)
    source: str = spec["source"]
    target_class: str = spec["target_class"]
    ctor_kwargs: dict = spec["ctor_kwargs"]
    input_shape = tuple(spec["input_shape"])
    input_dtype: str = spec.get("input_dtype", "float")
    tool: str = spec["tool"]
    module_name: str = spec["module_name"]

    cache_dir = Path(__file__).parent / ".cache" / "_runner_tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    init = cache_dir / "__init__.py"
    if not init.exists():
        init.write_text("")
    mod_path = cache_dir / f"{module_name}.py"
    mod_path.write_text(source)

    sys.path.insert(0, str(cache_dir.parent))
    try:
        spec_obj = importlib.util.spec_from_file_location(
            f"_runner_tmp.{module_name}", str(mod_path)
        )
        mod = importlib.util.module_from_spec(spec_obj)  # type: ignore[arg-type]
        spec_obj.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as e:
        json.dump({
            "verdict": "import-failed",
            "message": f"{type(e).__name__}: {e}",
        }, sys.stdout)
        return

    try:
        cls = getattr(mod, target_class)
    except AttributeError as e:
        json.dump({"verdict": "import-failed", "message": str(e)}, sys.stdout)
        return

    # Resolve ctor kwargs that may reference "config" presets.
    resolved: dict = {}
    for k, v in ctor_kwargs.items():
        if isinstance(v, dict) and v.get("__kind__") in ("tiny_gpt_config", "hf_config"):
            class _Cfg:
                pass
            cfg = _Cfg()
            for ck, cv in v["fields"].items():
                setattr(cfg, ck, cv)
            resolved[k] = cfg
        else:
            resolved[k] = v

    try:
        model = cls(**resolved)
        model.eval()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        verdict = "detected" if looks_like_shape_error(msg) else "tool-error"
        json.dump({"verdict": verdict, "message": msg[:500]}, sys.stdout)
        return

    # Build input tensor
    if input_dtype == "long":
        x = torch.zeros(input_shape, dtype=torch.long)
    else:
        x = torch.randn(input_shape)

    try:
        if tool == "fx":
            from torch.fx import symbolic_trace
            from torch.fx.passes.shape_prop import ShapeProp
            try:
                gm = symbolic_trace(model)
            except Exception as e:
                json.dump({
                    "verdict": "tool-error",
                    "message": f"fx-trace: {type(e).__name__}: {e}"[:500],
                }, sys.stdout)
                return
            sp = ShapeProp(gm)
            try:
                sp.propagate(x)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                verdict = "detected" if looks_like_shape_error(msg) else "tool-error"
                json.dump({"verdict": verdict, "message": msg[:500]}, sys.stdout)
                return
            json.dump({"verdict": "missed", "message": "ShapeProp completed"}, sys.stdout)
            return

        elif tool == "fakettensor":
            from torch._subclasses.fake_tensor import FakeTensorMode
            mode = FakeTensorMode(allow_non_fake_inputs=True)
            try:
                with mode:
                    fake_x = torch.empty(input_shape, dtype=x.dtype)
                    _ = model(fake_x)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                verdict = "detected" if looks_like_shape_error(msg) else "tool-error"
                json.dump({"verdict": verdict, "message": msg[:500]}, sys.stdout)
                return
            json.dump({"verdict": "missed", "message": "FakeTensor forward completed"}, sys.stdout)
            return
        else:
            json.dump({"verdict": "tool-error", "message": f"unknown tool {tool}"}, sys.stdout)
            return
    except Exception as e:
        json.dump({
            "verdict": "tool-error",
            "message": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:400]}",
        }, sys.stdout)
        return


if __name__ == "__main__":
    main()
