"""
build_block_corpus.py
=====================

Collects ≥150 standalone ``nn.Module`` blocks from torchvision, timm,
transformers, and (optionally) mamba_ssm.  For each block we record:

    {
      "id": str,                     # stable, sortable
      "library": str,                # torchvision / timm / transformers / mamba_ssm
      "library_version": str,        # __version__
      "library_sha": str,            # git rev parsed from site-packages or "unknown"
      "qualified_name": str,         # e.g. torchvision.models.resnet.BasicBlock
      "module_path": str,            # python module the class lives in
      "class_name": str,
      "source": str,                 # inspect.getsource(cls) (full class def)
      "source_sha256": str,
      "loc": int,
      "input_shapes": dict,          # heuristic, tagged with provenance
      "shape_provenance": str,       # "explicit" | "vision_default" | "transformer_default" | "ssm_default"
      "category": str,               # "vision_cnn" | "vision_vit" | "transformer" | "ssm" | "other"
    }

Output:  experiments_v5/v5_block_corpus.jsonl  (one JSON object per line)
         experiments_v5/v5_block_corpus_manifest.json  (versions + SHAs + counts)

Run:     python3.11 experiments_v5/build_block_corpus.py
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

ROOT = Path(__file__).resolve().parent
OUT_JSONL = ROOT / "v5_block_corpus.jsonl"
OUT_MANIFEST = ROOT / "v5_block_corpus_manifest.json"


# ---------------------------------------------------------------------------
# SHA pinning helpers
# ---------------------------------------------------------------------------
def _pkg_sha(pkg_dir: str, pkg_name: str) -> str:
    """Best-effort: read git rev only if pkg_dir is itself the toplevel of a
    git checkout whose remote URL contains *pkg_name*.  Otherwise we return
    a string that explicitly identifies the wheel install — we will NOT
    silently surface the SHA of an unrelated parent repo (e.g. homebrew
    holding /opt/homebrew/lib/... in a git repo), which would be misleading.
    """
    try:
        toplevel = subprocess.check_output(
            ["git", "-C", pkg_dir, "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        # Require pkg_dir == toplevel (i.e. this is THIS package's own git
        # checkout, not just installed inside someone else's git repo).
        if os.path.realpath(toplevel) != os.path.realpath(pkg_dir):
            return f"wheel-install (no upstream git checkout)"
        url = subprocess.check_output(
            ["git", "-C", pkg_dir, "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if pkg_name not in url.lower():
            return f"wheel-install (no upstream git checkout)"
        sha = subprocess.check_output(
            ["git", "-C", pkg_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"git:{sha}"
    except Exception:
        return "wheel-install (no upstream git checkout)"


def _pkg_info(pkg) -> Dict[str, str]:
    pkg_dir = os.path.dirname(pkg.__file__)
    version = getattr(pkg, "__version__", "unknown")
    sha = _pkg_sha(pkg_dir, pkg.__name__)
    # For wheel installs we anchor reproducibility on the *release tag* —
    # the version string uniquely identifies the upstream commit on PyPI.
    if sha.startswith("wheel-install"):
        sha = f"pypi:{pkg.__name__}=={version}"
    return {
        "version": version,
        "path": pkg_dir,
        "sha": sha,
    }


# ---------------------------------------------------------------------------
# Heuristic input shapes
# ---------------------------------------------------------------------------
VISION_CNN_DEFAULT = {"x": (1, 64, 32, 32)}
VISION_VIT_DEFAULT = {"x": (1, 196, 384)}
TRANSFORMER_DEFAULT = {"hidden_states": (1, 16, 64)}
SSM_DEFAULT = {"x": (1, 16, 64)}


def _infer_category(qualified_name: str) -> Tuple[str, Dict[str, tuple], str]:
    q = qualified_name.lower()
    if "vision_transformer" in q or "swin" in q or ".vit" in q or "beit" in q \
            or "deit" in q or "cait" in q or "xcit" in q or "convit" in q \
            or "twins" in q or "mvit" in q or "maxvit" in q or "focalnet" in q:
        return "vision_vit", VISION_VIT_DEFAULT, "vision_default"
    if "torchvision" in q or "timm.models" in q:
        return "vision_cnn", VISION_CNN_DEFAULT, "vision_default"
    if "mamba" in q:
        return "ssm", SSM_DEFAULT, "ssm_default"
    if "transformers.models" in q:
        return "transformer", TRANSFORMER_DEFAULT, "transformer_default"
    return "other", VISION_CNN_DEFAULT, "vision_default"


# ---------------------------------------------------------------------------
# Block extraction from a module
# ---------------------------------------------------------------------------
def _is_block(cls: type, mod_name: str) -> bool:
    """Heuristic: an nn.Module class defined in this exact module."""
    if not isinstance(cls, type):
        return False
    if not issubclass(cls, nn.Module):
        return False
    if cls is nn.Module:
        return False
    if getattr(cls, "__module__", "") != mod_name:
        return False
    if cls.__name__.startswith("_"):
        return False
    return True


def _extract_from_module(mod_name: str) -> List[Tuple[str, type]]:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        print(f"  [skip] {mod_name}: {type(e).__name__}: {e}", file=sys.stderr)
        return []
    out: List[Tuple[str, type]] = []
    for name in dir(mod):
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        if _is_block(obj, mod_name):
            out.append((name, obj))
    return out


# ---------------------------------------------------------------------------
# Module lists per library (curated; covers diverse architectures)
# ---------------------------------------------------------------------------
TORCHVISION_MODULES = [
    "torchvision.models.resnet",
    "torchvision.models.mobilenetv2",
    "torchvision.models.mobilenetv3",
    "torchvision.models.efficientnet",
    "torchvision.models.densenet",
    "torchvision.models.regnet",
    "torchvision.models.shufflenetv2",
    "torchvision.models.mnasnet",
    "torchvision.models.squeezenet",
    "torchvision.models.vision_transformer",
    "torchvision.models.swin_transformer",
    "torchvision.models.convnext",
    "torchvision.models.maxvit",
    "torchvision.models.video.resnet",
    "torchvision.models.segmentation.deeplabv3",
    "torchvision.models.segmentation.fcn",
    "torchvision.models.segmentation.lraspp",
    "torchvision.ops.misc",
    "torchvision.ops.feature_pyramid_network",
]

TIMM_MODULES = [
    "timm.models.resnet",
    "timm.models.vision_transformer",
    "timm.models.swin_transformer",
    "timm.models.swin_transformer_v2",
    "timm.models.convnext",
    "timm.models.efficientnet_blocks",
    "timm.models.mlp_mixer",
    "timm.models.beit",
    "timm.models.cait",
    "timm.models.coat",
    "timm.models.nfnet",
    "timm.models.regnet",
    "timm.models.xcit",
    "timm.models.maxxvit",
    "timm.models.focalnet",
    "timm.models.davit",
    "timm.models.mvitv2",
    "timm.models.crossvit",
    "timm.models.gcvit",
    "timm.models.deit",
    "timm.models.byobnet",
    "timm.layers.attention2d",
    "timm.layers.mlp",
]

TRANSFORMERS_MODULES = [
    "transformers.models.bert.modeling_bert",
    "transformers.models.gpt2.modeling_gpt2",
    "transformers.models.t5.modeling_t5",
    "transformers.models.llama.modeling_llama",
    "transformers.models.vit.modeling_vit",
    "transformers.models.whisper.modeling_whisper",
    "transformers.models.bart.modeling_bart",
    "transformers.models.distilbert.modeling_distilbert",
    "transformers.models.clip.modeling_clip",
    "transformers.models.mistral.modeling_mistral",
    "transformers.models.gemma.modeling_gemma",
    "transformers.models.falcon.modeling_falcon",
    "transformers.models.gpt_neox.modeling_gpt_neox",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.roberta.modeling_roberta",
    "transformers.models.albert.modeling_albert",
    "transformers.models.opt.modeling_opt",
    "transformers.models.bloom.modeling_bloom",
    "transformers.models.deberta.modeling_deberta",
    "transformers.models.electra.modeling_electra",
]

MAMBA_MODULES = [
    "mamba_ssm.modules.mamba_simple",
    "mamba_ssm.modules.mamba2",
    "mamba_ssm.modules.block",
]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _hash_source(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def collect(library: str, modules: List[str], pkg_info: Dict[str, str]
            ) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for mod_name in modules:
        for cls_name, cls in _extract_from_module(mod_name):
            qn = f"{mod_name}.{cls_name}"
            if qn in seen:
                continue
            seen.add(qn)
            try:
                src = inspect.getsource(cls)
            except (OSError, TypeError):
                continue
            cat, shapes, prov = _infer_category(qn)
            out.append({
                "id": f"{library}__{cls_name}__{_hash_source(qn)[:8]}",
                "library": library,
                "library_version": pkg_info["version"],
                "library_sha": pkg_info["sha"],
                "library_path": pkg_info["path"],
                "qualified_name": qn,
                "module_path": mod_name,
                "class_name": cls_name,
                "source": src,
                "source_sha256": _hash_source(src),
                "loc": len(src.splitlines()),
                "input_shapes": shapes,
                "shape_provenance": prov,
                "category": cat,
            })
    return out


def main():
    t0 = time.time()
    blocks: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "build_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "libraries": {},
        "module_lists": {},
        "counts": {},
    }

    for lib_name, mods in [
        ("torchvision", TORCHVISION_MODULES),
        ("timm", TIMM_MODULES),
        ("transformers", TRANSFORMERS_MODULES),
        ("mamba_ssm", MAMBA_MODULES),
    ]:
        try:
            pkg = importlib.import_module(lib_name)
        except Exception as e:
            print(f"[{lib_name}] not installed ({e}); skipping", file=sys.stderr)
            manifest["libraries"][lib_name] = {"status": f"skipped: {e!r}"}
            continue
        info = _pkg_info(pkg)
        info["status"] = "ok"
        manifest["libraries"][lib_name] = info
        manifest["module_lists"][lib_name] = mods
        b = collect(lib_name, mods, info)
        manifest["counts"][lib_name] = len(b)
        blocks.extend(b)
        print(f"[{lib_name}] {len(b)} blocks  v={info['version']}  sha={info['sha'][:8]}")

    manifest["counts"]["total"] = len(blocks)
    manifest["elapsed_s"] = round(time.time() - t0, 2)

    with OUT_JSONL.open("w") as f:
        for b in blocks:
            f.write(json.dumps(b) + "\n")
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(blocks)} blocks → {OUT_JSONL}")
    print(f"Wrote manifest        → {OUT_MANIFEST}")
    if len(blocks) < 150:
        print(f"WARNING: only {len(blocks)} blocks (<150).", file=sys.stderr)


if __name__ == "__main__":
    main()
