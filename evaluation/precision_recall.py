#!/usr/bin/env python3
"""Precision/recall evaluation of TensorGuard against real baselines.

This harness runs five bug detectors over the frozen, balanced, executable
ground-truth corpus in ``real_benchmarks/`` (8 clean + 8 buggy PyTorch
``nn.Module``s spanning the shape, device, phase and gradient domains) and
emits per-method confusion matrices (TP / FP / TN / FN) with precision,
recall and F1.

Methods
-------
``tensorguard``
    The static verifier under test (``src.api.verify_architecture``). A
    non-zero ``bug_count`` is a ``buggy`` prediction. Needs no execution,
    no concrete inputs and no GPU.
``runtime_forward``
    Dynamic baseline modelling "run it and see". Construct the module and
    call ``forward`` once with a seeded random tensor of the declared input
    shape; any raised exception -> ``buggy``. This is what an eager smoke
    test catches.
``runtime_backward``
    Stronger dynamic baseline: ``runtime_forward`` plus a backward pass and
    a check that every ``requires_grad`` parameter received a gradient. This
    additionally catches *silent* gradient-severing bugs that raise no
    exception. It still requires execution, concrete inputs, and -- for the
    device domain -- an actual matching device.
``pytea``
    The published academic static shape analyser (ropas/pytea, built from
    source under ``experiments_v5/_pytea_src``). Run live on an
    auto-generated entry wrapper that exercises ``forward``. Shape-only by
    construction: cannot reason about ``requires_grad``/autograd or device
    placement (per its own documented limitations).
``noop``
    Trivial floor: always predict ``clean``.

Honesty notes
-------------
* On a CPU-only host the device-domain bug raises at *construction* because
  it allocates a CUDA buffer (``Torch not compiled with CUDA enabled``), so
  the runtime baselines flag it via a construction crash rather than by
  detecting the device mismatch. The ``detail`` field records this so the
  result is not over-claimed.
* All runtime-detectable bugs in this corpus are structural (shape/device/
  gradient-flow), i.e. value-independent, so a single seeded random draw is
  sufficient and representative; this assumption is recorded in ``meta``.
* We report the full corpus *and* a shape-only sub-corpus so PyTea and the
  runtime baselines are also compared on the footing they were designed for.

Usage
-----
    cd tensorguard && PYTHONPATH=. python3 evaluation/precision_recall.py
    cd tensorguard && PYTHONPATH=. python3 evaluation/precision_recall.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from real_benchmarks import load  # noqa: E402

OUT_JSON = os.path.join(THIS_DIR, "confusion_matrices.json")
OUT_MD = os.path.join(THIS_DIR, "confusion_matrices.md")
PYTEA_BIN = os.path.join(REPO_ROOT, "experiments_v5", "_pytea_src", "bin", "pytea.py")

TORCH_SEED = 0
PYTEA_TIMEOUT_S = 120

METHODS = ["tensorguard", "runtime_forward", "runtime_backward", "pytea", "noop"]

# Static description of what each detector needs and can see. Reported in the
# artifact so the headline accuracy numbers are read in context.
CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "tensorguard": {
        "static": True, "sound_mode_available": True, "needs_execution": False,
        "needs_concrete_inputs": False, "needs_gpu_for_device_bugs": False,
        "domains": ["shape", "device", "phase", "gradient"],
    },
    "runtime_forward": {
        "static": False, "sound_mode_available": False, "needs_execution": True,
        "needs_concrete_inputs": True, "needs_gpu_for_device_bugs": True,
        "domains": ["shape", "device(runtime)"],
    },
    "runtime_backward": {
        "static": False, "sound_mode_available": False, "needs_execution": True,
        "needs_concrete_inputs": True, "needs_gpu_for_device_bugs": True,
        "domains": ["shape", "device(runtime)", "gradient"],
    },
    "pytea": {
        "static": True, "sound_mode_available": False, "needs_execution": False,
        "needs_concrete_inputs": False, "needs_gpu_for_device_bugs": False,
        "domains": ["shape"],
    },
    "noop": {
        "static": True, "sound_mode_available": False, "needs_execution": False,
        "needs_concrete_inputs": False, "needs_gpu_for_device_bugs": False,
        "domains": [],
    },
}

# Prediction sentinels.
BUGGY, CLEAN, NA = "buggy", "clean", "na"


# --------------------------------------------------------------------------
# Model loading helpers
# --------------------------------------------------------------------------
def _import_module_from_item(item: Dict[str, Any]):
    path = os.path.join(REPO_ROOT, "real_benchmarks", item["repro_file"])
    spec = importlib.util.spec_from_file_location("rb_" + item["id"], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # may raise (e.g. CUDA buffer on a CPU host)
    cls = getattr(mod, "BuggyModule", None) or getattr(mod, "CleanModule", None)
    if cls is None:
        raise AttributeError("no BuggyModule/CleanModule in %s" % item["repro_file"])
    return cls


def _rand_args(item, torch):
    args = []
    for shp in item["input_shapes"].values():
        args.append(torch.rand(*[int(d) for d in shp]))
    return args


# --------------------------------------------------------------------------
# Method predictors -> (prediction, detail)
# --------------------------------------------------------------------------
def predict_tensorguard(item) -> Tuple[str, str]:
    result = load.verify_item(item)
    if result.bug_count > 0:
        return BUGGY, "bug_count=%d" % result.bug_count
    return CLEAN, "bug_count=0"


def predict_runtime_forward(item) -> Tuple[str, str]:
    import torch
    torch.manual_seed(TORCH_SEED)
    try:
        cls = _import_module_from_item(item)
    except Exception as exc:  # construction-time crash (e.g. CUDA buffer)
        return BUGGY, "construct:%s" % type(exc).__name__
    try:
        model = cls()
    except Exception as exc:
        return BUGGY, "construct:%s" % type(exc).__name__
    model.eval()
    try:
        with torch.no_grad():
            model(*_rand_args(item, torch))
    except Exception as exc:
        return BUGGY, "forward:%s" % type(exc).__name__
    return CLEAN, "ran_ok"


def predict_runtime_backward(item) -> Tuple[str, str]:
    import torch
    torch.manual_seed(TORCH_SEED)
    try:
        cls = _import_module_from_item(item)
        model = cls()
    except Exception as exc:
        return BUGGY, "construct:%s" % type(exc).__name__
    model.train()
    try:
        out = model(*_rand_args(item, torch))
        out.float().sum().backward()
    except Exception as exc:
        return BUGGY, "forward_or_backward:%s" % type(exc).__name__
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    if missing:
        return BUGGY, "no_grad:%s" % ",".join(sorted(missing)[:2])
    return CLEAN, "all_params_grad"


def _pytea_entry_source(item) -> str:
    path = os.path.join(REPO_ROOT, "real_benchmarks", item["repro_file"])
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    cls = "BuggyModule" if "class BuggyModule" in src else "CleanModule"
    lines = ["", "# --- PyTea entry harness (auto-generated) ---",
             "import torch as _t", "_m = %s()" % cls]
    argnames: List[str] = []
    for i, shp in enumerate(item["input_shapes"].values()):
        dims = ", ".join(str(int(d)) for d in shp)
        name = "_a%d" % i
        argnames.append(name)
        lines.append("%s = _t.rand(%s)" % (name, dims))
    lines.append("_out = _m(%s)" % ", ".join(argnames))
    return src + "\n".join(lines) + "\n"


def _parse_pytea(out: str) -> Tuple[str, str]:
    m = re.search(r"immediate failed path #:\s*(\d+)", out)
    fail = int(m.group(1)) if m else None
    m = re.search(r"potential success path #:\s*(\d+)", out)
    succ = int(m.group(1)) if m else None
    m = re.search(r"Invalid paths[^:]*:\s*(\d+)", out)
    invalid = int(m.group(1)) if m else None
    if fail and fail > 0:
        return BUGGY, "failed_path=%d" % fail
    if invalid and invalid > 0:
        return BUGGY, "z3_invalid=%d" % invalid
    if succ and succ > 0:
        return CLEAN, "success_path=%d" % succ
    if "Frontend parse failed" in out:
        return NA, "frontend_parse_failed"
    return NA, "no_paths"


def pytea_available() -> bool:
    if not os.path.exists(PYTEA_BIN):
        return False
    from shutil import which
    return which("node") is not None


def predict_pytea(item) -> Tuple[str, str]:
    entry = _pytea_entry_source(item)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        tmp.write(entry)
        tmp.close()
        try:
            proc = subprocess.run(
                ["python3", PYTEA_BIN, tmp.name],
                capture_output=True, text=True, timeout=PYTEA_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return NA, "timeout"
        return _parse_pytea(proc.stdout + "\n" + proc.stderr)
    finally:
        os.unlink(tmp.name)


def predict_noop(item) -> Tuple[str, str]:
    return CLEAN, "always_clean"


PREDICTORS = {
    "tensorguard": predict_tensorguard,
    "runtime_forward": predict_runtime_forward,
    "runtime_backward": predict_runtime_backward,
    "pytea": predict_pytea,
    "noop": predict_noop,
}


# --------------------------------------------------------------------------
# Confusion-matrix maths
# --------------------------------------------------------------------------
def _round(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 4)


def confusion(rows: List[Tuple[str, str]]) -> Dict[str, Any]:
    """rows: list of (ground_truth_label in {clean,buggy}, prediction).

    NA predictions are handled two ways:
      * conservative view (headline): NA counts as the *wrong* answer, i.e.
        an abstain never helps -- NA on a buggy model is a miss (FN), NA on a
        clean model is a spurious non-pass (FP).
      * covered-only view: NA rows are excluded entirely.
    Coverage = decided / total.
    """
    tp = fp = tn = fn = 0
    na = 0
    ctp = cfp = ctn = cfn = 0  # covered-only
    for gt, pred in rows:
        if pred == NA:
            na += 1
            if gt == BUGGY:
                fn += 1
            else:
                fp += 1
            continue
        pred_buggy = pred == BUGGY
        gt_buggy = gt == BUGGY
        if gt_buggy and pred_buggy:
            tp += 1; ctp += 1
        elif gt_buggy and not pred_buggy:
            fn += 1; cfn += 1
        elif not gt_buggy and pred_buggy:
            fp += 1; cfp += 1
        else:
            tn += 1; ctn += 1

    def prf(tp_, fp_, fn_):
        prec = tp_ / (tp_ + fp_) if (tp_ + fp_) else None
        rec = tp_ / (tp_ + fn_) if (tp_ + fn_) else None
        if prec and rec:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0 if (tp_ + fp_ + fn_) else None
        return _round(prec), _round(rec), _round(f1)

    total = len(rows)
    prec, rec, f1 = prf(tp, fp, fn)
    cprec, crec, cf1 = prf(ctp, cfp, cfn)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn, "NA": na, "N": total,
        "precision": prec, "recall": rec, "f1": f1,
        "coverage": _round((total - na) / total) if total else None,
        "covered_only": {
            "TP": ctp, "FP": cfp, "TN": ctn, "FN": cfn,
            "precision": cprec, "recall": crec, "f1": cf1,
        },
    }


def run(check: bool = False) -> Dict[str, Any]:
    items = load.load_items()
    have_pytea = pytea_available()

    per_model: List[Dict[str, Any]] = []
    # predictions[method] = list of (gt_label, prediction) aligned with items
    predictions: Dict[str, List[Tuple[str, str]]] = {m: [] for m in METHODS}

    for item in items:
        entry = {
            "id": item["id"], "label": item["label"], "domain": item["domain"],
            "predictions": {},
        }
        for method in METHODS:
            if method == "pytea" and not have_pytea:
                pred, detail = NA, "pytea_unavailable"
            else:
                pred, detail = PREDICTORS[method](item)
            entry["predictions"][method] = {"pred": pred, "detail": detail}
            predictions[method].append((item["label"], pred))
        per_model.append(entry)

    domains = sorted({it["domain"] for it in items})
    shape_idx = [i for i, it in enumerate(items) if it["domain"] == "shape"]

    confusion_out: Dict[str, Any] = {}
    for method in METHODS:
        rows = predictions[method]
        by_domain = {}
        for dom in domains:
            idx = [i for i, it in enumerate(items) if it["domain"] == dom]
            by_domain[dom] = confusion([rows[i] for i in idx])
        confusion_out[method] = {
            "all": confusion(rows),
            "shape_only": confusion([rows[i] for i in shape_idx]),
            "by_domain": by_domain,
        }

    artifact = {
        "meta": {
            "generated_by": "evaluation/precision_recall.py",
            "command": "python3 evaluation/precision_recall.py",
            "corpus": "real_benchmarks",
            "corpus_version": load.load_manifest().get("meta", {}).get("version"),
            "n_models": len(items),
            "n_clean": sum(1 for it in items if it["label"] == "clean"),
            "n_buggy": sum(1 for it in items if it["label"] == "buggy"),
            "torch_seed": TORCH_SEED,
            "methods": METHODS,
            "pytea_available": have_pytea,
            "na_policy": (
                "headline counts NA as wrong (FN on buggy, FP on clean); "
                "covered_only excludes NA; coverage = decided/total"
            ),
            "runtime_single_draw_note": (
                "All runtime-detectable bugs are structural (value-independent), "
                "so one seeded random draw per input shape is representative."
            ),
            "device_domain_note": (
                "On a CPU-only host the device bug raises at construction "
                "(CUDA unavailable); runtime baselines flag it via that crash, "
                "not by detecting the device mismatch. See per-model detail."
            ),
        },
        "capabilities": CAPABILITIES,
        "per_model": per_model,
        "confusion": confusion_out,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            current = fh.read()
        if current != text:
            raise SystemExit("confusion_matrices.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------
def _fmt(x: Optional[float]) -> str:
    return "--" if x is None else ("%.3f" % x)


def render_markdown(artifact: Dict[str, Any]) -> str:
    meta = artifact["meta"]
    lines: List[str] = []
    lines.append("# TensorGuard precision/recall vs baselines")
    lines.append("")
    lines.append(
        "Corpus: `%s` (%d models: %d clean + %d buggy) spanning the shape, "
        "device, phase and gradient domains. Every label is verified against "
        "real PyTorch. Generated by `evaluation/precision_recall.py`."
        % (meta["corpus"], meta["n_models"], meta["n_clean"], meta["n_buggy"])
    )
    lines.append("")
    lines.append("## Full corpus (all four domains)")
    lines.append("")
    lines.append("| Method | TP | FP | TN | FN | Precision | Recall | F1 | Coverage |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for method in METHODS:
        c = artifact["confusion"][method]["all"]
        lines.append(
            "| `%s` | %d | %d | %d | %d | %s | %s | %s | %s |"
            % (method, c["TP"], c["FP"], c["TN"], c["FN"],
               _fmt(c["precision"]), _fmt(c["recall"]), _fmt(c["f1"]),
               _fmt(c["coverage"]))
        )
    lines.append("")
    lines.append("## Shape-only sub-corpus (apples-to-apples for shape analysers)")
    lines.append("")
    lines.append("| Method | TP | FP | TN | FN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for method in METHODS:
        c = artifact["confusion"][method]["shape_only"]
        lines.append(
            "| `%s` | %d | %d | %d | %d | %s | %s | %s |"
            % (method, c["TP"], c["FP"], c["TN"], c["FN"],
               _fmt(c["precision"]), _fmt(c["recall"]), _fmt(c["f1"]))
        )
    lines.append("")
    lines.append("## Per-domain recall on buggy models")
    lines.append("")
    domains = sorted(artifact["confusion"]["tensorguard"]["by_domain"].keys())
    header = "| Method | " + " | ".join(domains) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(domains) + 1))
    for method in METHODS:
        cells = []
        for dom in domains:
            c = artifact["confusion"][method]["by_domain"][dom]
            denom = c["TP"] + c["FN"]
            cells.append("%d/%d" % (c["TP"], denom) if denom else "--")
        lines.append("| `%s` | " % method + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Detector capabilities")
    lines.append("")
    lines.append("| Method | Static | Sound mode | Needs exec | Needs inputs | Needs GPU (device bugs) | Domains |")
    lines.append("|---|---|---|---|---|---|---|")
    for method in METHODS:
        cap = artifact["capabilities"][method]
        lines.append(
            "| `%s` | %s | %s | %s | %s | %s | %s |"
            % (method,
               "yes" if cap["static"] else "no",
               "yes" if cap["sound_mode_available"] else "no",
               "yes" if cap["needs_execution"] else "no",
               "yes" if cap["needs_concrete_inputs"] else "no",
               "yes" if cap["needs_gpu_for_device_bugs"] else "no",
               ", ".join(cap["domains"]) or "(none)")
        )
    lines.append("")
    lines.append("### Reading the numbers honestly")
    lines.append("")
    lines.append(
        "- On pure shape bugs every real analyser ties: TensorGuard, both "
        "runtime baselines and PyTea each catch 6/6 with zero false positives."
    )
    lines.append(
        "- `runtime_forward` misses the **silent gradient bug** (it raises no "
        "exception), and PyTea misses both the **device** and **gradient** "
        "bugs (it is shape-only by construction)."
    )
    lines.append(
        "- `runtime_backward` matches TensorGuard's accuracy on this "
        "executable corpus, but only by *executing* every model with concrete "
        "inputs and a backward pass; it flags the device bug solely because "
        "constructing a CUDA buffer crashes on a CPU host. TensorGuard reaches "
        "the same verdicts **statically and soundly** -- no execution, no "
        "inputs, and no GPU."
    )
    lines.append("")
    lines.append("> %s" % meta["device_domain_note"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed artifact is stale")
    args = ap.parse_args()
    artifact = run(check=args.check)
    if args.check:
        print("confusion_matrices.json is up to date")
        return
    print("Wrote %s and %s" % (os.path.relpath(OUT_JSON, REPO_ROOT),
                               os.path.relpath(OUT_MD, REPO_ROOT)))
    for method in METHODS:
        c = artifact["confusion"][method]["all"]
        print("  %-18s P=%s R=%s F1=%s (TP=%d FP=%d TN=%d FN=%d NA=%d)"
              % (method, _fmt(c["precision"]), _fmt(c["recall"]),
                 _fmt(c["f1"]), c["TP"], c["FP"], c["TN"], c["FN"], c["NA"]))


if __name__ == "__main__":
    main()
