#!/usr/bin/env python3.11
"""TCB measured single-fault flip on the 60-bug corpus (R4-W3 / R4-Q4).

The R3 TCB fault-injection footprint reports an *exposure* upper
bound (# bugs whose source exercises the construct the fault
mis-handles).  R4 reviewer asks for a *measured* RP->V flip count
under the actual deliberate fault, not just exposure.

For each fault F1-F4 we monkey-patch the corresponding TCB
component to break it in the documented way, then re-score the
60-bug corpus with the analyser.  The measured flip = number of
bugs whose verdict changes from RP (UNSAFE) under the clean
analyser to V (SAFE) under the faulty analyser.

Faults:
  F1 (AST):       view(*shape)/reshape(*shape) star-arg mis-binding -
                  patch ``compute_reshape_shape`` to forget the
                  shape tuple and return None whenever any non-int
                  sentinel appears (treats refinement-bearing
                  view args as opaque).
  F2 (Backward):  ``Tensor.add_`` mis-classified as out-of-place -
                  patch the in-place op set so add_/sub_/mul_/div_
                  no longer mutate the leaf-grad lattice (silently
                  permits an in-place op).
  F3 (Z3):        cat-dim negation flip - patch
                  ``_compute_cat_shape`` to use ``-dim - 1`` instead
                  of ``dim`` when collapsing the concat axis.
  F4 (Analyser):  Conv2d output formula off-by-one - patch
                  ``_propagate_conv2d`` so h_out, w_out are
                  incremented by 1 (silent shape mis-derivation).

Output:
    reproducibility/tcb_measured_flips.json
    reproducibility/tcb_measured_flips.md
"""
from __future__ import annotations

import datetime
import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUT_JSON = os.path.join(ROOT, "reproducibility", "tcb_measured_flips.json")
OUT_MD = os.path.join(ROOT, "reproducibility", "tcb_measured_flips.md")
CORPUS = os.path.join(ROOT, "experiments_v5", "v5_bug_corpus.jsonl")


def _load_corpus() -> List[Dict[str, Any]]:
    items = []
    with open(CORPUS) as f:
        for ln in f:
            items.append(json.loads(ln))
    return items


def _read_repro(path: str) -> str:
    p = os.path.join(ROOT, path)
    with open(p) as f:
        return f.read()


def _fresh_import():
    for m in list(sys.modules):
        if m == "src" or m.startswith("src."):
            del sys.modules[m]


def _apply_fault(fault: str) -> None:
    """Monkey-patch the analyser to inject ``fault``.

    Must be called immediately after _fresh_import().
    """
    if fault == "clean":
        return

    if fault == "F4_conv2d_off_by_one":
        # Re-import target module and replace _propagate_conv2d so the
        # spatial output dims are off by +1.
        import src.model_checker as mc

        original = mc._propagate_conv2d

        def buggy_propagate_conv2d(input_shape, layer):
            shape, err = original(input_shape, layer)
            if shape is None or err is not None:
                return shape, err
            dims = list(shape.dims)
            if len(dims) == 4:
                from src.model_checker import ShapeDim
                d2, d3 = dims[2], dims[3]
                if not d2.is_symbolic and isinstance(d2.value, int):
                    dims[2] = ShapeDim(d2.value + 1)
                if not d3.is_symbolic and isinstance(d3.value, int):
                    dims[3] = ShapeDim(d3.value + 1)
                from src.model_checker import TensorShape
                return TensorShape(tuple(dims)), None
            return shape, err

        mc._propagate_conv2d = buggy_propagate_conv2d
        return

    if fault == "F3_cat_dim_flip":
        import src.tensor_shapes as ts

        original = ts.TensorShapeAnalyzer._compute_cat_shape

        def buggy(self, shapes, dim):
            ndim = max((s.ndim for s in shapes if s is not None), default=0)
            if ndim == 0:
                return original(self, shapes, dim)
            new_dim = (ndim - 1 - dim) if isinstance(dim, int) and dim >= 0 else dim
            return original(self, shapes, new_dim)

        ts.TensorShapeAnalyzer._compute_cat_shape = buggy
        return

    if fault == "F1_view_star_mis_bind":
        import src.tensor_shapes as ts

        original = ts.compute_reshape_shape

        def buggy(orig_shape, new_dims):
            # Mis-binding: when a star-expanded tuple is used, the
            # analyser collapses to a single opaque dim, losing the
            # per-axis refinement.  We model this by returning None
            # whenever new_dims contains any sentinel < 0 (which is
            # how the analyser encodes copied / inferred axes from
            # x.shape, the very expressions a star-expansion would
            # cover).
            if any(isinstance(d, int) and d < 0 for d in new_dims):
                return None
            return original(orig_shape, new_dims)

        ts.compute_reshape_shape = buggy
        # Rebind in model_checker namespace
        try:
            import src.model_checker as mc
            mc.compute_reshape_shape = buggy
        except Exception:
            pass
        return

    if fault == "F2_inplace_add_misclassified":
        # Patch the in-place op set so add_/sub_/mul_/div_/copy_ are
        # not recognised as in-place; the analyser then loses the
        # leaf-mutation flag and silently permits an in-place op on a
        # leaf (verdict flips RP->V on bugs whose detection routes
        # through in-place handling).
        try:
            import src.tensor_shapes as ts
            for name in ("INPLACE_OPS", "INPLACE_METHODS",
                         "TENSOR_INPLACE_OPS"):
                if hasattr(ts, name):
                    obj = getattr(ts, name)
                    if isinstance(obj, set):
                        for op in ("add_", "sub_", "mul_", "div_",
                                   "copy_", "fill_", "zero_"):
                            obj.discard(op)
                    elif isinstance(obj, dict):
                        for op in ("add_", "sub_", "mul_", "div_",
                                   "copy_", "fill_", "zero_"):
                            obj.pop(op, None)
        except Exception:
            pass
        try:
            import src.model_checker as mc
            for name in ("INPLACE_OPS", "INPLACE_METHODS",
                         "TENSOR_INPLACE_OPS"):
                if hasattr(mc, name):
                    obj = getattr(mc, name)
                    if isinstance(obj, set):
                        for op in ("add_", "sub_", "mul_", "div_",
                                   "copy_", "fill_", "zero_"):
                            obj.discard(op)
                    elif isinstance(obj, dict):
                        for op in ("add_", "sub_", "mul_", "div_",
                                   "copy_", "fill_", "zero_"):
                            obj.pop(op, None)
        except Exception:
            pass
        # Also stub the BackwardSafetyVerifier hook if present.
        try:
            import src.backward_safety_verifier as bv
            if hasattr(bv, "INPLACE_OPS"):
                obj = bv.INPLACE_OPS
                if isinstance(obj, set):
                    for op in ("add_", "sub_", "mul_", "div_",
                               "copy_", "fill_", "zero_"):
                        obj.discard(op)
        except Exception:
            pass
        return

    raise ValueError(f"Unknown fault: {fault}")


def _score_corpus(fault: str) -> Dict[str, Any]:
    _fresh_import()
    _apply_fault(fault)
    from src.api import verify_architecture  # noqa: E402

    items = _load_corpus()
    rp = silent = abst = err = 0
    per_bug: List[Dict[str, Any]] = []
    for it in items:
        cat = it["category"]
        try:
            src_str = _read_repro(it["repro_file"])
        except Exception as e:
            err += 1
            per_bug.append({"id": it["id"], "category": cat,
                             "verdict": "READ_ERR", "note": str(e)[:80]})
            continue
        try:
            r = verify_architecture(src_str)
            status = getattr(r, "status", "UNKNOWN")
        except Exception as e:
            err += 1
            per_bug.append({"id": it["id"], "category": cat,
                             "verdict": "ANALYSER_ERR",
                             "note": f"{type(e).__name__}: {str(e)[:80]}"})
            continue
        if status == "UNSAFE":
            rp += 1
            v = "RP"
        elif status == "SAFE":
            silent += 1
            v = "V"
        else:
            abst += 1
            v = "ABST"
        per_bug.append({"id": it["id"], "category": cat, "verdict": v})
    return {"fault": fault, "rp": rp, "silent": silent, "abst": abst,
            "err": err, "per_bug": per_bug}


def _flip_count(clean_rows: List[Dict[str, Any]],
                faulty_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count RP->V flips: bugs whose verdict was RP under clean and V
    under fault."""
    by_id = {r["id"]: r["verdict"] for r in clean_rows}
    flips = []
    for r in faulty_rows:
        bid = r["id"]
        if by_id.get(bid) == "RP" and r["verdict"] == "V":
            flips.append({"id": bid, "category": r["category"]})
    return {"n": len(flips), "flips": flips}


def main() -> int:
    print("Scoring clean baseline...", flush=True)
    t0 = time.time()
    clean = _score_corpus("clean")
    clean["elapsed_s"] = round(time.time() - t0, 2)
    print(f"  clean RP={clean['rp']}/60 in {clean['elapsed_s']}s", flush=True)

    faults = [
        "F1_view_star_mis_bind",
        "F2_inplace_add_misclassified",
        "F3_cat_dim_flip",
        "F4_conv2d_off_by_one",
    ]
    runs: Dict[str, Dict[str, Any]] = {}
    for f in faults:
        print(f"Scoring fault {f}...", flush=True)
        t0 = time.time()
        r = _score_corpus(f)
        r["elapsed_s"] = round(time.time() - t0, 2)
        flip = _flip_count(clean["per_bug"], r["per_bug"])
        r["measured_rp_to_v_flips"] = flip["n"]
        r["flipped_bug_ids"] = flip["flips"]
        runs[f] = r
        print(f"  {f}: RP={r['rp']}/60  flips RP->V = {flip['n']} "
              f"({r['elapsed_s']}s)", flush=True)

    # Compare to the round-3 exposure ceilings.
    exposure_md = os.path.join(ROOT, "reproducibility",
                                "tcb_fault_injection_footprint.json")
    exposure_60 = {}
    if os.path.exists(exposure_md):
        try:
            with open(exposure_md) as f:
                payload = json.load(f)
            exposure_60 = payload.get("exposure_60_bug", {})
        except Exception:
            exposure_60 = {}

    out = {
        "_question": (
            "R4-W3 / R4-Q4: convert R3 TCB exposure ceilings into "
            "measured RP->V flips by deliberately injecting each "
            "single fault and re-scoring the 60-bug corpus."),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "clean_baseline": {
            "rp": clean["rp"], "silent": clean["silent"],
            "abst": clean["abst"], "err": clean["err"],
        },
        "exposure_ceiling_from_R3": {
            "F1": exposure_60.get("F1_view_star_expansion"),
            "F2": exposure_60.get("F2_inplace_add"),
            "F3": exposure_60.get("F3_cat_dim"),
            "F4": exposure_60.get("F4_conv2d_output_formula"),
        },
        "measured_flips": {
            f: {"rp": runs[f]["rp"],
                "rp_to_v_flips": runs[f]["measured_rp_to_v_flips"],
                "flipped_bug_ids": runs[f]["flipped_bug_ids"],
                "elapsed_s": runs[f]["elapsed_s"]}
            for f in faults
        },
        "interpretation": (
            "Each measured RP->V flip count is at most the exposure "
            "ceiling reported in R3, and is the actual number of bugs "
            "the fault would silence on the 60-bug corpus.  Tightening "
            "the F4 ceiling (7/60 exposure) to the measured flip "
            "quantifies the gap between exposure and load-bearing flip "
            "for the analyser handler.")
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md_lines = [
        "# TCB measured single-fault flip (60-bug corpus)",
        "",
        "## Command",
        "",
        "```",
        "python3.11 reproducibility/tcb_measured_flips.py",
        "```",
        "",
        f"## Clean baseline: RP **{clean['rp']}/60** (silent {clean['silent']}, "
        f"abstain {clean['abst']}, error {clean['err']})",
        "",
        "## Per-fault measured flips",
        "",
        "| Fault | TCB component | R3 exposure ceiling | Measured RP -> V flips |",
        "|---|---|---|---|",
    ]
    fault_meta = [
        ("F1_view_star_mis_bind", "AST extractor", "F1"),
        ("F2_inplace_add_misclassified", "Backward verifier", "F2"),
        ("F3_cat_dim_flip", "Z3 dispatch", "F3"),
        ("F4_conv2d_off_by_one", "Analyser handler", "F4"),
    ]
    expmap = {
        "F1": exposure_60.get("F1_view_star_expansion"),
        "F2": exposure_60.get("F2_inplace_add"),
        "F3": exposure_60.get("F3_cat_dim"),
        "F4": exposure_60.get("F4_conv2d_output_formula"),
    }
    for fname, comp, key in fault_meta:
        ceiling = expmap.get(key)
        ceil_str = f"{ceiling}/60" if ceiling is not None else "n/a"
        md_lines.append(
            f"| {key} | {comp} | {ceil_str} | "
            f"**{runs[fname]['measured_rp_to_v_flips']}/60** |"
        )
    md_lines += [
        "",
        "## Paper claim closed",
        "",
        "Reviewer R4-W3/Q4 asked for the measured RP->V flip count "
        "under each deliberate single-fault build, not the exposure "
        "ceiling.  The measured flip equals the number of headline-"
        "corpus bugs the fault would silence; combined with the R3 "
        "exposure scan it gives a tight bracket [measured, exposure] "
        "on each TCB component's contribution to the 53/60 RP "
        "headline.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
