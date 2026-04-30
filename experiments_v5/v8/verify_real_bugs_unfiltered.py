"""Verify the unfiltered post-freeze corpus end-to-end on TG, Pytea, and FakeTensorMode.

This addresses the round-3 reviewer's borderline ask: report TG / Pytea / FakeTensor
verdict triple on a sample that is *not* filtered for "TG-handlable", with a
pre-registered query and chronological inclusion rule (see
``experiments_v5/v8/REAL_BUG_PREREG_QUERY.md`` and ``manifest.json`` next to this
script).

Run
---
    PYTHONPATH=. python3 experiments_v5/v8/verify_real_bugs_unfiltered.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.api import verify_architecture  # noqa: E402

HERE = os.path.dirname(__file__)
UF_DIR = os.path.join(HERE, "real_bugs_unfiltered")
PF_DIR = os.path.join(HERE, "real_bugs_postfreeze")
MANIFEST_PATH = os.path.join(UF_DIR, "manifest.json")
OUT = os.path.join(ROOT, "reproducibility", "real_bugs_unfiltered.json")


# ---- Tool stubs --------------------------------------------------------

def _faketensor_verdict(repro_path: str, input_shapes: dict) -> str:
    """Run FakeTensorMode on a self-contained repro.

    Returns one of {"verified", "refuted", "abstain", "n/a"}.
    """
    try:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: F401
    except Exception:
        return "n/a"
    src = open(repro_path).read()
    ns: dict = {}
    try:
        exec(compile(src, repro_path, "exec"), ns)
    except Exception:
        return "n/a"
    BuggyModule = ns.get("BuggyModule")
    if BuggyModule is None:
        return "n/a"
    try:
        mod = BuggyModule()
    except Exception:
        return "n/a"
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            inputs = {}
            for k, shape in input_shapes.items():
                inputs[k] = torch.empty(*shape)
            try:
                _ = mod(**inputs)
                return "verified"
            except RuntimeError as e:
                msg = str(e)
                if any(s in msg for s in ("shape", "size", "dim", "broadcast",
                                           "BFloat16", "scalar type", "matmul")):
                    return "refuted"
                return "abstain"
            except Exception:
                return "abstain"
    except Exception:
        return "n/a"


def _pytea_verdict(repro_path: str, input_shapes: dict) -> str:
    """Stub Pytea verdict.

    The Pytea baseline is offline (commit 2022-04-26, not pip-installable on
    every box).  We mark every entry as ``n/a`` here when Pytea is not
    available; in the cached output we hand-fill the entries from the existing
    Pytea harness output for those PRs that fall in Pytea's catalogue.

    Returns one of {"verified", "refuted", "abstain", "n/a"}.
    """
    return "n/a"


# Hand-filled from the existing Pytea baseline harness on the modern subset
# (operators in Pytea's 2022 catalogue).  Entries not in this dict default to
# "n/a" (Pytea cannot run on the surface).
PYTEA_CACHE = {
    # In-fragment-literal entries that Pytea's catalogue covers:
    "rb_pf_003": "refuted",   # 3D LoRA + matmul: Pytea catches via Linear-in-features
    "rb_pf_004": "verified",  # Pytea silent-verifies (softmax over wrong dim is a value bug)
    "rb_uf_007": "refuted",   # view + Linear: catalogue-covered
    "rb_uf_008": "n/a",       # 3D conv decoder: outside Pytea's nn.Conv3d coverage
    "rb_uf_013": "verified",  # broadcast multiply: Pytea silent-skips on attribute lookup
    "rb_uf_015": "refuted",   # transpose+view+Linear: catalogue-covered
}


# ---- Driver -----------------------------------------------------------

def _resolve(item) -> str:
    f = item["file"]
    if f.startswith("../real_bugs_postfreeze/"):
        return os.path.join(PF_DIR, os.path.basename(f))
    return os.path.join(UF_DIR, os.path.basename(f))


def _tg_status(repro_path: str, input_shapes: dict) -> dict:
    src = open(repro_path).read()
    try:
        result = verify_architecture(
            src,
            input_shapes=input_shapes,
            high_confidence_only=False,
            filename=os.path.basename(repro_path),
        )
        n_bugs = len(result.bugs)
        max_conf = max((b.confidence for b in result.bugs), default=0.0)
        verdicts = sorted({(b.category.value if hasattr(b.category, "value") else str(b.category)) for b in result.bugs}) or ["none"]
        if max_conf >= 0.99:
            status = "RP_0.99"
        elif max_conf >= 0.80:
            status = "RP_0.80"
        elif n_bugs == 0:
            status = "silent_verified"
        else:
            status = "low_confidence"
        return {"n_bugs": n_bugs, "max_conf": max_conf,
                "verdicts": verdicts, "status": status}
    except Exception as e:
        return {"n_bugs": 0, "max_conf": 0.0,
                "verdicts": ["abstain_load_err"],
                "status": "abstain", "err": str(e)[:200]}


def main() -> int:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    records = []
    counts = {
        "TG": {"RP_0.99": 0, "RP_0.80": 0, "silent_verified": 0,
                "abstain": 0, "low_confidence": 0},
        "FakeTensor": {"verified": 0, "refuted": 0, "abstain": 0, "n/a": 0},
        "Pytea":     {"verified": 0, "refuted": 0, "abstain": 0, "n/a": 0},
    }
    for item in manifest["items"]:
        repro = _resolve(item)
        # input_shapes is in the repro source; load via exec
        ns: dict = {}
        try:
            exec(compile(open(repro).read(), repro, "exec"), ns)
        except Exception as e:
            input_shapes = {}
        else:
            input_shapes = ns.get("INPUT_SHAPES", {})
        tg = _tg_status(repro, input_shapes)
        counts["TG"][tg["status"]] = counts["TG"].get(tg["status"], 0) + 1
        ft = _faketensor_verdict(repro, input_shapes)
        counts["FakeTensor"][ft] = counts["FakeTensor"].get(ft, 0) + 1
        pt = PYTEA_CACHE.get(item["id"], _pytea_verdict(repro, input_shapes))
        counts["Pytea"][pt] = counts["Pytea"].get(pt, 0) + 1
        rec = {**item, "tg": tg, "faketensor": ft, "pytea": pt}
        records.append(rec)
        print(f"{item['id']:10s}  TG={tg['status']:18s}  FT={ft:9s}  Pytea={pt}")
    summary = {
        "regime": "no_synthesised_assume_M (user-visible); pre-registered query",
        "preregistration": "experiments_v5/v8/REAL_BUG_PREREG_QUERY.md",
        "n_total": len(records),
        "counts": counts,
        "per_item": records,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(json.dumps(counts, indent=2))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
