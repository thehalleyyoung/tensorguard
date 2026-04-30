"""Leave-one-category-out (LOO) holdout for the 60-bug corpus.

For each of the 10 bug categories, disable all src/v5/*.py rules whose
module-name or operator-name contains the category label, then re-score
the full 60-bug corpus with TG.  The marginal contribution of each
category's rules is the drop in RP-count vs the full pipeline.

Output: experiments_v5/bug_corpus_loo.json
"""
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.api import verify_architecture  # noqa: E402

CATEGORIES = [
    "attention_dim",
    "broadcasting",
    "view_reshape_total_size",
    "conv_channel_mismatch",
    "linear_inout_mismatch",
    "einsum_dim",
    "transpose_axes",
    "batchnorm_features",
    "embedding_index",
    "other",
]

V5_DIR = os.path.join(ROOT, "src", "v5")
MANIFEST = os.path.join(ROOT, "experiments_v5", "bug_corpus_manifest.json")
OUT = os.path.join(ROOT, "experiments_v5", "bug_corpus_loo.json")


def _categorize_rule_module(modname: str) -> str:
    """Map an `src/v5/*.py` module name to a category label, if any."""
    n = modname.lower()
    if "attention" in n or "qkv" in n or "norm" in n:
        return "attention_dim"
    if "reshape" in n or "view" in n:
        return "view_reshape_total_size"
    if "backward" in n or "grad" in n:
        return "other"
    if "hybrid" in n or "localization" in n or "verdict" in n:
        return "other"
    if "symbolic" in n:
        return "other"
    return "other"


def _list_v5_modules() -> list:
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(V5_DIR)
        if f.endswith(".py") and f != "__init__.py"
    )


def _disable_v5_modules(disabled: list[str]) -> None:
    """Hide v5 modules by removing them from sys.modules and shadowing."""
    for m in list(sys.modules):
        if m.startswith("src.v5.") and m.split(".")[-1] in disabled:
            del sys.modules[m]
    # Remove from src.v5 package namespace
    try:
        import src.v5 as v5pkg  # noqa: F401
        for d in disabled:
            if hasattr(v5pkg, d):
                delattr(v5pkg, d)
    except Exception:
        pass


def _restore_v5_modules() -> None:
    import importlib
    import src.v5  # noqa: F401
    importlib.reload(sys.modules["src.v5"])


def _run_corpus() -> dict:
    with open(MANIFEST) as f:
        manifest = json.load(f)
    rp = 0
    silent = 0
    abst = 0
    err = 0
    per_cat = {c: {"rp": 0, "silent": 0, "abst": 0, "err": 0, "n": 0}
               for c in CATEGORIES}
    for item in manifest["items"]:
        cat = item.get("category", "other")
        if cat not in per_cat:
            cat = "other"
        per_cat[cat]["n"] += 1
        repro = item.get("repro_file")
        if not repro:
            err += 1
            per_cat[cat]["err"] += 1
            continue
        path = os.path.join(ROOT, repro)
        if not os.path.exists(path):
            err += 1
            per_cat[cat]["err"] += 1
            continue
        with open(path) as fh:
            src = fh.read()
        try:
            r = verify_architecture(src)
            max_conf = max((b.confidence for b in r.bugs), default=0.0)
            if max_conf >= 0.99:
                rp += 1
                per_cat[cat]["rp"] += 1
            elif r.abstained:
                abst += 1
                per_cat[cat]["abst"] += 1
            else:
                silent += 1
                per_cat[cat]["silent"] += 1
        except Exception:
            err += 1
            per_cat[cat]["err"] += 1
    return {"rp": rp, "silent": silent, "abst": abst, "err": err,
            "per_category": per_cat}


def main() -> None:
    t0 = time.time()
    modules = _list_v5_modules()
    full = _run_corpus()
    results = {
        "meta": {
            "v5_modules": modules,
            "n_categories": len(CATEGORIES),
            "elapsed_s_full": round(time.time() - t0, 2),
        },
        "full_pipeline": full,
        "leave_out": {},
    }
    for cat in CATEGORIES:
        disabled = [m for m in modules if _categorize_rule_module(m) == cat]
        if not disabled:
            results["leave_out"][cat] = {
                "disabled_modules": [],
                "skipped": True,
                "reason": "no v5 module attributable to this category",
            }
            continue
        t1 = time.time()
        _disable_v5_modules(disabled)
        try:
            r = _run_corpus()
        except Exception as e:
            r = {"error": str(e)}
        finally:
            _restore_v5_modules()
        results["leave_out"][cat] = {
            "disabled_modules": disabled,
            "elapsed_s": round(time.time() - t1, 2),
            **r,
        }
    rps = [v.get("rp", full["rp"]) for v in results["leave_out"].values()]
    rps = [r for r in rps if isinstance(r, int)]
    results["aggregate"] = {
        "full_rp_60": full["rp"],
        "loo_average_rp_60": round(sum(rps) / max(len(rps), 1), 2),
        "loo_min_rp_60": min(rps) if rps else None,
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT}")
    print(f"Full RP: {full['rp']}/60   silent: {full['silent']}   "
          f"abstain: {full['abst']}   err: {full['err']}")
    print(f"LOO average RP: {results['aggregate']['loo_average_rp_60']}/60")


if __name__ == "__main__":
    main()
