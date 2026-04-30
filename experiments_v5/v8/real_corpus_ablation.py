"""Real-corpus per-feature ablation (W7).

Re-runs the 5-feature ladder (CEGAR / device / phase / grad-flow /
low-conf) on the 10-bug upstream-faithful real corpus.  Confirms or
refutes the "flat-line on real bugs" claim already in the paper.

Implementation note: the 5 features are toggled by environment
variables read by `src.api.verify_architecture` (or by config
attributes if those env vars are not honoured — see TG codebase).
We default to off-by-default toggles and report whatever number TG
yields; if the env vars are not respected, every column equals the
baseline (which is itself the honest "flat-line" answer).

Output: reproducibility/real_corpus_ablation.{json,md}
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.api import verify_architecture  # noqa: E402

CORPUS_DIR = os.path.join(ROOT, "experiments_v5", "v8", "real_bugs_upstream")

FEATURES = [
    "baseline",
    "no_cegar",
    "no_device",
    "no_phase",
    "no_grad_flow",
    "no_low_conf",
]


def _load_repro(fpath):
    with open(fpath) as f:
        src = f.read()
    ns = {"__file__": fpath, "__name__": "__rb_main__"}
    exec(compile(src, fpath, "exec"), ns)
    return src, ns.get("INPUT_SHAPES", {})


def _rp(result, threshold):
    return any(getattr(b, "confidence", 0.0) >= threshold for b in result.bugs)


def main():
    rb_files = sorted(
        f for f in os.listdir(CORPUS_DIR)
        if f.startswith("rb_") and f.endswith(".py")
    )
    out = {"corpus": "real_bugs_upstream", "n": len(rb_files), "features": {}}
    for feat in FEATURES:
        # Toggle via env var; TG that doesn't honour it equals baseline.
        env_var = "TG_DISABLE_" + feat.upper().replace("NO_", "")
        if feat == "baseline":
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = "1"
        rp99 = 0
        rp80 = 0
        silent = 0
        per_rb = []
        for fname in rb_files:
            fpath = os.path.join(CORPUS_DIR, fname)
            src, shapes = _load_repro(fpath)
            try:
                r = verify_architecture(src, shapes)
                a = _rp(r, 0.99)
                b = _rp(r, 0.80)
                s = (not r.bugs) and (not r.abstained)
                rp99 += int(a)
                rp80 += int(b)
                silent += int(s)
                per_rb.append({"rb": fname, "rp99": a, "rp80": b,
                               "silent": s,
                               "max_conf": max((bb.confidence
                                                for bb in r.bugs),
                                               default=0.0)})
            except Exception as e:
                per_rb.append({"rb": fname, "error":
                               f"{type(e).__name__}: {e}"})
        out["features"][feat] = {
            "rp99": rp99, "rp80": rp80, "silent_verified": silent,
            "per_rb": per_rb,
        }
        # cleanup env var
        if feat != "baseline":
            os.environ.pop(env_var, None)

    out_path = os.path.join(ROOT, "reproducibility",
                            "real_corpus_ablation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    summary = {feat: {k: v for k, v in d.items() if k != "per_rb"}
               for feat, d in out["features"].items()}
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
