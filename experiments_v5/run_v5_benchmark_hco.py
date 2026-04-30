"""Run v5 benchmark in HIGH-confidence-only regime; this is the
configuration that the paper's headline 57/206/225 row corresponds to.

This script is a thin wrapper around run_v5_benchmark.py — it injects
``high_confidence_only=True`` into the verifier call so its output
matches the paper's headline numbers and the L0 row of
feature_ablation.json.  The default ``run_v5_benchmark.py`` uses
``high_confidence_only=False`` and therefore produces 50/213/225 (more
refutations from the heuristic post-pass).  Both regimes are valid;
only the high-confidence regime is referenced by the paper.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import experiments_v5.run_v5_benchmark as base
from src.api import verify_architecture as _real_va

def _hco(*a, **kw):
    kw["high_confidence_only"] = True
    return _real_va(*a, **kw)

base.verify_architecture = _hco
base.OUT_JSON = base.OUT_JSON.parent / "v5_benchmark_results_hco.json"

if __name__ == "__main__":
    base.main()
