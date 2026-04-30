# Track-C Coverage Summary (v5)
_Generated 2026-04-28T14:46:14Z_
## Targets
- torchvision: 30
- injected bugs: 24
- hf blocks: 25
## Verdict counts
- BEFORE (existing analyzer): V=49  R=30  A=0  (total=79)
- AFTER  (with v5 imports):   V=49  R=30  A=0  (total=79)
- Δ: VERIFIED +0, REFUTED +0, ABSTAIN +0

> v5 imports did NOT change verdict counts on these targets — the new transfer rules register additional ops but the existing analyzer's coarse VERIFIED/ABSTAIN classification is dominated by parse-level recognition; reporting as-is per calibrated honesty.

## Evidence files (absolute paths)
- `v5_symbolic_config` → `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/src/v5/symbolic_config.py`
- `v5_qkv_unpacking` → `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/src/v5/qkv_unpacking.py`
- `v5_reshape_neg1` → `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/src/v5/reshape_neg1.py`
- `v5_attention_norms` → `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/src/v5/attention_norms.py`
- `tests_dir` → `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/tests/v5`
- `this_script` → `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/run_track_C_coverage.py`
- coverage JSON → `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/track_C_coverage.json`
