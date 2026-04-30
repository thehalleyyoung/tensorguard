# Track F — Benchmark v5: Summary

**Goal.** Build the largest static-shape-verifier evaluation in the literature
and run TensorGuard plus 5 baselines on it with calibrated honesty
(Verified / Refuted / Abstain / N/A reported per tool, per category).

## Files (absolute paths)

| Artifact | Path |
|---|---|
| Block-corpus builder            | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/build_block_corpus.py` |
| Bug-corpus builder              | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/build_bug_corpus.py` |
| TensorGuard v5 runner           | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/run_v5_benchmark.py` |
| Baseline-comparison runner      | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/run_baseline_comparison.py` |
| Block corpus (488 blocks)       | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/v5_block_corpus.jsonl` |
| Block-corpus manifest           | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/v5_block_corpus_manifest.json` |
| Bug corpus (60 bugs)            | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/v5_bug_corpus.jsonl` |
| Bug repros                      | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/bug_repros/` (60 files) |
| Bug-corpus protocol doc         | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/bug_corpus_protocol.md` |
| TensorGuard v5 results          | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/v5_benchmark_results.json` |
| Baseline-comparison results     | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/v5_baseline_comparison.json` |
| LaTeX section F                 | `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/docs/paper/sections_v5/F_benchmark.tex` |

## Headline numbers

### Block corpus (N = 488 standalone nn.Module subclasses)

| Tool | Verified | Refuted | Abstain | N/A |
|---|---:|---:|---:|---:|
| **TensorGuard (baseline)**       |  57 | 206 | 225 |   0 |
| torch.fx.symbolic_trace          |   0 |   0 |   7 | 481 |
| torch._subclasses.FakeTensorMode |   0 |   0 |   7 | 481 |
| torch.export                     |   0 |   3 |   4 | 481 |
| mypy + jaxtyping                 |   0 |   0 |   0 | 488 |
| beartype                         |   0 |   0 |   0 | 488 |

Provenance: torchvision 0.24.1 → 72 blocks · timm 1.0.26 → 127 · transformers 4.57.3 → 289. mamba_ssm: **skipped (not installable on this CPU-only machine)**.

### Bug corpus (N = 60 historical PyTorch shape bugs)

| Tool | Verified (silent miss!) | **Refuted (correct)** | Abstain | N/A |
|---|---:|---:|---:|---:|
| **TensorGuard (baseline)**       | 4 | **56 (93.3%)** | 0 | 0 |
| torch.fx.symbolic_trace          | 0 |   0 |  7 | 53 |
| torch._subclasses.FakeTensorMode | 0 |   0 |  7 | 53 |
| torch.export                     | 2 |   4 |  1 | 53 |
| mypy + jaxtyping                 | 0 |   0 |  0 | 60 |
| beartype                         | 0 |   0 |  0 | 60 |

The 4 TensorGuard silent misses are tagged in `v5_benchmark_results.json` with `"calibration_note": "VERIFIED_BUT_GROUND_TRUTH_BUGGY (silent miss)"`.

### TensorGuard top-10 abstain reasons on block corpus

| Reason | Count |
|---|---:|
| opaque_submodule (Unsupported layer kind UNKNOWN) | 100 |
| opaque_config_attr (`self.config.x`)              |  40 |
| tuple_returning_forward                           |  27 |
| unclassified                                      |  18 |
| qkv_unpack                                        |  12 |
| registered_buffer_or_parameter                    |   9 |
| external_library_call (sdpa, flash_attn, einsum)  |   7 |
| data_dependent_control_flow                       |   5 |
| dynamic_reshape (`x.view(-1, ...)`)               |   5 |
| python_loop_over_layers (`for l in self.layers`)  |   2 |

## Limitations (calibrated honesty)

1. **mamba_ssm skipped** — no pre-built CPU wheel; honestly excluded with `status: "skipped: ..."` in the manifest. Would add ~12–20 SSM blocks if installed.
2. **Track-C v5 modules not loaded** — `HAS_V5_EXT=False` in `meta`; we ran TensorGuard's baseline analyzer (`verify_architecture` from `src/api.py`). Track-C extensions would address most of the abstain categories above.
3. **Constructor synthesis** — baselines only attempt `cls()`; transformer blocks needing a `config` object are correctly reported `N/A` (this is what makes them N/A=481 on the block corpus). A future iteration could synthesise stub configs.
4. **TensorGuard refutes 206 unmodified library blocks** — these are the false-positive surface (not true bugs in production code). This is reported, not hidden, and motivates the v5 (Track-C) precision improvements.
5. **SHA pinning** — wheel installs are not git checkouts, so we record `pypi:<pkg>==<ver>` (which uniquely identifies the source archive). For all three libraries the function correctly detected this rather than reporting an unrelated parent-repo SHA.
6. **Bug corpus distribution** is skewed toward short-repro categories (view/reshape, broadcasting); see `bug_corpus_protocol.md` § "limitations".

## Reproduction

```bash
cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
python3.11 experiments_v5/build_block_corpus.py    # ~5 s
python3.11 experiments_v5/build_bug_corpus.py      # documents the prebuilt corpus
python3.11 experiments_v5/run_v5_benchmark.py      # ~25 s
python3.11 experiments_v5/run_baseline_comparison.py  # ~10 min (mostly mypy)
```
