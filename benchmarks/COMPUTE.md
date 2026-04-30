# Compute environment for TensorGuard benchmarks

All experiments reported in the paper run on commodity CPU. No GPU,
no accelerator, no distributed training.

## Hardware

- CPU: Apple M-series (arm64) / Intel x86_64 commodity laptop
- RAM: 16 GiB sufficient
- Disk: < 1 GiB scratch (cached source files)

## Software

- macOS 13+ or Linux x86_64
- Python 3.11 (tested with /opt/homebrew/bin/python3.11 = 3.11.15)
- Z3 (z3-solver Python package), torch >= 2.0, torchvision >= 0.16
- Lean 4 (toolchain pinned in lean/lean-toolchain); `lake build` < 10 s

## Wall-clock time per benchmark (single-thread)

| Benchmark                         | Driver                                | Time   |
|-----------------------------------|---------------------------------------|--------|
| Real-source torchvision (n=30)    | benchmarks/tv_realsource_benchmark.py | ~30 s  |
| Real-bug injection (n=24)         | benchmarks/injected_bugs.py           | ~3 min |
| Real-repo file-level (n=6)        | experiments/real_repo_eval.py         | ~30 s  |
| Curated 14-model smoke test       | verify_neurips_extended.py            | ~10 s  |
| Lean core (17 theorems, 0 sorry)  | cd lean && lake build                 | < 10 s |

## Determinism

All Python drivers set no randomness; tool subprocesses set
`torch.manual_seed(0)` defensively. Output JSONs are
byte-deterministic across runs on a given machine modulo
sub-millisecond timing fields.

## Reproducing

```sh
# 1. Install dependencies
pip install -r pyproject.toml  # or: pip install -e .

# 2. Run benchmarks
PYTHONPATH=. python3.11 benchmarks/tv_realsource_benchmark.py
PYTHONPATH=. python3.11 benchmarks/injected_bugs.py
PYTHONPATH=. python3.11 experiments/real_repo_eval.py
PYTHONPATH=. python3.11 verify_neurips_extended.py

# 3. Lean core
cd lean && lake build
```

Each driver writes a JSON file alongside the source file URL
and (for the injection benchmark) the exact source-line diff,
so a verifier can cross-check every reported number.
