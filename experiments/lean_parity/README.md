# Lean ↔ Python Parity Testing (Track G)

This directory contains the parity testing infrastructure for verifying that Lean transfer rules and Python implementations agree.

## Files

- `lean_rules_mirror.py`: Python mirrors of Lean shape transfer functions
- `run_parity.py`: Main test runner that generates random test cases and compares Lean vs Python

## Running

```bash
cd /path/to/tensorguard
python3.11 experiments/lean_parity/run_parity.py
```

This will run 1000 tests for each of 20+ operators and output results to `experiments/lean_parity_results.json`.
