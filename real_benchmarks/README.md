# TensorGuard Frozen Ground-Truth Benchmark Corpus

A small, **frozen, versioned** corpus of real PyTorch `nn.Module` architectures,
each labeled `clean` (should verify **SAFE**) or `buggy` (should verify
**UNSAFE**). It exists so TensorGuard's accuracy can be measured against a stable
ground truth that **cannot silently drift**: every repro file is content-addressed
by SHA-256 in [`manifest.json`](manifest.json), and [`load.py`](load.py)
re-verifies those hashes before use.

Current version: **1.0.0** — 16 models (8 clean / 8 buggy).

## Layout

| Path | Purpose |
| --- | --- |
| `corpus_def.py` | Single source of truth: every model's source, input shapes, label, and provenance. |
| `build_manifest.py` | Materializes `clean/*.py` + `buggy/*.py` and (re)generates `manifest.json` with per-file hashes. |
| `load.py` | Loads the manifest, **verifies every file hash**, and runs TensorGuard to check verdicts. |
| `manifest.json` | The frozen manifest (labels, provenance, expected verdicts, SHA-256 hashes). |
| `clean/`, `buggy/` | Generated standalone repro files (each has a provenance header + `INPUT_SHAPES`). |
| `VERSION` | Corpus version string. |

## Ground truth, proven against real code

The labels are not TensorGuard's opinion — they are grounded in real PyTorch
behavior:

* **6 of 8 buggy models raise a real `RuntimeError`** when executed in eager
  PyTorch (shape/channel/view/cat/matmul mismatches), e.g.
  `mat1 and mat2 shapes cannot be multiplied (32x256 and 128x10)`.
* **`buggy_device_mismatch`** only raises on a CUDA-enabled host
  (`Expected all tensors to be on the same device`); TensorGuard catches it
  statically with no GPU.
* **`buggy_gradient_detach`** is a **silent** bug — it raises *no* exception, so
  runtime testing misses it entirely; TensorGuard flags the severed gradient
  path statically. This is the headline case for static verification.

Buggy models drawn from tracked `pytorch/pytorch` issues carry the issue URL in
their `source_url`; the two canonical real-world failure modes (device mismatch,
gradient detach) are labeled `provenance_type: canonical_pattern`.

## Usage

```bash
# Verify corpus integrity + that every TensorGuard verdict matches its label:
python -m real_benchmarks.load

# Regenerate the repro files and manifest from corpus_def.py:
python -m real_benchmarks.build_manifest
```

## Freeze policy

The corpus is frozen. To change a model, add one, or relabel:

1. Edit `corpus_def.py`.
2. Bump `CORPUS_VERSION`.
3. Run `python -m real_benchmarks.build_manifest` to re-freeze hashes.
4. `tests/test_real_benchmarks.py` enforces hash-sync, determinism, and that
   every verdict still matches its frozen label.
