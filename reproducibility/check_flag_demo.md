# `check_flag_demo.json` --- live secondary-check verdict flips

## Command
```
python3 experiments_v5/run_check_flag_demo.py
```

## Inputs
- Three committed real-source examples in `examples/check_flag_demo/`
  (`device_mismatch_residual.py`, `phase_dependent_head.py`,
  `grad_checkpoint_block.py`).  Each is a small, self-contained
  `nn.Module` adapted from a real PyTorch pattern.
- All eight combinations of the three secondary-check flags
  (`check_devices`, `check_phases`, `check_gradients`).

## Outputs
- `reproducibility/check_flag_demo.json`: per-example, per-combo verdict
  (`REFUTED` / `VERIFIED`), bug counts, first error message, and
  per-call latency.

## Numbers cited in the paper
- 3/3 real-source examples have the property that toggling the
  corresponding secondary-check flag flips the overall verdict between
  `REFUTED` and `VERIFIED`, holding the other two flags off.
- For each example, the verdict produced with the primary flag on
  matches the documented expectation in the example's docstring
  (`flag_flips_verdict = true`, `expectation_met = true`).

## Paper claims that cite this artifact
- The 5-theory product domain (Shape × Device × Phase × Stride ×
  Permutation) is shown to be live end-to-end on real source: each of
  the three secondary-check flags is exercised by at least one
  committed example whose verdict is sensitive to that flag.
- The CEGAR contribution C5 is no longer a stress-benchmark proxy:
  the device, phase and gradient checks each demonstrably move at
  least one verdict on un-instantiated class source.
