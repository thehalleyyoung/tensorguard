# `preregistered_power_calc.json` --- power calculation for the pre-registered corpus

## Command
```
python3 reproducibility/preregistered_power_calc.py
```

## Inputs
- Observed proportions on the 15-module pre-registered corpus:
  TensorGuard 5/15 = 0.333, baselines 2/15 = 0.133.
- Two-sided two-proportion z-test (normal approximation), alpha = 0.05,
  target power = 0.80.

## Outputs
- `reproducibility/preregistered_power_calc.json`: required N per arm
  (69), total required N (138), shortfall vs current sample (54
  modules per arm).

## Paper claim
The pre-registered 5/15 vs 2/15 split is consistent with a true effect
of about 20 percentage points but the n=15 sample is far below the
n=69 per arm needed for 80% power against that effect. We report the
required N alongside the existing 5/15 result so the reader can see
the magnitude of the shortfall.
