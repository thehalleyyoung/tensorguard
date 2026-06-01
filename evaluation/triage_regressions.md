# Step 18 -- disagreement triage + minimal-reproducer regression suite

## Disagreement triage

Combined over the Step 15 clean fuzz population and the Step 16 injected-fault population:

| | Count |
|---|---|
| Clean models examined | 200 |
| Faulty models examined | 281 |
| Population total | 481 |
| False positives | 0 |
| False negatives | 0 |
| **Total disagreements** | **0** |

No TensorGuard/runtime disagreement was found, so there is nothing to fix; the regression suite instead freezes minimal bug reproducers.

## Regression suite (50 minimal reproducers + clean siblings)

Each buggy entry is verified to raise at runtime *and* be refuted by TensorGuard; each clean sibling is verified to run clean *and* be accepted. Replayed as parametrized tests by `tests/test_triage.py`.

| Fault category | Reproducers |
|---|---|
| `add_broadcast` | 6 |
| `cat_noncat_dim` | 5 |
| `conv_in` | 6 |
| `conv_kernel` | 6 |
| `flatten_linear` | 3 |
| `invalid_reshape` | 6 |
| `invalid_view` | 6 |
| `linear_in` | 6 |
| `matmul_inner` | 6 |
