# Track D — Backward shape & grad-flag verifier (v5)

## Files

- `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/src/v5/backward_shape.py`
- `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/src/v5/grad_flag_verifier.py`
- `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/tests/v5/test_backward_shape.py`
- `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/tests/v5/test_grad_flag.py`
- `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/track_D_run.py`
- `/Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/track_D_results.json`

No existing `src/` files were modified; all new code is under `src/v5/`.
The pre-existing `src/backward/` directory is empty.

## Theory

### Backward-shape soundness theorem (informal)
Given a verified forward graph `G`, if `verify_backward(G) = OK` then for
every node `v` with `φ_grad(v) = True`, the gradient produced by
`loss.backward()` has shape `primal(v).shape` modulo
`sum_to(grad, primal.shape)` reduction. Proof by reverse-topological
induction over `G`, using PyTorch's per-Function shape contract
(`grad_outputs[i].shape == y_i.shape`, `grad_inputs[j].shape == x_j.shape`
or `None`) as the per-step lemma. The verifier mirrors this induction
through the `SHAPE_RULES` table.

### Grad-flag soundness statement (informal)
Let `P` = parameters, `E` ⊆ `P` = expected-to-learn. Define `H ⊆ P` as
the set of params reverse-reachable from `loss` along edges that are
**not** marked `no_grad`, **not** behind `.detach()`, with
`requires_grad=True`. PyTorch sets `p.grad ≠ None` ⇔ `p ∈ H`. The
verifier returns OK iff `H = E` and additionally checks B3 (in-place on
leaf w/ grad) and B4 (no leaf has `requires_grad`).

## Numbers (from `track_D_results.json`)

| metric                                         | value       |
|------------------------------------------------|-------------|
| Property test: 500 random small models         | 500 / 500   |
| Static ↔ runtime agreement rate                | **1.0000**  |
| Real-bug case studies (BUG-A … BUG-H)          | **8 / 8**   |
| False positives on 50 clean models             | **0 / 50**  |
| Static unit tests (pytest)                     | **20 / 20** |
| Torch version                                  | 2.9.1       |

The 8 case studies correspond to canonical PyTorch silently-None /
wrong-shape grad bug patterns documented in issues such as #2769,
#4132, #7613, #20580, #39279, #56380, #69991, #82064 (no_grad scope,
.detach(), requires_grad=False, in-place-on-leaf, missing leaf,
in-place alias mutating a saved tensor, non-scalar loss, unused
parameter silently skipped by `optimizer.step()`).

## Reproduce

```
python3.11 -m pytest /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/tests/v5/ -q
python3.11 /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard/experiments_v5/track_D_run.py
```
