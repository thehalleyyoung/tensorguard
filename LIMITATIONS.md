# What TensorGuard can't do yet

TensorGuard is **sound**: when it says *verified safe*, it has proven the
absence of the bug classes it checks for the inputs you described. The price of
soundness is honesty about scope. This page is the honest map of what falls
*outside* the verifiable fragment today — and what happens when your model uses
one of those constructs.

> This page is kept in sync with the implementation by
> `tests/test_docs_limitations.py`: every category below is checked against
> `src/verifiable_fragment.UNSUPPORTED_CATEGORY_INFO`, so the list cannot drift.

## What happens at the boundary

TensorGuard never *guesses*. When a module uses a construct outside the
verifiable fragment, the verifier **abstains** on the affected region rather
than emitting a possibly-wrong verdict:

- In `--soundness-mode sound`, an out-of-fragment construct yields **UNKNOWN**
  (exit code 2) instead of a green check, so a CI gate fails closed.
- In the default `balanced` mode, TensorGuard verifies the parts of the model it
  *can* reason about and reports the opaque region as not-verified, never as a
  false "safe".

In other words: the constructs below are reasons a verdict may be **UNKNOWN**,
never reasons a real bug is silently mislabeled "safe".

## Constructs outside the verifiable fragment

These are detected by static AST analysis and/or `torch.fx` tracing and cause
TensorGuard to abstain on the affected code.

| Category | What it means |
| --- | --- |
| Data-dependent control flow | Branch (if/while) whose condition depends on a tensor value. |
| Data-dependent iteration | Loop whose trip count depends on runtime data (e.g. range(int(x.item()))). |
| Dynamic assertion | assert statement in forward (may reference tensor values). |
| Tensor-to-scalar conversion | .item() / .tolist() / .numpy() converts a tensor to a Python value. |
| Custom autograd function | Custom torch.autograd.Function subclass with opaque shape behaviour. |
| In-place mutation | In-place mutation that torch.fx cannot trace soundly. |
| TorchScript | torch.jit.script / scripted submodule opaque to torch.fx. |
| Opaque external call | Call into an external/undefined symbol opaque to the tracer. |
| Dynamic module construction | Submodules constructed dynamically at forward time. |
| Unsupported builtin | Python builtin not modelled by the shape semantics. |
| Other trace failure | Any other torch.fx trace failure not otherwise classified. |

## Other current limitations

- **Ambiguous-rank, annotation-free models.** For layers that accept more than
  one input rank (e.g. a model that is only `nn.Linear`s), TensorGuard cannot
  soundly infer the input shape and will abstain unless you pass `-s`/
  `input_shapes`. Convolutional and norm layers pin the rank, so those infer
  automatically. This is a deliberate soundness choice, not a missed bug.
- **Value-level numerical correctness.** TensorGuard checks shapes, devices,
  dtypes, train/eval phase, and gradient flow — not whether your math is
  *correct*. A model can be "verified safe" and still compute the wrong thing.
- **Layers outside the supported set.** Layer types the shape semantics do not
  model yet are treated as opaque (abstain), never assumed shape-preserving.

## How to push the boundary

- Run with `--soundness-mode sound` in CI so any abstention fails the gate and
  you find out exactly where the fragment ends.
- Refactor data-dependent control flow out of `forward` (e.g. into config-time
  branching) so the hot path stays inside the fragment.
- Pass explicit `input_shapes` for ambiguous-rank models to recover full
  coverage.

See [VERIFIABLE_FRAGMENT.md](VERIFIABLE_FRAGMENT.md) for the formal fragment
definition and [SOUNDNESS_CONTRACT.md](SOUNDNESS_CONTRACT.md) for exactly what a
green verdict guarantees.
