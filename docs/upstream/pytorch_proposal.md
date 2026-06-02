# Upstream proposal: an opt-in static verification hook for `torch.nn.Module`

**Status:** Draft RFC (reference implementation included)
**Reference implementation:** [`src/upstream_hook.py`](../../src/upstream_hook.py)
**Executable evidence:** [`reproducibility/upstream_hook_demo.py`](../../reproducibility/upstream_hook_demo.py)
→ [`reproducibility/upstream_hook_demo.md`](../../reproducibility/upstream_hook_demo.md)

## Summary

We propose that PyTorch expose an **opt-in, non-breaking** hook that lets any
`nn.Module` be *statically* verified for shape / device / dtype / phase /
gradient consistency before it executes. When a module opts in and is proven
safe, it runs exactly as today; when it is not, the user gets a precise
diagnostic at the module boundary instead of a deep `aten`-level runtime stack
trace (often only triggered on a specific input shape, device, or in `train()`
vs `eval()`). TensorGuard already implements the analysis; this proposal is
about the *integration surface*, not the solver.

## Motivation

Shape, device, and dtype mismatches are among the most common PyTorch runtime
errors, and they share three painful properties:

1. **They surface late.** A `Linear(5, 2)` fed a width-4 activation only fails
   when `forward` actually runs, deep inside the dispatcher, often after minutes
   of data loading or several training steps.
2. **They are input/-context-dependent.** A model can pass every CPU unit test
   and still fail on a different batch rank, on CUDA, or only in `eval()` (e.g.
   a `BatchNorm`/`Dropout` phase interaction), or only on the backward pass.
3. **The diagnostics are low-level.** The stack trace points at an `aten`
   kernel, not at the two `nn.Module` attributes whose feature dimensions
   disagree.

A static, *sound* checker eliminates whole classes of these errors before a
single kernel runs — but only if it is trivial to adopt. That adoption surface
is what PyTorch is uniquely positioned to provide.

## Design principles

- **Opt-in and non-breaking.** A module with no verifier attached behaves
  byte-for-byte as today. Nothing in the default path changes.
- **Zero changes to model code.** Verification works from the module's existing
  source / structure; no type annotations are required (though they can sharpen
  results).
- **Sound by abstention.** When the analysis cannot prove or refute a property
  (data-dependent control flow, opaque custom ops), it *abstains* (`UNKNOWN`)
  rather than emitting a false alarm. A single false positive in a default-on
  tool destroys trust, so abstention is a first-class verdict.
- **Boundary-level diagnostics.** Errors are reported against the offending
  `nn.Module` attributes, not the dispatcher.

## Proposed API surface

Three layers, from lowest to highest level. All three are implemented today in
[`src/upstream_hook.py`](../../src/upstream_hook.py) on top of stock
`register_forward_pre_hook`, demonstrating that **no core changes are required
for a prototype** — only for first-class ergonomics.

### 1. Instance verification

```python
result = torch.nn.utils.verify_module(model, input_shapes={"x": (2, 8)})
if result.verdict == "UNSAFE":
    print(result.bugs)
```

(Reference: `verify_nn_module`.) Extracts the module's structure and returns an
analysis result with a three-valued verdict (`SAFE` / `UNSAFE` / `UNKNOWN`).

### 2. Attached pre-hook

```python
handle = torch.nn.utils.attach_verifier(model, input_shapes={"x": (2, 8)})
model(x)            # verified once on first forward; precise error if unsafe
handle.remove()     # native-style handle, fully removable
```

(Reference: `attach_verifier`.) Registers a one-shot `forward_pre_hook`; on the
first forward the module is verified and a precise `ShapeVerificationError` is
raised *before* the crashing kernel runs. Transparent for modules proven safe.

### 3. Declarative decorator

```python
@torch.nn.verifiable(input_shapes={"x": (2, 8)})
class Net(nn.Module):
    ...
```

(Reference: `verifiable`.) Attaches the verifier at construction time for
modules that always want to be checked.

## Soundness contract & abstention semantics

The verdict is three-valued and abstention-aware (see
[`SOUNDNESS_CONTRACT.md`](../../SOUNDNESS_CONTRACT.md) and
[`VERIFIABLE_FRAGMENT.md`](../../VERIFIABLE_FRAGMENT.md)):

- `SAFE` — proven free of the checked classes of error for the given input
  contract.
- `UNSAFE` — a concrete counterexample (shape/device/dtype/phase/gradient)
  exists; reported with the offending location.
- `UNKNOWN` — the module (or a sub-region) lies outside the statically
  verifiable fragment; the tool abstains rather than guess.

This maps cleanly onto a default-on policy: ship with `raise_on_unsafe=True` but
*never* raise on `UNKNOWN`, so opt-in users are never blocked by abstention.

## Relationship to `torch.export` / Dynamo

`torch.export` already traces modules under symbolic shapes and *fails* on
data-dependent constructs (`GuardOnDataDependentSymNode`). That overlap is
exactly the boundary of the verifiable fragment (see
[`reproducibility/quant_export_safety.md`](../../reproducibility/quant_export_safety.md)):
the constructs that break export are the ones this checker abstains on. The two
are complementary — export gives a traced graph for the in-fragment subset,
while this hook gives an *ahead-of-execution soundness verdict* with
boundary-level diagnostics, including for properties (device, phase, gradient
flow) that export does not check.

## Backward compatibility & rollout

- **Phase 1 (today):** ship as a `torch.nn.utils` helper built on public hooks
  — exactly the reference implementation here. Zero risk.
- **Phase 2:** first-class `torch.nn.verifiable` decorator and a `Module.verify`
  method for ergonomics.
- **Phase 3 (optional):** an environment/`torch.set_*` flag to auto-attach the
  verifier in debug builds, off by default.

No phase changes default execution semantics for un-opted-in modules.

## Evidence

[`reproducibility/upstream_hook_demo.py`](../../reproducibility/upstream_hook_demo.py)
runs the reference hook against **real PyTorch** and records, byte-deterministically:

- a buggy chained-`Linear` module: real `forward` raises a `RuntimeError`, and
  the attached hook rejects it *before* execution with a one-line diagnostic;
- a clean module: verified `SAFE`, the hook is transparent, and the real
  forward runs and returns the expected shape;
- the `@verifiable` decorator: accepts the clean module and rejects the buggy
  one at construction-driven first forward.

Static rejection holds **iff** the real runtime fails, on every case.
