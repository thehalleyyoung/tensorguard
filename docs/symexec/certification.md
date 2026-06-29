# Safety certificates — certified *absence* of forced-failure shape bugs

Every other proof artifact in the symbolic-execution engine answers *"why is this
a bug?"*. A **safety certificate** answers the dual, far stronger question:

> on the covered fragment of a program, is it *proven* that **no** modeled
> forced-failure shape bug is reachable?

This is only credible because the engine is **sound** (every report is real,
machine-checked in Lean) *and* **relatively complete** (`completeness_contract`:
on the covered fragment, a genuine forced failure on known operands *would* have
been reported). Together, on that fragment, **absence of a report is a guarantee
of absence of the failure** — and a safety certificate makes that guarantee
transferable and checkable offline.

## What it contains

A `SafetyCertificate` (see `src/symexec/safety_certificate.py`) records:

- **`proven_safe`** — true iff no *sound* bug was reported (a sound bug is any
  report whose severity is not `warning`; heuristic / intent suspicions never
  bear on the verdict).
- one **obligation per `COMPLETE_FOR` kind**, each carrying its runtime
  precondition, its completeness condition, the count of reports for that kind,
  and the Lean **`…​.refute`** theorem (axiom-clean) that proves the soundness ⇐
  direction for that kind.
- the **covered fragment** (statement / value coverage) and the **abstain
  boundary** — exactly where the guarantee stops.
- a **determinism receipt**: the source SHA-256 and the analysis fingerprint.
- the **trusted axiom base** (`propext`, `Classical.choice`, `Quot.sound`; no
  `sorryAx`).

## Replayability

`verify_safety_certificate(cert, source)` trusts nothing in the certificate but
its claims. It re-hashes the source, re-runs the deterministic engine, and
confirms (a) the source matches, (b) the reproducibility fingerprint matches,
(c) no sound forced-failure bug exists, and (d) every obligation is discharged
with the precondition + Lean refutation the *current* contract assigns its kind.
A matching fingerprint plus an empty sound-bug set *is* the proof of absence.

The trust chain is test-enforced: every `COMPLETE_FOR` kind is a key in
`LEAN_REFUTATION_FOR`, and every theorem it cites appears in the Lean soundness
audit (`tests/test_lean_soundness.py::_AUDITED_THEOREMS`), so a certificate can
never name an unproven or unaudited theorem.

## CLI

```sh
# Gate CI: exit non-zero if any file cannot be certified safe.
python -m src.symexec.certify check  model.py layers.py

# Emit a replayable JSON certificate.
python -m src.symexec.certify emit   model.py -o model.cert

# Re-verify a certificate against a file, offline.
python -m src.symexec.certify verify model.py model.cert
```

## Python API

```python
from src.symexec import certify_file, verify_certificate_file

cert = certify_file("model.py")
if cert.proven_safe:
    assert verify_certificate_file(cert, "model.py").verified
```

`SymResult.safety_certificate(source, filename=...)` builds a certificate from an
already-analysed result; `render_safety_certificate(cert)` renders deterministic
Markdown; `dumps_safety_certificate` / `loads_safety_certificate` serialize it.

## The weights layer (data + code↔data contract)

Shape-safe *code* is only half of a transformer's safety story; the other half is
the **weights** it loads. `src/symexec/weights.py` certifies a **safetensors**
checkpoint torch-free, by reading its self-describing JSON header:

* **Data well-formedness** — every tensor names a known dtype and a non-negative
  shape, and the storage offsets tile the data buffer exactly (sorted, contiguous,
  non-overlapping, each span `prod(shape)·dtype_size`). A file passing this is
  exactly one a conforming safetensors loader accepts; malformed/truncated/corrupt
  checkpoints are caught.
* **Data finiteness** — every float tensor is scanned for `NaN`/`Inf` directly
  from its IEEE-754 all-ones-exponent bit pattern (no torch, no execution).
* **Code↔data contract** — against an expected `name → (dtype, shape)` contract
  (what `load_state_dict(strict=True)` enforces) it reports missing keys,
  unexpected keys, dtype and shape mismatches. A contract can be lifted from a
  *reference* checkpoint with `weights_contract_from_file`, so "this retrain
  matches the known-good architecture" is itself certifiable.

A `WeightsSafetyCertificate` is `proven_safe` iff it has no findings, and is
replayable: `verify_weights_certificate(cert, path)` re-hashes the file, re-reads
the structure, and reproduces the verdict (file SHA-256 + structural fingerprint
are the determinism receipt). A *safe* contract certificate self-verifies — its
expected contract is exactly its own tensor list.

```sh
# Certify a checkpoint's weights layer (exit non-zero if unsafe).
python -m src.symexec.certify weights model.safetensors

# Also require it to match a known-good architecture, and emit the certificate.
python -m src.symexec.certify weights model.safetensors \
    --expected reference.safetensors -o model.wcert

# Re-verify offline.
python -m src.symexec.certify weights-verify model.safetensors model.wcert
```

```python
from src.symexec import certify_weights_file, verify_weights_certificate

cert = certify_weights_file("model.safetensors")
if cert.proven_safe:
    assert verify_weights_certificate(cert, "model.safetensors").verified
```

### The code→data bridge: a contract derived from model code

A reference checkpoint is not always available. `src/symexec/model_contract.py`
derives the expected contract **from the model's own code**: given the source of
an `nn.Module` and a concrete construction, it drives the symbolic-execution
engine to run `__init__` (resolving every layer's integer hyper-parameters by
abstract interpretation) and reads off the resulting submodule tree, emitting each
standard layer's parameters named exactly as `state_dict` does (dotted paths,
`0/1/...` for `nn.Sequential`).

The derived contract is deliberately **sound and partial**:

* a parameter is emitted only when its existence and full shape are *forced* by
  the resolved hyper-parameters and statically-known flags (`bias=`,
  `elementwise_affine=`, `affine=`, `track_running_stats=`); otherwise the bridge
  **abstains** (recorded in `contract.abstained`) rather than guess — e.g. an
  `nn.ModuleList`/comprehension it cannot enumerate;
* it never claims to be exhaustive (it cannot see `register_buffer` / raw
  `nn.Parameter` / dynamically-built submodules), so it is checked with
  `contract_partial=True`: a *missing* derived tensor or a *shape mismatch* is a
  genuine `load_state_dict(strict=True)` failure, but an *extra* checkpoint tensor
  is never flagged. Dtypes are not constrained (a model fixes them at load time).

```sh
# Certify a checkpoint against the model that will load it — no reference needed.
python -m src.symexec.certify weights model.safetensors \
    --model gpt.py --construct "GPT(n_layer=12, n_embd=768)"
```

```python
from src.symexec import derive_model_contract, certify_weights_against_model

contract = derive_model_contract(open("gpt.py").read(), "GPT(n_layer=12, n_embd=768)")
cert, contract = certify_weights_against_model(
    "model.safetensors", open("gpt.py").read(), "GPT(n_layer=12, n_embd=768)"
)
assert cert.proven_safe  # every layer the engine resolved is present with the right shape
```
