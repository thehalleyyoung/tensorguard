# Threat model — TensorGuard weights-layer certifier

> Roadmap step **1** (`100_QUANTUM_LEAP_NEXT_STEPS.md`). This document is the
> normative specification the weights-layer tests are audited against. The
> machine-checked invariant (`tests/test_threat_model.py`) is: **every
> `WeightsFinding` kind emitted by `src/symexec/weights.py` and
> `src/symexec/model_contract.py` appears in the
> [Finding → runtime-failure table](#finding--runtime-failure-table) below, with a
> concrete runtime failure it rules out, and no table row is stale.**

## 1. Purpose & scope

The weights-layer certifier answers one question with a *proof*, not a heuristic:

> *If I hand these checkpoint bytes to the loader this model code will use, can the
> load — or the first forward/backward that touches the loaded tensors — fail for a
> reason that is decidable from the file's structure and the model's code?*

A `proven_safe` certificate (`certify_weights_file(...).proven_safe is True`) is a
claim that **no** modeled failure can occur. The modeled failures are exactly the
[`WeightsFinding` kinds](#finding--runtime-failure-table). Anything the certifier
cannot decide it **abstains** on (records, never silently assumes safe). This is the
zero-false-positive contract: a safe verdict is sound; an unsafe verdict carries a
witness; an undecidable input abstains.

In scope:

* **Data well-formedness** of a checkpoint container (today: safetensors; the
  roadmap adds GGUF/npz/pickle/ONNX). The bytes must describe a self-consistent set
  of tensors that a loader can actually deserialize.
* **Numerical hygiene** of float tensors (no NaN/Inf) — a *correctness* guarantee,
  not a load-time one.
* **Code ↔ data contract**: the checkpoint satisfies the `name → (shape[, dtype])`
  obligations of a `load_state_dict(strict=True)`, where the obligations come either
  from a *reference* checkpoint or are *derived from the model's own code*
  (`model_contract.py`).

Out of scope (explicit non-goals — see [§7](#7-non-goals--limitations)).

## 2. Actors

| Actor | Role | Trusts | Is trusted for |
| --- | --- | --- | --- |
| **Checkpoint author** | Produces the weight file (trainer, fine-tuner, converter, quantizer). | Their framework's serializer. | Nothing — the file is the artifact under test. |
| **Model author** | Writes the `nn.Module` code that will load the checkpoint. | The Python/PyTorch semantics. | The *contract* source when the model is the reference (`--model`). |
| **Loader** | The runtime that maps bytes → tensors → `state_dict` → module (`safetensors.load`, `torch.load`, `load_state_dict`). | The file's header + the model. | Faithful execution of the format/`load_state_dict` semantics we model. |
| **CI / release gate** | Runs the certifier to admit/deny an artifact. | The certifier's verdict + receipt. | Enforcing the exit-code policy. |
| **Verifier** | Independently replays a certificate (`verify_weights_certificate`). | The receipt (file SHA-256 + structural fingerprint). | Re-deriving the verdict from the file alone. |
| **Adversary** | Supplies a malicious or corrupt checkpoint (supply-chain). | — | Nothing; assumed hostile. |
| **TensorGuard certifier** | The analyzer (this code). | The standard library only (torch-free trust path). | Soundness: never emit `proven_safe` for a modeled failure. |

## 3. Trust boundaries (the pipeline)

Each arrow is a boundary where untrusted input is interpreted. A finding kind guards
a specific boundary; crossing a boundary is only sound once the prior one is.

```
                                ┌─ malformed_frame
file bytes ──(8-byte <Q len)──► header length prefix
   │                                │
   │                                ▼  (UTF-8 JSON parse)
   │                          header object ──► malformed_entry / unknown_dtype /
   │                                │            malformed_shape / malformed_offsets
   │                                ▼
   │                     typed tensor entries  ──► byte_length_mismatch
   │                                │
   ▼                                ▼  (offsets vs. data buffer)
data buffer [8+len, EOF) ──► storage layout ──► storage_out_of_bounds / storage_gap /
   │                                │            storage_overlap / storage_undercovered
   │                                ▼  (IEEE bit-class scan of float tensors)
   │                          tensor values   ──► non_finite_values
   │                                │
   ▼                                ▼  (name→shape[,dtype] vs. model obligations)
load_state_dict(strict) ──► code↔data contract ──► contract_missing_key /
                                                    contract_unexpected_key /
                                                    contract_shape_mismatch /
                                                    contract_dtype_mismatch
```

The **contract obligations** themselves come from `model_contract.py`, which derives
a *sound, partial, shape-only* `name → shape` map by symbolically executing the
model's `__init__` (see [§5](#5-the-codedata-contract--model_contractpy)). The four
`contract_*` findings are emitted by `weights.py::_check_contract`, whether the
obligations originate from a reference checkpoint or from model code.

## 4. Assets & security goals

* **Loadability** — the file can be deserialized and applied via
  `load_state_dict(strict=True)` without raising. Guarded by every non-`non_finite`
  finding.
* **No silent aliasing** — distinct named tensors do not share storage bytes.
  Guarded by `storage_overlap`.
* **Numerical integrity** — float weights are finite. Guarded by `non_finite_values`
  (a correctness, not a load, guarantee).
* **Determinism / non-repudiation** — the verdict is reproducible and bound to the
  exact file (file SHA-256) and reading (structural fingerprint).

## 5. The code↔data contract — `model_contract.py`

`derive_model_contract(source, construction)` drives the symbolic-execution engine to
run an `nn.Module.__init__`, resolving each standard layer's integer
hyper-parameters by abstract interpretation, and reads off the submodule tree to emit
the parameters PyTorch is *guaranteed* to register, named exactly as `state_dict`
(dotted paths; `0/1/…` for `nn.Sequential`).

Soundness mechanism — **abstention, not emission of new finding kinds.** The bridge
emits a `name → shape` obligation *only* when existence and full shape are forced by
resolved dims and statically-known flags (`bias=`, `elementwise_affine=`, `affine=`,
`track_running_stats=`). Otherwise it records an `abstained` entry `(path, reason)`
and emits nothing. Consequently `model_contract.py` itself raises **no** safety
verdict and contributes **no** `WeightsFinding` kinds of its own; its obligations are
checked through `weights.py::_check_contract(..., partial=True)`, which for a partial
contract asserts only *positive* obligations and never emits `contract_unexpected_key`
(a checkpoint may legitimately carry tensors the partial contract could not account
for). Dtypes are unconstrained by a code-derived contract (a model fixes parameter
dtype at load/`.to()` time), so `contract_dtype_mismatch` only fires for an explicit
non-`None` dtype obligation (e.g. a reference checkpoint).

This is why the machine-checked test scans **both** `weights.py` and
`model_contract.py` for `WeightsFinding(...)` constructions: today the union equals
the `weights.py` set, and the test will catch the day `model_contract.py` ever starts
emitting findings directly (forcing a doc update).

## 6. Finding → runtime-failure table

Each row maps a finding **kind** to the trust boundary it guards, what it detects, and
the concrete runtime failure a `proven_safe` certificate therefore rules out. The
"runtime failure" column names the actual exception/condition the loader would hit.

| kind | boundary | detects | runtime failure ruled out |
| --- | --- | --- | --- |
| `malformed_frame` | length prefix | missing/oversized 8-byte `<Q` header length, or a header that runs past EOF | `safetensors.SafetensorError` / `struct.error` on open — header cannot be read (`InvalidHeaderLength`/truncated file) |
| `malformed_entry` | header object | a tensor entry is not a JSON object (missing `dtype`/`shape`/`data_offsets` structure) | `safetensors.SafetensorError` deserializing the header (malformed metadata) → load aborts |
| `unknown_dtype` | header object | `dtype` is not a recognized safetensors dtype | `safetensors.SafetensorError` (`invalid dtype`) — loader cannot map the dtype to a tensor type |
| `malformed_shape` | header object | `shape` is not a list of non-negative ints | `safetensors.SafetensorError` / `OverflowError` building the tensor shape |
| `malformed_offsets` | header object | `data_offsets` is not an int `[begin, end]` with `0 <= begin <= end` | `safetensors.SafetensorError` (`invalid offset`) — slice bounds are nonsensical |
| `byte_length_mismatch` | tensor entries | declared byte span `end-begin` ≠ `numel(shape) * sizeof(dtype)` | `safetensors.SafetensorError` (`tensor invalid info` / size mismatch) — the slice cannot be reshaped to the declared tensor |
| `storage_out_of_bounds` | storage layout | a tensor's `end` offset exceeds the data buffer length | `safetensors.SafetensorError` (`offset out of bounds`) / `IndexError` — read past the buffer |
| `storage_gap` | storage layout | unreferenced bytes between consecutive tensors (offsets not contiguous) | `safetensors.SafetensorError` (`metadata incomplete buffer`) — safetensors requires a contiguous tiling |
| `storage_overlap` | storage layout | two tensors' byte ranges overlap | `safetensors.SafetensorError` (overlapping ranges) **or**, if loaded raw, silent storage **aliasing** — two params share memory and corrupt each other |
| `storage_undercovered` | storage layout | tiled tensors cover fewer bytes than the data buffer (trailing unaccounted bytes) | `safetensors.SafetensorError` (`metadata incomplete buffer`) — buffer not fully described |
| `non_finite_values` | tensor values | NaN/Inf bit patterns in a float tensor (all-ones IEEE exponent) | *no load error*, but a runtime **correctness** failure: NaN/Inf propagates → `loss = nan`, `RuntimeError` from anomaly/assert checks, or silently wrong outputs |
| `contract_missing_key` | code↔data contract | a tensor the model requires is absent from the checkpoint | `RuntimeError: Error(s) in loading state_dict: Missing key(s) in state_dict` (`strict=True`) |
| `contract_unexpected_key` | code↔data contract | the checkpoint carries a tensor the (full, non-partial) contract does not expect | `RuntimeError: Error(s) in loading state_dict: Unexpected key(s) in state_dict` (`strict=True`) |
| `contract_shape_mismatch` | code↔data contract | a tensor's shape differs from the model's parameter shape | `RuntimeError: ... size mismatch for <key>: copying a param with shape … from checkpoint, the shape in current model is …` |
| `contract_dtype_mismatch` | code↔data contract | a tensor's dtype differs from an explicit (non-`None`) dtype obligation | dtype/`RuntimeError` on a dtype-pinned tensor (e.g. integer index/`bool` buffer) — a silent unsafe cast or copy failure |

Notes on honesty of the mapping:

* `non_finite_values` is the **only** finding that does not prevent a load-time
  exception; it prevents a downstream numerical failure. The table says so
  explicitly. A `proven_safe` certificate's loadability guarantee comes from the
  other fourteen findings; its numerical guarantee comes from this one.
* `storage_overlap` prevents *either* a deserializer error *or* (under a permissive
  raw loader) silent aliasing — both are listed.
* `contract_unexpected_key` is suppressed for *partial* (code-derived) contracts by
  construction (§5); it only fires for full contracts (e.g. a reference checkpoint).

## 7. Non-goals & limitations

A `proven_safe` certificate makes **no** claim about any of the following; these are
deliberately out of the modeled-failure set (several are addressed by later roadmap
steps):

* **Semantic/behavioral correctness** — that the weights produce good predictions, or
  that the architecture matches intent beyond shapes.
* **Non-modeled formats** — anything other than safetensors today (GGUF/npz/pickle/
  ONNX are roadmap steps 26–30). An unsupported format **abstains**.
* **Code execution / malicious pickle** — safetensors carries no code, so it is not a
  concern for this format; the pickle/`.bin` path (roadmap 28/83) adds an explicit
  `pickle_executes_code` hard gate. *That kind does not exist yet and is therefore
  intentionally absent from the table.*
* **Dtype for code-derived contracts** — a model's parameter dtype is set at load
  time, so a code-derived (partial) contract does not constrain dtype.
* **Completeness of a partial contract** — extra checkpoint tensors are allowed; the
  partial contract asserts only what it could prove.
* **Quantization values, value ranges, sharding/FSDP/LoRA semantics** — roadmap
  phases C/D. Until implemented, the certifier abstains on the parts it cannot
  decide; it never certifies them safe by omission.

## 8. Soundness summary

Putting §3–§6 together, the end-to-end guarantee is:

> If `certify_weights_file(path, check_finite=True, expected=E)` returns
> `proven_safe = True`, then (a) the safetensors frame, header, tensor entries, and
> storage layout are well-formed (the loader can deserialize the file); (b) no float
> tensor contains NaN/Inf; and (c) the checkpoint satisfies every obligation in `E`
> — so `load_state_dict(strict=True)` against the corresponding model cannot raise a
> missing-key, unexpected-key, shape-mismatch (or, for pinned dtypes,
> dtype-mismatch) error. Every other outcome yields an *unsafe* verdict with a
> witnessing finding, or an *abstention*.

This document, the [table](#finding--runtime-failure-table), and
`tests/test_threat_model.py` together pin that guarantee: the table is exhaustive over
the emitted finding kinds, each row cites the concrete runtime failure it rules out,
and the test fails if code and documentation ever drift apart.
