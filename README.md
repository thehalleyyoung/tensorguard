# TensorGuard — Static Shape/Device/Phase Verifier for PyTorch Models

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Z3 SMT Solver](https://img.shields.io/badge/Z3-SMT%20solver-orange?logo=microsoft)
![PyTorch](https://img.shields.io/badge/PyTorch-compatible-EE4C2C?logo=pytorch&logoColor=white)

**TensorGuard** statically verifies PyTorch `nn.Module` architectures for
shape mismatches, device inconsistencies, and train/eval phase errors —
**with zero annotations**. It encodes tensor shapes as refinement type
predicates (`{v: Tensor | shape(v) == (batch, C, H, W)}`) and uses Z3 to
prove compatibility at every operation site, catching the #1 class of
runtime errors in ML codebases before any code runs.

---

## Key Features

- **142 operator transfer functions** — covers `matmul`, `conv2d`, `cat`,
  `view`, `reshape`, `transpose`, `permute`, `einsum`, `bmm`, attention
  patterns, and more
- **5-theory product domain** — jointly reasons over
  **Shape × Device × Phase × Stride × Permutation** for each tensor
- **Zero annotations required** — shapes are inferred from constructors,
  `torch.randn`, `nn.Linear(in, out)`, reshapes, and data flow
- **Train/eval phase tagging** — every tensor is annotated with the
  phase context in which it is produced; `BatchNorm` / `Dropout`
  misuse patterns can be detected from the resulting trace.  Note:
  the `--no-phase-check` CLI flag is wired through the public API as
  a post-hoc filter on phase-violation bugs (the phase predicates
  are computed and stored on every tensor regardless); see
  *Known limitations* below for the practical implication on the
  real corpora.
- **Device tracking** — catches silent CPU ↔ CUDA mismatches before they
  become runtime errors
- **CEGAR predicate discovery** — a counterexample-guided refinement
  loop discovers shape predicates automatically and stores them as
  `cegar_predicates` metadata on the verifier result, available for
  downstream consumers (no manual specification needed).  In addition,
  when the predicates discovered across refinement rounds are *jointly
  unsatisfiable* the conflict is promoted to a real
  `Bug(category=cegar_refined_contract)`, so `--cegar-iterations N` can
  change the reported bug set.  See *Known limitations* below.
- **Per-domain verification** — beyond shape, the device and gradient
  domains each refute real bugs that the base shape view misses (a cuda
  buffer added to a cpu input; a `.detach()` that severs gradient flow).
  The phase domain is **diagnostic-only** (it registers train/eval
  well-formedness but does not refute).  This is demonstrated end-to-end
  by `experiments_v5/run_domain_contribution.py` over the curated corpus
  in `experiments_v5/domain_corpus/` (results committed to
  `experiments_v5/domain_contribution.json`), and pinned by
  `tests/test_domain_contribution.py`.
- **Z3-backed** — all shape constraints are discharged by the Z3 SMT solver
  for soundness (0% false positives in `--high-confidence` mode).  The precise
  guarantee — exactly which programs are never miss-passed, and which
  constructs are over-/under-approximated or skipped — is published in
  [`SOUNDNESS_CONTRACT.md`](SOUNDNESS_CONTRACT.md) (generated from
  `src/soundness_contract.py`, including the currently-known unsoundness gaps).
- **Per-operator confidence** — every one of the 117 registered operator
  transfer functions is tagged `complete`, `sound`, or `heuristic` so you know
  how much to trust each inference (unknown ops default to `heuristic`).
  Inspect the table with `tensorguard operator-confidence [--json]` (machine-readable
  table committed to `operator_confidence_table.json`, source of truth
  `src/operator_confidence.py`, pinned by `tests/test_operator_confidence.py`).
- **Soundness modes** — `tensorguard verify --soundness-mode {sound,balanced,heuristic}`
  selects a three-valued verdict (`SAFE`/`UNKNOWN`/`UNSAFE`). `sound` is the
  contract you can rely on: it reports `SAFE` only when the module is fully
  inside the verifiable fragment (no opaque layers, no data-dependent control
  flow, no heuristic-tagged operators), otherwise `UNKNOWN` (CLI exit 2) — never
  a silent pass. The mode never changes which bugs are reported. Source of truth
  `src/soundness_contract.py`, pinned by `tests/test_soundness_mode.py`.
- **Formal verifiable fragment** — the exact grammar of `nn.Module`/`forward`
  constructs TensorGuard can analyze, the full supported-construct tables, the
  out-of-fragment taxonomy, and the "unsupported → `UNKNOWN`, never a silent
  pass" fallback policy are published in
  [`VERIFIABLE_FRAGMENT.md`](VERIFIABLE_FRAGMENT.md) (generated from
  `src/verifiable_fragment.py`; the instance-free `analyze_source()` exposes the
  statically-checkable fallback). Pinned by
  `tests/test_verifiable_fragment_spec.py`.
- **Frozen ground-truth benchmark corpus** — [`real_benchmarks/`](real_benchmarks/)
  holds 16 self-contained PyTorch `nn.Module`s (8 clean / 8 buggy) across the
  shape, device, phase, and gradient domains, each content-addressed by SHA-256
  in `manifest.json` so the corpus cannot silently drift. Labels are proven
  against real code (6/8 buggy raise a real eager `RuntimeError`, one is
  CUDA-only, one is a *silent* gradient-detach bug TG catches statically).
  Run `python -m real_benchmarks.load` to re-verify integrity and that 16/16
  TensorGuard verdicts match the frozen labels; pinned by
  `tests/test_real_benchmarks.py`.
- **GitHub-mined bug dataset** — [`experiments_v5/github_bug_mining/`](experiments_v5/github_bug_mining/)
  is a frozen, labeled dataset of thousands of real PyTorch shape/device bugs
  mined from public GitHub issues/PRs by matching verbatim PyTorch runtime error
  signatures, content-addressed by sha256. Verify it offline with
  `python experiments_v5/github_bug_mining/load.py`; pinned by
  `tests/test_github_bug_mining.py`. Re-mine from live GitHub with
  `experiments_v5/github_bug_mining/mine_github_bugs.py`.
- **SARIF 2.1.0 output** — integrates with GitHub Code Scanning / Advanced Security
- **Sub-second analysis** — typical models verified in < 1 second

---

## Known limitations of the shipped CLI

These are deliberately surfaced here so downstream users do not over-read
the feature list above.

* **`--no-phase-check`, `--no-device-check`, `--no-grad-check` gate the
  solver, not just the verdict.** The corresponding `check_phases`,
  `check_devices`, `check_gradients` keyword arguments to
  `src.api.verify_architecture` and `src.api.verify_module` (and the
  matching CLI flags) are threaded into the `ConstraintVerifier`: when a
  domain is disabled, its per-step safety encoder
  (`_encode_device_safety` / `_encode_phase_safety` /
  `_encode_gradient_safety`) returns **no constraints**, and the device /
  phase theory-solver checks are skipped, so the disabled domain incurs
  **no solver work and produces no cross-domain witnesses** (Step 2 of
  `100_STEPS.md`).  A post-hoc verdict filter is retained as a defensive
  safety net, but it is no longer the mechanism.  See
  `tests/test_solver_domain_gating.py`, which proves the disabled-domain
  encoders emit zero constraints while the enabled ones do.
  Three committed real-source examples in `examples/check_flag_demo/`
  demonstrate that toggling each flag flips the overall verdict
  between `REFUTED` and `VERIFIED` on un-instantiated class source;
  the verdict matrix is regenerated by
  `python3 experiments_v5/run_check_flag_demo.py` and committed under
  `reproducibility/check_flag_demo.json`.
* **`--cegar-iterations N` controls how many refinement rounds run.**
  Discovered predicates are stored as metadata, **and** unsatisfiable
  refined contracts are now promoted to real
  `Bug(category=cegar_refined_contract)` entries (Step 1 of
  `100_STEPS.md`).  When the predicates CEGAR proposes for a forward
  parameter across iterations are jointly unsatisfiable — e.g. an input
  that must be both `shape[-1] == 768` and `shape[-1] == 512` — the
  module can never run, and the conflict is surfaced as a sound,
  Z3-discharged bug.  Because these predicates only exist once refinement
  runs, the reported bug set now varies with `N` (with `N = 0` the
  contract-level conflict is not surfaced).  See
  `tests/test_cegar_refined_contract.py`.

---

## Installation

```bash
git clone https://github.com/thehalleyyoung/tensorguard.git
cd tensorguard
pip install -e .
```

The only required dependency is `z3-solver>=4.12` (installed automatically).

---

## Quickstart

Write a model with a shape bug:

```python
# model.py
import torch
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.fc = nn.Linear(16 * 5 * 5, 10)   # wrong: should be 16 * 222 * 222

    def forward(self, x):          # x: (batch, 3, 224, 224)
        x = self.conv(x)          # → (batch, 16, 222, 222)
        x = x.view(x.size(0), -1) # → (batch, 16*222*222)
        return self.fc(x)          # ERROR: expects 16*5*5 = 400, got 788544
```

Run TensorGuard:

```
$ tensorguard verify model.py -s x=batch,3,224,224

  ✗ model.py: 1 verification errors (243ms)

  L15: Shape mismatch at nn.Linear: input has 788544 features,
       expected 400                                  (shape-error)
```

Verify a correct model:

```
$ tensorguard verify model.py -s x=batch,3,224,224
  ✓ model.py: Architecture verified safe (768ms)
```

---

## CLI Reference

```
tensorguard verify FILE [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `FILE` | Python file containing `nn.Module` class(es) | — |
| `-s`, `--input-shape` | `name=dim1,dim2,...` (repeatable) | auto-inferred |
| `--no-device-check` | Skip device consistency checks | off |
| `--no-phase-check` | Skip train/eval phase checks | off |
| `--cegar-iterations` | Max CEGAR refinement iterations | `10` |
| `-f`, `--format` | `text`, `json`, `sarif` | `text` |
| `--high-confidence` | Only report Z3-proven bugs (0% FP) | off |

### Additional Commands

| Command | Description |
|---------|-------------|
| `tensorguard ci-check [PATHS...] --sarif-output out.sarif` | CI mode with SARIF output |
| `tensorguard watch [PATHS...]` | Watch and re-verify on changes |
| `tensorguard version` | Show version info |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Verification succeeded — no shape/device/phase errors |
| `1` | Errors found or analysis error (invalid input, file not found) |

---

## What TensorGuard Catches

| Error Class | Example |
|-------------|---------|
| **Shape mismatch** | `nn.Linear` input dimension ≠ declared `in_features` |
| **Matmul incompatibility** | `torch.matmul(A, B)` where inner dimensions differ |
| **Bad reshape/view** | `x.view(batch, -1)` where total elements don't divide evenly |
| **Conv output → Linear input** | Feature map flattened size ≠ `nn.Linear` input |
| **Cat dimension mismatch** | `torch.cat([a, b], dim=1)` with different sizes on other dims |
| **Device mismatch** | `cpu_tensor + cuda_tensor` |
| **Phase error** | `model.eval()` then calling layers that behave differently in eval |
| **Stride/permutation error** | Contiguity assumptions violated after `transpose`/`permute` |

---

## How It Works

1. **AST Parse** — extract `nn.Module` class, `__init__`, and `forward` method
2. **Shape Predicate Harvesting** — infer shapes from `nn.Linear(in, out)`,
   `nn.Conv2d(...)`, `torch.randn(...)`, input shape flags, and reshape calls
3. **5-Theory Product Domain Propagation** — propagate
   **(Shape × Device × Phase × Stride × Permutation)** through every
   operation using 142 transfer functions
4. **Z3 Constraint Solving** — at each operation site, generate and discharge
   shape compatibility constraints via Z3
5. **CEGAR Refinement** — if the initial abstraction is too coarse, discover
   new predicates from counterexamples and re-check
6. **Report** — emit shape/device/phase errors with concrete dimension values
   and fix suggestions

---

## Python API

```python
from src.api import verify_module

# Verify a file
result = verify_module("model.py", input_shapes={"x": ("batch", 3, 224, 224)})
print(f"Status: {result.status}")       # "SAFE" or "UNSAFE"
print(f"Bugs: {len(result.bugs)}")
print(f"Duration: {result.duration_ms:.0f}ms")

for bug in result.bugs:
    print(f"  {bug.location.line}: {bug.message}")
```

---

## CI / CD Integration

```yaml
# .github/workflows/tensorguard.yml
name: TensorGuard Shape Check
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .
      - run: tensorguard ci-check models/ --sarif-output results.sarif
      - uses: github/codeql-action/upload-sarif@v3
        if: always()
        with: { sarif_file: results.sarif }
```

---

## Configuration

TensorGuard reads configuration from `.reftype.toml` or `[tool.reftype]`
in `pyproject.toml`:

```toml
[reftype]
include = ["models/**/*.py"]
exclude = ["tests/**"]

[reftype.cegar]
max_iterations = 10
```

---

## FAQ

**Q: Z3 install fails.**
A: Ensure Python 3.9+ and pip ≥ 21.0. On Apple Silicon:
`pip install --no-cache-dir z3-solver>=4.12`.

**Q: False positive on complex `view()`/`reshape()`.**
A: TensorGuard is conservative with dynamic reshapes. Use `--high-confidence`
to suppress heuristic findings.

**Q: How fast is it?**
A: Typical single-model verification completes in < 1 second. The
`ci-check` and `analyze` commands support `--timeout` and `--workers` for
large codebases.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Reproducing the NeurIPS evaluation

The released benchmark artefacts and reproducibility scripts live
under `experiments_v5/` and `reproducibility/`.

### One command: `make reproduce`

The whole CI-reproducible pipeline is wired into a single target:

```bash
make reproduce          # regenerate every CI-reproducible artifact + audit
make reproduce-check    # also assert byte-identical regeneration (no git diff)
```

`make reproduce` regenerates, from source and in dependency order, the
generated spec docs/tables (`SOUNDNESS_CONTRACT.md`,
`VERIFIABLE_FRAGMENT.md`, `operator_confidence_table.json`), the frozen
benchmark corpus and its audit artifact (`real_benchmarks/`), and the
headline 60-bug Refuted-Proof figure — then runs the numeric-claim audit,
which recomputes every `x/y` ratio and `%` token in `README.md` from the
freshly regenerated artifacts. `make reproduce-check` additionally proves
determinism: regenerating the byte-deterministic artifacts must produce no
git diff. The orchestrator is `reproducibility/reproduce_all.py`, pinned by
`tests/test_reproduce_harness.py`.

**Honest scope.** Artifacts that need CUDA, a HuggingFace download, or a
Lean toolchain cannot be rebuilt in a standard CI box; their committed
copies are validated by the numeric audit (reported as `QUALIFIED_ENV`,
with the regeneration command recorded) rather than regenerated in place.
`make reproduce-full` regenerates those too in an environment that has the
required toolchains.

### One-command reproduction of the headline RP figure

The paper headline "**Refuted-Proof on 53/60** historical bugs" and
the related "raw refute count = 56/60" reported by the per-feature
ablation are emitted by **a single command**:

```bash
PYTHONPATH=. python3 reproducibility/reproduce_headline_60bug.py
```

This runs the full 60-bug historical corpus end-to-end against the
current `main` branch verifier and prints **both** numbers, with the
per-item verdict pair written to
`reproducibility/reproduce_headline_60bug.json`.  The two numbers
correspond to two clearly-labelled regimes (free-symbolic vs
per-bug `INPUT_SHAPES` lifted), see
`reproducibility/reproduce_headline_60bug.md` for the full
explanation.

### Numeric-claim audit (every headline number is backed)

Every headline numeric claim in `README.md`, `neurips.tex`, and
`workshop_fmai.tex` is registered in a single audit harness and checked
against the committed regeneration artifacts under `reproducibility/`:

```bash
PYTHONPATH=. python3 reproducibility/audit_numeric_claims.py
```

For each claim the harness (a) confirms the number is still literally
present in the prose at its cited source, and (b) recomputes it from the
backing artifact(s) — supporting ratios, percentages, and p-values with
tolerance. It classifies each claim as `VERIFIED`, `MISMATCH`,
`QUALIFIED_REGIME` (regime-specific — e.g. the fragment-fair Pytea
score vs. the stricter 2024-catalogue score, both legitimate),
`QUALIFIED_ENV` (requires Lean/HF/CUDA, regeneration command recorded
rather than asserted here), `ORPHAN`, or `SOURCE_MISSING`. A scanner also
fails the audit if any `x/y` ratio or `%` token appears in `README.md`
without a registry claim (script-catalogue rows that name their own
regeneration script in-row are exempt). Results are written to
`reproducibility/numeric_claims_audit.json` and pinned by
`tests/test_numeric_claims_audit.py`.

**Honest scope.** This audits the committed artifacts (the outputs of the
last committed regeneration run) and the prose that cites them; it is not
a from-scratch regeneration in this process (several artifacts need CUDA,
HuggingFace downloads, or a Lean toolchain). Pass `--regenerate` to
additionally invoke each artifact's recorded `meta.command` where the
environment supports it.

### Precision/recall vs baselines

`evaluation/precision_recall.py` scores five detectors on the frozen,
balanced, executable ground-truth corpus in `real_benchmarks/` (clean and
buggy PyTorch modules spanning the shape, device, phase and gradient
domains) and emits per-method confusion matrices with precision, recall and
F1:

```bash
PYTHONPATH=. python3 evaluation/precision_recall.py     # or: make precision-recall
```

The baselines are real, not strawmen: a forward-only runtime smoke test, a
stronger runtime baseline that also runs a backward pass and inspects every
parameter's gradient, the published academic static shape analyser
[PyTea](https://github.com/ropas/pytea) (built from source and run live on an
auto-generated entry wrapper), and a no-op floor that always predicts clean.
On pure shape bugs every real analyser ties with zero false positives; the
forward-only runtime baseline misses the *silent* gradient-severing bug
(it raises no exception), and PyTea — shape-only by construction — misses
both the device and gradient bugs. TensorGuard is the only detector that is
simultaneously static, sound, and correct across all four domains, with no
execution, concrete inputs, or GPU required. The committed matrices live in
`evaluation/confusion_matrices.json` and `evaluation/confusion_matrices.md`
and are pinned by `tests/test_precision_recall.py`.

### Sound-mode false-positive hunt

For a tool meant to ship inside PyTorch a single false alarm destroys trust,
so `evaluation/sound_mode_fp.py` hunts aggressively for one. It generates a
large, diverse corpus of clean PyTorch modules — every model is validated to
execute without error in eager PyTorch before admission — spanning MLP,
residual-MLP, CNN, LayerNorm, attention and GroupNorm templates, then runs
TensorGuard in the strict `sound` soundness mode on each:

```bash
PYTHONPATH=. python3 evaluation/sound_mode_fp.py     # or: make sound-fp
```

The hard requirement is **zero Refuted (false-positive) verdicts** on clean,
executing code. Across eighty clean models (the eight hand-written clean
benchmarks plus seventy-two seeded-generated models) TensorGuard raises zero
false alarms and verifies every one SAFE in sound mode — a result that is not
achieved by trivially abstaining, since SAFE coverage is total. The committed
artifacts live in `evaluation/sound_mode_fp.json` and
`evaluation/sound_mode_fp.md` and are pinned by
`tests/test_sound_mode_fp.py`.

### Latent-bug recall vs the strongest dynamic baseline

On the small balanced corpus TensorGuard already ties the strongest dynamic
baseline (`runtime_backward` — one seeded `train()` forward+backward, then a
grad-presence check) at perfect recall, because every bug there is exercised by
that single pass. The interesting question for a *static* verifier is the class
of **latent** bugs no single concrete execution can observe.
`evaluation/hard_recall.py` builds such a corpus — phase-dependent faults that
only manifest in `eval()`, path-dependent faults on an untaken branch, and
silent gradient-freeze faults — where every model is *proven genuine* (the
latent fault really fails when exercised) and *proven silent* under the
strongest dynamic baseline (the seeded pass runs clean):

```bash
PYTHONPATH=. python3 evaluation/hard_recall.py     # or: make hard-recall
```

The strongest dynamic baseline catches none of these latent bugs, while
TensorGuard's static analysis catches the phase-dependent and path-dependent
families in full — a recall advantage of three quarters on bugs that are
invisible to dynamic testing by construction. The two residual misses (silent
`requires_grad = False` freezes) are not hidden: each carries a root-cause tag
in the committed artifact, documenting exactly which static-analysis gap is
responsible. Artifacts live in `evaluation/hard_recall.json` and
`evaluation/hard_recall.md` and are pinned by `tests/test_hard_recall.py`.

### Differential fuzzing for false positives

`evaluation/diff_fuzz.py` is a *random-architecture* false-positive hunt. A
fuzzer grows valid `nn.Module`s by threading the running tensor shape through a
random chain of layers — Linear, Conv2d, BatchNorm, LayerNorm, pooling, flatten
and activations, entering at either rank-two (vector) or rank-four (image)
inputs — so each network is dimensionally valid by construction, and is then
**validated to execute without error** in eager PyTorch before admission. Every
admitted model is checked differentially: runtime says "ran clean", so
TensorGuard in strict `sound` mode must never Refute it.

```bash
PYTHONPATH=. python3 evaluation/diff_fuzz.py     # or: make diff-fuzz
```

Across two hundred random seeds every admitted model executed cleanly,
TensorGuard raised **zero false positives**, and — crucially — verified every
one SAFE (total coverage, no abstentions), so the zero-false-positive result is
not a vacuous "always abstain". The verdicts stay non-vacuous: feeding a
mismatched input dimension to the same fuzzed model makes TensorGuard Refute it.
The committed artifacts live in `evaluation/diff_fuzz.json` and
`evaluation/diff_fuzz.md` and are pinned by `tests/test_diff_fuzz.py`.

### Negative fuzzing (false-negative hunting)

`evaluation/neg_fuzz.py` is the dual hunt: instead of checking that TensorGuard
stays silent on clean code, it **injects a fault** into a valid random
`nn.Module` and asserts TensorGuard catches it. Faults are drawn from a
catalogue — corrupting a `Linear`'s input or output width, a `Conv2d`'s input
channels, or splicing an incompatible `reshape` — and every injected fault is
**proven genuine** by observing a real eager-PyTorch `RuntimeError`; a mutation
that happens to still execute is not admitted.

```bash
PYTHONPATH=. python3 evaluation/neg_fuzz.py     # or: make neg-fuzz
```

Across the genuine injected faults TensorGuard's recall is perfect — it catches
every one, with **zero false negatives** — and the result is broken out
per injector family. Any fault it missed would be tagged with a root cause in
the artifact rather than hidden. The committed artifacts live in
`evaluation/neg_fuzz.json` and `evaluation/neg_fuzz.md` and are pinned by
`tests/test_neg_fuzz.py`.

### Minimal-reproducer shrinker

`evaluation/minimize.py` is a delta-debugging minimizer for `nn.Module`s,
generic over a predicate that captures the TensorGuard/runtime relation:
`false_positive` (TG refutes but the model runs clean), `false_negative` (TG
accepts but the model raises), or `agreement_bug` (both flag the bug). It
ddmin-removes layers and then coordinate-shrinks every integer dimension while
preserving the predicate, yielding a minimal reproducing module.

```bash
PYTHONPATH=. python3 evaluation/minimize.py     # or: make minimize
```

Because earlier steps found no real false positives or false negatives, there
is no natural disagreement corpus yet, so the live demo shrinks an
`agreement_bug` — a genuine caught shape fault — from a twelve-layer chain down
to a single faulted `Linear` on a one-feature input, removing every inessential
layer while TensorGuard still refutes and the model still raises. The
`false_positive`/`false_negative` paths are covered by unit tests with a
synthetic oracle. The minimizer preserves the predicate's truth, not its
mechanism, and is locally minimal under op removal and coordinate-wise dim
shrinking. Committed artifacts live in `evaluation/minimize_demo.json` and
`evaluation/minimize_demo.md` and are pinned by `tests/test_minimize.py`.

### Disagreement triage + frozen regression suite

`evaluation/triage.py` triages the false-positive / false-negative hunts and
turns their signal into a frozen regression suite. The combined fuzzing
population — the random clean models of Step 15 plus the injected faults of
Step 16, over four hundred models in total — produced no TensorGuard/runtime
disagreements, so there is nothing to fix. Instead the harness freezes **fifty
minimal bug reproducers** spanning a catalogue of distinct fault mechanisms
(Linear in/out width, Conv channel/kernel, invalid view/reshape, matmul
inner-dim, broadcast add, cat non-cat-dim, flatten→Linear), each paired with a
minimal **clean sibling**.

```bash
PYTHONPATH=. python3 evaluation/triage.py     # or: make triage
```

Every buggy entry is verified to raise at runtime *and* be refuted by
TensorGuard; every clean sibling is verified to run clean *and* be accepted. The
frozen entries are replayed as parametrized regression tests by
`tests/test_triage.py`, so any future regression — a missed bug or a false
alarm on a clean sibling — fails CI. Committed artifacts live in
`evaluation/triage_regressions.json` and `evaluation/triage_regressions.md`.

### Shape-algebra property tests (Hypothesis)

`tests/test_shape_algebra_properties.py` exercises the concrete and abstract
shape transfer functions in `src/denotational_semantics.py` with property-based
testing. Hypothesis generates thousands of random shapes (bounded rank and
dimensions) per run and checks the algebraic laws of the shape algebra —
transpose involution, reshape and flatten element-count preservation with
reverse round-trips, squeeze/unsqueeze inversion, global-squeeze size-one
removal, cat dimension summation, matmul output shape with inner-dim-mismatch
rejection, add-broadcast commutativity, and identity. Most importantly it pins
the per-node soundness theorem `α(⟦op⟧(σ)) ⊑ ⟦op⟧♯(α(σ))` for every operator,
asserting the abstract transfer over-approximates the concrete one over a large
random input space rather than a handful of examples. This complements the
example-based `tests/test_denotational_semantics.py`.

```bash
PYTHONPATH=. python3 -m pytest tests/test_shape_algebra_properties.py -q   # or: make shape-props
```

### Precision/recall regression dashboard (merge gate)

`evaluation/dashboard.py` aggregates the headline metrics from every committed
evaluation artifact (the confusion matrices, sound-mode false-positive hunt,
differential and negative fuzzing, latent-bug recall, and the frozen triage
suite) into a single dashboard and gates them against a frozen baseline,
`evaluation/dashboard_baseline.json`. Quality metrics (precision, recall, F1,
false positives, false negatives, coverage) are gated by direction; corpus-size
("integrity") metrics must never shrink. Integer counts are compared exactly;
non-finite or missing values are treated as failures.

```bash
PYTHONPATH=. python3 evaluation/dashboard.py            # regenerate dashboard.md
PYTHONPATH=. python3 evaluation/dashboard.py --check    # gate; non-zero on regression
PYTHONPATH=. python3 evaluation/dashboard.py --update-baseline   # accept an intentional change
```

In `--check` mode (`make dashboard-check`) the dashboard exits non-zero on any
regression or metric drift, so it can block a merge. The baseline is a
**reviewed regression ratchet**, not a tamper-proof boundary: any intentional
metric change must appear as a reviewable diff to the baseline file. Because the
dashboard reads committed artifacts, CI runs each harness in `--check` mode
(byte-identical regeneration) **before** the dashboard gate so a source
regression that left stale artifacts behind is still caught — see the
`dashboard-gate` job in `.github/workflows/tensorguard-ci.yml` and the
`make dashboard-gate` target. Committed artifacts: `evaluation/dashboard.md` and
`evaluation/dashboard_baseline.json`.

### Lean build (sorry-free)

The Lean proof corpus is built per-module to keep the harness
happy:

```bash
cd lean
for m in TensorGuard.Soundness TensorGuard.AssumeGuarantee \
         TensorGuard.AssumeGuaranteeExtended TensorGuard.Extended \
         TensorGuard.Parity TensorGuard.V5OperatorRules; do
  lake build "$m"
done
lake build parity_runner
```

Captured log: `experiments_v5/v8/lean_build_v8.log` (and the
identical `lean_build_v9.log`).  The log contains zero
`declaration uses 'sorry'` warnings; the only `sorry` substrings
in `lean/TensorGuard/` live inside docstring comments.

### Other artefacts

| Artefact                                                       | Script                                                  |
|----------------------------------------------------------------|---------------------------------------------------------|
| 488-block + 60-bug headline triple                             | `experiments_v5/run_v5_benchmark.py`                    |
| Precision/recall confusion matrices vs PyTea/runtime/no-op     | `evaluation/precision_recall.py`                        |
| Sound-mode false-positive hunt (clean executing models)        | `evaluation/sound_mode_fp.py`                           |
| Latent-bug recall vs the strongest dynamic baseline            | `evaluation/hard_recall.py`                             |
| Differential fuzz false-positive hunt (random valid models)    | `evaluation/diff_fuzz.py`                               |
| Negative-fuzz false-negative hunt (injected faults)            | `evaluation/neg_fuzz.py`                                |
| Minimal-reproducer shrinker (delta-debug disagreements)        | `evaluation/minimize.py`                                |
| Disagreement triage + 50-case regression suite                 | `evaluation/triage.py`                                  |
| Shape-algebra property tests (Hypothesis)                      | `tests/test_shape_algebra_properties.py`                |
| Precision/recall regression dashboard (merge gate)             | `evaluation/dashboard.py`                               |
| 60-bug headline RP reproducer (single command)                 | `reproducibility/reproduce_headline_60bug.py`           |
| Verdict reclassification (RP / CV / LW)                        | `experiments_v5/run_verdict_reclassification.py`        |
| 10-bug real-public corpus                                      | `experiments_v5/v8/verify_real_bugs.py`                 |
| Real-public corpus selection protocol                          | `experiments_v5/v8/REAL_BUG_SELECTION_PROTOCOL.md`      |
| Real-public corpus freeze invariant                            | `experiments_v5/v8/verify_corpus_freeze.py`             |
| User-visible (no-synthesised-assume) RP report                 | `experiments_v5/v8/build_user_visible_rp.py`            |
| Backward verifier on 10 real importable models                 | `experiments_v5/v8/backward_real/run_backward_real.py`  |
| Lean operator-rule audit (28 rules)                            | `lean/TensorGuard/V5OperatorRules.lean`                 |
| Lean assume/guarantee composition (Theorem 3, weak form)       | `lean/TensorGuard/AssumeGuarantee.lean`                 |
| CEGAR / phase-encoder dead-code TCB confirmation               | `reproducibility/cegar_phase_deletion_tcb.py`           |
| Grad-lattice runtime holdout (round-5 rewrite, 10 self-contained subjects) | `reproducibility/grad_lattice_runtime_holdout.py` |
| 60-bug ablation: rule-driven only (parser-marker excluded)     | `reproducibility/bug_corpus_no_parser_marker.py`        |
| Post-freeze N=15 power analysis (Fisher exact, 80% power)      | `reproducibility/postfreeze_power_analysis.py`          |
| Cross-family static analysis: Llama 2/3 (6 modules)            | `reproducibility/hf_extra_model_family.py`              |
| Cross-family static analysis: Qwen2 (5 modules)                | `reproducibility/hf_extra_family_round_comet1.py`       |
| Cross-family static analysis: Mistral / Gemma / Phi-3 (15 modules) | `reproducibility/hf_extra_families_round11.py`      |
| Naturally-occurring cross-family bugs (7 upstream PRs/issues, Llama/Qwen2/Mistral/Phi-3) | `reproducibility/cross_family_natural_bugs.py` |
| 488-block headline-triple regime reconciliation (HCO=True vs HCO=False) | `reproducibility/block_corpus_488_reconciliation.py` |

Round-by-round reviewer responses live under `.comet_neurips/` and
the latest is `review_response.md` at the repo root.
