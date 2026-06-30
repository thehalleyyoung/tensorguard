# TensorGuard

**Catch the bugs that waste your GPU hours — before you launch the run.**

TensorGuard is a **static analyzer for PyTorch**. It reads your model and training
code and finds the shape mismatches, device errors, broken checkpoints, and silent
training-killers that normally only surface *after* you've waited in the queue,
allocated 8 GPUs, and loaded the dataset.

```python
import tensorguard as tg

bugs = tg.verify_architecture(open("model.py").read(), input_shapes={"x": (1, 784)}).bugs
for b in bugs:
    print(b.message)
# [SHAPE-INCOMPATIBLE] Linear expects last dim=128, got 256
```

No training run. No GPU. **No `import torch` required** — TensorGuard analyzes your
source code symbolically, so it runs in milliseconds in CI on a machine that has
never had PyTorch installed.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](#install)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Soundness: Lean-verified](https://img.shields.io/badge/soundness-Lean--verified-orange)](#why-you-can-trust-it)
[![Zero false positives](https://img.shields.io/badge/sound%20mode-zero%20false%20positives-brightgreen)](#three-modes-you-choose-the-tradeoff)

---

## Why TensorGuard

| | |
|---|---|
| ⚡ **Fast & torch-free** | Pure static analysis. Runs in CI in milliseconds — no model instantiation, no GPU, no `torch` install. |
| 🎯 **Zero false positives** | In `sound` mode every reported bug is a *real* bug. When TensorGuard isn't sure, it **abstains** instead of guessing. |
| 🔍 **Zero annotations** | Shapes are inferred from `nn.Linear(in, out)`, `torch.randn`, reshapes, and dataflow. You write nothing extra. |
| 🧠 **It explains itself** | Every finding comes with the input→op→shape→failure chain, a fix suggestion, and (optionally) a replayable proof certificate. |
| 📐 **Lean-verified core** | The soundness of the engine is backed by **80+ machine-checked Lean proofs**. |
| 🔌 **Plugs into your workflow** | CLI, GitHub Action, SARIF code-scanning, LSP (VS Code / Neovim / JetBrains), pre-commit, and Jupyter magic. |

---

## Install

```bash
git clone https://github.com/thehalleyyoung/tensorguard
cd tensorguard && pip install -e .
```

That's it — the only hard dependency is the Z3 SMT solver (pulled in automatically).
PyTorch is **not** required to run TensorGuard.

```python
import tensorguard
from tensorguard import verify_architecture, analyze
```

---

## 60-second tour

### 1. Verify a model architecture

```bash
tensorguard verify model.py -s x=1,784
```

```
✗ model.py: 1 verification issue (97ms)
  ERROR: Layer fc2 (line 8) expects input dimension 128, but receives (1, 256)
    --> line 8, col 8
    |         return self.fc2(self.fc1(x))
    |         ^
    note: Layer fc2 defined here with in_features=128 (line 6)
    fix: change fc2 to accept (1, 256)
```

Add `--fix` to get a concrete one-line patch, or `--fix --write` to apply it:

```bash
tensorguard verify model.py -s x=1,784 --fix --write
#   - self.fc2 = nn.Linear(128, 10)
#   + self.fc2 = nn.Linear(256, 10)
```

### 2. Let it fix the code — *and re-verify its own fix*

Most "auto-fix" tools emit a suggestion and hope. TensorGuard does something
stronger: for every repairable finding it proposes a minimal, canonical edit,
**re-runs the analyzer on the patched source**, and only surfaces the fix when
the targeted bug is gone *and* no new bug was introduced. So every fix is
*machine-verified*, not guessed.

```bash
tensorguard fix model.py            # show verified unified diffs
tensorguard fix model.py --write    # apply them in place
tensorguard fix src/ --format json  # machine-readable, for CI bots
tensorguard fix src/ --format patch # one `git apply`-able patch per file
tensorguard fix model.py --format sarif   # SARIF "Apply suggested fix" for code-scanning
tensorguard fix model.py --explain  # show each fix's originating finding + re-verification proof
```

```diff
=== model.py ===
  [✓ verified] missing_super_init (line 5) — insert-super-init
     class Net(nn.Module):
         def __init__(self):
+            super().__init__()
             self.fc = nn.Linear(3, 4)
  [✓ verified] direct_forward_call (line 9) — forward-to-call
-        return self.fc.forward(x)
+        return self.fc(x)
```

Current repair strategies: insert a missing `super().__init__()`, rewrite
`module.forward(x)` → `module(x)`, `.data` → `.detach()`, an `nn.Linear`'s
`in_features` to the dim actually flowing in, reshape→`-1` flatten, and
non-negative dimensions. When a fix can't be made *unambiguously*, TensorGuard
abstains rather than risk a wrong edit.

### 3. Scan a whole project

```bash
tensorguard analyze src/ --format sarif -o results.sarif   # GitHub code-scanning ready
tensorguard analyze-package my_project/                    # directory summary
```

### 4. Use it from Python

```python
import tensorguard as tg

# whole-file analysis
result = tg.analyze_file("model.py")
print(result.bug_count, "issues")

# architecture verification with a known input
result = tg.verify_architecture(src, input_shapes={"x": (1, 3, 224, 224)})
for bug in result.bugs:
    print(bug.category, bug.location.line, bug.message)
```

---

## What it catches

### 🔺 Shape & dimension bugs

Mismatched layers, bad matmuls, impossible reshapes, broadcast errors, out-of-range
axes, `cat`/`stack` mismatches, negative/zero dimensions, and more — across **140+
operator transfer functions** (`matmul`, `conv2d`, `view`/`reshape`, `einsum`,
`permute`, `gather`/`scatter`, `repeat_interleave`, `fold`/`unfold`, attention
patterns, RNN/LSTM/GRU state contracts…).

```python
from src.symexec import analyze_source

src = '''
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5, 6)
    return a @ b          # 4 vs 5
'''
print(analyze_source(src).bugs[0].message)
# matmul contracted-dim mismatch: (3, 4) @ (5, 6) (4 vs 5); RuntimeError at runtime
```

### 🖥️ Device & dtype bugs

Silent CPU ↔ CUDA mismatches and dtype/autocast violations caught before the kernel
raises. TensorGuard jointly reasons over a **5-theory product domain**:
**Shape × Device × Phase × Stride × Permutation**.

### 🧪 "Why isn't it training?" — intent checks

Some bugs never raise an exception; your script runs to completion and the model just
silently fails to learn. TensorGuard's intent detectors catch the classic
training-loop mistakes:

```python
from src.symexec import analyze_source, SymConfig

src = '''
def train(model, loader, optimizer):
    for x, y in loader:
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()        # forgot optimizer.zero_grad()
'''
bug = analyze_source(src, config=SymConfig.heuristic()).bugs[0]
print(bug.kind.value)   # missing_zero_grad
```

| Detector | Fires when… | Consequence |
|---|---|---|
| `missing_zero_grad` | `.backward()` + `step()` but no `zero_grad()` | gradients accumulate across iterations |
| `step_without_backward` | `zero_grad()`/`step()` but no `.backward()` | no gradients computed; model never learns |
| `backward_without_step` | `.backward()` but no `step()` | gradients computed, never applied |
| `backward_no_grad` | `.backward()` on a detached tensor (`loss.detach().backward()`) | raises *"does not require grad"*; no gradients flow |

Plus footgun checks: calling `module.forward(x)` instead of `module(x)`, missing
`super().__init__()`, `.data` autograd bypass, discarded out-of-place transforms,
`.backward()` on a non-scalar, `.numpy()` on a grad tensor, and more.

### 📦 "Will my checkpoint load?" — the weights layer

TensorGuard derives the exact `state_dict` shape contract **from your model code**, so
you can certify a checkpoint matches a model *without loading either*.

```python
from src.symexec.model_contract import derive_model_contract

contract = derive_model_contract(open("model.py").read(), "Net()")
print(contract.params)
# {'fc1.bias': (256,), 'fc1.weight': (256, 784),
#  'fc2.bias': (10,),  'fc2.weight': (10, 256)}
```

Deployment- and resume-time gates cover the things that break *after* training:

- **Checkpoint `state_dict` compatibility** — missing/unexpected keys, tied weights,
  tensor-parallel shards, dtype drift (`verify_checkpoint_state_dict`)
- **LoRA / PEFT adapters** — rank, `target_modules`, merged state, quantized bases
  (`verify_lora_adapter_compatibility`)
- **Optimizer state** — AdamW / Adafactor / fused / sharded resume (`verify_optimizer_state`)
- **Export & serving** — GGUF/llama.cpp (`verify_gguf_export_contract`), ONNX
  (`verify_onnx_export_contract`), `torch.compile` guard parity, CUDA-graph capture
  eligibility (`verify_cuda_graph_capture_eligibility`), and TorchServe/FastAPI
  request→model schema gates (`verify_serving_schema`)

### 📚 Real-library contracts (differentially tested against the real thing)

`einops` (`rearrange`/`reduce`/`repeat`), `nn.MultiheadAttention` & SDPA,
`torch.linalg` (solve/inv/cholesky/SVD/eig/QR), `grid_sample`/`affine_grid`,
`torchvision.transforms.v2`, `embedding_bag` & TorchRec jagged embeddings,
`torch.distributions` shapes, named tensors, complex/FFT, `vmap` & `torch.func`
autodiff transforms, loss target/reduction contracts, and `torch.sparse` layouts —
each verified by shape-only models **differentially checked against real `torch`/`einops`**.

```python
import tensorguard as tg

tg.verify_einops(...)               # non-divisible patch embeds caught
tg.verify_multihead_attention(...)  # broken q/k/v/mask contracts caught
tg.verify_linalg_solve(...)         # non-square / broadcast errors caught
# ...and verify_fft, verify_vmap, verify_loss, verify_embedding_bag, verify_sparse_*, etc.
```

---

## Three modes — you choose the tradeoff

```
sound  ⊆  balanced  ⊆  heuristic
```

| Mode | Use it for | Guarantee |
|---|---|---|
| **`sound`** | CI gates, pre-merge checks | **Zero false positives.** Every finding is real; abstains when unsure. |
| **`balanced`** *(default)* | day-to-day development | Sound shape reasoning; abstains at unmodeled boundaries. |
| **`heuristic`** | exploratory "what might be wrong here?" triage | Surfaces low-confidence suspicions and intent checks. *May over-warn.* |

```python
from src.symexec import analyze_source, SymConfig

analyze_source(src, config=SymConfig.sound())                    # only certainties
analyze_source(src, config=SymConfig.heuristic(min_confidence=0.6))  # max recall
```

```bash
tensorguard verify model.py --soundness-mode sound
```

---

## Drop it into your workflow

**GitHub Action** — findings appear natively in the PR "Files changed" tab and the
Security → Code scanning view:

```yaml
# .github/workflows/tensorguard.yml
- uses: ./   # tensorguard action
  with:
    paths: "src/"
    soundness-mode: sound
    fail-on: error
    sarif-output: tensorguard.sarif
```

**Pre-commit:**

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: tensorguard
      name: tensorguard
      entry: tensorguard-precommit
      language: system
      types: [python]
```

**Editor / LSP** — real-time squiggles in VS Code, Neovim, and JetBrains
(`editors/`), or run the server directly:

```bash
tensorguard server         # stdio JSON-RPC LSP
```

**Jupyter** — analyze a cell as you write it:

```python
%load_ext src.symexec.notebook
%%tensorguard --mode heuristic
# ... model code in this cell is analyzed (and the findings shown) before it runs ...
```

**SARIF / CI everywhere** — `tensorguard analyze --format sarif`, plus a `ci-check`
subcommand with proper exit codes, baselines, and PR-comment output.

---

## Why you can trust it

- **Refutation soundness, proved in Lean.** The engine's core guarantees are backed by
  **82 machine-checked Lean proof files** (`lean/`) — including the abstract-domain
  transfer functions, the CEGAR refinement bound, and the mode-dependent soundness
  fragment. See [`SOUNDNESS_CONTRACT.md`](SOUNDNESS_CONTRACT.md).
- **Torch-free trust path.** TensorGuard never imports or executes your model. It can't
  trigger side effects, download weights, or run arbitrary code — it only reads source.
- **Proof-carrying findings.** A reported bug can ship with a replayable certificate and
  a concrete counterexample, so you (or a reviewer) can independently confirm it.
- **CEGAR predicate discovery.** A counterexample-guided refinement loop discovers shape
  predicates automatically — no manual specification — with a Lean-checked termination bound.

---

## Command reference

```
tensorguard verify <file>          # verify an nn.Module architecture
tensorguard analyze <path>         # scan files/dirs for bugs (text/json/sarif)
tensorguard analyze-package <dir>  # whole-package analysis with summary
tensorguard symexec <file>         # run the symbolic-execution engine directly
tensorguard fix <file>             # apply machine-verified repairs (--write to apply)
tensorguard explain <file>         # HTML inference-chain report
tensorguard watch <path>           # re-analyze incrementally on change
tensorguard ci-check <path>        # CI mode with exit codes
tensorguard server                 # start the LSP server
tensorguard adoption-recipes       # ready-made CI/pre-commit/editor configs
```

Run `tensorguard <command> --help` for full flags. Common ones: `--soundness-mode
{sound,balanced,heuristic}`, `--format {text,json,sarif}`, `--input-shape/-s
name=d1,d2,...`, `--fix [--write]`, `--explain`, `--no-phase-check`,
`--cegar-iterations N`.

---

## Python API at a glance

```python
import tensorguard as tg

tg.analyze(source)                       # analyze a source string
tg.analyze_file(path)                    # analyze one file
tg.analyze_directory(path)               # analyze a tree
tg.verify_architecture(source, input_shapes=...)  # full architecture verification
tg.verify_module(path, input_shapes=...)          # verify a module file
tg.quick_check(source)                   # bool: any high-confidence bugs?
```

```python
from src.symexec import (
    analyze_source, analyze_package,      # the symbolic engine
    SymConfig,                            # mode selection
    derive_model_contract,                # weights/state_dict contract
    certify_weights_against_model,        # certify a checkpoint vs a model
    to_sarif, to_lsp_diagnostics, to_github_annotations,  # output adapters
)
```

The full verifier surface (`verify_einops`, `verify_multihead_attention`,
`verify_checkpoint_state_dict`, `verify_lora_adapter_compatibility`,
`verify_optimizer_state`, `verify_quantization`, `verify_sparse_*`, `verify_linalg_*`,
`verify_func_*`, and many more) is exported from the top-level `tensorguard` package.

---

## Learn more

- [`GETTING_STARTED.md`](GETTING_STARTED.md) — first-run walkthrough
- [`docs/symexec/engine.md`](docs/symexec/engine.md) — the symbolic engine, modes, and bug taxonomy
- [`SOUNDNESS_CONTRACT.md`](SOUNDNESS_CONTRACT.md) — exactly what is and isn't guaranteed
- [`LIMITATIONS.md`](LIMITATIONS.md) — where TensorGuard abstains and why
- [`API.md`](API.md) / [`README_REFERENCE.md`](README_REFERENCE.md) — the complete, exhaustive reference
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — add an operator, write a Lean proof, ship a stub

## License

MIT — see [`LICENSE`](LICENSE).
