# The Symbolic-Execution Engine (`src.symexec`)

A self-contained, **torch-free** abstract-interpretation engine that finds
shape / rank / device / control-flow defects in PyTorch code *without running
it* — no model is instantiated and no tensor is ever allocated. It interprets a
program over an abstract domain of tensor shapes and integer intervals, proves
when an operation must fail for every concretization, and reports it.

This page is a practical guide. The terse per-symbol API reference lives in
[`API.md`](../../API.md#symbolic-execution-engine-srcsymexec).

---

## Why a second engine?

TensorGuard's primary checker (`src.model_checker`) verifies an annotated
`nn.Module` against declared input shapes. The symbolic-execution engine is
complementary:

- It needs **no annotations and no input shapes** — it runs the file's own
  `if __name__ == "__main__":` demo and analyses every free function and
  `forward`/`__call__` method directly.
- It reasons across **ordinary Python** (loops, branches, comprehensions,
  unpacking, integer arithmetic, containers), not just layer wiring.
- It is **torch-free**: it imports nothing heavy and can be embedded anywhere.

Both engines share the same north star: **soundness by abstention**. Anywhere a
value leaves the modeled fragment, the engine yields `Top` and emits *no*
report. Every finding is therefore a Z3-proved or concretely-forced runtime
failure — **zero false positives by construction** (in the default mode).

---

## Quick start

```python
from src.symexec import analyze_source

src = '''
import torch

def f():
    a = torch.randn(2, 3)
    b = torch.randn(4, 5)
    return a @ b          # 3 != 4: contracted-dim mismatch
'''

result = analyze_source(src, filename="demo.py")
for bug in result.bugs:
    print(f"{bug.line}:{bug.col} [{bug.kind.value}] {bug.message}")
    print("  confidence:", round(bug.confidence, 2))
```

```
7:11 [matmul_dim_mismatch] matmul contracted-dim mismatch: (2, 3) @ (4, 5) (3 vs 4); RuntimeError at runtime
  confidence: 0.99
```

Analyse a file on disk with `analyze_file("demo.py")`.

---

## Understanding a `SymResult`

```python
result = analyze_source(src)

result.bugs                 # the findings (SymBug list)
result.functions_analyzed   # how many top-level functions were analysed
result.ran_main             # did the shipped __main__ harness run?
result.abstentions.summary()# where/why the engine declined to reason
result.fingerprint()        # deterministic SHA-256 reproducibility receipt
print(result.explain("demo.py"))   # full source->...->sink derivation per bug
```

The **abstain ledger** is the key to interpreting a clean result. "No bugs" can
mean *"proved safe"* or *"abstained immediately"* — the ledger tells you which:

```python
print(result.abstentions.summary())
# abstentions: 3 (unknown_rank=1, unknown_shape=2)
```

and the complementary **coverage meter** (`result.coverage`) reports how much of
the program was interpreted with concrete abstract information.

---

## Soundness modes

The same analysis can sit at different points on the precision/recall curve via
`SymConfig`. The three modes have **nesting** report sets:

```
sound  ⊆  balanced  ⊆  heuristic
```

```python
from src.symexec import analyze_source, SymConfig

analyze_source(src, config=SymConfig.balanced())   # default; historic behaviour
analyze_source(src, config=SymConfig.sound())      # max precision (a subset)
analyze_source(src, config=SymConfig.heuristic())  # max recall (a superset)
```

| Mode | Use it for | What changes |
|------|------------|--------------|
| `balanced` *(default)* | Everyday analysis. | Nothing — byte-identical to the historic engine. Every proven finding; nothing filtered. |
| `sound` | A release gate that must never cry wolf. | Keeps a path-conditioned report only when Z3 *positively confirms* the path is satisfiable (drops it on `unknown`/missing solver), and discards weak-prior findings (`confidence < 0.85`). |
| `heuristic` | Exploratory "what might be wrong here?" triage. | Surfaces clearly-labelled, low-confidence *suspicions* at sites where balanced abstains. **May produce false positives.** |

You can override individual knobs:

```python
# Heuristic recall, but still drop anything below 0.6 confidence:
cfg = SymConfig.heuristic(min_confidence=0.6)
```

The default (`balanced`) preserves every reproducibility fingerprint, so adding
`config=` to an existing call without choosing a non-default mode changes
nothing.

### Worked example: a heuristic suspicion

```python
src = '''
import torch
def g(n):
    a = torch.randn(7)     # concrete trailing dim 7
    b = torch.randn(n)     # symbolic, unconstrained
    return a + b           # broadcasts only if n in {1, 7}
'''

from src.symexec import analyze_source, SymConfig

print(analyze_source(src, config=SymConfig.balanced()).bugs)   # []  (sound: abstains)
heur = analyze_source(src, config=SymConfig.heuristic())
print(heur.bugs[0].kind.value, round(heur.bugs[0].confidence, 2))
# broadcast_mismatch 0.53   ("suspected ...; heuristic mode")
```

### Intent checks: "why isn't it training?"

Some mistakes never raise an exception — the script runs to completion but the
model silently fails to learn. In `heuristic` mode the engine surfaces a pack of
**training-loop hygiene** checks (all `severity="warning"`, suppressed in
`sound`/`balanced`):

| Kind | Fires when a loop … | Consequence |
|------|---------------------|-------------|
| `missing_zero_grad` | calls `.backward()` **and** `optimizer.step()` but never `zero_grad()` | gradients accumulate across iterations; every update uses the sum of all prior batches' gradients |
| `step_without_backward` | calls `optimizer.zero_grad()`/`step()` but never `.backward()` | no gradients are computed; parameters never update |
| `backward_without_step` | calls `.backward()` but no `optimizer.step()` | gradients are computed and then never applied |

A related check, `backward_no_grad`, fires when `.backward()` is called on a
tensor that provably does not require grad (e.g. `loss.detach().backward()`):
PyTorch raises *"does not require grad and does not have a grad_fn"* and no
gradients flow. It fires only on positive non-grad provenance, never on tensors
whose grad status is merely unknown.

```python
src = '''
import torch
def train(model, loader, optimizer):
    for x, y in loader:
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()          # forgot optimizer.zero_grad()
'''
from src.symexec import analyze_source, SymConfig
heur = analyze_source(src, config=SymConfig.heuristic())
print(heur.bugs[0].kind.value)    # missing_zero_grad
```

These are intentionally heuristic: a correct loop that places `zero_grad()`,
`backward()` and `step()` together never fires, and a lone `scheduler.step()`
(no gradient signal) is not mistaken for a training step.

---

## Whole-package analysis

Real models span many files. Single-file analysis abstains at the import
boundary; `analyze_package` resolves cross-file imports so a mismatch that only
manifests when one module *uses* another is caught.

```python
from src.symexec import analyze_package

pkg = analyze_package("my_project/")
for path, bug in pkg.all_bugs():
    print(path, bug.line, bug.message)

# Cross-file call graph: "module:symbol" -> ["module:symbol", ...]
for caller, callees in pkg.call_graph().items():
    print(caller, "->", callees)
```

For large projects, the **parallel driver** is a drop-in replacement whose
output is byte-identical to the serial run:

```python
from src.symexec import analyze_package_parallel
pkg = analyze_package_parallel("my_project/", workers=8, backend="process")
```

---

## Incremental re-analysis (editor / watch loops)

Keep an `IncrementalCache` across runs to re-analyse only the files affected by
an edit (the file itself, or any file that directly imports a project symbol
that changed):

```python
from src.symexec import IncrementalCache, analyze_package_incremental

cache = IncrementalCache()
pkg, stats = analyze_package_incremental("proj/", cache)   # cold: all analysed
# ... developer edits proj/layers.py ...
pkg, stats = analyze_package_incremental("proj/", cache)   # warm
print("reused:", len(stats.reused), "reanalyzed:", len(stats.reanalyzed))
```

The reused result is the exact cached object, so reuse can never change a
verdict; the output is byte-identical to a fresh `analyze_package`.

---

## Closing the loop: calibration telemetry (opt-in)

Every report carries a *calibrated* confidence. To check whether those
confidences match reality — and get advice on tuning them — opt in to
telemetry, label outcomes, and read the calibration report:

```python
from src.symexec import TelemetrySink, analyze_source

sink = TelemetrySink().enable()       # OFF by default — nothing is collected
                                      # until you call .enable()

# Label each analysis's findings as confirmed real bugs or false positives:
sink.record_result(analyze_source(true_positive_src), outcome=True)
sink.record_result(analyze_source(false_positive_src), outcome=False)

report = sink.report()
print(report.summary())               # e.g. ECE=0.04, Brier=0.06, kinds=3
for s in report.suggestions:           # advisory per-kind prior deltas
    print(s.kind, "delta", round(s.delta, 3))
```

Three guarantees:

- **Opt-in.** A `TelemetrySink` is disabled until `.enable()`; every ingestion
  method is a no-op otherwise.
- **Anonymous.** A `CalibrationRecord` stores only the bug *kind*, the
  confidence, and the evidence flags — **no file path, message, or location**.
- **Advisory & side-effect-free.** Enabling telemetry never changes which bugs
  report or any fingerprint, and the suggested prior adjustments are *not*
  auto-applied (you apply them by hand to `src/symexec/confidence.py`).

Records round-trip as newline-delimited JSON; the caller owns any file I/O:

```python
from src.symexec import records_to_jsonl, records_from_jsonl
open("calib.jsonl", "w").write(records_to_jsonl(sink.records))
records = records_from_jsonl(open("calib.jsonl").read())
```

---

## Editor integrations (Jupyter & VS Code)

Run the engine inside a Jupyter notebook and get findings attributed back to the
cell that produced them:

```python
from src.symexec import analyze_notebook

res = analyze_notebook("analysis.ipynb")   # path, JSON string, or parsed dict
res                                          # rich HTML table inline in Jupyter
for f in res.findings:
    print(f"cell {f.cell_index}, line {f.cell_line}: {f.bug.message}")
```

Code cells are concatenated into one virtual module (IPython `%magic` / `!shell`
lines are blanked in place so line numbers are preserved) and each finding's
global line is mapped back to `(cell, line)`. Inside a kernel you can also
`%load_ext src.symexec.notebook` to enable a `%%tensorguard` cell magic that
analyses and displays findings for the cell it decorates (`--mode heuristic`,
`--no-run` supported).

For **VS Code** (or any LSP client), build the `publishDiagnostics` notification
that drives the Problems panel and inline squiggles:

```python
from src.symexec import analyze_file, to_publish_diagnostics

note = to_publish_diagnostics(analyze_file("model.py"), "file:///abs/model.py")
# -> {"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{...}}
```

Re-publishing with an empty `diagnostics` list (a clean result) clears the
markers after a fix.

## Exporting findings

```python
result = analyze_source(src, filename="demo.py")
result.to_dict("demo.py")               # stable JSON object
result.to_sarif("demo.py")              # SARIF 2.1.0 for GitHub Code Scanning
result.to_lsp_diagnostics("file:///demo.py")  # editor squiggles
result.to_github_annotations("demo.py")        # CI ::error:: annotations
```

---

## Proof-carrying bug certificates & replay

Every forced-failure report is *refutation sound* — it fires only when a runtime
precondition is violated on operands the abstract state pins to concrete values
(the machine-checked `refute`/`witness` lemmas under
`lean/TensorGuard/Symexec/`). `certify` distils that into a compact, replayable
**certificate**: the bug's identity, the *named runtime precondition* that must
hold, and the concrete witness operands on which it is violated. The
precondition vocabulary mirrors the Lean `Ok` predicates one-for-one
(`dims_equal`, `broadcast_compat`, `numel_match`, `index_in_range`,
`arity_match`, `divisor_nonzero`, `dim_nonneg`, `feature_match`).

```python
result = analyze_source(src, filename="demo.py")
certs = result.certificates("demo.py")     # one BugCertificate per report
text = dumps_certificates(certs)            # deterministic JSON proof artifact

# Independent re-derivation — no engine, just the precondition vocabulary:
for r in replay_text(text):
    print(r.status, r.detail)               # "verified" | "refuted" | "unchecked"
```

`replay` re-evaluates the precondition on the witness without re-running the
analysis: `verified` (precondition violated ⇒ forced failure re-derived),
`refuted` (precondition holds ⇒ a tampered/invalid certificate), or `unchecked`
(claim-only certificate with no recoverable numeric witness). This is the
executable dual of the Lean `refute` lemma.

---

## Soundness contract in one paragraph

A report is emitted only when the engine has a *proof* the operation fails for
every value consistent with the abstract state on the current path: a forced
concrete mismatch, or a Z3-discharged symbolic one that survives the feasibility
gate. Outside the modeled fragment the engine returns `Top` and stays silent.
The `sound` mode tightens this further (positive-feasibility required); the
`heuristic` mode deliberately relaxes it for triage and labels every relaxed
finding as a *suspicion*. The default `balanced` mode is exactly the proven,
zero-false-positive engine.

For the converse direction — *which real bugs the engine is guaranteed to
find* — see the relative-completeness characterization in
[`completeness.md`](completeness.md) (source of truth:
`src/symexec/completeness_contract.py`): within the modeled operator set, a
forced failure is reported whenever the operands the detector depends on are
known (non-`⊤`) and the operation is reached. Completeness and soundness are the
two directions of one "report iff provable on known operands" contract.
