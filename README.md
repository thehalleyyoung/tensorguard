# RefType — Refinement Type Inference for Dynamic Languages

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Z3 SMT Solver](https://img.shields.io/badge/Z3-SMT%20solver-orange?logo=microsoft)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-lightgrey)

**RefType** is a CEGAR-based refinement type inference engine for Python (and TypeScript) that statically catches null-deref, index-out-of-bounds, division-by-zero, type mismatches, unreachable code, unused refinements, and tensor shape errors — all without annotations. It infers dependent types with predicates (e.g. `{x: int | x > 0}`) using liquid type analysis backed by Z3, and includes **TensorGuard**, a PyTorch `nn.Module` shape/device/phase verifier that jointly reasons across five property domains (Shape × Device × Phase × Stride × Permutation) with 331 operator transfer functions.

---

## Key Features

- **CEGAR-based refinement type inference** — counterexample-guided abstraction refinement loop that discovers dependent types automatically
- **Liquid type analysis with Z3** — predicate abstraction and SMT-backed constraint solving for sound type inference
- **9 bug classes** — null-deref, index-out-of-bounds, division-by-zero, type-mismatch, unreachable-code, unused-refinement, shape-error, attribute-error, precondition-violation
- **Function contract inference** — automatically infers preconditions and postconditions for every function
- **PyTorch `nn.Module` verification (TensorGuard)** — 331 operators, 5-theory product domain (Shape × Device × Phase × Stride × Permutation), multi-phase train/eval analysis
- **SARIF output** — SARIF 2.1.0 for GitHub Code Scanning and Advanced Security integration
- **Incremental caching** — content-hash-based cache avoids re-analyzing unchanged files
- **Parallel multi-file analysis** — configurable worker pool for large codebases
- **`.pyi` / `.d.ts` stub generation** — export inferred contracts as Python type stubs or TypeScript declarations
- **Plugin system** — extensible analysis hooks for custom bug classes and domain-specific checks

---

## Installation

```bash
git clone <repo-url>
cd refinement-type-inference-dynamic-lang
pip install -e .
```

The only required dependency is `z3-solver>=4.12` (installed automatically).

**Optional development dependencies:**

```bash
pip install -e ".[dev]"   # adds pytest, mypy
```

> **Note:** The `pyproject.toml` entry point is `tensorguard = "src.cli.main:main"`, but the
> CLI program name used throughout is **`reftype`** with subcommands.

---

## Quickstart (30 seconds)

### CLI — analyze a project

```bash
reftype analyze my_project/
```

```
my_project/utils.py:17:8  division-by-zero  [error]
  Division by `n` which may be zero.
  Fix: add a guard `if n != 0` before the division.

my_project/data.py:42:4  index-out-of-bounds  [error]
  Index `i` into list `items` may exceed len(items)-1.
  Fix: clamp or check `i < len(items)`.

Found 2 bugs in 14 functions (387 lines) — 0.8s
```

### Python API — analyze source code

```python
from src.api import analyze, liquid_analyze

# Flow-sensitive analysis
result = analyze("def f(x): return 1 / x", filename="example.py")
for bug in result.bugs:
    print(bug)  # division-by-zero at line 1

# Z3-backed liquid type analysis (more precise)
result = liquid_analyze("def g(xs, i): return xs[i]", filename="index.py")
for bug in result.bugs:
    print(bug)  # index-out-of-bounds at line 1
```

---

## Full CLI Reference

The CLI is invoked as `reftype <subcommand>`. All subcommands accept `-h` for help.

### `reftype analyze`

Analyze one or more files or directories for refinement type bugs.

```
reftype analyze [paths...] [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `paths` | Files or directories to analyze (positional) | `.` |
| `-l`, `--language` | Language: `python`, `typescript`, or `auto` | `auto` |
| `-f`, `--format` | Output format: `pyi`, `dts`, `sarif`, `html`, `json` | terminal text |
| `-o`, `--output` | Write output to file instead of stdout | stdout |
| `-v`, `--verbose` | Increase verbosity (repeat for more: `-vv`) | off |
| `-c`, `--config` | Path to config file | auto-detected |
| `--include` | Glob patterns for files to include | `**/*.py` |
| `--exclude` | Glob patterns for files to exclude | `__pycache__/**`, `.venv/**` |
| `--max-functions` | Maximum functions to analyze per file | unlimited |
| `--timeout` | Per-file timeout in seconds | `300.0` |
| `-w`, `--workers` | Parallel workers (0 = auto-detect CPU count) | `0` |
| `--incremental` | Enable incremental caching | off |
| `--baseline` | Baseline SARIF/JSON file for diff | none |
| `--no-color` | Disable colored terminal output | off |
| `--fail-on-new-bugs` | Exit 1 only for bugs not in baseline | off |
| `--stubs-dir` | Directory of `.pyi` stub files to use as known types | none |
| `--mypy-baseline` | mypy output file (text or JSON) for baseline comparison | none |
| `--pyright-baseline` | pyright JSON output file for baseline comparison | none |

**Example:**

```bash
reftype analyze src/ tests/ \
  -l python \
  -f sarif \
  -o results.sarif \
  -w 4 \
  --incremental \
  --exclude "tests/fixtures/**"
```

### `reftype analyze-package`

Analyze an entire Python package with an aggregated summary.

```
reftype analyze-package DIRECTORY [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `DIRECTORY` | Package directory to analyze (positional, required) | — |
| `--requirements` | `requirements.txt` or `pyproject.toml` for type stub awareness | none |
| `--output-format` | Output format: `text`, `json`, `sarif` | `text` |
| `-o`, `--output` | Write output to file | stdout |
| `--include` | Glob patterns for files to include | `**/*.py` |
| `--exclude` | Glob patterns for files to exclude | standard excludes |
| `-w`, `--workers` | Parallel workers (0 = auto-detect) | `0` |
| `--timeout` | Per-file timeout in seconds | `300.0` |
| `-v`, `--verbose` | Show per-file details | off |
| `-c`, `--config` | Path to config file | auto-detected |
| `--stubs-dir` | Directory of `.pyi` stub files to use as known types | none |
| `--mypy-baseline` | mypy output file (text or JSON) for baseline comparison | none |
| `--pyright-baseline` | pyright JSON output file for baseline comparison | none |

**Example:**

```bash
reftype analyze-package my_project/ \
  --requirements requirements.txt \
  --output-format sarif \
  -o report.sarif
```

**Example with stubs and mypy baseline:**

```bash
# Compare refinement type results against mypy's findings
mypy my_project/ --no-error-summary > mypy_output.txt
reftype analyze-package my_project/ \
  --stubs-dir my_project/stubs \
  --mypy-baseline mypy_output.txt
```

```
── mypy baseline comparison (12 issues) ──
  Checker errors confirmed   : 8
  Checker false positives     : 4
  New issues (refinement type): 3
```

### `reftype verify`

Verify a PyTorch `nn.Module` architecture across all five property domains.

```
reftype verify FILE [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `FILE` | Python file containing an `nn.Module` class (positional) | — |
| `-s`, `--input-shape` | Input tensor shapes: `name=dim1,dim2,...` (repeatable) | auto-inferred |
| `--no-device-check` | Skip device consistency checks | off |
| `--no-phase-check` | Skip train/eval phase checks | off |
| `--cegar-iterations` | Maximum CEGAR refinement iterations | `10` |
| `-f`, `--format` | Output format: `text`, `json`, `sarif` | `text` |
| `--high-confidence` | Only report high-confidence bugs | off |

**Example:**

```bash
reftype verify model.py -s x=batch,3,224,224 --format json
```

### `reftype watch`

Watch files for changes and re-analyze incrementally.

```
reftype watch [paths...] [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `paths` | Files or directories to watch (positional) | `.` |
| `-l`, `--language` | Language: `python`, `typescript`, `auto` | `auto` |
| `--debounce` | Debounce interval in seconds | `0.5` |
| `--editor` | Editor integration: `vim`, `emacs`, `vscode` | none |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Analysis succeeded, no bugs found |
| `1` | Analysis succeeded, bugs found (or new bugs when `--fail-on-new-bugs`) |
| `2` | Analysis error (invalid input, config error, timeout, etc.) |

---

## Supported Input Formats

| Input | Description | Detection |
|-------|-------------|-----------|
| Python files (`.py`) | Single-file or multi-file analysis | Auto or `-l python` |
| TypeScript files (`.ts`, `.tsx`) | Single-file or multi-file analysis | Auto or `-l typescript` |
| Individual files | Analyze a specific file | Pass file path |
| Directories / packages | Recursive discovery of source files | Pass directory path |
| `.reftype.toml` | Primary config file | Auto-discovered in project root |
| `pyproject.toml` `[tool.reftype]` | Config section in standard Python config | Auto-discovered |
| `package.json` `"reftype"` key | Config in Node.js project manifest | Auto-discovered |
| `requirements.txt` | Dependency list for type stub awareness | Via `--requirements` flag |

**Config file search order:** `.reftype.toml` → `pyproject.toml [tool.reftype]` → `package.json "reftype"`.

**Example `.reftype.toml`:**

```toml
[reftype]
language = "python"
include = ["**/*.py"]
exclude = ["__pycache__/**", ".venv/**"]
min_severity = "warning"
timeout = 300.0

[reftype.cegar]
max_iterations = 50
interpolation_enabled = true

[reftype.incremental]
enabled = true
cache_dir = ".reftype-cache"
```

---

## Python API Overview

All public functions are available from `src.api`. The nn.Module verifier lives in
`src.model_checker`.

### `analyze(source, filename)` — basic flow-sensitive analysis

```python
from src.api import analyze

result = analyze(
    source='def f(x): return 1 / x',
    filename='example.py',
)
print(result.bugs)         # [Bug(class='division-by-zero', line=1, ...)]
print(result.functions_analyzed)  # 1
```

### `liquid_analyze(source, filename)` — Z3-backed liquid type analysis

```python
from src.api import liquid_analyze

result = liquid_analyze(
    source='def get(xs, i): return xs[i]',
    filename='index.py',
)
# More precise: uses predicate abstraction + Z3 to infer {i: int | 0 <= i < len(xs)}
```

### `analyze_shapes(source)` — tensor shape analysis

```python
from src.api import analyze_shapes

result = analyze_shapes("import torch; x = torch.randn(4, 3); y = x @ torch.randn(3, 5)")
print(result.shapes)  # inferred tensor shapes at each program point
```

### `verify_model(source, input_shapes)` — nn.Module verification

```python
from src.model_checker import verify_model

result = verify_model(
    source=open("resnet.py").read(),
    input_shapes={"x": ("batch", 3, 224, 224)},
)
if result.safe:
    print(result.certificate.pretty())     # Z3-backed safety certificate
else:
    print(result.counterexample.pretty())  # concrete failing dimensions
```

### `verify_module(module, input_shapes)` — live module verification via FX/Dynamo

```python
from src.model_checker import verify_module

import torch.nn as nn
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
result = verify_module(model, input_shapes={"x": ("batch", 128)})
print(result.safe)  # True
```

### Result Data Classes

```python
@dataclass
class AnalysisResult:
    bugs: List[Bug]
    guards_harvested: int
    functions_analyzed: int
    lines_analyzed: int
    duration_ms: float

@dataclass
class Bug:
    bug_class: str        # e.g. "null-deref", "division-by-zero"
    message: str
    filename: str
    line: int
    column: int
    severity: str         # "error", "warning", "info", "hint"
    fix_suggestion: str

@dataclass
class FunctionContract:
    name: str
    preconditions: List[str]   # e.g. ["{x: int | x > 0}"]
    postconditions: List[str]  # e.g. ["{return: int | return >= 0}"]
    refinement_type: str       # full refinement type signature
```

---

## Architecture Overview

```
 Source (.py / .ts)
        │
        ▼
 ┌─────────────────┐
 │   AST Parse      │   Language-specific front end
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ Guard Extraction │   Harvest path predicates from if/assert/match
 └────────┬────────┘
          ▼
 ┌──────────────────────────────────────────────────┐
 │              CEGAR Loop                          │
 │  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
 │  │ Abstract  │→ │  Check   │→ │   Refine      │  │
 │  │ (initial  │  │ (find    │  │ (add new      │  │
 │  │  types)   │  │  cex)    │  │  predicates)  │  │
 │  └──────────┘  └──────────┘  └───────────────┘  │
 │       ↑                            │             │
 │       └────────────────────────────┘  (iterate)  │
 └─────────────────────┬────────────────────────────┘
                       ▼
 ┌──────────────────────────────────────────────────┐
 │  Liquid Type Inference → Z3 Verification        │
 │  Predicate abstraction + SMT constraint solving  │
 └─────────────────────┬────────────────────────────┘
                  ┌────┴────┐
                  ▼         ▼
           Bugs found    Contracts
           (with fix     (pre/post
            suggestions)  conditions)
```

For **nn.Module verification**, the pipeline additionally includes multi-phase graph
extraction (`if self.training:` branches), 5-theory product domain propagation
(T_shape × T_device × T_phase × T_stride × T_perm), and optional IC3/PDR for
parametric certificates valid for all input sizes.

---

## Bug Classes

| Bug Class | Severity | Description | Fix Suggestion |
|-----------|----------|-------------|----------------|
| `null-deref` | error | Access attribute or call method on a possibly-`None` value | Add an `if x is not None` guard |
| `index-out-of-bounds` | error | List/array index may exceed valid range | Clamp index or check `i < len(xs)` |
| `division-by-zero` | error | Divisor may be zero | Add `if d != 0` guard before division |
| `type-mismatch` | error | Operand types are incompatible (e.g. `str + int`) | Cast or convert to matching type |
| `unreachable-code` | warning | Branch is statically unreachable given inferred refinements | Remove dead code or fix condition |
| `unused-refinement` | info | A guard narrows a type but the refinement is never used | Remove unnecessary guard or use the value |
| `shape-error` | error | Tensor dimension mismatch in matrix ops or nn layers | Fix layer dimensions or reshape input |
| `attribute-error` | error | Access to undefined attribute on an object | Check attribute exists or fix name |
| `precondition-violation` | warning | Caller does not satisfy inferred precondition of callee | Ensure arguments meet the function contract |

Severity levels: `error` > `warning` > `info` > `hint`. Filter with `--min-severity` in
config or by checking `bug.severity` in the API.

---

## Examples

### Analyze a single file

```bash
reftype analyze utils.py -v
```

```
utils.py:12:4  null-deref  [error]
  `user` may be None when accessed at `.name`.
  Fix: add `if user is not None` before access.

utils.py:31:17  division-by-zero  [warning]
  `count` may be zero in expression `total / count`.
  Fix: guard with `if count != 0`.

Analyzed 5 functions, inferred 3 contracts — 0.4s
```

### Analyze a package with SARIF output

```bash
reftype analyze-package src/ \
  --requirements requirements.txt \
  --output-format sarif \
  -o results.sarif

# Upload to GitHub Code Scanning:
# gh api repos/{owner}/{repo}/code-scanning/sarifs \
#   -f "sarif=@results.sarif" -f "ref=refs/heads/main"
```

### Verify a PyTorch model

```python
from src.model_checker import verify_model

source = """
import torch, torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(512, 256)
        self.eval_head = nn.Linear(128, 10)   # BUG: expects 128, gets 256

    def forward(self, x):
        h = self.backbone(x)
        return self.eval_head(h)              # shape mismatch: 256 != 128
"""

result = verify_model(source, input_shapes={"x": ("batch", 512)})
print(result.safe)    # False
print(result.pretty())
# ✗ Model is UNSAFE — eval_head expects in_features=128 but receives 256
```

### Use incremental caching

```bash
# First run: full analysis, populates cache
reftype analyze src/ --incremental
# Found 3 bugs in 42 files — 4.2s

# Second run: only re-analyzes changed files
reftype analyze src/ --incremental
# Found 1 bug in 2 files (40 cached) — 0.3s
```

---

## FAQ / Troubleshooting

**Q: Z3 is not found or `z3-solver` fails to install.**
A: Ensure Python 3.9+ and pip ≥ 21.0. On Apple Silicon, try `pip install --no-cache-dir z3-solver>=4.12`.

**Q: The tool reports `shape-error` but my model runs fine at runtime.**
A: RefType is conservative — complex `view()`/`reshape()` with 4+ symbolic dims can cause false positives. File an issue with a minimal reproducer.

**Q: How do I suppress a specific warning?**
A: Set `min_severity = "error"` in `.reftype.toml`, or add `# reftype: ignore` on the line.

**Q: Analysis is slow on a large codebase.**
A: Use `--incremental` for caching, `-w 0` for parallel analysis, `--timeout 60` to cap per-file time, and `--exclude` test directories.

**Q: Can I use RefType in CI?**
A: Yes. Use `reftype analyze-package . --output-format sarif -o results.sarif` and upload to GitHub Code Scanning. Add `--fail-on-new-bugs` with `--baseline` to fail only on regressions.

**Q: What PyTorch operators does TensorGuard support?**
A: 331 operators including `Linear`, `Conv1d/2d/3d`, `BatchNorm`, `MultiheadAttention`, `LSTM`, `GRU`, `Transformer`, `einops`, and all standard activations/pooling. Unsupported operators produce `UNKNOWN`. See `src/typing_rules.py`.

**Q: TypeScript support — how complete is it?**
A: Covers core bug classes (null-deref, index-out-of-bounds, division-by-zero, type-mismatch, unreachable-code). Auto-detected from `.ts`/`.tsx` extensions; use `--format dts` to export `.d.ts` stubs.

---

## License

MIT — see [LICENSE](LICENSE) for details.
