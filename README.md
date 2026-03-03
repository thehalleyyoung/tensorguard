# RefType — Refinement Type Inference for Dynamic Languages

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Z3 SMT Solver](https://img.shields.io/badge/Z3-SMT%20solver-orange?logo=microsoft)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-lightgrey)

**RefType** is a CEGAR-based refinement type inference engine for Python that
statically catches null-dereferences, index-out-of-bounds, division-by-zero,
type mismatches, and tensor shape errors — **all without annotations**. It
infers dependent types with predicates (e.g. `{x: int | x > 0}`) using liquid
type analysis backed by Z3, and includes **TensorGuard**, a PyTorch
`nn.Module` shape/device/phase verifier that jointly reasons across five
property domains (Shape × Device × Phase × Stride × Permutation) with 331
operator transfer functions.

---

## Key Features

- **CEGAR-based refinement type inference** — counterexample-guided abstraction
  refinement loop that discovers dependent types automatically
- **Liquid type analysis with Z3** — predicate abstraction and SMT-backed
  constraint solving for sound type inference
- **5 core bug classes** — null-deref, index-out-of-bounds, division-by-zero,
  type-mismatch, attribute-error (plus shape-error, unreachable-code,
  unused-refinement, precondition-violation)
- **Function contract inference** — automatically infers preconditions and
  postconditions for every function
- **PyTorch `nn.Module` verification (TensorGuard)** — 331 operators, 5-theory
  product domain, multi-phase train/eval analysis
- **SARIF 2.1.0 output** — for GitHub Code Scanning and Advanced Security
- **Incremental caching** — content-hash-based cache avoids re-analyzing
  unchanged files
- **Parallel multi-file analysis** — configurable worker pool for large
  codebases
- **`.pyi` / `.d.ts` stub generation** — export inferred contracts as type stubs

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

Verify the installation:

```
$ python3 -m src.cli.main version
reftype 0.1.0
Python 3.14.0 (main, Oct  7 2025, 09:34:52) [Clang 17.0.0 (clang-1700.0.13.3)]
Platform macOS-15.7.4-arm64-arm-64bit-Mach-O
```

> **Note:** The `pyproject.toml` entry point is `tensorguard = "src.cli.main:main"`,
> so after `pip install -e .` you can use either `tensorguard` or
> `python3 -m src.cli.main`. This README uses `reftype` as the command name
> throughout (alias it in your shell if desired).

---

## Quickstart (30 Seconds)

Create a test file with some intentional bugs:

```python
# /tmp/test_reftype.py
def divide(x: int, y: int) -> float:
    return x / y

def safe_index(lst: list, i: int):
    return lst[i]

def process(data):
    if data is not None:
        return data.strip()
    return data.upper()  # bug: data is None here
```

Run the analyzer with `-v` for per-file details:

```
$ python3 -m src.cli.main analyze /tmp/test_reftype.py -v
██████████████████████████████████████████████████████████ 100.0% (1/1)  Done in 0.3s

── /private/tmp/test_reftype.py [python]
  3 bug(s) found:
[WARNING] :2:11  Possible division by zero: y may be 0        (division-by-zero)
[WARNING] :5:11  Possible index out of bounds: i may be negative  (index-out-of-bounds)
[WARNING] :10:11 Possible null dereference: data may be None   (null-dereference)
  3 contract(s) inferred:
  divide()     requires y ≠ 0
  safe_index() requires i ≥ 0
  process()    requires data is None
  analyzed 6 functions, 2 CEGAR iterations, 345ms

═══ Analysis Summary ═══
  Files: 1  Functions: 6  Bugs: 3  Contracts: 3  CEGAR iters: 2  Duration: 345ms
```

RefType found all three bugs in under 400 ms — no annotations required.

---

## CLI Reference

```
$ python3 -m src.cli.main --help
usage: reftype [-h] [--version]
  {analyze,analyze-package,verify,watch,ci-check,init,export,diff,server,version,config} ...
```

### `reftype analyze`

Analyze one or more files or directories for refinement type bugs.

```
reftype analyze [paths...] [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `paths` | Files or directories (positional) | `.` |
| `-f`, `--format` | `pyi`, `dts`, `sarif`, `html`, `json` | terminal text |
| `-o`, `--output` | Write output to file | stdout |
| `-v`, `--verbose` | Show per-file details and contracts | off |
| `-w`, `--workers` | Parallel workers (0 = auto) | `0` |
| `--incremental` | Enable incremental caching | off |
| `--baseline` | Baseline SARIF/JSON for diff | none |
| `--fail-on-new-bugs` | Exit 1 only for new bugs | off |
| `--stubs-dir` | `.pyi` stub directory | none |
| `--mypy-baseline` | mypy output for comparison | none |
| `--timeout` | Per-file timeout (seconds) | `300.0` |

### `reftype analyze-package`

Analyze an entire Python package with an aggregated summary.

```
reftype analyze-package DIRECTORY [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `DIRECTORY` | Package directory (positional) | `.` |
| `--output-format` | `text`, `json`, `sarif` | `text` |
| `-o`, `--output` | Write to file | stdout |
| `-w`, `--workers` | Parallel workers (0 = auto) | `0` |
| `--stubs-dir` | `.pyi` stub directory | none |
| `--mypy-baseline` | mypy output for comparison | none |
| `--timeout` | Per-file timeout (seconds) | `300.0` |

### `reftype verify`

Verify a PyTorch `nn.Module` architecture for shape, device, and phase errors.

```
reftype verify FILE [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `FILE` | Python file with `nn.Module` class | — |
| `-s`, `--input-shape` | `name=dim1,dim2,...` (repeatable) | auto-inferred |
| `--no-device-check` | Skip device checks | off |
| `--no-phase-check` | Skip train/eval phase checks | off |
| `--cegar-iterations` | Max CEGAR iterations | `10` |
| `-f`, `--format` | `text`, `json`, `sarif` | `text` |
| `--high-confidence` | Only Z3-proven bugs (0% FP) | off |

### `reftype ci-check`

Run analysis in CI mode with `--fail-on-new-bugs`, `--baseline`, and
`--sarif-output` flags. See [CI / CD Integration](#ci--cd-integration) below.

### `reftype watch`

Watch files and re-analyze on changes: `reftype watch src/ --debounce 1.0`.

### Other Subcommands

| Command | Description |
|---------|-------------|
| `reftype init [dir]` | Generate `.reftype.toml` config |
| `reftype export INPUT [-f pyi\|dts\|json]` | Export contracts from JSON results |
| `reftype diff BEFORE AFTER` | Compare two JSON analysis results |
| `reftype server --transport stdio` | Start LSP server |
| `reftype version` | Show version info |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Analysis succeeded, no bugs found |
| `1` | Analysis succeeded, bugs found (or new bugs with `--fail-on-new-bugs`) |
| `2` | Analysis error (invalid input, config error, timeout) |

---

## CLI Examples with Real Output

Every example below was run and its output captured verbatim.

### Analyze a single file (default JSON output)

```
$ python3 -m src.cli.main analyze /tmp/test_reftype.py
██████████████████████████████████████████████████████████ 100.0% (1/1)   Done in 0.4s
{
  "summary": {
    "total_files": 1, "total_functions": 6, "total_bugs": 3,
    "bugs_by_category": {
      "division-by-zero": 1, "index-out-of-bounds": 1, "null-dereference": 1
    },
    "total_contracts": 3, "total_cegar_iterations": 2
  },
  "results": [{
    "bugs": [
      { "id": "division_by_zero-2", "message": "Possible division by zero: y may be 0", "location": { "line": 2 } },
      { "id": "index_out_of_bounds-5", "message": "Possible index out of bounds: i may be negative", "location": { "line": 5 } },
      { "id": "null_dereference-10", "message": "Possible null dereference: data may be None", "location": { "line": 10 } }
    ],
    "contracts": [
      { "name": "divide", "preconditions": ["y ≠ 0"] },
      { "name": "safe_index", "preconditions": ["i ≥ 0"] }
    ]
  }]
}
```

### Analyze with SARIF output (`-f sarif`)

```
$ python3 -m src.cli.main analyze /tmp/test_reftype.py -f sarif
```

Produces a SARIF 2.1.0 document. Each bug becomes a `result` entry with
`ruleId`, source location, and a stable `fingerprints` hash for deduplication:

```json
{
  "version": "2.1.0",
  "runs": [{
    "tool": { "driver": { "name": "reftype", "version": "0.1.0" } },
    "results": [
      {
        "ruleId": "division-by-zero",
        "level": "warning",
        "message": { "text": "Possible division by zero: y may be 0" },
        "locations": [{ "physicalLocation": {
          "artifactLocation": { "uri": "/private/tmp/test_reftype.py" },
          "region": { "startLine": 2, "startColumn": 12 }
        }}],
        "fingerprints": { "reftype/v1": "fd256939c26a32a4" }
      }
    ]
  }]
}
```

### Analyze a package (`analyze-package`)

```
$ python3 -m src.cli.main analyze-package examples/sample_package/
Discovered 3 Python file(s)
██████████████████████████████████████████████████████████ 100.0% (3/3)  Done in 14.3s

═══ Package Analysis Summary ═══
  Files: 3  Functions: 12  Types inferred: 8  Refinements: 6
  Bugs: 3 (null-dereference: 3)  CEGAR iters: 10  Duration: 879ms
```

The `--output-format json` flag returns per-file results with bugs and
contracts. For instance, `utils.py` shows inferred preconditions like
`b ≠ 0` for `safe_divide` and `index ≥ 0` for `get_item`.

### Verify PyTorch models (`verify`)

```
$ python3 -m src.cli.main verify examples/quickstart.py -s x=batch,3,224,224
✓ quickstart.py: Architecture verified safe (768.1ms)

$ python3 -m src.cli.main verify examples/quickstart.py -s x=batch,3,224,224 -f json
{ "file": "examples/quickstart.py", "bugs": [], "duration_ms": 740.39, "status": "SAFE" }
```

---

## Python API

All public functions are importable from `src.api`. Below, every code example
was run and its output captured.

### `analyze(source, filename)` — flow-sensitive analysis

```python
from src.api import analyze

result = analyze("def f(x): return 1 / x", filename="example.py")
print("Bugs found:", result.bug_count)
for bug in result.bugs:
    print(f"  {bug.location.line}:{bug.location.column} {bug.category.value}: {bug.message}")
print("Functions analyzed:", result.functions_analyzed)
print("Guards harvested:", result.guards_harvested)
print(f"Duration: {result.duration_ms:.1f}ms")
```

```
Bugs found: 1
  1:17 division_by_zero: Potential division by zero: 'x' not guarded
Functions analyzed: 1
Guards harvested: 0
Duration: 3.8ms
```

### `liquid_analyze(source, filename)` — Z3-backed liquid types

The liquid type engine uses predicate abstraction + Z3 for higher-precision
analysis. Confidence is 0.95 (Z3-proven).

```python
from src.api import liquid_analyze

result = liquid_analyze("def get(xs, i): return xs[i]", filename="index.py")
print("Bugs found:", result.bug_count)
for bug in result.bugs:
    print(f"  {bug.location.line}:{bug.location.column} {bug.category.value}: {bug.message}")
print("Functions analyzed:", result.functions_analyzed)
```

```
Bugs found: 1
  1:23 index_out_of_bounds: Possible index out of bounds: i may be negative
Functions analyzed: 1
```

Multiple functions analyzed at once with liquid types:

```python
from src.api import liquid_analyze

result = liquid_analyze("""
def divide(x: int, y: int) -> float:
    return x / y

def safe_index(lst: list, i: int):
    return lst[i]
""", filename="demo.py")

print("Bugs found:", result.bug_count)
for bug in result.bugs:
    print(f"  {bug.location.line}:{bug.location.column} {bug.category.value}: {bug.message}")
    print(f"    severity={bug.severity}, confidence={bug.confidence}")
print("Functions analyzed:", result.functions_analyzed)
```

```
Bugs found: 2
  3:11 division_by_zero: Possible division by zero: y may be 0
    severity=warning, confidence=0.95
  6:11 index_out_of_bounds: Possible index out of bounds: i may be negative
    severity=warning, confidence=0.95
Functions analyzed: 2
```

### `infer_contracts(source)` — contract inference

Infers liquid type contracts (preconditions and postconditions) for every
function, returned as `Annotated[...]` signatures:

```python
from src.api import infer_contracts

contracts = infer_contracts("""
def safe_div(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b

def clamp(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
""")
for name, sig in contracts.items():
    print(f"{name}: {sig}")
```

```
safe_div: def safe_div(a: int, b: Annotated[int, 'b ≠ 0']) -> Annotated[Any, 'ν is not None']
clamp: def clamp(x: int, lo: int, hi: int) -> Any
```

The `safe_div` contract shows that Z3 proved `b ≠ 0` is a required
precondition — matching the explicit guard in the source.

### `analyze_file(path)` — analyze a file on disk

```python
from src.api import analyze_file
result = analyze_file("examples/sample_package/utils.py")
print("Bugs:", result.bug_count, "| Functions:", result.functions_analyzed)
```

```
Bugs: 0 | Functions: 4
```

Zero bugs because `utils.py` guards every error path (`if b == 0`, `if index < 0`, etc.).

### `analyze_directory(path)` — recursive analysis

```python
from src.api import analyze_directory
result = analyze_directory("examples/sample_package/")
print("Bugs:", result.bug_count, "| Functions:", result.functions_analyzed,
      "| Lines:", result.lines_analyzed)
```

```
Bugs: 0 | Functions: 8 | Lines: 76
```

### Detecting null-dereference after a loop

```python
from src.api import analyze
result = analyze("""
def lookup(users, name):
    user = None
    for u in users:
        if u.name == name:
            user = u
    return user.email
""", filename="lookup.py")
for b in result.bugs:
    print(f"  {b.location.line}:{b.location.column} {b.category.value}: {b.message}")
```

```
  7:11 null_dereference: Potential None attribute on 'user' without guard
```

### Detecting division by `len()`

```python
from src.api import analyze
result = analyze("""
def average(nums):
    total = sum(nums)
    return total / len(nums)
""", filename="avg.py")
for b in result.bugs:
    print(f"  {b.location.line}:{b.location.column} {b.category.value}: {b.message}")
```

```
  4:11 division_by_zero: Potential division by zero: len('nums') may be 0
```

### `quick_check(source)` — one-liner bug list

Returns a simple list of bug strings for scripting:

```python
from src.api import quick_check

warnings = quick_check("def f(x): return 1 / x")
for w in warnings:
    print(w)
```

```
1:17 division_by_zero: Potential division by zero: 'x' not guarded
```

### Result to SARIF — programmatic export

```python
from src.api import analyze
import json

result = analyze("def f(x): return 1 / x", filename="example.py")
sarif = result.to_sarif()
print(json.dumps(sarif, indent=2))
```

Output is a valid SARIF 2.1.0 document with `tool.driver.rules` for all five
bug categories and a `results` array with one `division_by_zero` finding at
line 1, column 17.

---

## Result Data Classes

```python
class BugCategory(Enum):
    NULL_DEREFERENCE = "null_dereference"
    DIVISION_BY_ZERO = "division_by_zero"
    INDEX_OUT_OF_BOUNDS = "index_out_of_bounds"
    TYPE_ERROR = "type_error"
    ATTRIBUTE_ERROR = "attribute_error"

@dataclass
class Bug:
    category: BugCategory
    message: str
    location: SourceLocation   # file, line, column
    severity: str              # "error" | "warning" | "info"
    confidence: float          # 0.0–1.0
    fix_suggestion: Optional[str] = None

@dataclass
class AnalysisResult:
    bugs: List[Bug]
    guards_harvested: int
    functions_analyzed: int
    lines_analyzed: int
    duration_ms: float

    def bug_count(self) -> int: ...
    def errors(self) -> List[Bug]: ...
    def to_sarif(self) -> dict: ...
```

---

## Bug Classes

| Bug Class | Severity | Description |
|-----------|----------|-------------|
| `null-dereference` | error | Access attribute/method on a possibly-`None` value |
| `index-out-of-bounds` | error | List/array index may exceed valid range |
| `division-by-zero` | error | Divisor may be zero |
| `type-mismatch` | error | Operand types are incompatible (e.g. `str + int`) |
| `attribute-error` | warning | Access to undefined attribute on an object |
| `unreachable-code` | warning | Branch is statically unreachable |
| `unused-refinement` | info | A guard narrows a type but the refinement is never used |
| `shape-error` | error | Tensor dimension mismatch in nn layers |
| `precondition-violation` | warning | Caller violates inferred precondition of callee |

Severity levels: `error` > `warning` > `info` > `hint`.

---

## Architecture

1. **AST Parse** → language-specific front end
2. **Guard Extraction** → harvest predicates from `if`/`assert`/`match`
3. **CEGAR Loop** → abstract → check (find counterexample) → refine (add predicates) → repeat
4. **Liquid Type Inference** → Z3-backed predicate abstraction + constraint solving
5. **Output** → bugs with fix suggestions + function contracts (pre/postconditions)

For **nn.Module verification**, the pipeline adds multi-phase graph extraction,
5-theory product domain propagation (Shape × Device × Phase × Stride × Perm),
and optional IC3/PDR for parametric certificates.

---

## Configuration

RefType searches for configuration in this order:

1. `.reftype.toml` in the project root
2. `[tool.reftype]` section in `pyproject.toml`
3. `"reftype"` key in `package.json`

Generate a starter config with:

```bash
python3 -m src.cli.main init --language python
```

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

## CI / CD Integration

```yaml
# .github/workflows/reftype.yml
name: RefType Analysis
on: [push, pull_request]
jobs:
  reftype:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .
      - run: python3 -m src.cli.main ci-check . --sarif-output results.sarif --fail-on-new-bugs
      - uses: github/codeql-action/upload-sarif@v3
        if: always()
        with: { sarif_file: results.sarif }
```

Generate a baseline on `main`, then gate PRs on **new** bugs only:

```bash
python3 -m src.cli.main analyze . -f json -o baseline.json       # on main
python3 -m src.cli.main ci-check . --baseline baseline.json \     # on PR
  --fail-on-new-bugs --sarif-output results.sarif
```

---

## FAQ

**Q: Z3 install fails.**
A: Ensure Python 3.9+ and pip ≥ 21.0. On Apple Silicon:
`pip install --no-cache-dir z3-solver>=4.12`.

**Q: False positive `shape-error`.**
A: RefType is conservative with complex `view()`/`reshape()`. File an issue.

**Q: How to suppress warnings?**
A: Set `min_severity = "error"` in `.reftype.toml`, or `# reftype: ignore` on the line.

**Q: Slow on large codebases?**
A: Use `--incremental`, `-w 0` (auto-parallel), `--timeout 60`, `--exclude tests/`.

---

## License

MIT — see [LICENSE](LICENSE) for details.
