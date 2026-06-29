# TensorGuard API Reference

Static tensor shape verification for PyTorch `nn.Module` architectures using Z3-backed symbolic constraint propagation.

## Installation

```bash
pip install -e .
```

---

## Core Verification (`src.model_checker`)

### `verify_model()`

One-shot verification of an `nn.Module` defined in source code.

```python
from src.model_checker import verify_model

result = verify_model(
    source=open("my_model.py").read(),
    input_shapes={"x": ("batch", 3, 224, 224)},
)
if result.safe:
    print(result.certificate.pretty())
else:
    print(result.counterexample.pretty())
```

```python
def verify_model(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
    high_confidence_only: bool = False,
    verification_mode: str = "bounded",
    symbolic_dims: Optional[Dict[str, str]] = None,
    produce_certificates: bool = False,
    return_kripke: bool = False,
    use_kb_normalization: bool = False,
) -> VerificationResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | *(required)* | Python source code containing an `nn.Module` subclass |
| `input_shapes` | `Dict[str, tuple] \| None` | `None` | Shape tuples; ints for concrete dims, strings for symbolic dims |
| `default_device` | `Device` | `Device.CPU` | Default device for input tensors |
| `default_phase` | `Phase` | `Phase.TRAIN` | `TRAIN` or `EVAL` |
| `max_k` | `int \| None` | `None` | Maximum verification depth (defaults to number of graph steps) |
| `constraints` | `Dict[str, str \| int] \| None` | `None` | Relational constraints between symbolic dims (e.g. `{"embed_dim": "heads * head_dim", "heads": 8}`) |
| `high_confidence_only` | `bool` | `False` | Only report HIGH-confidence (Z3-proven) violations |
| `verification_mode` | `str` | `"bounded"` | `"bounded"` for BMC, `"unbounded"` for IC3/PDR |
| `symbolic_dims` | `Dict[str, str] \| None` | `None` | Only for `verification_mode="unbounded"` |
| `use_kb_normalization` | `bool` | `False` | Apply Knuth–Bendix constraint normalization before solving |
| `produce_certificates` | `bool` | `False` | Generate proof certificates with inference chains |
| `return_kripke` | `bool` | `False` | Include the Kripke structure in the result |

**Returns:** `VerificationResult`

---

### `extract_computation_graph()`

Extract an AST-based computation graph from an `nn.Module` source string.

```python
from src.model_checker import extract_computation_graph

graph = extract_computation_graph(source)
print(graph.pretty())
```

```python
def extract_computation_graph(source: str) -> ComputationGraph
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `str` | Python source code containing an `nn.Module` subclass |

**Returns:** `ComputationGraph`

**Raises:** `ValueError` if no `nn.Module` subclass is found.

> When the source contains multiple `nn.Module` classes, selects the root model (the one not used as a submodule by another class) and inlines submodule calls.

---

### `extract_kripke_structure()`

Extract a formal Kripke structure K = (S, S₀, R, AP, L) from a computation graph.

```python
from src.model_checker import extract_kripke_structure

kripke = extract_kripke_structure(graph, input_shapes={"x": ("batch", 784)})
assert kripke.is_safe()
```

```python
def extract_kripke_structure(
    graph: ComputationGraph,
    input_shapes: Dict[str, Any],
    initial_device: Device = Device.CPU,
    initial_phase: Phase = Phase.EVAL,
) -> KripkeStructure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `ComputationGraph` | *(required)* | Computation graph from `extract_computation_graph()` |
| `input_shapes` | `Dict[str, Any]` | *(required)* | Input shape specification |
| `initial_device` | `Device` | `Device.CPU` | Initial device for input tensors |
| `initial_phase` | `Phase` | `Phase.EVAL` | Initial phase |

**Returns:** `KripkeStructure`

---

### `VerificationResult`

```python
@dataclass
class VerificationResult:
    safe: bool
    certificate: Optional[SafetyCertificate] = None
    counterexample: Optional[CounterexampleTrace] = None
    graph: Optional[ComputationGraph] = None
    errors: List[str] = field(default_factory=list)
    verification_time_ms: float = 0.0
    confidence: Confidence = Confidence.HIGH
    min_confidence_threshold: Confidence = Confidence.LOW
    dynamic_features: Dict[str, Any] = field(default_factory=dict)
    dynamic_feature_warnings: List[str] = field(default_factory=list)
    proof_certificate: Optional[ProofCertificate] = None
    kripke_structure: Optional[KripkeStructure] = None
    unsupported_op_tracker: Optional[UnsupportedOpTracker] = None
```

**Methods:**

- `filter_by_confidence(min_level: Confidence = Confidence.MEDIUM) -> VerificationResult` — Return a copy with violations below the confidence threshold removed.
- `pretty() -> str` — Human-readable summary.

---

### `SafetyCertificate`

```python
@dataclass
class SafetyCertificate:
    model_name: str
    properties: List[str]
    k: int
    symbolic_bindings: Dict[str, str] = field(default_factory=dict)
    checked_steps: int = 0
    verification_time_ms: float = 0.0
    z3_queries: int = 0
    z3_total_time_ms: float = 0.0
    z3_sat_count: int = 0
    z3_unsat_count: int = 0
    theories_used: List[str] = field(default_factory=list)
    product_domains: List[str] = field(default_factory=list)
    proof_certificate: Optional[ProofCertificate] = None
```

**Methods:**

- `smtlib_certificate() -> str` — SMT-LIB 2.6 verification conditions (assertion witnesses checkable by any SMT solver).
- `pretty() -> str` — Human-readable summary.
- `to_dict() -> dict` — JSON-serializable dictionary.

---

### `CounterexampleTrace`

```python
@dataclass
class CounterexampleTrace:
    model_name: str
    violations: List[SafetyViolation] = field(default_factory=list)
    failing_step: int = -1
    states: List[ModelState] = field(default_factory=list)
    concrete_dims: Dict[str, int] = field(default_factory=dict)
```

**Methods:**

- `pretty() -> str` — Human-readable trace with computation path and violations.

---

### `KripkeStructure`

```python
@dataclass
class KripkeStructure:
    states: List[KripkeState]
    initial_state_idx: int
    transitions: List[KripkeTransition]
    atomic_propositions: FrozenSet[str] = frozenset({
        "shape_safe", "device_consistent", "gradient_valid", "phase_correct"
    })
    labeling: Dict[int, FrozenSet[str]] = field(default_factory=dict)
```

**Properties:**

- `num_states -> int`
- `num_transitions -> int`
- `initial_state -> KripkeState`

**Methods:**

- `is_safe() -> bool` — Check universal safety: ∀s ∈ reachable(S₀). `shape_safe` ∈ L(s).
- `get_violation_trace() -> Optional[List[KripkeTransition]]` — BFS trace from S₀ to an unsafe state.
- `to_dict() -> Dict[str, Any]` — JSON-serializable dictionary.

---

### Enums

```python
class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Device(Enum):
    CPU = "cpu"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"
    CUDA_2 = "cuda:2"
    CUDA_3 = "cuda:3"

class Phase(Enum):
    TRAIN = auto()
    EVAL = auto()
```

`Device` also provides `Device.from_string(s: str) -> Device` for parsing device strings.

---

## FX-Based Verification (`src.fx_extractor`)

### `verify_module()`

Verify a live `nn.Module` instance by tracing it with `torch.fx` (or TorchDynamo).

```python
import torch.nn as nn
from src.fx_extractor import verify_module

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
result = verify_module(model, input_shapes={"x": ("batch", 784)})
print(result.safe)  # True
```

```python
def verify_module(
    module: nn.Module,
    input_shapes: Optional[Dict[str, tuple]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
    high_confidence_only: bool = False,
    class_name: Optional[str] = None,
    backend: str = "auto",
) -> VerificationResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `nn.Module` | *(required)* | The model instance to verify |
| `input_shapes` | `Dict[str, tuple] \| None` | `None` | Input shape specification (same format as `verify_model`) |
| `default_device` | `Device` | `Device.CPU` | Default device for input tensors |
| `default_phase` | `Phase` | `Phase.TRAIN` | Default phase |
| `max_k` | `int \| None` | `None` | Maximum verification depth |
| `constraints` | `Dict[str, str \| int] \| None` | `None` | Relational constraints between symbolic dimensions |
| `high_confidence_only` | `bool` | `False` | Only report Z3-proven violations |
| `class_name` | `str \| None` | `None` | Override class name (defaults to module's class name) |
| `backend` | `str` | `"auto"` | Graph capture backend: `"auto"`, `"dynamo"`, or `"fx"` |

**Returns:** `VerificationResult`

> **Backend selection** (`"auto"`): Tries TorchDynamo first (best coverage for dynamic control flow), falls back to `torch.fx.symbolic_trace`, then AST-based extraction.

---

### `fx_trace_to_graph()`

Convert a `torch.fx.GraphModule` to a `ComputationGraph`.

```python
import torch
from src.fx_extractor import fx_trace_to_graph

traced = torch.fx.symbolic_trace(model)
graph = fx_trace_to_graph(traced)
```

```python
def fx_trace_to_graph(
    traced: torch.fx.GraphModule,
    class_name: Optional[str] = None,
) -> ComputationGraph
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traced` | `torch.fx.GraphModule` | *(required)* | Traced graph module |
| `class_name` | `str \| None` | `None` | Override for the class name in the graph |

**Returns:** `ComputationGraph`

---

## Tensor Shape Analysis (`src.tensor_shapes`)

### `TensorShapeAnalyzer`

Lower-level AST-based tensor shape verifier using liquid types and Z3.

```python
from src.tensor_shapes import TensorShapeAnalyzer

analyzer = TensorShapeAnalyzer(timeout_ms=5000)
result = analyzer.analyze_source(source)
for err in result.errors:
    print(f"L{err.line}: {err.message}")
```

```python
class TensorShapeAnalyzer(ast.NodeVisitor):
    def __init__(self, timeout_ms: int = 5000): ...
    def analyze_source(self, source: str) -> ShapeAnalysisResult: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout_ms` | `int` | `5000` | Z3 solver timeout in milliseconds |

#### `analyze_source(source: str) -> ShapeAnalysisResult`

Analyze Python source for tensor shape errors. Performs three passes: class analysis (nn.Module layers), function analysis, and module-level code.

### `ShapeAnalysisResult`

```python
@dataclass
class ShapeAnalysisResult:
    errors: List[ShapeError] = field(default_factory=list)
    shapes: Dict[str, TensorShape] = field(default_factory=dict)
    constraints_generated: int = 0
    constraints_checked: int = 0
    functions_analyzed: int = 0
    analysis_time_ms: float = 0.0
```

**Methods:**

- `summary() -> str` — Human-readable summary.

---

## Typing Rules (`src.typing_rules`)

Formal typing rules for tensor operations, applied as refinement type judgements. Each rule checks preconditions and returns the output `TensorType`.

### `TensorType`

```python
@dataclass(frozen=True)
class TensorType:
    shape: Tuple[Union[int, str], ...]  # concrete or symbolic dimensions
    device: str = "any"
    dtype: str = "float32"
```

**Properties/Methods:**

- `ndim -> int`
- `concrete_numel() -> Optional[int]` — Element count if all dims are concrete.
- `is_concrete() -> bool`

### `verify_rule()`

Apply a named typing rule with Z3 well-formedness checking.

```python
from src.typing_rules import verify_rule, TensorType

success, result_type = verify_rule(
    "T-LINEAR",
    input_types={"x": TensorType(shape=(32, 784))},
    params={"in_features": 784, "out_features": 256},
)
```

```python
def verify_rule(
    rule_name: str,
    input_types: Dict[str, TensorType],
    params: Dict[str, Any],
) -> Tuple[bool, Optional[TensorType]]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `rule_name` | `str` | One of the supported rule names (see below) |
| `input_types` | `Dict[str, TensorType]` | Named input tensor types |
| `params` | `Dict[str, Any]` | Rule-specific parameters |

**Returns:** `(success, result_type)` — `success` is `True` iff the rule applied without error and the result passes Z3 well-formedness checks.

**Raises:** `ValueError` if `rule_name` is not recognized.

### Individual Typing Rules

Each rule can be called directly. All raise `TypingRuleError` on precondition violations.

| Function | Signature | Description |
|----------|-----------|-------------|
| `apply_t_linear` | `(input_type, in_features, out_features) -> TensorType` | `nn.Linear`: last dim must equal `in_features` |
| `apply_t_conv2d` | `(input_type, out_channels, kernel_size, stride?, padding?, dilation?) -> TensorType` | `nn.Conv2d`: 4-D input required |
| `apply_t_broadcast` | `(type_a, type_b) -> TensorType` | NumPy/PyTorch broadcasting (right-aligned) |
| `apply_t_reshape` | `(input_type, new_shape) -> TensorType` | Element-count preservation; at most one `-1` dim |
| `apply_t_cat` | `(input_types, dim?) -> TensorType` | Concatenation; non-cat dims must match |
| `apply_t_matmul` | `(type_a, type_b) -> TensorType` | Batched matmul with inner-dim matching |
| `apply_t_reduce` | `(input_type, dim, keepdim?) -> TensorType` | Reduction (sum, mean, etc.) along a dimension |
| `apply_t_embed` | `(input_type, num_embeddings, embedding_dim) -> TensorType` | `nn.Embedding`: appends `embedding_dim` |

---

## Operator Coverage (`src.stdlib.modern_ops`)

### `get_all_covered_ops()`

Return the merged dictionary of all operators with shape transfer functions (original + modern ops).

```python
from src.stdlib.modern_ops import get_all_covered_ops

ops = get_all_covered_ops()
print(f"{len(ops)} operators covered")
# e.g. {'torch.matmul': 'matmul', 'torch.nn.Linear': 'linear', ...}
```

```python
def get_all_covered_ops() -> Dict[str, str]
```

**Returns:** `Dict[str, str]` — Mapping from operator name to category.

Also exposes `get_all_covered_ops.get_unsupported_op_shape` for retrieving the fallback handler for unsupported ops.

---

## CLI (`src.cli.main`)

### `reftype analyze-package [DIRECTORY]`

Analyze an entire Python package or directory. Recursively finds `.py` files, respects `.gitignore` patterns, processes all files, and shows an aggregated summary.

```bash
reftype analyze-package my_project/
reftype analyze-package src/ --output-format json -o results.json
reftype analyze-package . --requirements requirements.txt --output-format sarif
```

| Flag | Description |
|------|-------------|
| `DIRECTORY` | Directory to analyze (default: `.`) |
| `--requirements FILE` | Path to `requirements.txt` or `pyproject.toml` for library type stubs |
| `--output-format` | Output format: `text` (default), `json`, `sarif` |
| `-o, --output` | Output file (default: stdout) |
| `--include` | Glob patterns to include (default: `**/*.py`) |
| `--exclude` | Additional glob patterns to exclude |
| `-w, --workers N` | Parallel workers (0 = auto-detect) |
| `--timeout N` | Per-file timeout in seconds (default: 300) |
| `-v, --verbose` | Show per-file details |
| `-c, --config` | Config file path |
| `--stubs-dir PATH` | Directory containing `.pyi` stub files to use as known types |
| `--mypy-baseline FILE` | mypy output file (text or JSON) for baseline comparison |
| `--pyright-baseline FILE` | pyright JSON output file for baseline comparison |

**Output formats:**

- **`text`** — Human-readable summary with files analyzed, types inferred, refinements found, and bugs.
- **`json`** — Machine-readable JSON with full results, summary, and optional `requirements` section.
- **`sarif`** — SARIF 2.1.0 format for GitHub Code Scanning / GitHub Advanced Security integration.

**`--requirements` flag:**

When provided, parses a `requirements.txt` or `pyproject.toml` to extract dependency names and reports which dependencies have known type stubs (numpy, pandas, torch, scipy, requests, flask, django, sqlalchemy, tensorflow, pydantic).

**`--stubs-dir` flag:**

Scans the given directory for `.pyi` stub files, parses type annotations using the `ast` module, and provides them as "known types" to the inference engine. This allows the tool to start from existing type information and refine it further. Typeshed stubs are also auto-detected if available.

**`--mypy-baseline` flag:**

Imports mypy output as a baseline for comparison. Supports both mypy's default text format (`file:line: error: message [code]`) and JSON output (from `--output json`). After analysis, prints a comparison showing how many mypy errors were confirmed by refinement types, how many are likely false positives, and how many new issues were found.

**`--pyright-baseline` flag:**

Imports pyright's JSON output as a baseline. Same comparison logic as `--mypy-baseline`.

**Skipped directories:** `venv/`, `.venv/`, `__pycache__/`, `.git/`, `node_modules/`, `.mypy_cache/`, `.pytest_cache/`, `dist/`, `build/`, `*.egg-info/`.

**Exit codes:** `0` = no bugs, `1` = bugs found, `2` = error.

---

### `reftype verify FILE`

Verify an `nn.Module` source file from the command line.

```bash
reftype verify model.py --input-shape x=batch,3,224,224
reftype verify model.py -s x=batch,784 --high-confidence --format json
```

| Flag | Description |
|------|-------------|
| `FILE` | Python file containing an `nn.Module` class |
| `--input-shape, -s` | Input shape as `name=dim1,dim2,...` (repeatable) |
| `--no-device-check` | Disable device consistency checking |
| `--no-phase-check` | Disable train/eval phase checking |
| `--cegar-iterations N` | Max CEGAR refinement iterations (default: 10) |
| `--format, -f` | Output format: `text` (default), `json`, `sarif` |
| `--high-confidence` | Only report Z3-proven bugs (0% FP for CI/CD gating) |

**Exit codes:** `0` = safe, `1` = unsafe or error.

---

## Symbolic-Execution Engine (`src.symexec`)

A self-contained, **torch-free** abstract-interpretation / symbolic-execution
engine that finds shape, rank, device and control-flow defects in PyTorch code
by interpreting it over an abstract domain — no model instantiation, no tensors
allocated. It is **sound by abstention**: anywhere a value leaves the modeled
fragment the engine returns `Top` and emits *no* report, so every finding is a
Z3-proved or concretely-forced runtime failure (zero false positives by
construction).

### `analyze_source()` / `analyze_file()`

```python
from src.symexec import analyze_source, analyze_file, SymConfig

result = analyze_source(open("model.py").read(), filename="model.py")
for bug in result.bugs:
    print(bug.kind.value, bug.line, bug.message, bug.confidence)

result = analyze_file("model.py")              # convenience wrapper over a path
```

```python
def analyze_source(
    source: str,
    filename: str = "<unknown>",
    budget_ms: Optional[float] = None,   # coarse per-file wall-clock guard
    config: Optional[SymConfig] = None,  # soundness mode (Step 86)
) -> SymResult: ...

def analyze_file(path: str, config: Optional[SymConfig] = None) -> SymResult: ...
```

`budget_ms` is defence-in-depth for pathologically large *files*: the engine's
per-construct iteration caps (`ITERATION_CAPS`) already bound the cost of any
single unit deterministically. Exceeding the budget stops analysis at a
top-level unit boundary (lost coverage, never a false report) and records a
`RESOURCE_BUDGET` abstention.

### `SymResult`

The result object returned by every analysis entry point.

| Member | Description |
|--------|-------------|
| `bugs` | `List[SymBug]` — the findings, canonically ordered by `(line, col, kind, message)`. |
| `functions_analyzed` | Count of top-level functions analysed. |
| `ran_main` | Whether the shipped `if __name__ == "__main__":` harness ran. |
| `abstentions` | `AbstainLedger` — structured coverage of *where/why* the engine declined to reason. |
| `coverage` | `CoverageMeter` — how much of the program was interpreted with non-`Top` values. |
| `fingerprint()` | Deterministic SHA-256 digest over the finding set + abstain profile (a reproducibility receipt). |
| `footprint()` | Full `ProofFootprint` (digest + bug/abstain counts + coverage profile). |
| `explain(filename)` | Full `--explain` provenance view (source→…→sink derivation + counterexample / certificate / minimal conditions). |
| `to_dict(filename)` | Stable JSON object (confidence, provenance, fingerprint, abstain coverage). |
| `to_sarif(filename)` | Complete SARIF 2.1.0 log for GitHub Code Scanning. |
| `to_lsp_diagnostics(uri)` | LSP `Diagnostic[]` for inline editor squiggles. |
| `to_github_annotations(filename)` | GitHub Actions `::error file=…::` annotation commands. |
| `certificates(filename)` | `List[BugCertificate]` — one replayable proof-carrying certificate per report (Step 94). |
| `replay(filename)` | `List[ReplayResult]` — independently re-derive each report's verdict from its certificate (Step 95). |

### `SymBug`

A single finding (`src.symexec.bugs`). Frozen dataclass with: `kind`
(`SymBugKind` enum), `message`, `line`, `col`, `function`, `severity`,
`confidence` (calibrated, `< 1.0`), `fix_suggestion`, `evidence`.
`bug.to_api_bug(filename)` converts to the public `src.api.Bug` type.

Bug kinds (`SymBugKind`): `matmul_dim_mismatch`, `broadcast_mismatch`,
`layer_dim_mismatch`, `reshape_size_mismatch`, `cat_shape_mismatch`,
`einsum_dim_mismatch`, `einops_pattern_mismatch`, `axis_out_of_range`,
`tensor_index_oob`, `rank_index_error`, `negative_dimension`,
`channel_axis_mismatch`, `division_by_zero`, `none_propagation`,
`unpack_arity_mismatch`, `return_arity_contract`, `axis_name_construction`,
`device_mismatch` (two operands on statically-known and different device types,
e.g. `cpu` vs `cuda` — a forced `RuntimeError`; abstains on any unknown device
and normalises `cuda`==`cuda:0`), `item_on_nonscalar` (`tensor.item()` when the
element count is statically known and not 1 — a forced `RuntimeError`; abstains
on any unknown/symbolic dim), `inplace_on_leaf` (an in-place op `add_`/`mul_`/…
applied to a leaf tensor that requires grad — a forced `RuntimeError`; abstains
unless both `requires_grad` and leaf-status are positively known, and never
flags the permitted `requires_grad_`/`detach_`), `bool_on_nonscalar`
(`if t:`/`while t:`/`not t` on a tensor whose element count is statically known
and not 1 — a forced `RuntimeError`; abstains on any unknown/symbolic dim),
`backward_on_nonscalar` (`tensor.backward()` with no `gradient=` argument on a
non-scalar tensor whose `requires_grad` is positively known — a forced
`RuntimeError`; abstains unless `requires_grad` is True and the element count is
known and not 1, and never flags calls that pass an explicit gradient),
`numpy_on_grad` (`tensor.numpy()` on a tensor whose `requires_grad` is positively
known — a forced `RuntimeError`; `.detach().numpy()` is never flagged),
`requires_grad_non_float` (a tensor constructor setting `requires_grad=True` on a
known integer/bool dtype — a forced `RuntimeError`; abstains on any unknown
dtype).

**Intent (heuristic-only) findings.** Beyond the forced-failure kinds above,
`heuristic` mode surfaces patterns that do not crash but are almost certainly
bugs. These are **suppressed in `sound`/`balanced`** so the zero-false-positive
guarantee of those modes is preserved, and carry `severity="warning"`:
`discarded_tensor_result` (a bare statement like `x.cuda()` / `x.to(...)` /
`x.reshape(...)` whose new-tensor result is discarded — a silent no-op the author
likely meant to assign back), `direct_forward_call` (`module.forward(x)` called
directly instead of `module(x)`, which bypasses `nn.Module.__call__` and its
registered hooks), `tensor_data_access` (accessing `tensor.data`, which bypasses
autograd tracking — prefer `.detach()`), `missing_super_init` (an `nn.Module`
subclass whose `__init__` never calls `super().__init__()`, so parameter and
submodule registration silently breaks), `tensor_copy_construct`
(`torch.tensor(existing_tensor)`, which copy-constructs and detaches silently —
emits a UserWarning at runtime; prefer `sourceTensor.clone().detach()`).

### Soundness modes — `SymConfig` (Step 86)

Reporting policy. The three named modes have **nesting** report sets:
`sound ⊆ balanced ⊆ heuristic`.

```python
from src.symexec import SymConfig, analyze_source

analyze_source(src, config=SymConfig.sound())       # max precision (subset)
analyze_source(src, config=SymConfig.balanced())    # default == historic behaviour
analyze_source(src, config=SymConfig.heuristic())   # max recall (superset)
analyze_source(src, config=SymConfig.for_mode("sound", min_confidence=0.9))
```

| Mode | `min_confidence` | `require_feasibility` | `enable_heuristics` | Meaning |
|------|------------------|------------------------|----------------------|---------|
| `balanced` *(default)* | `0.0` | `False` | `False` | Byte-identical to the historic engine; every proven finding, nothing filtered. |
| `sound` | `0.85` | `True` | `False` | Keep a path-conditioned report only when Z3 *positively confirms* feasibility; drop weak-prior findings. A strict subset. |
| `heuristic` | `0.0` | `False` | `True` | Surface clearly-labelled, low-confidence *suspicions* where balanced abstains (may be false positives). A superset. |

`SymConfig(mode, min_confidence, require_feasibility, enable_heuristics, budget_ms)`
is a frozen dataclass; `.with_overrides(**knobs)` returns a revalidated copy.
The default (`DEFAULT_CONFIG`) is `balanced` and preserves every reproducibility
fingerprint.

### Whole-package analysis (`src.symexec.package`)

```python
from src.symexec import analyze_package

pkg = analyze_package("my_project/", config=SymConfig.balanced())
for path, bug in pkg.all_bugs():
    print(path, bug.line, bug.message)
graph = pkg.call_graph()   # cross-file module:symbol -> [module:symbol] edges
```

Resolves cross-file imports so a shape mismatch that only manifests when one
module *uses* another (e.g. an `Encoder` imported into `model.py`) is caught.
`analyze_package(root, *, budget_ms=None, config=None) -> PackageResult`.

### Incremental analysis (`src.symexec.incremental`)

```python
from src.symexec import IncrementalCache, analyze_package_incremental

cache = IncrementalCache()
pkg, stats = analyze_package_incremental("proj/", cache)   # cold: all analysed
# ... edit one file ...
pkg, stats = analyze_package_incremental("proj/", cache)   # warm: reuses unchanged
print(stats.reused, stats.reanalyzed)
```

The atomic unit is the **file**, but invalidation is symbol-dependency-aware: a
file is recomputed only when its own source *or* a directly-imported project
symbol changed. Output is byte-identical to a fresh `analyze_package`.
`analyze_source_incremental(source, filename, cache) -> (SymResult, reused)` for
single files.

### Parallel driver (`src.symexec.parallel`)

```python
from src.symexec import analyze_package_parallel

pkg = analyze_package_parallel("proj/", workers=8, backend="process")
```

`backend` is `"process"` (default — real parallelism), `"thread"`, or
`"serial"`. Output is byte-identical to serial `analyze_package` (each module is
analysed independently; the merge is order-independent). The process backend
transparently falls back to serial if a pool cannot be created.

### Calibration telemetry (`src.symexec.telemetry`) — opt-in

```python
from src.symexec import TelemetrySink, analyze_source

sink = TelemetrySink().enable()                 # opt-in; disabled by default
sink.record_result(analyze_source(src), outcome=True)   # label confirmed/FP
report = sink.report()
print(report.summary())                          # ECE, Brier, per-kind precision
for s in report.suggestions:                     # advisory prior adjustments
    print(s.kind, s.delta)
```

`CalibrationRecord`s carry **no source content** (kind + confidence + evidence
flags only). Telemetry is side-effect-free: enabling it never changes which bugs
report or any fingerprint, and prior-adjustment suggestions are advisory (never
auto-applied). Records serialize via `records_to_jsonl` / `records_from_jsonl`
(the caller owns any file I/O).

### Third-party stubs (`src.symexec.stubs`)

A declarative, report-free registry (`STUB_REGISTRY`) of return-shape summaries
for common pure library calls (`torch.relu`, `torch.nn.functional.*`,
`numpy.zeros`, …) so the engine can keep reasoning through them instead of
abstaining. Stubs only encode shape transforms that hold for *every* runtime
input and never emit bugs themselves.

### Editor integrations (`src.symexec.notebook`, `src.symexec.integrations`)

```python
from src.symexec import analyze_notebook, to_publish_diagnostics, analyze_file

# Jupyter: analyse a .ipynb (path / JSON string / dict); findings map to cells.
nb = analyze_notebook("analysis.ipynb")
nb                                   # rich HTML table inline in a notebook
for f in nb.findings:
    print(f.cell_index, f.cell_line, f.bug.message)

# VS Code / any LSP client: a publishDiagnostics JSON-RPC notification.
note = to_publish_diagnostics(analyze_file("model.py"), "file:///abs/model.py")
```

`analyze_notebook(nb, filename="<notebook>", *, budget_ms=None, config=None) ->
NotebookResult` concatenates code cells into one virtual module (IPython
`%magic`/`!shell` lines blanked in place to preserve line numbers) and attributes
each finding to `(cell_index, cell_line)`. `NotebookResult` exposes `findings`
(`CellFinding[]`), `by_cell()`, `summary()`, and `to_html()` / `_repr_html_()`.
Inside a kernel, `%load_ext src.symexec.notebook` registers a `%%tensorguard`
cell magic.

`to_publish_diagnostics(result, uri) -> dict` wraps `to_lsp_diagnostics` in a
complete `textDocument/publishDiagnostics` notification; an empty diagnostics
list (clean result) clears prior markers.

### Proof-carrying certificates & replay (`src.symexec.certificate`, `src.symexec.replay`)

```python
from src.symexec import analyze_source, dumps_certificates, replay_text

result = analyze_source(src, filename="demo.py")
certs = result.certificates("demo.py")    # one BugCertificate per report
proof = dumps_certificates(certs)          # deterministic JSON proof artifact

for r in replay_text(proof):               # independent re-derivation, no engine
    print(r.status, r.detail)              # "verified" | "refuted" | "unchecked"
```

Each `BugCertificate` names the violated runtime `predicate` (from the fixed
`PRECONDITIONS` vocabulary, which mirrors the Lean `Ok` predicates:
`dims_equal`, `broadcast_compat`, `numel_match`, `index_in_range`,
`arity_match`, `divisor_nonzero`, `dim_nonneg`, `feature_match`) and the concrete
witness `operands` on which it is violated (`None` for a *claim-only*
certificate). `replay(cert)` / `replay_all` / `replay_text(json)` re-evaluate the
precondition on the witness **without re-running the analysis**: `verified` (the
precondition is violated ⇒ the forced failure is re-derived), `refuted` (the
precondition holds ⇒ a tampered/invalid certificate), or `unchecked` (claim-only
— nothing numeric to re-derive). This is the executable dual of the machine-
checked Lean `refute`/`witness` lemmas. Certificates are *diagnostic*: they
neither change which bugs report nor perturb the proof fingerprint.

### Runnable reproducers (`src.symexec.repro`)

```python
from src.symexec import analyze_source, confirm

result = analyze_source(src, filename="model.py")
for rep in result.repros():                 # one ReproScript per covered report
    print(rep.script)                       # a minimal, self-contained program
    print(confirm(rep).confirmed)           # True: it raises the predicted error
```

`generate_repro(bug)` / `result.repros()` synthesise, per bug kind, a **minimal
runnable program** that constructs concrete tensors matching the witness in the
(fingerprinted, stable) report message and invokes the offending op, so it
deterministically raises the predicted `RuntimeError` / `IndexError` /
`ZeroDivisionError`. `confirm(rep) -> ReproResult` executes it in an isolated
namespace and checks the predicted exception is actually raised — an empirical
soundness layer beyond the proof fingerprint and the Lean proofs. Covered kinds:
matmul, broadcast, reshape, axis-OOB, list/tensor index-OOB, div-by-zero,
nn.Linear, cat, einsum, negative-dim. `confirm` needs torch; generation does not.
Reproducers are *diagnostic* — they never change which bugs report or the
fingerprint.

### Verified auto-repair (`src.symexec.autofix`)

```python
from src.symexec import repair

for fix in repair(source, filename="model.py"):   # only re-verified fixes
    assert fix.verified
    print(fix.strategy, fix.description)
    print(fix.diff)                                # a git-apply-able unified diff
```

`repair(source) -> List[VerifiedFix]` analyses the source, proposes a minimal,
canonical edit per report (`propose_fix`), then **re-runs the engine on the
patched source** (`verify_fix`) and returns a fix only when the targeted bug is
gone *and* no new bug kind appears — so every returned fix is **machine-verified,
not guessed**. Edits are line-local (line numbers preserved) and ship with a
unified diff. Current strategies: `reshape-flatten` (mismatched
`reshape`/`view` target → `-1`) and `negdim-abs` (negative constructor dim →
non-negative); unmatched kinds yield nothing rather than a risky rewrite. Pass
`verified_only=False` to also see rejected candidates with the reason. Repair
edits user source only — it never affects analysis of the original program or the
fingerprint.

### Inferred shape contracts → jaxtyping `.pyi` stubs (`src.symexec.contracts`)

```python
from src.symexec import contracts_to_pyi, infer_contracts

print(contracts_to_pyi(source))         # a ready-to-ship .pyi stub
contracts = infer_contracts(source)     # structured FunctionContract objects
```

TensorGuard already *consumes* jaxtyping/torchtyping annotations to seed an
analysis; `contracts` closes the loop and **produces** them. For every top-level
function and each `forward`/`__call__` method it runs the engine with
annotation-seeded parameters and reads off the **input** abstraction (echoed from
the annotation) and the analysis-**derived** output abstraction, rendering both
as jaxtyping annotations (`Float[Tensor, "b 7"]`) in a `.pyi` stub. The contract
is *inferential/advisory*: a dimension gets a concrete size or a shared symbolic
name only when the analysis tracked it (e.g. a batch dim carried through a
`Linear` whose output feature is computed), otherwise it degrades honestly to an
anonymous axis or a bare `Tensor`. `infer_contracts` returns structured
`FunctionContract`/`ParamContract`/`TensorSpec` objects; `to_pyi` renders a list
of them. The module is torch-free, pure, and never emits a diagnostic, so it
does not affect analysis of the original program or the fingerprint.

### "Why is this safe?" — positive safety reports (`src.symexec.safety`)

```python
from src.symexec.engine import analyze_source

result = analyze_source(source, filename="model.py")
print(result.safety(filename="model.py"))     # a positive safety report
```

Every other surface answers *"why is this a bug?"*; a *sound* analyser can also
answer the dual question only sound tools can — *"what did you prove is safe?"*
`result.safety()` (module `src.symexec.safety`: `safety_report`,
`render_safety_report`, `explain_safety`) re-presents facts the analysis already
computed: the **verdict** (whether any sound forced-failure bug was provable),
the **covered fragment** (the coverage profile — how much of the file was
reasoned about with known, non-⊤ values), the **guarantee** (the
relative-completeness clauses from `completeness_contract.COMPLETE_FOR` — bug
kinds whose *absence of a report on the covered fragment is a positive
guarantee*), and the **boundary** (the abstain ledger marking exactly where that
guarantee stops). The claim is honestly scoped: it covers the complete-for kinds
on the covered fragment, and abstentions name every place that scope ends.
Heuristic/intent warnings never bear on the verdict. `SafetyReport.to_dict()`
gives a JSON-ready snapshot. The module is pure and read-only — it never runs the
engine, mutates a result, or emits a diagnostic, so it cannot affect the
fingerprint.
