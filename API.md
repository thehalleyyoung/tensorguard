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
