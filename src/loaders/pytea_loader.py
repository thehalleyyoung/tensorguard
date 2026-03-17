"""
PyTEA Test Case Loader.

Parses Python files from the PyTEA benchmark suite and converts them into
TensorGuard verification targets.  Each test case is classified as either
containing a known shape bug or being correct, based on heuristic analysis
of the code and any available annotations.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class PyTEATestCase:
    """A single PyTEA test case prepared for TensorGuard verification."""

    name: str
    source: str
    file_path: str
    category: str  # "basics", "mnist", "dcgan", etc.
    has_known_bug: bool
    bug_description: Optional[str] = None
    # Extracted nn.Module classes found in this file
    module_classes: List[str] = field(default_factory=list)
    # Inferred input shapes for each module class
    input_shapes: Dict[str, Dict[str, tuple]] = field(default_factory=dict)


# Patterns that indicate intentional shape bugs in PyTEA tests.
# Each pattern uses a negative lookahead to avoid matching narrative
# comments like "# error was caused since ..." which describe past bugs
# rather than annotating current code as buggy.
_BUG_INDICATORS = [
    re.compile(r"#\s*fail(?!\s+\w{2,}\s+\w)", re.IGNORECASE),
    re.compile(r"#\s*error(?!\s+\w{2,}\s+\w)", re.IGNORECASE),
    re.compile(r"#\s*bug(?!\s+\w{2,}\s+\w)", re.IGNORECASE),
    re.compile(r"#\s*shape\s*mismatch", re.IGNORECASE),
    re.compile(r"torch\.[a-z]*x{2,}\w*\("),  # e.g. torch.xx() — nonsense call
    re.compile(r"#\s*wrong(?!\s+\w{2,}\s+\w)", re.IGNORECASE),
]


def _classify_pytea_file(source: str, file_path: str) -> Tuple[bool, Optional[str]]:
    """Determine whether a PyTEA test file contains a known shape bug."""
    for pat in _BUG_INDICATORS:
        m = pat.search(source)
        if m:
            line_no = source[:m.start()].count("\n") + 1
            return True, f"Bug indicator at line {line_no}: {m.group().strip()}"

    basename = os.path.basename(file_path).lower()
    if "bug" in basename or "fail" in basename or "error" in basename:
        return True, f"Filename indicates bug: {basename}"

    return False, None


def _extract_module_classes(tree: ast.Module) -> List[str]:
    """Find nn.Module subclass names in an AST."""
    classes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Attribute):
                base_name = base.attr
            elif isinstance(base, ast.Name):
                base_name = base.id
            if base_name in ("Module", "nn.Module"):
                classes.append(node.name)
                break
    return classes


def _infer_input_shapes(tree: ast.Module, class_name: str) -> Dict[str, tuple]:
    """Heuristically infer input shapes for an nn.Module subclass.

    Looks at the forward() method's parameters, __init__ layer definitions,
    and any tensor creation calls near the model invocation.
    """
    shapes: Dict[str, tuple] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue

        init_info: Dict[str, Dict] = {}
        forward_params: List[str] = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "__init__":
                    init_info = _extract_init_layers(item)
                elif item.name == "forward":
                    forward_params = [
                        a.arg for a in item.args.args if a.arg != "self"
                    ]

        if not forward_params:
            continue

        first_layer = _find_first_layer(init_info)
        if first_layer:
            kind = first_layer.get("kind", "")
            if kind == "Linear":
                in_f = first_layer.get("in_features", 784)
                shapes[forward_params[0]] = ("batch", in_f)
            elif kind in ("Conv2d", "Conv1d"):
                in_c = first_layer.get("in_channels", 3)
                if kind == "Conv2d":
                    # 28×28 for single-channel (MNIST-like), 32×32 otherwise
                    spatial = 28 if in_c == 1 else 32
                    shapes[forward_params[0]] = ("batch", in_c, spatial, spatial)
                else:
                    shapes[forward_params[0]] = ("batch", in_c, 128)
            elif kind == "Embedding":
                shapes[forward_params[0]] = ("batch", "seq_len")
            elif kind in ("LSTM", "GRU", "RNN"):
                in_s = first_layer.get("input_size", 128)
                shapes[forward_params[0]] = ("batch", "seq_len", in_s)
        else:
            # Default: assume image-like input
            shapes[forward_params[0]] = ("batch", 3, 32, 32)

    return shapes


def _extract_init_layers(init_func: ast.FunctionDef) -> Dict[str, Dict]:
    """Extract layer definitions from __init__."""
    layers: Dict[str, Dict] = {}
    for node in ast.walk(init_func):
        if not isinstance(node, ast.Assign):
            continue
        if not node.targets:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Attribute) and
                isinstance(target.value, ast.Name) and
                target.value.id == "self"):
            continue

        attr_name = target.attr
        call = node.value
        if not isinstance(call, ast.Call):
            continue

        func_name = ""
        if isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        elif isinstance(call.func, ast.Name):
            func_name = call.func.id

        info: Dict = {"kind": func_name}
        args = call.args
        if func_name == "Linear" and len(args) >= 2:
            info["in_features"] = _safe_int(args[0])
            info["out_features"] = _safe_int(args[1])
        elif func_name in ("Conv2d", "Conv1d") and len(args) >= 2:
            info["in_channels"] = _safe_int(args[0])
            info["out_channels"] = _safe_int(args[1])
        elif func_name == "Embedding" and len(args) >= 2:
            info["num_embeddings"] = _safe_int(args[0])
            info["embedding_dim"] = _safe_int(args[1])
        elif func_name in ("LSTM", "GRU", "RNN") and len(args) >= 2:
            info["input_size"] = _safe_int(args[0])
            info["hidden_size"] = _safe_int(args[1])

        layers[attr_name] = info

    return layers


def _find_first_layer(layers: Dict[str, Dict]) -> Optional[Dict]:
    """Find the first data-processing layer (skip dropout/relu/norm)."""
    skip = {"Dropout", "ReLU", "BatchNorm1d", "BatchNorm2d", "LayerNorm",
            "Identity", "Softmax", "GELU", "SiLU", "Tanh", "Sigmoid"}
    for info in layers.values():
        if info.get("kind") not in skip:
            return info
    return None


def _safe_int(node: ast.expr) -> int:
    """Extract an integer from an AST node, defaulting to 0."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # Python ≤3.7
        return node.n  # type: ignore[return-value]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_int(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _safe_int(node.left) * _safe_int(node.right)
    return 0


class PyTEALoader:
    """Load PyTEA benchmark test cases from a directory tree."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"PyTEA test directory not found: {root_dir}")

    def load_all(self) -> List[PyTEATestCase]:
        """Load all .py files under the root directory as test cases."""
        cases: List[PyTEATestCase] = []
        for py_file in sorted(self.root_dir.rglob("*.py")):
            case = self._load_file(py_file)
            if case is not None:
                cases.append(case)
        return cases

    def _load_file(self, path: Path) -> Optional[PyTEATestCase]:
        """Load a single PyTEA test file."""
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        # Skip __init__.py and non-PyTorch files
        if path.name == "__init__.py":
            return None
        if "import torch" not in source and "from torch" not in source:
            return None

        # Determine category from directory structure
        rel = path.relative_to(self.root_dir)
        parts = rel.parts
        if len(parts) >= 2:
            category = parts[1] if parts[0] == "benchmarks" else parts[0]
        else:
            category = "basics"

        has_bug, bug_desc = _classify_pytea_file(source, str(path))

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        module_classes = _extract_module_classes(tree)
        input_shapes: Dict[str, Dict[str, tuple]] = {}
        for cls_name in module_classes:
            shapes = _infer_input_shapes(tree, cls_name)
            if shapes:
                input_shapes[cls_name] = shapes

        name = str(rel).replace(os.sep, "/").removesuffix(".py")
        return PyTEATestCase(
            name=name,
            source=source,
            file_path=str(path),
            category=category,
            has_known_bug=has_bug,
            bug_description=bug_desc,
            module_classes=module_classes,
            input_shapes=input_shapes,
        )
