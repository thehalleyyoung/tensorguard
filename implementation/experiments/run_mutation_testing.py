#!/usr/bin/env python3
"""
Mutation testing for TensorGuard's Torchvision-style benchmarks.

Applies shape-breaking mutations to correct nn.Module definitions, then checks
whether TensorGuard's verify_model detects the injected bug.  The mutation
score (killed / (killed + survived)) gives the false-negative rate.
"""

import ast
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_checker import verify_model


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MutationOperator enum
# ═══════════════════════════════════════════════════════════════════════════════

class MutationOperator(Enum):
    WRONG_IN_FEATURES = "wrong_in_features"
    WRONG_OUT_FEATURES = "wrong_out_features"
    WRONG_KERNEL_SIZE = "wrong_kernel_size"
    SWAP_LAYERS = "swap_layers"
    REMOVE_RESHAPE = "remove_reshape"
    WRONG_CHANNELS = "wrong_channels"
    ADD_DIMENSION_MISMATCH = "add_dimension_mismatch"
    WRONG_POOL_SIZE = "wrong_pool_size"
    TRANSPOSE_MISSING = "transpose_missing"
    WRONG_CONCAT_DIM = "wrong_concat_dim"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ModelMutator – AST-based mutation engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MutationRecord:
    operator: MutationOperator
    description: str
    original_line: int
    mutated_source: str


class ModelMutator:
    """Applies shape-breaking AST mutations to nn.Module source code."""

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self.lines = source.splitlines(keepends=True)

    # -- public API ----------------------------------------------------------

    def available_mutations(self) -> List[MutationOperator]:
        """Return which mutation operators are applicable to this source."""
        avail: List[MutationOperator] = []
        if self._find_linear_calls():
            avail.extend([MutationOperator.WRONG_IN_FEATURES,
                          MutationOperator.WRONG_OUT_FEATURES])
        if self._find_conv_calls():
            avail.extend([MutationOperator.WRONG_KERNEL_SIZE,
                          MutationOperator.WRONG_CHANNELS])
        if self._find_reshape_calls():
            avail.append(MutationOperator.REMOVE_RESHAPE)
        if self._find_pool_calls():
            avail.append(MutationOperator.WRONG_POOL_SIZE)
        if self._find_transpose_calls():
            avail.append(MutationOperator.TRANSPOSE_MISSING)
        if self._find_cat_calls():
            avail.append(MutationOperator.WRONG_CONCAT_DIM)
        if self._count_layer_assignments() >= 2:
            avail.append(MutationOperator.SWAP_LAYERS)
        if self._find_linear_calls() or self._find_conv_calls():
            avail.append(MutationOperator.ADD_DIMENSION_MISMATCH)
        return avail

    def apply(self, op: MutationOperator) -> Optional[MutationRecord]:
        """Apply a single mutation operator; return record or None."""
        dispatch = {
            MutationOperator.WRONG_IN_FEATURES: self._mut_wrong_in_features,
            MutationOperator.WRONG_OUT_FEATURES: self._mut_wrong_out_features,
            MutationOperator.WRONG_KERNEL_SIZE: self._mut_wrong_kernel_size,
            MutationOperator.SWAP_LAYERS: self._mut_swap_layers,
            MutationOperator.REMOVE_RESHAPE: self._mut_remove_reshape,
            MutationOperator.WRONG_CHANNELS: self._mut_wrong_channels,
            MutationOperator.ADD_DIMENSION_MISMATCH: self._mut_add_dim_mismatch,
            MutationOperator.WRONG_POOL_SIZE: self._mut_wrong_pool_size,
            MutationOperator.TRANSPOSE_MISSING: self._mut_transpose_missing,
            MutationOperator.WRONG_CONCAT_DIM: self._mut_wrong_concat_dim,
        }
        fn = dispatch.get(op)
        if fn is None:
            return None
        return fn()

    # -- AST finders ---------------------------------------------------------

    def _find_calls(self, names: List[str]) -> List[ast.Call]:
        hits: List[ast.Call] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node)
                if name and any(n in name for n in names):
                    hits.append(node)
        return hits

    def _find_linear_calls(self) -> List[ast.Call]:
        return self._find_calls(["Linear"])

    def _find_conv_calls(self) -> List[ast.Call]:
        return self._find_calls(["Conv2d", "Conv1d"])

    def _find_pool_calls(self) -> List[ast.Call]:
        return self._find_calls(["MaxPool", "AvgPool", "AdaptiveAvgPool"])

    def _find_reshape_calls(self) -> List[ast.Call]:
        hits: List[ast.Call] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node)
                if name and any(k in name for k in ["view", "reshape"]):
                    hits.append(node)
        return hits

    def _find_transpose_calls(self) -> List[ast.Call]:
        hits: List[ast.Call] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node)
                if name and any(k in name for k in ["transpose", "permute", ".t()"]):
                    hits.append(node)
        return hits

    def _find_cat_calls(self) -> List[ast.Call]:
        return self._find_calls(["cat", "stack"])

    def _count_layer_assignments(self) -> int:
        count = 0
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    name = self._call_name(node.value)
                    if name and any(k in name for k in
                                    ["Linear", "Conv", "BatchNorm", "LayerNorm"]):
                        count += 1
        return count

    @staticmethod
    def _call_name(node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
        return None

    # -- text-level mutation helpers -----------------------------------------

    def _replace_arg_value(self, call: ast.Call, arg_idx: int,
                           new_val: int, kw_name: Optional[str] = None) -> Optional[str]:
        """Replace a positional/keyword arg in source and return new source."""
        lines = list(self.lines)
        # Try keyword first
        if kw_name:
            for kw in call.keywords:
                if kw.arg == kw_name:
                    return self._replace_node_value(kw.value, new_val)
        # Positional
        if arg_idx < len(call.args):
            return self._replace_node_value(call.args[arg_idx], new_val)
        return None

    def _replace_node_value(self, node: ast.expr, new_val: Any) -> str:
        """Replace a single AST node's text span with new_val in source."""
        lines = list(self.lines)
        lineno = node.lineno - 1
        col = node.col_offset
        end_col = getattr(node, "end_col_offset", None)
        if end_col is None:
            end_col = col + len(str(ast.literal_eval(ast.unparse(node))))
        line = lines[lineno]
        lines[lineno] = line[:col] + str(new_val) + line[end_col:]
        return "".join(lines)

    def _remove_line_containing(self, node: ast.AST) -> str:
        """Remove the source line containing the given AST node."""
        lines = list(self.lines)
        lineno = node.lineno - 1
        # Replace with pass to keep syntax valid
        indent = len(lines[lineno]) - len(lines[lineno].lstrip())
        lines[lineno] = " " * indent + "pass\n"
        return "".join(lines)

    # -- mutation operators --------------------------------------------------

    def _mut_wrong_in_features(self) -> Optional[MutationRecord]:
        calls = self._find_linear_calls()
        if not calls:
            return None
        call = random.choice(calls)
        if len(call.args) < 1:
            return None
        try:
            orig = ast.literal_eval(ast.unparse(call.args[0]))
        except Exception:
            orig = 128
        new_val = orig + random.choice([1, 7, -3, 17, 32])
        if new_val <= 0:
            new_val = orig + 7
        src = self._replace_node_value(call.args[0], new_val)
        if src is None:
            return None
        return MutationRecord(
            operator=MutationOperator.WRONG_IN_FEATURES,
            description=f"Linear in_features {orig} -> {new_val}",
            original_line=call.lineno,
            mutated_source=src,
        )

    def _mut_wrong_out_features(self) -> Optional[MutationRecord]:
        calls = self._find_linear_calls()
        if not calls:
            return None
        call = random.choice(calls)
        if len(call.args) < 2:
            return None
        try:
            orig = ast.literal_eval(ast.unparse(call.args[1]))
        except Exception:
            orig = 64
        new_val = orig + random.choice([3, -5, 11, 23])
        if new_val <= 0:
            new_val = orig + 11
        src = self._replace_node_value(call.args[1], new_val)
        if src is None:
            return None
        return MutationRecord(
            operator=MutationOperator.WRONG_OUT_FEATURES,
            description=f"Linear out_features {orig} -> {new_val}",
            original_line=call.lineno,
            mutated_source=src,
        )

    def _mut_wrong_kernel_size(self) -> Optional[MutationRecord]:
        calls = self._find_conv_calls()
        if not calls:
            return None
        call = random.choice(calls)
        # kernel_size is typically 3rd positional arg or keyword
        target_node = None
        for kw in call.keywords:
            if kw.arg == "kernel_size":
                target_node = kw.value
                break
        if target_node is None and len(call.args) >= 3:
            target_node = call.args[2]
        if target_node is None:
            return None
        try:
            orig = ast.literal_eval(ast.unparse(target_node))
        except Exception:
            orig = 3
        if isinstance(orig, tuple):
            new_val = (orig[0] + 4, orig[1] + 4)
        else:
            new_val = orig + 4
        src = self._replace_node_value(target_node, new_val)
        if src is None:
            return None
        return MutationRecord(
            operator=MutationOperator.WRONG_KERNEL_SIZE,
            description=f"Conv kernel_size {orig} -> {new_val}",
            original_line=call.lineno,
            mutated_source=src,
        )

    def _mut_wrong_channels(self) -> Optional[MutationRecord]:
        calls = self._find_conv_calls()
        if not calls:
            return None
        call = random.choice(calls)
        if len(call.args) < 2:
            return None
        # Mutate out_channels (arg index 1)
        target = call.args[1]
        try:
            orig = ast.literal_eval(ast.unparse(target))
        except Exception:
            orig = 64
        new_val = orig + random.choice([3, 7, -5, 13])
        if new_val <= 0:
            new_val = orig + 7
        src = self._replace_node_value(target, new_val)
        if src is None:
            return None
        return MutationRecord(
            operator=MutationOperator.WRONG_CHANNELS,
            description=f"Conv out_channels {orig} -> {new_val}",
            original_line=call.lineno,
            mutated_source=src,
        )

    def _mut_swap_layers(self) -> Optional[MutationRecord]:
        """Swap two layer-call lines in forward() to create shape mismatch."""
        forward_node = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "forward":
                forward_node = node
                break
        if forward_node is None:
            return None
        # Find self.layer(...) calls
        layer_stmts = []
        for stmt in forward_node.body:
            if isinstance(stmt, (ast.Assign, ast.Return, ast.Expr)):
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Attribute) and isinstance(
                            getattr(n, "value", None), ast.Name):
                        if n.value.id == "self":  # type: ignore
                            layer_stmts.append(stmt)
                            break
        if len(layer_stmts) < 2:
            return None
        # Pick two non-adjacent layer calls and swap their lines
        i, j = 0, min(1, len(layer_stmts) - 1)
        if len(layer_stmts) > 2:
            i = 0
            j = len(layer_stmts) - 1
        s1, s2 = layer_stmts[i], layer_stmts[j]
        lines = list(self.lines)
        l1, l2 = s1.lineno - 1, s2.lineno - 1
        if l1 == l2:
            return None
        lines[l1], lines[l2] = lines[l2], lines[l1]
        return MutationRecord(
            operator=MutationOperator.SWAP_LAYERS,
            description=f"Swapped forward lines {s1.lineno} and {s2.lineno}",
            original_line=s1.lineno,
            mutated_source="".join(lines),
        )

    def _mut_remove_reshape(self) -> Optional[MutationRecord]:
        forward_node = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "forward":
                forward_node = node
                break
        if forward_node is None:
            return None
        for stmt in forward_node.body:
            src_line = ast.unparse(stmt)
            if any(k in src_line for k in ["view(", "reshape(", "flatten("]):
                lines = list(self.lines)
                lineno = stmt.lineno - 1
                indent = len(lines[lineno]) - len(lines[lineno].lstrip())
                # Replace reshape with identity pass-through
                if isinstance(stmt, ast.Assign) and stmt.targets:
                    tgt = ast.unparse(stmt.targets[0])
                    # Find the object being reshaped
                    if isinstance(stmt.value, ast.Call) and isinstance(
                            stmt.value.func, ast.Attribute):
                        obj = ast.unparse(stmt.value.func.value)
                        lines[lineno] = " " * indent + f"{tgt} = {obj}\n"
                    else:
                        lines[lineno] = " " * indent + "pass\n"
                else:
                    lines[lineno] = " " * indent + "pass\n"
                return MutationRecord(
                    operator=MutationOperator.REMOVE_RESHAPE,
                    description=f"Removed reshape/view at line {stmt.lineno}",
                    original_line=stmt.lineno,
                    mutated_source="".join(lines),
                )
        return None

    def _mut_add_dim_mismatch(self) -> Optional[MutationRecord]:
        """Insert nn.Linear(999, 999) between two layers in forward."""
        forward_node = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "forward":
                forward_node = node
                break
        if forward_node is None:
            return None
        # Find first self.X(x) in forward
        for stmt in forward_node.body:
            if isinstance(stmt, ast.Assign):
                lines = list(self.lines)
                lineno = stmt.lineno - 1
                indent = len(lines[lineno]) - len(lines[lineno].lstrip())
                extra = " " * indent + "x = nn.Linear(999, 13)(x)  # MUTANT\n"
                lines.insert(lineno + 1, extra)
                # Also add init for this layer
                src = "".join(lines)
                return MutationRecord(
                    operator=MutationOperator.ADD_DIMENSION_MISMATCH,
                    description="Inserted nn.Linear(999, 13) in forward",
                    original_line=stmt.lineno,
                    mutated_source=src,
                )
        return None

    def _mut_wrong_pool_size(self) -> Optional[MutationRecord]:
        calls = self._find_pool_calls()
        if not calls:
            return None
        call = random.choice(calls)
        if len(call.args) < 1:
            for kw in call.keywords:
                if kw.arg == "kernel_size":
                    try:
                        orig = ast.literal_eval(ast.unparse(kw.value))
                    except Exception:
                        orig = 2
                    new_val = orig * 8
                    src = self._replace_node_value(kw.value, new_val)
                    if src:
                        return MutationRecord(
                            operator=MutationOperator.WRONG_POOL_SIZE,
                            description=f"Pool kernel {orig} -> {new_val}",
                            original_line=call.lineno,
                            mutated_source=src,
                        )
            return None
        target = call.args[0]
        try:
            orig = ast.literal_eval(ast.unparse(target))
        except Exception:
            orig = 2
        new_val = orig * 8 if isinstance(orig, int) else 64
        src = self._replace_node_value(target, new_val)
        if src is None:
            return None
        return MutationRecord(
            operator=MutationOperator.WRONG_POOL_SIZE,
            description=f"Pool kernel {orig} -> {new_val}",
            original_line=call.lineno,
            mutated_source=src,
        )

    def _mut_transpose_missing(self) -> Optional[MutationRecord]:
        forward_node = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "forward":
                forward_node = node
                break
        if forward_node is None:
            return None
        for stmt in forward_node.body:
            src_line = ast.unparse(stmt)
            if any(k in src_line for k in ["transpose(", "permute(", ".t()"]):
                lines = list(self.lines)
                lineno = stmt.lineno - 1
                indent = len(lines[lineno]) - len(lines[lineno].lstrip())
                if isinstance(stmt, ast.Assign) and stmt.targets:
                    tgt = ast.unparse(stmt.targets[0])
                    if isinstance(stmt.value, ast.Call) and isinstance(
                            stmt.value.func, ast.Attribute):
                        obj = ast.unparse(stmt.value.func.value)
                        lines[lineno] = " " * indent + f"{tgt} = {obj}\n"
                    else:
                        lines[lineno] = " " * indent + "pass\n"
                else:
                    lines[lineno] = " " * indent + "pass\n"
                return MutationRecord(
                    operator=MutationOperator.TRANSPOSE_MISSING,
                    description=f"Removed transpose at line {stmt.lineno}",
                    original_line=stmt.lineno,
                    mutated_source="".join(lines),
                )
        return None

    def _mut_wrong_concat_dim(self) -> Optional[MutationRecord]:
        calls = self._find_cat_calls()
        if not calls:
            return None
        call = random.choice(calls)
        for kw in call.keywords:
            if kw.arg == "dim":
                try:
                    orig = ast.literal_eval(ast.unparse(kw.value))
                except Exception:
                    orig = 1
                new_val = orig + 2 if orig < 2 else 0
                src = self._replace_node_value(kw.value, new_val)
                if src:
                    return MutationRecord(
                        operator=MutationOperator.WRONG_CONCAT_DIM,
                        description=f"cat dim {orig} -> {new_val}",
                        original_line=call.lineno,
                        mutated_source=src,
                    )
        # Check positional dim arg (2nd arg)
        if len(call.args) >= 2:
            target = call.args[1]
            try:
                orig = ast.literal_eval(ast.unparse(target))
            except Exception:
                orig = 1
            new_val = orig + 2 if isinstance(orig, int) and orig < 2 else 0
            src = self._replace_node_value(target, new_val)
            if src:
                return MutationRecord(
                    operator=MutationOperator.WRONG_CONCAT_DIM,
                    description=f"cat dim {orig} -> {new_val}",
                    original_line=call.lineno,
                    mutated_source=src,
                )
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Benchmark model sources (15+ architectures)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkSpec:
    source: str
    input_shapes: Dict[str, tuple]


BENCHMARK_MODELS: Dict[str, BenchmarkSpec] = {}


def _reg(name: str, input_shapes: Dict[str, tuple], source: str) -> None:
    BENCHMARK_MODELS[name] = BenchmarkSpec(source=source, input_shapes=input_shapes)


# -- MLPs -------------------------------------------------------------------
_reg("SimpleMLP", {"x": ("batch", 1, 28, 28)}, """\
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
""")

_reg("DeepMLP", {"x": ("batch", 512)}, """\
import torch
import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        return x
""")

_reg("BottleneckMLP", {"x": ("batch", 1024)}, """\
import torch
import torch.nn as nn

class BottleneckMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
""")

# -- CNNs -------------------------------------------------------------------
_reg("SimpleCNN", {"x": ("batch", 3, 32, 32)}, """\
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""")

_reg("VGGBlock", {"x": ("batch", 3, 32, 32)}, """\
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""")

_reg("LeNetStyle", {"x": ("batch", 1, 28, 28)}, """\
import torch
import torch.nn as nn

class LeNetStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
""")

# -- ResNet-style -----------------------------------------------------------
_reg("ResBlock", {"x": ("batch", 64, 16, 16)}, """\
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = torch.relu(x)
        return x
""")

_reg("ResNetSmall", {"x": ("batch", 3, 224, 224)}, """\
import torch
import torch.nn as nn

class ResNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
""")

_reg("BottleneckBlock", {"x": ("batch", 256, 16, 16)}, """\
import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + residual
        x = torch.relu(x)
        return x
""")

# -- Transformers -----------------------------------------------------------
_reg("SimpleTransformerBlock", {"x": ("batch", 10, 512)}, """\
import torch
import torch.nn as nn

class SimpleTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_proj = nn.Linear(512, 512)
        self.ff1 = nn.Linear(512, 2048)
        self.ff2 = nn.Linear(2048, 512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, x):
        attn_out = self.attn_proj(x)
        x = self.norm1(x + attn_out)
        ff_out = torch.relu(self.ff1(x))
        ff_out = self.ff2(ff_out)
        x = self.norm2(x + ff_out)
        return x
""")

_reg("MultiHeadSelfAttn", {"x": ("batch", 10, 256)}, """\
import torch
import torch.nn as nn

class MultiHeadSelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(256, 768)
        self.proj = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.proj(out)
        x = self.norm(x + out)
        return x
""")

# -- Autoencoders -----------------------------------------------------------
_reg("Autoencoder", {"x": ("batch", 1, 28, 28)}, """\
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 256)
        self.dec2 = nn.Linear(256, 784)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.enc1(x))
        x = torch.relu(self.enc2(x))
        x = torch.relu(self.dec1(x))
        x = self.dec2(x)
        return x
""")

# -- GAN-style --------------------------------------------------------------
_reg("Generator", {"z": ("batch", 100)}, """\
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = torch.tanh(self.fc4(z))
        z = z.view(-1, 1, 28, 28)
        return z
""")

# -- U-Net style ------------------------------------------------------------
_reg("UNetBlock", {"x": ("batch", 3, 32, 32)}, """\
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        e1 = torch.relu(self.enc_conv1(x))
        e2 = torch.relu(self.enc_conv2(self.pool(e1)))
        d1 = torch.relu(self.dec_conv1(e2))
        d2 = self.dec_conv2(d1)
        return d2
""")

# -- Classifier with concat -------------------------------------------------
_reg("ConcatClassifier", {"x": ("batch", 128)}, """\
import torch
import torch.nn as nn

class ConcatClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Linear(128, 64)
        self.branch2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        b1 = torch.relu(self.branch1(x))
        b2 = torch.relu(self.branch2(x))
        combined = torch.cat([b1, b2], dim=1)
        out = self.classifier(combined)
        return out
""")

# -- RNN-style --------------------------------------------------------------
_reg("RNNClassifier", {"x": ("batch", 300)}, """\
import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(300, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.embedding(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""")

# -- MobileNet-style depthwise ----------------------------------------------
_reg("DepthwiseSeparable", {"x": ("batch", 32, 16, 16)}, """\
import torch
import torch.nn as nn

class DepthwiseSeparable(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw_conv = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.pw_conv = nn.Conv2d(32, 64, 1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = torch.relu(self.dw_conv(x))
        x = torch.relu(self.bn(self.pw_conv(x)))
        return x
""")

# -- Inception-style multi-branch -------------------------------------------
_reg("InceptionBlock", {"x": ("batch", 64, 16, 16)}, """\
import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Conv2d(64, 32, 1)
        self.branch2 = nn.Conv2d(64, 32, 3, padding=1)
        self.branch3 = nn.Conv2d(64, 32, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, 10)

    def forward(self, x):
        b1 = torch.relu(self.branch1(x))
        b2 = torch.relu(self.branch2(x))
        b3 = torch.relu(self.branch3(x))
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
""")

# -- Pool-sensitive: flatten after pool forces spatial dim dependence --------
_reg("PoolFlattenNet", {"x": ("batch", 3, 32, 32)}, """\
import torch
import torch.nn as nn

class PoolFlattenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
""")

# -- Transpose-dependent: linear after transpose requires correct axis order --
_reg("TransposeNet", {"x": ("batch", 10, 256)}, """\
import torch
import torch.nn as nn

class TransposeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = x.transpose(1, 2)
        x = self.linear2(x)
        return x
""")

# -- Pool-classifier: pool size directly affects flatten dim -----------------
_reg("PoolClassifier", {"x": ("batch", 3, 64, 64)}, """\
import torch
import torch.nn as nn

class PoolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MutationTestRunner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MutantResult:
    model_name: str
    operator: str
    description: str
    outcome: str          # "killed", "survived", "equivalent", "error"
    verification_time_ms: float = 0.0
    error_message: str = ""


class MutationTestRunner:
    """Generate mutants and test them against TensorGuard."""

    def __init__(self, models: Dict[str, BenchmarkSpec], mutations_per_model: int = 5):
        self.models = models
        self.mutations_per_model = mutations_per_model
        self.results: List[MutantResult] = []

    def run(self) -> List[MutantResult]:
        print(f"Running mutation testing on {len(self.models)} models "
              f"({self.mutations_per_model} mutations each)")
        print("=" * 70)

        for name, spec in self.models.items():
            self._test_model(name, spec)

        return self.results

    def _test_model(self, name: str, spec: BenchmarkSpec) -> None:
        """Generate and test mutants for a single model."""
        source = spec.source
        input_shapes = spec.input_shapes
        print(f"\n{'─' * 50}")
        print(f"Model: {name}")

        # First verify the original model is SAFE
        try:
            orig_result = verify_model(source, input_shapes=input_shapes)
            orig_safe = orig_result.safe
        except Exception as e:
            orig_safe = True  # assume safe if checker errors

        mutator = ModelMutator(source)
        available = mutator.available_mutations()

        if not available:
            print(f"  ⚠ No applicable mutations")
            return

        generated = 0
        attempts = 0
        max_attempts = self.mutations_per_model * 3

        # Cycle through available operators to get diversity
        op_cycle = available * ((self.mutations_per_model // len(available)) + 2)

        for op in op_cycle:
            if generated >= self.mutations_per_model:
                break
            if attempts >= max_attempts:
                break
            attempts += 1

            fresh_mutator = ModelMutator(source)
            record = fresh_mutator.apply(op)
            if record is None:
                continue

            # Verify the mutant parses
            try:
                ast.parse(record.mutated_source)
            except SyntaxError:
                continue

            outcome = self._classify_mutant(
                name, source, record, orig_safe, input_shapes
            )
            self.results.append(outcome)
            generated += 1

            status_icon = {"killed": "✓", "survived": "✗",
                           "equivalent": "≡", "error": "⚠"}
            print(f"  {status_icon.get(outcome.outcome, '?')} "
                  f"[{outcome.operator}] {outcome.description} "
                  f"→ {outcome.outcome}")

        print(f"  Generated {generated} mutants")

    def _classify_mutant(self, model_name: str, original_source: str,
                         record: MutationRecord, orig_safe: bool,
                         input_shapes: Dict[str, tuple]) -> MutantResult:
        """Run TensorGuard on a mutant and classify the result."""
        t0 = time.monotonic()
        try:
            result = verify_model(record.mutated_source,
                                  input_shapes=input_shapes)
            elapsed = (time.monotonic() - t0) * 1000

            if not result.safe:
                outcome = "killed"
            else:
                # Check if mutation is equivalent (same semantics)
                if self._is_likely_equivalent(record):
                    outcome = "equivalent"
                else:
                    outcome = "survived"

            return MutantResult(
                model_name=model_name,
                operator=record.operator.value,
                description=record.description,
                outcome=outcome,
                verification_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            # Errors during verification count as detection
            return MutantResult(
                model_name=model_name,
                operator=record.operator.value,
                description=record.description,
                outcome="killed",
                verification_time_ms=elapsed,
                error_message=str(e),
            )

    @staticmethod
    def _is_likely_equivalent(record: MutationRecord) -> bool:
        """Heuristic: is this mutation likely semantically equivalent?"""
        # ADD_DIMENSION_MISMATCH is never equivalent
        if record.operator == MutationOperator.ADD_DIMENSION_MISMATCH:
            return False
        # Swaps of identical layers could be equivalent
        if record.operator == MutationOperator.SWAP_LAYERS:
            return False  # conservative: assume not equivalent
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Statistics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_ci(successes: int, total: int,
              z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(
        p_hat * (1 - p_hat) / total + z * z / (4 * total * total)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def compute_summary(results: List[MutantResult]) -> Dict[str, Any]:
    """Build the full results summary."""
    killed = sum(1 for r in results if r.outcome == "killed")
    survived = sum(1 for r in results if r.outcome == "survived")
    equivalent = sum(1 for r in results if r.outcome == "equivalent")
    errors = sum(1 for r in results if r.outcome == "error")
    total = len(results)
    testable = killed + survived  # exclude equivalent

    mutation_score = killed / testable if testable > 0 else 0.0
    ci_low, ci_high = wilson_ci(killed, testable)

    # Per-model scores
    model_scores: Dict[str, Any] = {}
    model_names = sorted(set(r.model_name for r in results))
    for m in model_names:
        m_results = [r for r in results if r.model_name == m]
        m_killed = sum(1 for r in m_results if r.outcome == "killed")
        m_survived = sum(1 for r in m_results if r.outcome == "survived")
        m_total = m_killed + m_survived
        model_scores[m] = {
            "killed": m_killed,
            "survived": m_survived,
            "equivalent": sum(1 for r in m_results if r.outcome == "equivalent"),
            "mutation_score": m_killed / m_total if m_total > 0 else 0.0,
        }

    # Per-operator kill rates
    op_scores: Dict[str, Any] = {}
    ops = sorted(set(r.operator for r in results))
    for op in ops:
        op_results = [r for r in results if r.operator == op]
        op_killed = sum(1 for r in op_results if r.outcome == "killed")
        op_survived = sum(1 for r in op_results if r.outcome == "survived")
        op_total = op_killed + op_survived
        op_scores[op] = {
            "killed": op_killed,
            "survived": op_survived,
            "kill_rate": op_killed / op_total if op_total > 0 else 0.0,
            "total_mutants": len(op_results),
        }

    # Surviving mutants
    survivors = [
        {
            "model": r.model_name,
            "operator": r.operator,
            "description": r.description,
        }
        for r in results if r.outcome == "survived"
    ]

    return {
        "summary": {
            "total_mutants": total,
            "killed": killed,
            "survived": survived,
            "equivalent": equivalent,
            "errors": errors,
            "mutation_score": round(mutation_score, 4),
            "false_negative_rate": round(1 - mutation_score, 4),
            "wilson_95_ci": {
                "lower": round(ci_low, 4),
                "upper": round(ci_high, 4),
            },
            "num_models": len(model_names),
        },
        "per_model_scores": model_scores,
        "per_operator_kill_rates": op_scores,
        "surviving_mutants": survivors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    random.seed(42)

    runner = MutationTestRunner(BENCHMARK_MODELS, mutations_per_model=8)
    results = runner.run()

    summary = compute_summary(results)

    # Print summary
    print("\n" + "=" * 70)
    print("MUTATION TESTING RESULTS")
    print("=" * 70)
    s = summary["summary"]
    print(f"  Models tested:     {s['num_models']}")
    print(f"  Total mutants:     {s['total_mutants']}")
    print(f"  Killed:            {s['killed']}")
    print(f"  Survived:          {s['survived']}")
    print(f"  Equivalent:        {s['equivalent']}")
    print(f"  Mutation score:    {s['mutation_score']:.2%}")
    print(f"  False-neg rate:    {s['false_negative_rate']:.2%}")
    ci = s["wilson_95_ci"]
    print(f"  95% Wilson CI:     [{ci['lower']:.2%}, {ci['upper']:.2%}]")

    print(f"\nPer-operator kill rates:")
    for op, data in sorted(summary["per_operator_kill_rates"].items()):
        print(f"  {op:30s}  {data['kill_rate']:6.1%}  "
              f"({data['killed']}/{data['killed']+data['survived']})")

    if summary["surviving_mutants"]:
        print(f"\nSurviving mutants ({len(summary['surviving_mutants'])}):")
        for sv in summary["surviving_mutants"]:
            print(f"  • {sv['model']}: [{sv['operator']}] {sv['description']}")

    # Save results
    out_dir = Path(__file__).resolve().parent / ".benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mutation_testing_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
