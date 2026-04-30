"""
Abstain reason taxonomy classifier for TensorGuard analysis results.

Classifies each abstention into one of:
- OPAQUE_SUBMODULE: submodule passed via __init__ arg
- CONFIG_INDIRECTION: config.X style attribute on opaque object
- DATA_DEPENDENT_CONTROL: if x.size(-1) > 0:
- UNSUPPORTED_OP: operator not in the DSL
- UNRESOLVED_HELPER: non-DSL helper call
- RNN_RECURRENCE: RNN/LSTM/GRU
- CUSTOM_AUTOGRAD: autograd.Function subclass
- OTHER: catch-all
"""
from __future__ import annotations
import ast
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api import AnalysisResult


def classify_abstain(result: AnalysisResult, source: str) -> str:
    """
    Classify why TensorGuard abstained on this module.
    
    Args:
        result: AnalysisResult from verify_architecture
        source: source code of the module
        
    Returns:
        One of the taxonomy labels
    """
    if not result.abstained:
        return "N/A"
    
    # Parse source AST
    try:
        tree = ast.parse(source)
    except:
        return "OTHER"
    
    # Check for RNN/LSTM/GRU
    if _has_rnn_recurrence(tree, source):
        return "RNN_RECURRENCE"
    
    # Check for custom autograd
    if _has_custom_autograd(tree, source):
        return "CUSTOM_AUTOGRAD"
    
    # Check for opaque submodules (modules passed as init args)
    if _has_opaque_submodule(tree, source):
        return "OPAQUE_SUBMODULE"
    
    # Check for config indirection (config.X patterns)
    if _has_config_indirection(tree, source):
        return "CONFIG_INDIRECTION"
    
    # Check for data-dependent control flow
    if _has_data_dependent_control(tree, source):
        return "DATA_DEPENDENT_CONTROL"
    
    # Check for unsupported ops
    if _has_unsupported_op(tree, source):
        return "UNSUPPORTED_OP"
    
    # Check for unresolved helper calls
    if _has_unresolved_helper(tree, source):
        return "UNRESOLVED_HELPER"
    
    return "OTHER"


def _has_rnn_recurrence(tree: ast.AST, source: str) -> bool:
    """Check for RNN/LSTM/GRU."""
    rnn_patterns = ['nn.LSTM', 'nn.GRU', 'nn.RNN', 'torch.lstm', 'torch.gru']
    for pattern in rnn_patterns:
        if pattern in source:
            return True
    
    # Also check for recurrent loops
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Check if loop variable is used in indexing previous timestep
            if _looks_like_recurrent_loop(node):
                return True
    return False


def _looks_like_recurrent_loop(node: ast.For) -> bool:
    """Heuristic: loop with state updates that reference previous iterations."""
    # Look for patterns like: h[t] = f(h[t-1], x[t])
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            if isinstance(child.slice, ast.BinOp):
                if isinstance(child.slice.op, ast.Sub):
                    return True
    return False


def _has_custom_autograd(tree: ast.AST, source: str) -> bool:
    """Check for autograd.Function subclass."""
    if 'autograd.Function' in source or 'Function.apply' in source:
        return True
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Attribute):
                    if base.attr == 'Function':
                        return True
    return False


def _has_opaque_submodule(tree: ast.AST, source: str) -> bool:
    """Check for modules passed as __init__ args."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            # Check for nn.Module typed args or args stored as self.X
            for arg in node.args.args[1:]:  # Skip self
                arg_name = arg.arg
                # Look for patterns like: self.layer = layer
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute):
                                if isinstance(stmt.value, ast.Name):
                                    if stmt.value.id == arg_name:
                                        # This is storing an arg as instance attr
                                        return True
    
    # Also check for common patterns in source
    opaque_patterns = [
        r'self\.\w+\s*=\s*\w+\s*#.*(?:module|layer|block)',
        r'def __init__\(.*,\s*\w+:\s*nn\.Module',
    ]
    for pattern in opaque_patterns:
        if re.search(pattern, source, re.IGNORECASE):
            return True
    
    return False


def _has_config_indirection(tree: ast.AST, source: str) -> bool:
    """Check for config.X style attributes."""
    config_patterns = [
        r'config\.\w+',
        r'cfg\.\w+',
        r'self\.config\.\w+',
        r'self\.cfg\.\w+',
    ]
    for pattern in config_patterns:
        if re.search(pattern, source):
            return True
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in ('config', 'cfg'):
                    return True
            elif isinstance(node.value, ast.Attribute):
                if node.value.attr in ('config', 'cfg'):
                    return True
    
    return False


def _has_data_dependent_control(tree: ast.AST, source: str) -> bool:
    """Check for data-dependent control flow (if x.size(...) etc)."""
    # Look for if statements that check tensor properties
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check if condition involves .size(), .shape, .dim(), etc.
            if _condition_checks_tensor_property(node.test):
                return True
    
    # Also check source patterns
    control_patterns = [
        r'if\s+\w+\.size\(',
        r'if\s+\w+\.shape',
        r'if\s+\w+\.dim\(',
        r'if\s+len\(\w+\.shape\)',
    ]
    for pattern in control_patterns:
        if re.search(pattern, source):
            return True
    
    return False


def _condition_checks_tensor_property(node: ast.AST) -> bool:
    """Check if condition involves tensor shape/size queries."""
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            if child.attr in ('size', 'shape', 'dim', 'numel', 'ndim'):
                return True
        elif isinstance(child, ast.Call):
            if isinstance(child.func, ast.Attribute):
                if child.func.attr in ('size', 'dim'):
                    return True
    return False


def _has_unsupported_op(tree: ast.AST, source: str) -> bool:
    """Check for operators not in the DSL."""
    # Common unsupported patterns
    unsupported_patterns = [
        r'torch\.fft\.',
        r'torch\.linalg\.',
        r'F\.grid_sample',
        r'torch\.einsum',
        r'\.unfold\(',
        r'\.fold\(',
        r'torch\.stft',
        r'torch\.istft',
    ]
    for pattern in unsupported_patterns:
        if re.search(pattern, source):
            return True
    
    return False


def _has_unresolved_helper(tree: ast.AST, source: str) -> bool:
    """Check for calls to helper functions not in the DSL."""
    # Look for function calls to non-standard helpers
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                # If it's not a builtin or torch op, might be unresolved
                if not _is_known_function(func_name):
                    return True
    
    return False


def _is_known_function(name: str) -> bool:
    """Check if function name is a known builtin or standard library."""
    known = {
        'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'sum', 'max', 'min', 'abs', 'pow', 'round', 'int', 'float',
        'str', 'bool', 'list', 'dict', 'tuple', 'set',
        'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
    }
    return name in known or name.startswith('torch') or name.startswith('F')
