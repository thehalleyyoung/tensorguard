"""Symbolic config attribute handling for shape analysis.

When __init__ parameters like `config.hidden_size` are used in layer
dimensions, we create fresh symbolic variables instead of abstaining.
This enables verification of transformer blocks with config-based dims.
"""

from __future__ import annotations
from typing import List, Dict, Set, Any, Optional
import ast

# Global registry for symbolic config fields
_SYMBOLIC_CONFIG_FIELDS: Set[str] = set()


def symbolic_config(field_names: List[str]) -> None:
    """Declare config fields that should be treated as symbolic dimensions.
    
    Example:
        symbolic_config(["hidden_size", "num_heads", "intermediate_size"])
        
    Then nn.Linear(config.hidden_size, 3*config.hidden_size) will be
    treated as Linear(d, 3*d) symbolically.
    
    Args:
        field_names: List of attribute names to treat as symbolic dims.
    """
    global _SYMBOLIC_CONFIG_FIELDS
    _SYMBOLIC_CONFIG_FIELDS.update(field_names)


def detect_symbolic_config_attrs(
    init_node: ast.FunctionDef,
    param_names: Set[str] = {"config", "cfg", "args"}
) -> Set[str]:
    """Auto-detect config attributes used in layer dimensions.
    
    Heuristic: any attribute access on a parameter named 'config', 'cfg',
    or 'args' that flows into a layer constructor's dimension argument
    should be made symbolic.
    
    Args:
        init_node: The AST of the __init__ method.
        param_names: Parameter names to look for (default: config, cfg, args).
        
    Returns:
        Set of attribute names like {"hidden_size", "num_heads"}.
    """
    detected = set()
    
    # Find which params match our heuristic
    config_params = set()
    for arg in init_node.args.args:
        if arg.arg in param_names:
            config_params.add(arg.arg)
    
    if not config_params:
        return detected
    
    class ConfigAttrVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            # Look for nn.Linear(config.X, ...) patterns
            for arg in node.args:
                if isinstance(arg, ast.Attribute):
                    if isinstance(arg.value, ast.Name) and arg.value.id in config_params:
                        detected.add(arg.attr)
                elif isinstance(arg, ast.BinOp):
                    # Handle config.X * 3 or 3 * config.X patterns
                    for child in ast.walk(arg):
                        if isinstance(child, ast.Attribute):
                            if isinstance(child.value, ast.Name) and child.value.id in config_params:
                                detected.add(child.attr)
            self.generic_visit(node)
    
    visitor = ConfigAttrVisitor()
    visitor.visit(init_node)
    return detected


def resolve_config_attr(
    attr_node: ast.Attribute,
    config_param_name: str = "config",
    symbolic_fields: Optional[Set[str]] = None,
    scalar_attrs: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """Resolve a config.field attribute access.
    
    If field is in symbolic_fields, return a symbolic name.
    Otherwise, try to resolve from scalar_attrs or return None.
    
    Args:
        attr_node: AST node for the attribute access.
        config_param_name: Name of the config parameter (e.g., "config").
        symbolic_fields: Set of fields to treat as symbolic.
        scalar_attrs: Dict of known scalar values.
        
    Returns:
        Either a symbolic string like "d_hidden_size", an int, or None.
    """
    if not isinstance(attr_node, ast.Attribute):
        return None
    
    if not isinstance(attr_node.value, ast.Name):
        return None
    
    if attr_node.value.id != config_param_name:
        return None
    
    field = attr_node.attr
    
    # Check if this should be symbolic
    global _SYMBOLIC_CONFIG_FIELDS
    all_symbolic = (symbolic_fields or set()) | _SYMBOLIC_CONFIG_FIELDS
    
    if field in all_symbolic:
        return f"d_{field}"
    
    # Try to resolve from scalar_attrs
    if scalar_attrs:
        key = f"{config_param_name}.{field}"
        if key in scalar_attrs:
            return scalar_attrs[key]
    
    return None


def make_expression_symbolic(
    node: ast.expr,
    config_param_name: str = "config",
    symbolic_fields: Optional[Set[str]] = None,
    scalar_attrs: Optional[Dict[str, Any]] = None
) -> Any:
    """Convert an AST expression to symbolic form if it contains config attrs.
    
    Examples:
        config.hidden_size → "d_hidden_size"
        3 * config.hidden_size → "3*d_hidden_size"
        config.hidden_size // 2 → "d_hidden_size//2"
    
    Args:
        node: AST expression node.
        config_param_name: Name of config parameter.
        symbolic_fields: Fields to make symbolic.
        scalar_attrs: Known scalar values.
        
    Returns:
        Symbolic string, concrete int, or None.
    """
    if isinstance(node, ast.Attribute):
        return resolve_config_attr(node, config_param_name, symbolic_fields, scalar_attrs)
    
    elif isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    
    elif hasattr(ast, "Num") and isinstance(node, ast.Num):  # Python <3.8 legacy node
        return node.n  # type: ignore[attr-defined]
    
    elif isinstance(node, ast.BinOp):
        left = make_expression_symbolic(node.left, config_param_name, symbolic_fields, scalar_attrs)
        right = make_expression_symbolic(node.right, config_param_name, symbolic_fields, scalar_attrs)
        
        if left is None or right is None:
            return None
        
        # Build symbolic expression
        op_str = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
        }.get(type(node.op))
        
        if op_str:
            # If both are ints, compute
            if isinstance(left, int) and isinstance(right, int):
                ops = {
                    "+": lambda a, b: a + b,
                    "-": lambda a, b: a - b,
                    "*": lambda a, b: a * b,
                    "//": lambda a, b: a // b if b != 0 else None,
                    "%": lambda a, b: a % b if b != 0 else None,
                }
                result = ops.get(op_str, lambda a, b: None)(left, right)
                if result is not None:
                    return result
            
            # Otherwise build symbolic string
            left_str = str(left) if isinstance(left, int) else left
            right_str = str(right) if isinstance(right, int) else right
            return f"({left_str}{op_str}{right_str})"
    
    return None
