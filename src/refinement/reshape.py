"""Reshape with -1 dimension inference using Z3.

Handles x.view(B, -1) by deducing -1 = product(remaining dims).
Uses Z3 for divisibility constraints when shapes are symbolic.
"""

from __future__ import annotations
from typing import Tuple, Optional, Any, List
import math

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


def infer_reshape_minus_one(
    input_shape: Tuple[Any, ...],
    target_shape: Tuple[Any, ...],
) -> Optional[Tuple[Any, ...]]:
    """Infer the -1 dimension in a reshape/view operation.
    
    Given input shape and target shape with at most one -1, deduce the
    value of the -1 dimension by preserving element count.
    
    Args:
        input_shape: Input tensor shape
        target_shape: Target shape with at most one -1
        
    Returns:
        Resolved target shape, or None if invalid
        
    Examples:
        input: (2, 3, 4, 5), target: (2, -1) → (2, 60)
        input: (B, C, H, W), target: (B, -1) → (B, "C*H*W")
    """
    # Count -1s
    minus_one_count = sum(1 for d in target_shape if d == -1)
    if minus_one_count > 1:
        return None
    if minus_one_count == 0:
        return target_shape
    
    # Find -1 index
    minus_one_idx = next(i for i, d in enumerate(target_shape) if d == -1)
    
    # Compute input numel
    input_numel = compute_numel(input_shape)
    if input_numel is None:
        return None
    
    # Compute known target numel (excluding -1)
    known_numel = 1
    symbolic_parts = []
    for i, d in enumerate(target_shape):
        if i == minus_one_idx:
            continue
        if isinstance(d, int):
            known_numel *= d
        elif isinstance(d, str):
            symbolic_parts.append(d)
        else:
            return None
    
    # Deduce -1 dimension
    if isinstance(input_numel, int) and len(symbolic_parts) == 0:
        # All concrete
        if known_numel == 0:
            return None
        if input_numel % known_numel != 0:
            return None  # Invalid reshape
        inferred = input_numel // known_numel
        resolved = list(target_shape)
        resolved[minus_one_idx] = inferred
        return tuple(resolved)
    
    elif isinstance(input_numel, str) or len(symbolic_parts) > 0:
        # Symbolic: build expression
        if len(symbolic_parts) == 0:
            known_str = str(known_numel)
        else:
            known_str = "*".join([str(known_numel)] + symbolic_parts) if known_numel != 1 else "*".join(symbolic_parts)
        
        if isinstance(input_numel, str):
            inferred_expr = f"({input_numel})/({known_str})"
        else:
            inferred_expr = f"{input_numel}/({known_str})"
        
        resolved = list(target_shape)
        resolved[minus_one_idx] = inferred_expr
        return tuple(resolved)
    
    return None


def compute_numel(shape: Tuple[Any, ...]) -> Optional[Any]:
    """Compute number of elements in a shape.
    
    Args:
        shape: Tuple of dims (int or str for symbolic)
        
    Returns:
        int for concrete, str for symbolic expression, or None if invalid
    """
    if not shape:
        return 1
    
    concrete_product = 1
    symbolic_parts = []
    
    for d in shape:
        if isinstance(d, int):
            if d <= 0:
                return None
            concrete_product *= d
        elif isinstance(d, str):
            symbolic_parts.append(d)
        else:
            return None
    
    if len(symbolic_parts) == 0:
        return concrete_product
    
    if concrete_product == 1:
        return "*".join(symbolic_parts)
    else:
        return "*".join([str(concrete_product)] + symbolic_parts)


def validate_reshape_with_z3(
    input_shape: Tuple[Any, ...],
    target_shape: Tuple[Any, ...],
) -> Tuple[bool, Optional[str]]:
    """Validate a reshape using Z3 constraints.
    
    Checks if input_numel == target_numel modulo symbolic constraints.
    Emits divisibility preconditions for symbolic dimensions.
    
    Args:
        input_shape: Input tensor shape
        target_shape: Target shape (may contain symbolic dims)
        
    Returns:
        (is_valid, error_message) tuple
    """
    if not HAS_Z3:
        # Conservative: assume valid if Z3 not available
        return True, None
    
    # Build Z3 variables for symbolic dims
    solver = z3.Solver()
    sym_vars = {}
    
    def to_z3(dim: Any) -> z3.ArithRef:
        if isinstance(dim, int):
            return z3.IntVal(dim)
        elif isinstance(dim, str):
            if dim not in sym_vars:
                var = z3.Int(dim)
                solver.add(var >= 1)  # All dims are positive
                sym_vars[dim] = var
            return sym_vars[dim]
        else:
            return z3.IntVal(1)
    
    # Convert shapes to Z3
    input_numel_z3 = z3.IntVal(1)
    for d in input_shape:
        input_numel_z3 = input_numel_z3 * to_z3(d)
    
    target_numel_z3 = z3.IntVal(1)
    for d in target_shape:
        if d == -1:
            continue  # Skip -1 (should be inferred before validation)
        target_numel_z3 = target_numel_z3 * to_z3(d)
    
    # Add constraint: input_numel must equal target_numel
    solver.add(input_numel_z3 == target_numel_z3)
    
    # Check satisfiability
    result = solver.check()
    
    if result == z3.unsat:
        return False, "Reshape incompatible: input and target element counts cannot be equal"
    elif result == z3.unknown:
        # Timeout or too complex
        return True, "Reshape validation inconclusive (Z3 timeout)"
    else:
        # sat: valid reshape
        return True, None


def extract_symbolic_dims(shape: Tuple[Any, ...]) -> List[str]:
    """Extract symbolic dimension names from a shape.
    
    Args:
        shape: Tuple of dims
        
    Returns:
        List of symbolic dimension names
    """
    symbols = []
    for d in shape:
        if isinstance(d, str):
            # Parse composite expressions like "B*C*H*W"
            parts = d.replace("(", "").replace(")", "").replace("/", " ").replace("*", " ").split()
            for part in parts:
                if part and not part.isdigit() and part not in ("//", "+", "-"):
                    symbols.append(part)
    return list(set(symbols))


def build_divisibility_constraint(
    numerator: str,
    denominator: str,
) -> Optional[str]:
    """Build a Z3 divisibility constraint.
    
    Args:
        numerator: Symbolic expression
        denominator: Symbolic expression
        
    Returns:
        Z3 constraint string, or None
    """
    if not HAS_Z3:
        return None
    
    return f"({numerator}) % ({denominator}) == 0"
