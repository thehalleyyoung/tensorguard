"""
Custom Z3 Theory Plugin for Tensor Axis Permutation Constraints.

Encodes axis identity and reordering relationships for transpose and
permute operations on tensor shapes.  For a tensor with shape
(d0, d1, ..., dn), a permutation π produces an output tensor with:

    output[i] = input[π(i)]

Formal Specification
--------------------

**Signature** Σ_perm = (S, F, P) where:
  - Sorts S = {Dim, Perm} (Dim: positive integers ≥ 1; Perm: finite
    sequences of distinct indices in [0, n))
  - Functions F:
    apply_perm  : Shape × Perm → Shape     (general permutation)
    transpose   : Shape × Idx × Idx → Shape (swap two axes)
    identity    : n → Perm                   (identity permutation [0..n))
    compose     : Perm × Perm → Perm         (permutation composition)
    inverse     : Perm → Perm                (permutation inverse)
  - Predicates P:
    valid_perm  : Perm × n                   (indices in [0,n), all distinct)
    axis_eq     : Shape × Shape × Idx        (dimension i is equal)

**Axioms**:
  A1 (permute def):     apply_perm(s, π)[i] = s[π[i]]  ∀i < n
  A2 (transpose def):   transpose(s, a, b) = apply_perm(s, swap(a,b))
                         where swap(a,b)[i] = b if i=a, a if i=b, i otherwise
  A3 (element count):   numel(apply_perm(s, π)) = numel(s)
  A4 (rank preserve):   len(apply_perm(s, π)) = len(s)
  A5 (identity):        apply_perm(s, identity(n)) = s
  A6 (compose):         apply_perm(s, compose(π₁, π₂)) =
                         apply_perm(apply_perm(s, π₂), π₁)
  A7 (inverse):         compose(π, inverse(π)) = identity(n)
  A8 (valid perm):      valid_perm(π, n) ⟺ |π| = n ∧ π ⊆ [0,n) ∧ distinct(π)

**Decision procedure**: For the quantifier-free fragment with concrete
  ranks and permutations, satisfiability reduces to QF-LIA (decidable).
  The UserPropagator eagerly propagates dimension equalities when input
  dimensions are fixed, using the known permutation mapping.

**Soundness**: Propagated equalities follow directly from A1-A2.
  Conflicts arise only when concrete values violate axis identity.

This module provides:
  - A Z3 UserPropagateBase that eagerly propagates dimension values
    through permutation and transpose operations.
  - Constraint generation functions for transpose and permute.
  - High-level ``PermutationTheoryPlugin`` class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Z3 import with graceful fallback
# ---------------------------------------------------------------------------

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pure permutation helpers
# ═══════════════════════════════════════════════════════════════════════════


def is_valid_permutation(perm: Tuple[int, ...], n: int) -> bool:
    """Check that perm is a valid permutation of [0, n).

    Args:
        perm: Candidate permutation tuple.
        n: Expected length.

    Returns:
        True iff perm has length n, all values in [0,n), and all distinct.
    """
    if len(perm) != n:
        return False
    return set(perm) == set(range(n))


def apply_concrete_permutation(
    shape: Tuple[int, ...], perm: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Apply a permutation to a concrete shape.

    Args:
        shape: Input shape, e.g. (2, 3, 4).
        perm: Permutation indices, e.g. (2, 0, 1).

    Returns:
        Permuted shape, e.g. (4, 2, 3).
    """
    return tuple(shape[p] for p in perm)


def apply_concrete_transpose(
    shape: Tuple[int, ...], dim0: int, dim1: int
) -> Tuple[int, ...]:
    """Apply a transpose (swap two axes) to a concrete shape.

    Args:
        shape: Input shape.
        dim0: First axis to swap.
        dim1: Second axis to swap.

    Returns:
        Shape with dim0 and dim1 swapped.
    """
    lst = list(shape)
    lst[dim0], lst[dim1] = lst[dim1], lst[dim0]
    return tuple(lst)


def compose_permutations(
    p1: Tuple[int, ...], p2: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compose two permutations: result[i] = p1[p2[i]]."""
    return tuple(p1[p2[i]] for i in range(len(p2)))


def inverse_permutation(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute the inverse permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def swap_permutation(n: int, dim0: int, dim1: int) -> Tuple[int, ...]:
    """Construct the permutation that swaps dim0 and dim1."""
    perm = list(range(n))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return tuple(perm)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Trail for backtracking
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _PermTrailFrame:
    """Snapshot of permutation theory state at a push point."""

    fixed_vars: Dict[int, int]


# ═══════════════════════════════════════════════════════════════════════════
# 3. PermutationPropagator — Z3 UserPropagateBase implementation
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    class PermutationPropagator(z3.UserPropagateBase):
        """Z3 theory propagator for axis permutation constraints.

        Implements eager propagation of dimension equalities through
        transpose and permute operations.  Supports:

        - Transpose: output_dims[dim0] = input_dims[dim1] and vice versa;
          all other dims preserved.
        - Permute: output_dims[i] = input_dims[perm[i]] for all i.
        """

        def __init__(self, s: z3.Solver) -> None:
            super().__init__(s)

            self._vars: Dict[int, z3.ExprRef] = {}
            self._fixed: Dict[int, int] = {}
            self._trail: List[_PermTrailFrame] = []

            # Constraint storage
            # transpose: (input_dims, dim0, dim1, output_dims)
            self._transposes: List[
                Tuple[List[z3.ExprRef], int, int, List[z3.ExprRef]]
            ] = []
            # permute: (input_dims, perm, output_dims)
            self._permutations: List[
                Tuple[List[z3.ExprRef], Tuple[int, ...], List[z3.ExprRef]]
            ] = []

            self.add_fixed(self._on_fixed)
            self.add_final(self._on_final)
            self.add_created(self._on_created)

        # ---------------------------------------------------------------
        # Variable registration
        # ---------------------------------------------------------------

        def _register_var(self, v: z3.ExprRef) -> None:
            vid = v.get_id()
            if vid not in self._vars:
                self._vars[vid] = v
                self.add(v)

        # ---------------------------------------------------------------
        # Backtracking
        # ---------------------------------------------------------------

        def push(self) -> None:
            self._trail.append(
                _PermTrailFrame(fixed_vars=dict(self._fixed))
            )

        def pop(self, num_scopes: int) -> None:
            for _ in range(num_scopes):
                if self._trail:
                    frame = self._trail.pop()
                    self._fixed = frame.fixed_vars

        # ---------------------------------------------------------------
        # Callbacks
        # ---------------------------------------------------------------

        def _on_created(self, var: z3.ExprRef) -> None:
            self._vars[var.get_id()] = var

        def _on_fixed(self, var: z3.ExprRef, value: z3.ExprRef) -> None:
            """Eagerly propagate when Z3 fixes a dimension variable."""
            vid = var.get_id()
            try:
                concrete = value.as_long()
            except (AttributeError, z3.Z3Exception):
                return
            self._fixed[vid] = concrete

            for inp_dims, d0, d1, out_dims in self._transposes:
                self._propagate_transpose(inp_dims, d0, d1, out_dims)

            for inp_dims, perm, out_dims in self._permutations:
                self._propagate_permutation(inp_dims, perm, out_dims)

        def _on_final(self) -> None:
            """Final consistency check."""
            for inp_dims, d0, d1, out_dims in self._transposes:
                self._check_transpose_final(inp_dims, d0, d1, out_dims)
            for inp_dims, perm, out_dims in self._permutations:
                self._check_permutation_final(inp_dims, perm, out_dims)

        # ---------------------------------------------------------------
        # Transpose propagation
        # ---------------------------------------------------------------

        def _propagate_transpose(
            self,
            inp_dims: List[z3.ExprRef],
            dim0: int,
            dim1: int,
            out_dims: List[z3.ExprRef],
        ) -> None:
            """Propagate dimension equalities for a transpose operation."""
            n = len(inp_dims)
            for i in range(n):
                if i == dim0:
                    src_idx = dim1
                elif i == dim1:
                    src_idx = dim0
                else:
                    src_idx = i

                src_val = self._fixed.get(inp_dims[src_idx].get_id())
                dst_val = self._fixed.get(out_dims[i].get_id())

                if src_val is not None and dst_val is None:
                    self.propagate(
                        out_dims[i] == z3.IntVal(src_val),
                        ids=[inp_dims[src_idx]],
                    )
                elif src_val is not None and dst_val is not None:
                    if src_val != dst_val:
                        self.conflict(
                            deps=[inp_dims[src_idx], out_dims[i]]
                        )
                        return

        def _check_transpose_final(
            self,
            inp_dims: List[z3.ExprRef],
            dim0: int,
            dim1: int,
            out_dims: List[z3.ExprRef],
        ) -> None:
            """Final check that transpose is consistent."""
            n = len(inp_dims)
            for i in range(n):
                src_idx = dim1 if i == dim0 else (dim0 if i == dim1 else i)
                sv = self._fixed.get(inp_dims[src_idx].get_id())
                dv = self._fixed.get(out_dims[i].get_id())
                if sv is not None and dv is not None and sv != dv:
                    self.conflict(
                        deps=list(inp_dims) + list(out_dims)
                    )
                    return

        # ---------------------------------------------------------------
        # Permutation propagation
        # ---------------------------------------------------------------

        def _propagate_permutation(
            self,
            inp_dims: List[z3.ExprRef],
            perm: Tuple[int, ...],
            out_dims: List[z3.ExprRef],
        ) -> None:
            """Propagate dimension equalities for a permute operation."""
            for i, p in enumerate(perm):
                src_val = self._fixed.get(inp_dims[p].get_id())
                dst_val = self._fixed.get(out_dims[i].get_id())

                if src_val is not None and dst_val is None:
                    self.propagate(
                        out_dims[i] == z3.IntVal(src_val),
                        ids=[inp_dims[p]],
                    )
                elif src_val is not None and dst_val is not None:
                    if src_val != dst_val:
                        self.conflict(
                            deps=[inp_dims[p], out_dims[i]]
                        )
                        return

        def _check_permutation_final(
            self,
            inp_dims: List[z3.ExprRef],
            perm: Tuple[int, ...],
            out_dims: List[z3.ExprRef],
        ) -> None:
            """Final check that permutation is consistent."""
            for i, p in enumerate(perm):
                sv = self._fixed.get(inp_dims[p].get_id())
                dv = self._fixed.get(out_dims[i].get_id())
                if sv is not None and dv is not None and sv != dv:
                    self.conflict(
                        deps=list(inp_dims) + list(out_dims)
                    )
                    return

    # ═══════════════════════════════════════════════════════════════════════
    # 4. High-level constraint builders
    # ═══════════════════════════════════════════════════════════════════════

    def apply_permutation(
        prop: PermutationPropagator,
        input_dims: List[z3.ExprRef],
        perm: Tuple[int, ...],
        output_dims: List[z3.ExprRef],
    ) -> z3.ExprRef:
        """Assert permutation constraint: output_dims[i] = input_dims[perm[i]].

        Args:
            prop: The permutation propagator.
            input_dims: Z3 Int vars for input shape.
            perm: Permutation indices.
            output_dims: Z3 Int vars for output shape.

        Returns:
            Z3 Bool conjunction encoding the permutation.

        Raises:
            ValueError: If perm is not a valid permutation of [0, n).
        """
        n = len(input_dims)
        if len(output_dims) != n:
            return z3.BoolVal(False)
        if not is_valid_permutation(perm, n):
            return z3.BoolVal(False)

        for v in list(input_dims) + list(output_dims):
            prop._register_var(v)
        prop._permutations.append(
            (list(input_dims), perm, list(output_dims))
        )

        clauses = []
        for i, p in enumerate(perm):
            clauses.append(output_dims[i] == input_dims[p])
        return z3.And(*clauses) if clauses else z3.BoolVal(True)

    def apply_transpose(
        prop: PermutationPropagator,
        input_dims: List[z3.ExprRef],
        dim0: int,
        dim1: int,
        output_dims: List[z3.ExprRef],
    ) -> z3.ExprRef:
        """Assert transpose constraint: swap dim0 and dim1.

        Args:
            prop: The permutation propagator.
            input_dims: Z3 Int vars for input shape.
            dim0: First axis to swap.
            dim1: Second axis to swap.
            output_dims: Z3 Int vars for output shape.

        Returns:
            Z3 Bool conjunction encoding the transpose.

        Raises:
            ValueError: If dim0 or dim1 is out of range.
        """
        n = len(input_dims)
        if len(output_dims) != n:
            return z3.BoolVal(False)
        if not (0 <= dim0 < n and 0 <= dim1 < n):
            return z3.BoolVal(False)

        for v in list(input_dims) + list(output_dims):
            prop._register_var(v)
        prop._transposes.append(
            (list(input_dims), dim0, dim1, list(output_dims))
        )

        clauses = []
        for i in range(n):
            if i == dim0:
                clauses.append(output_dims[i] == input_dims[dim1])
            elif i == dim1:
                clauses.append(output_dims[i] == input_dims[dim0])
            else:
                clauses.append(output_dims[i] == input_dims[i])
        return z3.And(*clauses) if clauses else z3.BoolVal(True)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. PermutationTheoryPlugin — convenience wrapper
    # ═══════════════════════════════════════════════════════════════════════

    class PermutationTheoryPlugin:
        """High-level integration wrapper for attaching the permutation
        theory to any Z3 Solver.

        Usage::

            solver = z3.Solver()
            plugin = PermutationTheoryPlugin(solver)
            d0, d1, d2 = z3.Ints("d0 d1 d2")
            o0, o1, o2 = z3.Ints("o0 o1 o2")
            solver.add(plugin.apply_transpose([d0,d1,d2], 0, 2, [o0,o1,o2]))
            solver.add(d0 == 2, d1 == 3, d2 == 4)
            assert solver.check() == z3.sat
            # o0 == 4, o1 == 3, o2 == 2
        """

        def __init__(self, solver: z3.Solver) -> None:
            self.solver = solver
            self.propagator = PermutationPropagator(solver)

        def apply_permutation(
            self,
            input_dims: List[z3.ExprRef],
            perm: Tuple[int, ...],
            output_dims: List[z3.ExprRef],
        ) -> z3.ExprRef:
            """Assert permutation: output_dims[i] = input_dims[perm[i]]."""
            return apply_permutation(
                self.propagator, input_dims, perm, output_dims
            )

        def apply_transpose(
            self,
            input_dims: List[z3.ExprRef],
            dim0: int,
            dim1: int,
            output_dims: List[z3.ExprRef],
        ) -> z3.ExprRef:
            """Assert transpose: swap dim0 and dim1."""
            return apply_transpose(
                self.propagator, input_dims, dim0, dim1, output_dims
            )
