"""
Explicit interface predicates between T_perm and T_stride.

When a permutation reorders tensor axes, it also reorders the stride
pattern.  This module provides the explicit interface predicate
``stride_after_permute`` that mediates the T_perm / T_stride interaction,
ensuring signature disjointness is maintained.

The key insight: without this interface, the perm→stride relationship
is implicit (a semantic overlap between T_perm and T_stride that
violates the Nelson-Oppen signature disjointness precondition).
By making it an explicit constraint emitted by the permutation
propagator, the interaction becomes part of the shared Dim sort
managed by the combination procedure.

References
----------
- Nelson & Oppen (1979). Signature disjointness requirement.
- Tinelli & Zarba (2005). Extension to non-stably-infinite theories.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


def stride_after_permute_concrete(
    old_strides: Tuple[int, ...], perm: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute new strides after applying a permutation (concrete).

    When a permutation reorders axes, strides are reordered identically:
        new_strides[i] = old_strides[perm[i]]

    This is the fundamental connection between T_perm and T_stride.

    Args:
        old_strides: Original stride tuple, e.g. (12, 4, 1).
        perm: Permutation indices, e.g. (2, 0, 1).

    Returns:
        New stride tuple, e.g. (1, 12, 4).
    """
    if len(old_strides) != len(perm):
        raise ValueError(
            f"Stride length {len(old_strides)} != perm length {len(perm)}"
        )
    return tuple(old_strides[p] for p in perm)


def shape_after_permute_concrete(
    shape: Tuple[int, ...], perm: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute new shape after applying a permutation (concrete).

    Args:
        shape: Original shape, e.g. (2, 3, 4).
        perm: Permutation indices, e.g. (2, 0, 1).

    Returns:
        Permuted shape, e.g. (4, 2, 3).
    """
    return tuple(shape[p] for p in perm)


def is_contiguous_after_permute(
    shape: Tuple[int, ...],
    perm: Tuple[int, ...],
) -> bool:
    """Check if a contiguous tensor remains contiguous after permutation.

    A contiguous (row-major) tensor has strides[i] = ∏_{j>i} shape[j].
    After permutation, the tensor is contiguous iff perm is the identity.
    """
    return perm == tuple(range(len(perm)))


if HAS_Z3:

    def stride_after_permute_symbolic(
        old_strides: List[z3.ExprRef],
        perm: Tuple[int, ...],
        new_strides: List[z3.ExprRef],
    ) -> z3.ExprRef:
        """Assert the perm→stride interface constraint symbolically.

        Emits: ∧_i new_strides[i] = old_strides[perm[i]]

        This is the Z3-level interface predicate that the permutation
        propagator should emit whenever it applies a permutation, to
        ensure T_stride sees the stride reordering.

        Args:
            old_strides: Z3 Int vars for original strides.
            perm: Concrete permutation indices.
            new_strides: Z3 Int vars for permuted strides.

        Returns:
            Z3 conjunction encoding the stride reordering.
        """
        n = len(old_strides)
        if len(new_strides) != n or len(perm) != n:
            return z3.BoolVal(False)
        if not (set(perm) == set(range(n))):
            return z3.BoolVal(False)

        clauses = []
        for i in range(n):
            clauses.append(new_strides[i] == old_strides[perm[i]])

        return z3.And(*clauses) if clauses else z3.BoolVal(True)

    def contiguous_after_permute_constraint(
        shape: List[z3.ExprRef],
        perm: Tuple[int, ...],
        new_strides: List[z3.ExprRef],
    ) -> z3.ExprRef:
        """Assert that new_strides match contiguous layout for permuted shape.

        This combines the permutation and contiguous stride constraints:
        1. Permute shape: new_shape[i] = shape[perm[i]]
        2. Contiguous: new_strides[n-1] = 1, new_strides[i] = new_strides[i+1] * new_shape[i+1]

        Args:
            shape: Z3 Int vars for original shape.
            perm: Concrete permutation.
            new_strides: Z3 Int vars for output strides.

        Returns:
            Z3 conjunction asserting contiguous strides for the permuted shape.
        """
        n = len(shape)
        if len(new_strides) != n or len(perm) != n:
            return z3.BoolVal(False)

        # Build permuted shape symbolically
        new_shape = [shape[perm[i]] for i in range(n)]

        # Assert contiguous strides for new_shape
        clauses = [new_strides[n - 1] == z3.IntVal(1)]
        for i in range(n - 2, -1, -1):
            clauses.append(
                new_strides[i] == new_strides[i + 1] * new_shape[i + 1]
            )

        return z3.And(*clauses) if clauses else z3.BoolVal(True)
