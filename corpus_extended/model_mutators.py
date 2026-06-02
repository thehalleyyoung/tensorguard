"""Deterministic source-level mutation operators for clean models (Step 112).

Mutation testing turns the "no false alarms" story (Step 111, clean corpus) into
its dual: if we *inject* a genuine bug into an otherwise-clean model, does the
verifier catch it? A mutant that runtime-fails but is still reported ``SAFE`` is
a *survived* mutant -- a hole in coverage (or, worse, an unsoundness). The
fraction of genuine-bug mutants the verifier reports ``UNSAFE`` is its *kill
rate*.

This module provides the mutation *operators*. Each operator is a pure function
``source -> Optional[str]`` that rewrites the first applicable site in a model's
source text and returns the mutated source, or ``None`` if the operator does not
apply to that model. Operators are deliberately small, local and deterministic
(first-match only, no randomness) so the resulting corpus of mutants is fully
reproducible and each mutant differs from its parent by a single, explainable
edit.

The harness (``reproducibility/mutation_clean_models.py``) is responsible for
*validating* that a mutant is a genuine runtime bug (forward pass raises under
real PyTorch) before scoring it -- an operator may produce a still-valid model
(e.g. bumping the out-features of the final layer of a one-layer net), and those
non-bugs are discarded rather than counted.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional

# A mutation operator: model source -> mutated source (or None if inapplicable).
MutationOperator = Callable[[str], Optional[str]]


def _bump_second_int(pattern: str, source: str, delta: int = 1) -> Optional[str]:
    """Bump the 2nd captured integer of the first ``pattern`` match by ``delta``.

    ``pattern`` must capture two integer groups; the second is rewritten. Returns
    None if the pattern does not match.
    """
    m = re.search(pattern, source)
    if m is None:
        return None
    new_val = int(m.group(2)) + delta
    start, end = m.span(2)
    return source[:start] + str(new_val) + source[end:]


def _bump_first_int(pattern: str, source: str, delta: int = 1) -> Optional[str]:
    """Bump the 1st captured integer of the first ``pattern`` match by ``delta``."""
    m = re.search(pattern, source)
    if m is None:
        return None
    new_val = int(m.group(1)) + delta
    start, end = m.span(1)
    return source[:start] + str(new_val) + source[end:]


# ``nn.Linear(in, out)`` -- two leading positional ints.
_LINEAR_RE = r"nn\.Linear\(\s*(\d+)\s*,\s*(\d+)"
# ``nn.Conv2d(in, out, ...)`` -- two leading positional ints.
_CONV2D_RE = r"nn\.Conv2d\(\s*(\d+)\s*,\s*(\d+)"


def linear_out_bump(source: str) -> Optional[str]:
    """Increase the out_features of the first ``nn.Linear`` by one.

    Breaks the chain whenever a downstream layer consumes that activation with a
    concrete expected width (the common case in multi-layer nets).
    """
    return _bump_second_int(_LINEAR_RE, source)


def linear_in_bump(source: str) -> Optional[str]:
    """Increase the in_features of the first ``nn.Linear`` by one.

    Breaks whenever the tensor feeding that layer has a concrete last dim.
    """
    return _bump_first_int(_LINEAR_RE, source)


def conv_out_bump(source: str) -> Optional[str]:
    """Increase the out_channels of the first ``nn.Conv2d`` by one."""
    return _bump_second_int(_CONV2D_RE, source)


def conv_in_bump(source: str) -> Optional[str]:
    """Increase the in_channels of the first ``nn.Conv2d`` by one."""
    return _bump_first_int(_CONV2D_RE, source)


_FORWARD_RE = re.compile(
    r"(\n(?P<indent>[ \t]+)def forward\(self,\s*(?P<arg>[A-Za-z_]\w*)[^)]*\):\n)"
)


def dtype_long_cast(source: str) -> Optional[str]:
    """Insert ``<arg> = <arg>.long()`` as the first statement of ``forward``.

    Feeding an integer tensor into a float ``Linear``/``Conv`` matmul raises at
    runtime; this exercises the dtype domain rather than the shape domain.
    """
    m = _FORWARD_RE.search(source)
    if m is None:
        return None
    indent = m.group("indent")
    arg = m.group("arg")
    body_indent = indent + "    "
    insert = f"{body_indent}{arg} = {arg}.long()\n"
    end = m.end()
    return source[:end] + insert + source[end:]


# Registry: name -> operator. Order is the canonical reporting order.
OPERATORS: Dict[str, MutationOperator] = {
    "linear_out_bump": linear_out_bump,
    "linear_in_bump": linear_in_bump,
    "conv_out_bump": conv_out_bump,
    "conv_in_bump": conv_in_bump,
    "dtype_long_cast": dtype_long_cast,
}

# Which abstract domain each operator is designed to exercise.
OPERATOR_DOMAIN: Dict[str, str] = {
    "linear_out_bump": "shape",
    "linear_in_bump": "shape",
    "conv_out_bump": "shape",
    "conv_in_bump": "shape",
    "dtype_long_cast": "dtype",
}
