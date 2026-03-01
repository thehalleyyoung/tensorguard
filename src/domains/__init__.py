"""
Abstract Domains — Reduced product of numeric, type-tag, nullity, and string domains.

Each domain is equipped with abstraction (α) and concretization (γ) maps that
form a sound abstract interpretation framework.  For the guard predicate domain,
this constitutes a Galois connection in the Cousot-Cousot sense; for other
domains (e.g., the neuro-symbolic pipeline lattice), it is a practical
monotone abstraction that preserves soundness without the full Galois
insertion/connection properties.
"""

from src.domains.base import AbstractDomain, AbstractValue, Lattice
from src.domains.intervals import IntervalDomain, Interval, Bound
from src.domains.typetags import TypeTagDomain, TypeTagSet
from src.domains.nullity import NullityDomain, NullityValue
from src.domains.product import ReducedProductDomain, ProductValue

__all__ = [
    "AbstractDomain",
    "AbstractValue",
    "Lattice",
    "IntervalDomain",
    "Interval",
    "Bound",
    "TypeTagDomain",
    "TypeTagSet",
    "NullityDomain",
    "NullityValue",
    "ReducedProductDomain",
    "ProductValue",
]
