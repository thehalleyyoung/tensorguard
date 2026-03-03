"""Utility functions with refinement type patterns."""

from typing import List, Optional


def safe_divide(a: float, b: float) -> float:
    """Division with a non-zero precondition on b."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


def get_item(items: List[str], index: int) -> str:
    """Index access with bounds checking."""
    if index < 0 or index >= len(items):
        raise IndexError(f"Index {index} out of range [0, {len(items)})")
    return items[index]


def parse_positive_int(value: str) -> int:
    """Parse a string as a positive integer."""
    result = int(value)
    if result <= 0:
        raise ValueError(f"Expected positive integer, got {result}")
    return result


def first_or_none(items: List[str]) -> Optional[str]:
    """Return first element or None if empty."""
    if len(items) == 0:
        return None
    return items[0]
