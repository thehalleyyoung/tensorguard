"""Step 78 — deprecation utilities backing the SemVer stability guarantee.

TensorGuard follows Semantic Versioning (see ``DEPRECATION_POLICY.md``). Public
API or CLI surface is never removed without first being deprecated for at least
one minor release, emitting a ``DeprecationWarning`` that names the replacement
and the version in which the symbol will be removed.

These helpers make that policy mechanical and testable:

* :func:`deprecated` — decorator that warns when a function/method is called.
* :func:`deprecated_alias` — expose an old name that forwards to a new callable.
* :func:`warn_deprecated_parameter` — warn when a deprecated keyword is passed.
* :func:`parse_version` / :func:`is_compatible` — minimal SemVer helpers used by
  the stability tests to reason about the public version.
"""

from __future__ import annotations

import functools
import re
import warnings
from typing import Any, Callable, Optional, Tuple

_SEMVER_RE = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$"
)


def parse_version(version: str) -> Tuple[int, int, int]:
    """Parse ``"MAJOR.MINOR.PATCH"`` (ignoring any pre-release/build suffix)."""
    m = _SEMVER_RE.match(version.strip())
    if not m:
        raise ValueError(f"not a semantic version: {version!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def is_compatible(have: str, want: str) -> bool:
    """True if ``have`` satisfies ``want`` under SemVer (same MAJOR, >= MINOR/PATCH).

    For ``0.x`` versions the MINOR acts as the breaking-change axis, mirroring the
    common SemVer convention for pre-1.0 software.
    """
    h = parse_version(have)
    w = parse_version(want)
    if h[0] != w[0]:
        return False
    if h[0] == 0:
        # pre-1.0: MINOR is the breaking axis
        return h[1] == w[1] and h[2] >= w[2]
    return h >= w


def _format(name: str, since: Optional[str], removed_in: Optional[str],
            replacement: Optional[str], extra: Optional[str]) -> str:
    parts = [f"{name} is deprecated"]
    if since:
        parts.append(f"since {since}")
    if removed_in:
        parts.append(f"and will be removed in {removed_in}")
    msg = " ".join(parts) + "."
    if replacement:
        msg += f" Use {replacement} instead."
    if extra:
        msg += f" {extra}"
    return msg


def deprecated(
    *,
    since: Optional[str] = None,
    removed_in: Optional[str] = None,
    replacement: Optional[str] = None,
    extra: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """Mark a callable deprecated; calling it emits a ``DeprecationWarning``."""

    def decorate(func: Callable) -> Callable:
        message = _format(
            getattr(func, "__qualname__", func.__name__),
            since, removed_in, replacement, extra,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__deprecated__ = message  # type: ignore[attr-defined]
        return wrapper

    return decorate


def deprecated_alias(
    new_func: Callable,
    *,
    old_name: str,
    since: Optional[str] = None,
    removed_in: Optional[str] = None,
) -> Callable:
    """Return a wrapper exposing *old_name* that forwards to ``new_func``."""
    message = _format(
        old_name, since, removed_in,
        getattr(new_func, "__qualname__", new_func.__name__), None,
    )

    @functools.wraps(new_func)
    def wrapper(*args: Any, **kwargs: Any):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return new_func(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__deprecated__ = message  # type: ignore[attr-defined]
    return wrapper


def warn_deprecated_parameter(
    name: str,
    *,
    since: Optional[str] = None,
    removed_in: Optional[str] = None,
    replacement: Optional[str] = None,
) -> None:
    """Emit a ``DeprecationWarning`` for a deprecated keyword argument."""
    warnings.warn(
        _format(f"parameter '{name}'", since, removed_in, replacement, None),
        DeprecationWarning,
        stacklevel=3,
    )
