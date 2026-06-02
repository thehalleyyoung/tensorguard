"""Step 78 — deprecation helpers emit the right warnings."""

import warnings

import pytest

from src.deprecation import (
    deprecated,
    deprecated_alias,
    is_compatible,
    parse_version,
    warn_deprecated_parameter,
)


def test_deprecated_decorator_warns_and_runs():
    @deprecated(since="0.1.0", removed_in="0.2.0", replacement="new_fn")
    def old_fn(x):
        return x * 2

    with pytest.warns(DeprecationWarning) as rec:
        assert old_fn(3) == 6
    msg = str(rec[0].message)
    assert "old_fn is deprecated" in msg
    assert "0.2.0" in msg
    assert "new_fn" in msg


def test_deprecated_alias_forwards():
    def new_fn(a, b):
        return a + b

    legacy = deprecated_alias(new_fn, old_name="legacy_fn", removed_in="0.2.0")
    with pytest.warns(DeprecationWarning):
        assert legacy(2, 5) == 7
    assert legacy.__name__ == "legacy_fn"


def test_warn_deprecated_parameter():
    with pytest.warns(DeprecationWarning) as rec:
        warn_deprecated_parameter(
            "old_kw", since="0.1.0", removed_in="0.2.0", replacement="new_kw"
        )
    assert "parameter 'old_kw'" in str(rec[0].message)


def test_parse_version():
    assert parse_version("0.1.0") == (0, 1, 0)
    assert parse_version("1.2.3-rc1") == (1, 2, 3)
    with pytest.raises(ValueError):
        parse_version("not-a-version")


def test_is_compatible_pre_1_0():
    # pre-1.0: MINOR is the breaking axis
    assert is_compatible("0.1.5", "0.1.0")
    assert not is_compatible("0.2.0", "0.1.0")
    assert not is_compatible("1.0.0", "0.1.0")


def test_is_compatible_post_1_0():
    assert is_compatible("1.4.0", "1.2.0")
    assert not is_compatible("2.0.0", "1.2.0")


def test_no_warning_when_not_called():
    @deprecated(removed_in="0.2.0")
    def f():
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        assert hasattr(f, "__deprecated__")  # defining/decorating must not warn
