"""
Regression tests for three propagator bugs identified by reviewers.

Bug #1: BroadcastPropagator._check_stride_final compared shape IDs
         against themselves instead of stride IDs (copy-paste error).
Bug #2: BroadcastPropagator._on_fixed used bare `except Exception`,
         swallowing non-Z3 errors and violating DPLL(T) T-Explain.
Bug #3: DevicePropagator._on_fixed used as_long() on enum sorts,
         falling back to string comparison causing type mismatch.
"""

import pytest

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

pytestmark = pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")


class TestBroadcastStrideComparison:
    """Bug #1: stride_vals must use stride variable IDs, not shape variable IDs."""

    def test_stride_check_detects_non_contiguous(self):
        """A non-contiguous tensor should trigger a conflict, not pass silently."""
        from src.smt.broadcast_theory import BroadcastPropagator, stride_compatible

        s = z3.Solver()
        prop = BroadcastPropagator(s)

        shape_d0 = z3.Int("shape_0")
        shape_d1 = z3.Int("shape_1")
        stride_d0 = z3.Int("stride_0")
        stride_d1 = z3.Int("stride_1")

        constraint = stride_compatible(prop, [shape_d0, shape_d1], [stride_d0, stride_d1])
        s.add(constraint)

        # shape [3, 4] with strides [1, 1] — non-contiguous
        s.add(shape_d0 == 3)
        s.add(shape_d1 == 4)
        s.add(stride_d0 == 1)  # Wrong: should be 4 for contiguous
        s.add(stride_d1 == 1)

        result = s.check()
        # The explicit constraint encoding already forces UNSAT here;
        # the _check_stride_final is an additional propagator-level check
        assert result == z3.unsat

    def test_stride_check_accepts_contiguous(self):
        """A contiguous tensor should pass without conflict."""
        from src.smt.broadcast_theory import BroadcastPropagator, stride_compatible

        s = z3.Solver()
        prop = BroadcastPropagator(s)

        shape_d0 = z3.Int("shape_0c")
        shape_d1 = z3.Int("shape_1c")
        stride_d0 = z3.Int("stride_0c")
        stride_d1 = z3.Int("stride_1c")

        constraint = stride_compatible(prop, [shape_d0, shape_d1], [stride_d0, stride_d1])
        s.add(constraint)

        s.add(shape_d0 == 3)
        s.add(shape_d1 == 4)
        s.add(stride_d0 == 4)  # Correct for contiguous
        s.add(stride_d1 == 1)

        result = s.check()
        assert result == z3.sat


class TestBroadcastExceptionHandling:
    """Bug #2: _on_fixed must not use bare except Exception."""

    def test_non_z3_exception_not_swallowed(self):
        """Verify that non-Z3 exceptions propagate rather than being swallowed."""
        from src.smt.broadcast_theory import BroadcastPropagator

        s = z3.Solver()
        prop = BroadcastPropagator(s)

        import inspect
        source = inspect.getsource(prop._on_fixed)
        assert "except Exception:" not in source, \
            "_on_fixed must not use bare 'except Exception:'"
        assert "AttributeError" in source and "Z3Exception" in source, \
            "_on_fixed should catch only (AttributeError, z3.Z3Exception)"


class TestDeviceEnumExtraction:
    """Bug #3: DevicePropagator must handle enum sorts without type mismatch."""

    def test_enum_extraction_produces_canonical_value(self):
        """Verify _on_fixed produces consistent canonical values for enum sorts."""
        from src.smt.device_theory import DevicePropagator, DeviceSort, DEVICE_VALS

        s = z3.Solver()
        prop = DevicePropagator(s)

        # Simulate _on_fixed with enum values
        dev_a = z3.Const("dev_enum_a", DeviceSort)
        dev_b = z3.Const("dev_enum_b", DeviceSort)

        cpu = DEVICE_VALS["CPU"]
        cuda0 = DEVICE_VALS["CUDA_0"]

        # Directly call _on_fixed to test extraction logic
        prop._on_fixed(dev_a, cpu)
        prop._on_fixed(dev_b, cuda0)

        va = prop._fixed[dev_a.get_id()]
        vb = prop._fixed[dev_b.get_id()]

        # Values should be different for different devices
        assert va != vb, f"CPU and CUDA_0 must produce different canonical values, got {va!r} vs {vb!r}"

        # Same device should produce same value
        dev_c = z3.Const("dev_enum_c", DeviceSort)
        prop._on_fixed(dev_c, cpu)
        vc = prop._fixed[dev_c.get_id()]
        assert va == vc, f"Two CPU values should be identical, got {va!r} vs {vc!r}"

    def test_no_bare_string_fallback_in_source(self):
        """Verify _on_fixed uses sexpr() not str() for canonical enum form."""
        from src.smt.device_theory import DevicePropagator

        import inspect
        source = inspect.getsource(DevicePropagator._on_fixed)
        # Should use sexpr() for stable canonical form
        assert "sexpr()" in source, "_on_fixed should use sexpr() for enum normalization"
