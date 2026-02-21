"""Tests for GPU synchronization.

Target: ~10 cases.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy._metal_backend import MetalBackend


# ======================================================================
# Synchronization tests
# ======================================================================

class TestSynchronize:
    def test_no_pending_noop(self):
        """synchronize() with no pending work should be a no-op."""
        backend = MetalBackend()
        backend.synchronize()  # should not raise
        backend.synchronize()  # double call should also be fine

    def test_after_single_op(self):
        """After a single op, get() returns correct result."""
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = cp.sqrt(a)
        result = b.get()  # implicitly syncs
        npt.assert_allclose(result, np.sqrt([1.0, 2.0, 3.0]), rtol=1e-5)

    def test_after_chained_ops(self):
        """After chained ops (sqrt(exp(a))), result is correct."""
        a = cp.array([0.0, 1.0, 2.0], dtype=np.float32)
        b = cp.sqrt(cp.exp(a))
        result = b.get()
        expected = np.sqrt(np.exp([0.0, 1.0, 2.0]))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_get_implicit_sync(self):
        """.get() implicitly synchronizes."""
        a = cp.array([2.0, 4.0, 6.0], dtype=np.float32)
        b = a + a  # GPU op
        result = b.get()
        npt.assert_allclose(result, [4.0, 8.0, 12.0], rtol=1e-5)

    def test_set_implicit_sync(self):
        """.set() implicitly synchronizes."""
        a = cp.empty(3, dtype=np.float32)
        new_data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        a.set(new_data)
        npt.assert_allclose(a.get(), new_data, rtol=1e-5)

    def test_multiple_arrays_one_sync(self):
        """Multiple arrays, one synchronize call."""
        a = cp.array([1.0, 2.0], dtype=np.float32)
        b = cp.array([3.0, 4.0], dtype=np.float32)
        c = a + b
        d = a * b
        cp.synchronize()
        npt.assert_allclose(c.get(), [4.0, 6.0], rtol=1e-5)
        npt.assert_allclose(d.get(), [3.0, 8.0], rtol=1e-5)

    def test_back_to_back_sync(self):
        """Second synchronize() is a no-op."""
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = cp.sqrt(a)
        cp.synchronize()
        cp.synchronize()  # second call should be no-op
        npt.assert_allclose(b.get(), np.sqrt([1.0, 2.0, 3.0]), rtol=1e-5)

    def test_has_pending_false_after_sync(self):
        """_has_pending should be False after synchronize."""
        a = cp.array([1.0, 2.0], dtype=np.float32)
        _ = cp.sqrt(a)
        backend = MetalBackend()
        backend.synchronize()
        assert not backend._has_pending

    def test_has_pending_false_initially(self):
        """_has_pending should be False when no work has been submitted."""
        backend = MetalBackend()
        backend.synchronize()  # clear any pending
        assert not backend._has_pending

    def test_explicit_sync_then_get(self):
        """Explicit synchronize() followed by get() works correctly."""
        a = cp.array([1.0, 4.0, 9.0], dtype=np.float32)
        b = cp.sqrt(a)
        cp.synchronize()
        result = b.get()
        npt.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-5)
