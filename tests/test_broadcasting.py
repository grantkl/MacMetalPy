"""Tests for broadcasting logic and broadcasted operations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import macmetalpy as cp
from macmetalpy._broadcasting import broadcast_shapes


class TestBroadcastShapes:
    def test_same_shapes(self):
        assert broadcast_shapes((3, 4), (3, 4)) == (3, 4)

    def test_trailing_broadcast(self):
        assert broadcast_shapes((3, 4), (4,)) == (3, 4)

    def test_leading_broadcast(self):
        assert broadcast_shapes((3, 1), (1, 4)) == (3, 4)

    def test_3d_broadcast(self):
        assert broadcast_shapes((2, 3, 4), (4,)) == (2, 3, 4)

    def test_scalar_broadcast(self):
        assert broadcast_shapes((3, 4), (1,)) == (3, 4)

    def test_empty_shapes(self):
        assert broadcast_shapes(()) == ()

    def test_single_shape(self):
        assert broadcast_shapes((3, 4)) == (3, 4)

    def test_multiple_shapes(self):
        assert broadcast_shapes((1,), (3,), (1,)) == (3,)

    def test_complex_broadcast(self):
        assert broadcast_shapes((1, 3, 1), (2, 1, 4)) == (2, 3, 4)

    def test_incompatible_raises(self):
        with pytest.raises(ValueError, match="cannot broadcast"):
            broadcast_shapes((3,), (4,))

    def test_incompatible_2d_raises(self):
        with pytest.raises(ValueError, match="cannot broadcast"):
            broadcast_shapes((2, 3), (4, 5))


class TestBroadcastedOperations:
    def test_2d_plus_1d(self):
        a_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        b_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a = cp.array(a_np)
        b = cp.array(b_np)
        result = a + b
        expected = a_np + b_np
        assert result.get().shape == expected.shape
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_column_plus_row(self):
        a_np = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        b_np = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        a = cp.array(a_np)
        b = cp.array(b_np)
        result = a + b
        expected = a_np + b_np
        assert result.get().shape == (3, 4)
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_3d_plus_1d(self):
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        b_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a = cp.array(a_np)
        b = cp.array(b_np)
        result = a + b
        expected = a_np + b_np
        assert result.get().shape == expected.shape
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_scalar_broadcast(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a = cp.array(a_np)
        result = a + 10.0
        expected = a_np + 10.0
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_mul_broadcast(self):
        a_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        b_np = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        a = cp.array(a_np)
        b = cp.array(b_np)
        result = a * b
        expected = a_np * b_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_sub_broadcast(self):
        a_np = np.ones((3, 1), dtype=np.float32)
        b_np = np.ones((1, 4), dtype=np.float32) * 2
        a = cp.array(a_np)
        b = cp.array(b_np)
        result = a - b
        expected = a_np - b_np
        assert result.get().shape == (3, 4)
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_incompatible_shapes_raise(self):
        a = cp.array(np.zeros((3,), dtype=np.float32))
        b = cp.array(np.zeros((4,), dtype=np.float32))
        with pytest.raises((ValueError, RuntimeError)):
            _ = a + b
