"""Tests for gap-filling manipulation functions: block, insert, broadcast_shapes, asfortranarray."""

import numpy as np
import pytest

import macmetalpy as cp
from conftest import FLOAT_DTYPES, NUMERIC_DTYPES, assert_eq, make_arg


# ====================================================================
# block
# ====================================================================

class TestBlock:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1d_list(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([4, 5, 6], dtype=dtype)
        result = cp.block([cp.array(a), cp.array(b)])
        expected = np.block([a, b])
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d_blocks(self, dtype):
        a = np.ones((2, 2), dtype=dtype)
        b = np.zeros((2, 3), dtype=dtype)
        c = np.zeros((3, 2), dtype=dtype)
        d = np.ones((3, 3), dtype=dtype)
        result = cp.block([[cp.array(a), cp.array(b)],
                           [cp.array(c), cp.array(d)]])
        expected = np.block([[a, b], [c, d]])
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_single_array(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        result = cp.block(cp.array(a))
        expected = np.block(a)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_nested_2x2(self, dtype):
        a = np.eye(2, dtype=dtype)
        b = np.zeros((2, 2), dtype=dtype)
        result = cp.block([[cp.array(a), cp.array(b)],
                           [cp.array(b), cp.array(a)]])
        expected = np.block([[a, b], [b, a]])
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mixed_numpy_macmetalpy(self, dtype):
        a = np.array([1, 2], dtype=dtype)
        b = np.array([3, 4], dtype=dtype)
        result = cp.block([a, cp.array(b)])
        expected = np.block([a, b])
        assert_eq(result, expected, dtype=dtype)


# ====================================================================
# insert
# ====================================================================

class TestInsert:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d_single(self, dtype):
        a_np = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = cp.insert(cp.array(a_np), 2, 99)
        expected = np.insert(a_np, 2, 99)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d_multiple_values(self, dtype):
        a_np = np.array([1, 2, 3], dtype=dtype)
        result = cp.insert(cp.array(a_np), 1, [10, 20])
        expected = np.insert(a_np, 1, [10, 20])
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_at_beginning(self, dtype):
        a_np = np.array([1, 2, 3], dtype=dtype)
        result = cp.insert(cp.array(a_np), 0, 99)
        expected = np.insert(a_np, 0, 99)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_at_end(self, dtype):
        a_np = np.array([1, 2, 3], dtype=dtype)
        result = cp.insert(cp.array(a_np), 3, 99)
        expected = np.insert(a_np, 3, 99)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d_axis0(self, dtype):
        a_np = make_arg((2, 3), dtype)
        vals = np.array([10, 20, 30], dtype=dtype)
        result = cp.insert(cp.array(a_np), 1, vals, axis=0)
        expected = np.insert(a_np, 1, vals, axis=0)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d_axis1(self, dtype):
        a_np = make_arg((2, 3), dtype)
        vals = np.array([10, 20], dtype=dtype)
        result = cp.insert(cp.array(a_np), 1, vals, axis=1)
        expected = np.insert(a_np, 1, vals, axis=1)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_no_axis_flattens(self, dtype):
        a_np = make_arg((2, 3), dtype)
        result = cp.insert(cp.array(a_np), 2, 99)
        expected = np.insert(a_np, 2, 99)
        assert_eq(result, expected, dtype=dtype)


# ====================================================================
# broadcast_shapes
# ====================================================================

class TestBroadcastShapes:
    def test_simple(self):
        result = cp.broadcast_shapes((3,), (3,))
        expected = np.broadcast_shapes((3,), (3,))
        assert result == expected

    def test_1d_broadcast(self):
        result = cp.broadcast_shapes((1,), (3,))
        expected = np.broadcast_shapes((1,), (3,))
        assert result == expected

    def test_2d_broadcast(self):
        result = cp.broadcast_shapes((2, 1), (1, 3))
        expected = np.broadcast_shapes((2, 1), (1, 3))
        assert result == expected

    def test_different_ndim(self):
        result = cp.broadcast_shapes((3,), (2, 3))
        expected = np.broadcast_shapes((3,), (2, 3))
        assert result == expected

    def test_three_shapes(self):
        result = cp.broadcast_shapes((1,), (2, 1), (2, 3))
        expected = np.broadcast_shapes((1,), (2, 1), (2, 3))
        assert result == expected

    def test_scalar(self):
        result = cp.broadcast_shapes((), (3,))
        expected = np.broadcast_shapes((), (3,))
        assert result == expected

    def test_incompatible(self):
        with pytest.raises(ValueError):
            cp.broadcast_shapes((2,), (3,))

    def test_3d_broadcast(self):
        result = cp.broadcast_shapes((2, 1, 3), (1, 4, 1))
        expected = np.broadcast_shapes((2, 1, 3), (1, 4, 1))
        assert result == expected


# ====================================================================
# asfortranarray
# ====================================================================

class TestAsfortranarray:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((2, 3), dtype)
        result = cp.asfortranarray(cp.array(a_np))
        # For macmetalpy, just check it returns a valid contiguous copy with correct data
        assert_eq(result, a_np, dtype=dtype)
        assert result.shape == (2, 3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1d(self, dtype):
        a_np = make_arg((5,), dtype)
        result = cp.asfortranarray(cp.array(a_np))
        assert_eq(result, a_np, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype_override(self, dtype):
        a_np = np.array([1, 2, 3], dtype=np.int32)
        result = cp.asfortranarray(cp.array(a_np), dtype=dtype)
        expected = np.asfortranarray(a_np, dtype=dtype)
        assert result.dtype == np.dtype(dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_from_numpy_input(self):
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = cp.asfortranarray(a_np)
        assert_eq(result, a_np, dtype=np.float32)
