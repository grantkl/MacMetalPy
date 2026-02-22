"""Tests for comparison ufunc function-call forms (cp.equal, cp.not_equal, etc.).

These test the function-call API (e.g. cp.equal(a, b)) as opposed to the
operator forms (a == b) which are covered in test_comparison.py.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import macmetalpy as cp


# ====================================================================
# cp.equal
# ====================================================================

class TestEqual:
    def test_basic_int32(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        b = cp.array([1, 3, 3, 2, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        b_np = np.array([1, 3, 3, 2, 5], dtype=np.int32)
        result = cp.equal(a, b)
        expected = np.equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_basic_float32(self):
        a = cp.array([1.0, 2.5, 3.0, 4.5], dtype=cp.float32)
        b = cp.array([1.0, 2.0, 3.0, 4.5], dtype=cp.float32)
        a_np = np.array([1.0, 2.5, 3.0, 4.5], dtype=np.float32)
        b_np = np.array([1.0, 2.0, 3.0, 4.5], dtype=np.float32)
        result = cp.equal(a, b)
        expected = np.equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_scalar_broadcast(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = cp.equal(a, 3)
        expected = np.equal(a_np, 3)
        assert_array_equal(result.get(), expected)

    def test_nan_handling(self):
        a = cp.array([1.0, float('nan'), 3.0, float('nan')], dtype=cp.float32)
        b = cp.array([1.0, float('nan'), float('nan'), 3.0], dtype=cp.float32)
        a_np = np.array([1.0, float('nan'), 3.0, float('nan')], dtype=np.float32)
        b_np = np.array([1.0, float('nan'), float('nan'), 3.0], dtype=np.float32)
        result = cp.equal(a, b)
        expected = np.equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_mixed_dtypes(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([1.0, 2.5, 3.0], dtype=cp.float32)
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([1.0, 2.5, 3.0], dtype=np.float32)
        result = cp.equal(a, b)
        expected = np.equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_result_dtype_is_bool(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([1, 2, 3], dtype=cp.int32)
        result = cp.equal(a, b)
        assert result.dtype == np.bool_


# ====================================================================
# cp.not_equal
# ====================================================================

class TestNotEqual:
    def test_basic_int32(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        b = cp.array([1, 3, 3, 2, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        b_np = np.array([1, 3, 3, 2, 5], dtype=np.int32)
        result = cp.not_equal(a, b)
        expected = np.not_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_basic_float32(self):
        a = cp.array([1.0, 2.5, 3.0, 4.5], dtype=cp.float32)
        b = cp.array([1.0, 2.0, 3.0, 4.5], dtype=cp.float32)
        a_np = np.array([1.0, 2.5, 3.0, 4.5], dtype=np.float32)
        b_np = np.array([1.0, 2.0, 3.0, 4.5], dtype=np.float32)
        result = cp.not_equal(a, b)
        expected = np.not_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_scalar_broadcast(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = cp.not_equal(a, 3)
        expected = np.not_equal(a_np, 3)
        assert_array_equal(result.get(), expected)

    def test_nan_handling(self):
        a = cp.array([1.0, float('nan'), 3.0, float('nan')], dtype=cp.float32)
        b = cp.array([1.0, float('nan'), float('nan'), 3.0], dtype=cp.float32)
        a_np = np.array([1.0, float('nan'), 3.0, float('nan')], dtype=np.float32)
        b_np = np.array([1.0, float('nan'), float('nan'), 3.0], dtype=np.float32)
        result = cp.not_equal(a, b)
        expected = np.not_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_mixed_dtypes(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([1.0, 2.5, 3.0], dtype=cp.float32)
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([1.0, 2.5, 3.0], dtype=np.float32)
        result = cp.not_equal(a, b)
        expected = np.not_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_result_dtype_is_bool(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([4, 5, 6], dtype=cp.int32)
        result = cp.not_equal(a, b)
        assert result.dtype == np.bool_


# ====================================================================
# cp.less
# ====================================================================

class TestLess:
    def test_basic_int32(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        b = cp.array([2, 2, 4, 3, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        b_np = np.array([2, 2, 4, 3, 5], dtype=np.int32)
        result = cp.less(a, b)
        expected = np.less(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_basic_float32(self):
        a = cp.array([1.0, 2.5, 3.0, 4.5], dtype=cp.float32)
        b = cp.array([1.5, 2.0, 3.0, 5.0], dtype=cp.float32)
        a_np = np.array([1.0, 2.5, 3.0, 4.5], dtype=np.float32)
        b_np = np.array([1.5, 2.0, 3.0, 5.0], dtype=np.float32)
        result = cp.less(a, b)
        expected = np.less(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_scalar_broadcast(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = cp.less(a, 3)
        expected = np.less(a_np, 3)
        assert_array_equal(result.get(), expected)

    def test_nan_handling(self):
        a = cp.array([1.0, float('nan'), 3.0, float('nan')], dtype=cp.float32)
        b = cp.array([2.0, 1.0, float('nan'), float('nan')], dtype=cp.float32)
        a_np = np.array([1.0, float('nan'), 3.0, float('nan')], dtype=np.float32)
        b_np = np.array([2.0, 1.0, float('nan'), float('nan')], dtype=np.float32)
        result = cp.less(a, b)
        expected = np.less(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_mixed_dtypes(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([1.5, 1.5, 3.5], dtype=cp.float32)
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([1.5, 1.5, 3.5], dtype=np.float32)
        result = cp.less(a, b)
        expected = np.less(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_result_dtype_is_bool(self):
        a = cp.array([1.0, 2.0], dtype=cp.float32)
        b = cp.array([3.0, 4.0], dtype=cp.float32)
        result = cp.less(a, b)
        assert result.dtype == np.bool_


# ====================================================================
# cp.less_equal
# ====================================================================

class TestLessEqual:
    def test_basic_int32(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        b = cp.array([2, 2, 4, 3, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        b_np = np.array([2, 2, 4, 3, 5], dtype=np.int32)
        result = cp.less_equal(a, b)
        expected = np.less_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_basic_float32(self):
        a = cp.array([1.0, 2.5, 3.0, 4.5], dtype=cp.float32)
        b = cp.array([1.0, 2.0, 3.0, 5.0], dtype=cp.float32)
        a_np = np.array([1.0, 2.5, 3.0, 4.5], dtype=np.float32)
        b_np = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
        result = cp.less_equal(a, b)
        expected = np.less_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_scalar_broadcast(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = cp.less_equal(a, 3)
        expected = np.less_equal(a_np, 3)
        assert_array_equal(result.get(), expected)

    def test_nan_handling(self):
        a = cp.array([1.0, float('nan'), 3.0, float('nan')], dtype=cp.float32)
        b = cp.array([1.0, 1.0, float('nan'), float('nan')], dtype=cp.float32)
        a_np = np.array([1.0, float('nan'), 3.0, float('nan')], dtype=np.float32)
        b_np = np.array([1.0, 1.0, float('nan'), float('nan')], dtype=np.float32)
        result = cp.less_equal(a, b)
        expected = np.less_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_mixed_dtypes(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([1.0, 1.5, 3.0], dtype=cp.float32)
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([1.0, 1.5, 3.0], dtype=np.float32)
        result = cp.less_equal(a, b)
        expected = np.less_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_result_dtype_is_bool(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([3, 2, 1], dtype=cp.int32)
        result = cp.less_equal(a, b)
        assert result.dtype == np.bool_


# ====================================================================
# cp.greater
# ====================================================================

class TestGreater:
    def test_basic_int32(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        b = cp.array([2, 2, 4, 3, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        b_np = np.array([2, 2, 4, 3, 5], dtype=np.int32)
        result = cp.greater(a, b)
        expected = np.greater(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_basic_float32(self):
        a = cp.array([1.5, 2.0, 3.0, 5.0], dtype=cp.float32)
        b = cp.array([1.0, 2.5, 3.0, 4.5], dtype=cp.float32)
        a_np = np.array([1.5, 2.0, 3.0, 5.0], dtype=np.float32)
        b_np = np.array([1.0, 2.5, 3.0, 4.5], dtype=np.float32)
        result = cp.greater(a, b)
        expected = np.greater(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_scalar_broadcast(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = cp.greater(a, 3)
        expected = np.greater(a_np, 3)
        assert_array_equal(result.get(), expected)

    def test_nan_handling(self):
        a = cp.array([2.0, float('nan'), float('nan'), float('nan')], dtype=cp.float32)
        b = cp.array([1.0, 1.0, float('nan'), 3.0], dtype=cp.float32)
        a_np = np.array([2.0, float('nan'), float('nan'), float('nan')], dtype=np.float32)
        b_np = np.array([1.0, 1.0, float('nan'), 3.0], dtype=np.float32)
        result = cp.greater(a, b)
        expected = np.greater(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_mixed_dtypes(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([0.5, 2.5, 2.5], dtype=cp.float32)
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([0.5, 2.5, 2.5], dtype=np.float32)
        result = cp.greater(a, b)
        expected = np.greater(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_result_dtype_is_bool(self):
        a = cp.array([5, 4, 3], dtype=cp.int32)
        b = cp.array([1, 2, 3], dtype=cp.int32)
        result = cp.greater(a, b)
        assert result.dtype == np.bool_


# ====================================================================
# cp.greater_equal
# ====================================================================

class TestGreaterEqual:
    def test_basic_int32(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        b = cp.array([2, 2, 4, 3, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        b_np = np.array([2, 2, 4, 3, 5], dtype=np.int32)
        result = cp.greater_equal(a, b)
        expected = np.greater_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_basic_float32(self):
        a = cp.array([1.0, 2.5, 3.0, 5.0], dtype=cp.float32)
        b = cp.array([1.0, 2.0, 3.0, 4.5], dtype=cp.float32)
        a_np = np.array([1.0, 2.5, 3.0, 5.0], dtype=np.float32)
        b_np = np.array([1.0, 2.0, 3.0, 4.5], dtype=np.float32)
        result = cp.greater_equal(a, b)
        expected = np.greater_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_scalar_broadcast(self):
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = cp.greater_equal(a, 3)
        expected = np.greater_equal(a_np, 3)
        assert_array_equal(result.get(), expected)

    def test_nan_handling(self):
        a = cp.array([1.0, float('nan'), float('nan'), float('nan')], dtype=cp.float32)
        b = cp.array([1.0, 1.0, float('nan'), 3.0], dtype=cp.float32)
        a_np = np.array([1.0, float('nan'), float('nan'), float('nan')], dtype=np.float32)
        b_np = np.array([1.0, 1.0, float('nan'), 3.0], dtype=np.float32)
        result = cp.greater_equal(a, b)
        expected = np.greater_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_mixed_dtypes(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([1.0, 1.5, 3.0], dtype=cp.float32)
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([1.0, 1.5, 3.0], dtype=np.float32)
        result = cp.greater_equal(a, b)
        expected = np.greater_equal(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_result_dtype_is_bool(self):
        a = cp.array([1, 2, 3], dtype=cp.int32)
        b = cp.array([3, 2, 1], dtype=cp.int32)
        result = cp.greater_equal(a, b)
        assert result.dtype == np.bool_
