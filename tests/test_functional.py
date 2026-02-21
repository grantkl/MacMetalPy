"""Tests for functional programming tools -- vectorize, apply_along_axis,
apply_over_axes.

Ref: numpy.vectorize, numpy.apply_along_axis, numpy.apply_over_axes
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp
from macmetalpy.ndarray import ndarray
from macmetalpy.functional import vectorize, apply_along_axis, apply_over_axes


# ====================================================================
# vectorize
# ====================================================================

class TestVectorize:
    def test_basic_scalar_function(self):
        def myfunc(x):
            return x * 2 + 1

        vfunc = vectorize(myfunc)
        a = cp.array([1.0, 2.0, 3.0, 4.0])
        result = vfunc(a)
        expected = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float32)
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_two_arguments(self):
        def add(x, y):
            return x + y

        vfunc = vectorize(add)
        a = cp.array([1.0, 2.0, 3.0])
        b = cp.array([10.0, 20.0, 30.0])
        result = vfunc(a, b)
        expected = np.array([11.0, 22.0, 33.0], dtype=np.float32)
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_otypes(self):
        def to_int(x):
            return int(x)

        vfunc = vectorize(to_int, otypes=[np.int32])
        a = cp.array([1.5, 2.7, 3.1])
        result = vfunc(a)
        assert isinstance(result, ndarray)
        assert result.dtype == np.int32

    def test_excluded(self):
        def power(base, exp):
            return base ** exp

        vfunc = vectorize(power, excluded=['exp'])
        a = cp.array([1.0, 2.0, 3.0])
        result = vfunc(a, exp=2)
        expected = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_2d_input(self):
        def double(x):
            return x * 2

        vfunc = vectorize(double)
        a = cp.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        result = vfunc(a)
        expected = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_preserves_shape(self):
        def identity(x):
            return x

        vfunc = vectorize(identity)
        a = cp.zeros((2, 3, 4))
        result = vfunc(a)
        assert result.shape == (2, 3, 4)

    def test_with_numpy_array_input(self):
        def double(x):
            return x * 2

        vfunc = vectorize(double)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = vfunc(a)
        # Should still return macmetalpy ndarray
        assert isinstance(result, ndarray)


# ====================================================================
# apply_along_axis
# ====================================================================

class TestApplyAlongAxis:
    def test_1d_function_axis0(self):
        a = cp.array(np.array([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0]], dtype=np.float32))
        result = apply_along_axis(np.sum, 0, a)
        expected = np.apply_along_axis(np.sum, 0,
                                        np.array([[1.0, 2.0, 3.0],
                                                   [4.0, 5.0, 6.0]], dtype=np.float32))
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_1d_function_axis1(self):
        a = cp.array(np.array([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0]], dtype=np.float32))
        result = apply_along_axis(np.sum, 1, a)
        expected = np.apply_along_axis(np.sum, 1,
                                        np.array([[1.0, 2.0, 3.0],
                                                   [4.0, 5.0, 6.0]], dtype=np.float32))
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_sorted_along_axis(self):
        a_np = np.array([[3.0, 1.0, 2.0],
                          [6.0, 4.0, 5.0]], dtype=np.float32)
        a = cp.array(a_np)
        result = apply_along_axis(sorted, 1, a)
        expected = np.apply_along_axis(sorted, 1, a_np)
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_custom_function(self):
        def my_func(x):
            return np.array([x.sum(), x.mean()])

        a_np = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=np.float32)
        a = cp.array(a_np)
        result = apply_along_axis(my_func, 1, a)
        expected = np.apply_along_axis(my_func, 1, a_np)
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_3d_input(self):
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        a = cp.array(a_np)
        result = apply_along_axis(np.sum, 2, a)
        expected = np.apply_along_axis(np.sum, 2, a_np)
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ====================================================================
# apply_over_axes
# ====================================================================

class TestApplyOverAxes:
    def test_sum_over_single_axis(self):
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        a = cp.array(a_np)
        result = apply_over_axes(np.sum, a, [0])
        expected = np.apply_over_axes(np.sum, a_np, [0])
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_sum_over_two_axes(self):
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        a = cp.array(a_np)
        result = apply_over_axes(np.sum, a, [0, 2])
        expected = np.apply_over_axes(np.sum, a_np, [0, 2])
        assert isinstance(result, ndarray)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_preserves_ndim(self):
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        a = cp.array(a_np)
        result = apply_over_axes(np.sum, a, [0])
        # apply_over_axes should preserve ndim (keepdims behavior)
        assert result.ndim == 3
