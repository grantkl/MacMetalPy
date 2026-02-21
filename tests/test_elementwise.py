"""Tests for elementwise arithmetic operators and math functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp


@pytest.fixture
def float_arrays():
    a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b_np = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    return cp.array(a_np), cp.array(b_np), a_np, b_np


@pytest.fixture
def int_arrays():
    a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    b_np = np.array([2, 3, 4, 5, 6], dtype=np.int32)
    return cp.array(a_np), cp.array(b_np), a_np, b_np


@pytest.fixture
def positive_float_array():
    np_arr = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    return cp.array(np_arr), np_arr


class TestArithmeticFloat:
    def test_add(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a + b
        expected = a_np + b_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_sub(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a - b
        expected = a_np - b_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_mul(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a * b
        expected = a_np * b_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_div(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a / b
        expected = a_np / b_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_pow(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = a ** b
        expected = a_np ** b_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)


class TestArithmeticScalar:
    def test_add_scalar(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = a + 10.0
        expected = a_np + 10.0
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_sub_scalar(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = a - 1.0
        expected = a_np - 1.0
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_mul_scalar(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = a * 3.0
        expected = a_np * 3.0
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_div_scalar(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = a / 2.0
        expected = a_np / 2.0
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_pow_scalar(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = a ** 2.0
        expected = a_np ** 2.0
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)


class TestReverseOperators:
    def test_radd(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = 2.0 + a
        expected = 2.0 + a_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_rsub(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = 10.0 - a
        expected = 10.0 - a_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_rmul(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = 2.0 * a
        expected = 2.0 * a_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_rtruediv(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = 10.0 / a
        expected = 10.0 / a_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_rpow(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = 2.0 ** a
        expected = 2.0 ** a_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)


class TestUnaryOperators:
    def test_neg(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = -a
        expected = -a_np
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_abs(self):
        np_arr = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
        a = cp.array(np_arr)
        result = abs(a)
        expected = np.abs(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)


class TestMathFunctions:
    def test_sqrt(self, positive_float_array):
        gpu, np_arr = positive_float_array
        result = cp.sqrt(gpu)
        expected = np.sqrt(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_exp(self, float_arrays):
        a, _, a_np, _ = float_arrays
        result = cp.exp(a)
        expected = np.exp(a_np)
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_log(self, positive_float_array):
        gpu, np_arr = positive_float_array
        result = cp.log(gpu)
        expected = np.log(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_abs_function(self):
        np_arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = cp.abs(gpu)
        expected = np.abs(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)

    def test_power(self, float_arrays):
        a, b, a_np, b_np = float_arrays
        result = cp.power(a, b)
        expected = np.power(a_np, b_np)
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)


class TestArithmeticInt:
    def test_add_int(self, int_arrays):
        a, b, a_np, b_np = int_arrays
        result = a + b
        expected = a_np + b_np
        assert_array_equal(result.get(), expected)

    def test_sub_int(self, int_arrays):
        a, b, a_np, b_np = int_arrays
        result = a - b
        expected = a_np - b_np
        assert_array_equal(result.get(), expected)

    def test_mul_int(self, int_arrays):
        a, b, a_np, b_np = int_arrays
        result = a * b
        expected = a_np * b_np
        assert_array_equal(result.get(), expected)
