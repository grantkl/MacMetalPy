"""Tests for untested math/elementwise top-level APIs.

Covers: bitwise ops, arithmetic ops, math functions, complex ops, and logical ops.
Each API gets 2-3 tests covering basic cases and edge cases.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp


# ====================================================================
# Bitwise operations (int32 arrays)
# ====================================================================

class TestBitwiseAnd:
    def test_basic(self):
        a_np = np.array([0b1100, 0b1010, 0b1111, 0b0000], dtype=np.int32)
        b_np = np.array([0b1010, 0b1100, 0b0101, 0b1111], dtype=np.int32)
        result = cp.bitwise_and(cp.array(a_np), cp.array(b_np)).get()
        expected = np.bitwise_and(a_np, b_np)
        assert_array_equal(result, expected)

    def test_all_ones_and_zeros(self):
        a_np = np.array([0xFFFFFFFF, 0x00000000, 0xAAAAAAAA], dtype=np.uint32)
        b_np = np.array([0x00000000, 0xFFFFFFFF, 0x55555555], dtype=np.uint32)
        result = cp.bitwise_and(cp.array(a_np), cp.array(b_np)).get()
        expected = np.bitwise_and(a_np, b_np)
        assert_array_equal(result, expected)

    def test_negative_values(self):
        a_np = np.array([-1, -2, -128, 127], dtype=np.int32)
        b_np = np.array([127, -1, 255, -128], dtype=np.int32)
        result = cp.bitwise_and(cp.array(a_np), cp.array(b_np)).get()
        expected = np.bitwise_and(a_np, b_np)
        assert_array_equal(result, expected)


class TestBitwiseOr:
    def test_basic(self):
        a_np = np.array([0b1100, 0b1010, 0b0000], dtype=np.int32)
        b_np = np.array([0b0011, 0b0101, 0b0000], dtype=np.int32)
        result = cp.bitwise_or(cp.array(a_np), cp.array(b_np)).get()
        expected = np.bitwise_or(a_np, b_np)
        assert_array_equal(result, expected)

    def test_identity(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        zeros = np.zeros(5, dtype=np.int32)
        result = cp.bitwise_or(cp.array(a_np), cp.array(zeros)).get()
        assert_array_equal(result, a_np)

    def test_2d(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.int32)
        b_np = np.array([[4, 3], [2, 1]], dtype=np.int32)
        result = cp.bitwise_or(cp.array(a_np), cp.array(b_np)).get()
        expected = np.bitwise_or(a_np, b_np)
        assert_array_equal(result, expected)


class TestBitwiseXor:
    def test_basic(self):
        a_np = np.array([0b1100, 0b1010, 0b1111], dtype=np.int32)
        b_np = np.array([0b1010, 0b1100, 0b1111], dtype=np.int32)
        result = cp.bitwise_xor(cp.array(a_np), cp.array(b_np)).get()
        expected = np.bitwise_xor(a_np, b_np)
        assert_array_equal(result, expected)

    def test_self_xor_is_zero(self):
        a_np = np.array([1, 42, 255, -1], dtype=np.int32)
        result = cp.bitwise_xor(cp.array(a_np), cp.array(a_np)).get()
        expected = np.zeros_like(a_np)
        assert_array_equal(result, expected)

    def test_xor_with_zero(self):
        a_np = np.array([7, 13, 99], dtype=np.int32)
        zeros = np.zeros(3, dtype=np.int32)
        result = cp.bitwise_xor(cp.array(a_np), cp.array(zeros)).get()
        assert_array_equal(result, a_np)


class TestBitwiseNot:
    def test_basic(self):
        a_np = np.array([0, 1, -1, 127, -128], dtype=np.int32)
        result = cp.bitwise_not(cp.array(a_np)).get()
        expected = np.bitwise_not(a_np)
        assert_array_equal(result, expected)

    def test_double_not(self):
        a_np = np.array([42, -7, 0, 255], dtype=np.int32)
        a_cp = cp.array(a_np)
        result = cp.bitwise_not(cp.bitwise_not(a_cp)).get()
        assert_array_equal(result, a_np)


class TestBitwiseInvert:
    def test_basic(self):
        a_np = np.array([0, 1, -1, 255], dtype=np.int32)
        result = cp.bitwise_invert(cp.array(a_np)).get()
        expected = np.invert(a_np)
        assert_array_equal(result, expected)

    def test_matches_bitwise_not(self):
        a_np = np.array([10, 20, 30, 40], dtype=np.int32)
        a_cp = cp.array(a_np)
        not_result = cp.bitwise_not(a_cp).get()
        invert_result = cp.bitwise_invert(a_cp).get()
        assert_array_equal(not_result, invert_result)

    def test_int32_large(self):
        a_np = np.array([0, 1, 255, 65535], dtype=np.int32)
        result = cp.bitwise_invert(cp.array(a_np)).get()
        expected = np.invert(a_np)
        assert_array_equal(result, expected)


class TestLeftShift:
    def test_basic(self):
        a_np = np.array([1, 2, 4, 8], dtype=np.int32)
        b_np = np.array([1, 2, 3, 4], dtype=np.int32)
        result = cp.left_shift(cp.array(a_np), cp.array(b_np)).get()
        expected = np.left_shift(a_np, b_np)
        assert_array_equal(result, expected)

    def test_shift_by_zero(self):
        a_np = np.array([1, 42, 255], dtype=np.int32)
        zeros = np.zeros(3, dtype=np.int32)
        result = cp.left_shift(cp.array(a_np), cp.array(zeros)).get()
        assert_array_equal(result, a_np)

    def test_powers_of_two(self):
        a_np = np.array([1, 1, 1, 1], dtype=np.int32)
        shifts = np.array([0, 1, 2, 3], dtype=np.int32)
        result = cp.left_shift(cp.array(a_np), cp.array(shifts)).get()
        expected = np.array([1, 2, 4, 8], dtype=np.int32)
        assert_array_equal(result, expected)


class TestRightShift:
    def test_basic(self):
        a_np = np.array([16, 32, 64, 128], dtype=np.int32)
        b_np = np.array([1, 2, 3, 4], dtype=np.int32)
        result = cp.right_shift(cp.array(a_np), cp.array(b_np)).get()
        expected = np.right_shift(a_np, b_np)
        assert_array_equal(result, expected)

    def test_shift_by_zero(self):
        a_np = np.array([42, 99, 255], dtype=np.int32)
        zeros = np.zeros(3, dtype=np.int32)
        result = cp.right_shift(cp.array(a_np), cp.array(zeros)).get()
        assert_array_equal(result, a_np)

    def test_shift_to_zero(self):
        a_np = np.array([1, 3, 7, 15], dtype=np.int32)
        shifts = np.array([1, 2, 3, 4], dtype=np.int32)
        result = cp.right_shift(cp.array(a_np), cp.array(shifts)).get()
        expected = np.right_shift(a_np, shifts)
        assert_array_equal(result, expected)


# ====================================================================
# Arithmetic operations (float32 arrays)
# ====================================================================

class TestSubtract:
    def test_basic(self):
        a_np = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32)
        b_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = cp.subtract(cp.array(a_np), cp.array(b_np)).get()
        expected = np.subtract(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_self_subtract(self):
        a_np = np.array([1.0, 2.5, -3.7], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.subtract(a_cp, a_cp).get()
        expected = np.zeros(3, dtype=np.float32)
        assert_allclose(result, expected, rtol=1e-5)

    def test_negative_values(self):
        a_np = np.array([-1.0, -2.0, 3.0], dtype=np.float32)
        b_np = np.array([1.0, -3.0, -4.0], dtype=np.float32)
        result = cp.subtract(cp.array(a_np), cp.array(b_np)).get()
        expected = np.subtract(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestTrueDivide:
    def test_basic(self):
        a_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        b_np = np.array([2.0, 4.0, 5.0, 8.0], dtype=np.float32)
        result = cp.true_divide(cp.array(a_np), cp.array(b_np)).get()
        expected = np.true_divide(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_fractional_result(self):
        a_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        b_np = np.array([3.0, 7.0, 11.0], dtype=np.float32)
        result = cp.true_divide(cp.array(a_np), cp.array(b_np)).get()
        expected = np.true_divide(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_divide_by_one(self):
        a_np = np.array([5.5, -3.2, 0.0], dtype=np.float32)
        ones = np.ones(3, dtype=np.float32)
        result = cp.true_divide(cp.array(a_np), cp.array(ones)).get()
        assert_allclose(result, a_np, rtol=1e-5)


class TestFloorDivide:
    def test_basic(self):
        a_np = np.array([7.0, 10.0, 15.5, 20.9], dtype=np.float32)
        b_np = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = cp.floor_divide(cp.array(a_np), cp.array(b_np)).get()
        expected = np.floor_divide(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_exact_division(self):
        a_np = np.array([4.0, 9.0, 16.0], dtype=np.float32)
        b_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        result = cp.floor_divide(cp.array(a_np), cp.array(b_np)).get()
        expected = np.floor_divide(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_negative_values(self):
        a_np = np.array([-7.0, -10.0, 7.0], dtype=np.float32)
        b_np = np.array([2.0, 3.0, -2.0], dtype=np.float32)
        result = cp.floor_divide(cp.array(a_np), cp.array(b_np)).get()
        expected = np.floor_divide(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestFmod:
    def test_basic(self):
        a_np = np.array([7.0, 10.0, 15.5], dtype=np.float32)
        b_np = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        result = cp.fmod(cp.array(a_np), cp.array(b_np)).get()
        expected = np.fmod(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_exact_multiple(self):
        a_np = np.array([6.0, 12.0, 20.0], dtype=np.float32)
        b_np = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        result = cp.fmod(cp.array(a_np), cp.array(b_np)).get()
        expected = np.fmod(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_negative_dividend(self):
        a_np = np.array([-7.0, -10.0, -3.5], dtype=np.float32)
        b_np = np.array([3.0, 4.0, 2.0], dtype=np.float32)
        result = cp.fmod(cp.array(a_np), cp.array(b_np)).get()
        expected = np.fmod(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestFmax:
    def test_basic(self):
        a_np = np.array([1.0, 5.0, 3.0, 7.0], dtype=np.float32)
        b_np = np.array([4.0, 2.0, 6.0, 1.0], dtype=np.float32)
        result = cp.fmax(cp.array(a_np), cp.array(b_np)).get()
        expected = np.fmax(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_with_nan(self):
        a_np = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
        b_np = np.array([np.nan, 2.0, np.nan, np.nan], dtype=np.float32)
        result = cp.fmax(cp.array(a_np), cp.array(b_np)).get()
        expected = np.fmax(a_np, b_np)
        # fmax ignores NaN when possible
        assert_allclose(result[:3], expected[:3], rtol=1e-5)
        assert np.isnan(result[3]) and np.isnan(expected[3])

    def test_equal_values(self):
        a_np = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        b_np = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        result = cp.fmax(cp.array(a_np), cp.array(b_np)).get()
        assert_allclose(result, a_np, rtol=1e-5)


class TestFmin:
    def test_basic(self):
        a_np = np.array([1.0, 5.0, 3.0, 7.0], dtype=np.float32)
        b_np = np.array([4.0, 2.0, 6.0, 1.0], dtype=np.float32)
        result = cp.fmin(cp.array(a_np), cp.array(b_np)).get()
        expected = np.fmin(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_with_nan(self):
        a_np = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
        b_np = np.array([np.nan, 2.0, np.nan, np.nan], dtype=np.float32)
        result = cp.fmin(cp.array(a_np), cp.array(b_np)).get()
        expected = np.fmin(a_np, b_np)
        assert_allclose(result[:3], expected[:3], rtol=1e-5)
        assert np.isnan(result[3]) and np.isnan(expected[3])

    def test_negative_values(self):
        a_np = np.array([-1.0, -5.0, 3.0], dtype=np.float32)
        b_np = np.array([-3.0, -2.0, -1.0], dtype=np.float32)
        result = cp.fmin(cp.array(a_np), cp.array(b_np)).get()
        expected = np.fmin(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestFloatPower:
    def test_basic(self):
        a_np = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        b_np = np.array([2.0, 3.0, 0.5, 1.0], dtype=np.float32)
        result = cp.float_power(cp.array(a_np), cp.array(b_np)).get()
        expected = np.float_power(a_np.astype(np.float64), b_np.astype(np.float64)).astype(np.float32)
        assert_allclose(result, expected, rtol=1e-5)

    def test_power_of_zero(self):
        a_np = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        zeros = np.zeros(3, dtype=np.float32)
        result = cp.float_power(cp.array(a_np), cp.array(zeros)).get()
        expected = np.ones(3, dtype=np.float32)
        assert_allclose(result, expected, rtol=1e-5)

    def test_power_of_one(self):
        a_np = np.array([3.0, 7.0, 100.0], dtype=np.float32)
        ones = np.ones(3, dtype=np.float32)
        result = cp.float_power(cp.array(a_np), cp.array(ones)).get()
        assert_allclose(result, a_np, rtol=1e-5)


# ====================================================================
# Math functions (float32 arrays)
# ====================================================================

class TestLogaddexp:
    def test_basic(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        result = cp.logaddexp(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logaddexp(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_equal_values(self):
        a_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = cp.logaddexp(cp.array(a_np), cp.array(a_np)).get()
        expected = np.logaddexp(a_np, a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_negative_values(self):
        a_np = np.array([-1.0, -2.0], dtype=np.float32)
        b_np = np.array([-3.0, -1.0], dtype=np.float32)
        result = cp.logaddexp(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logaddexp(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestLogaddexp2:
    def test_basic(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        result = cp.logaddexp2(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logaddexp2(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_equal_values(self):
        a_np = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        result = cp.logaddexp2(cp.array(a_np), cp.array(a_np)).get()
        expected = np.logaddexp2(a_np, a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_negative_values(self):
        a_np = np.array([-1.0, -2.0], dtype=np.float32)
        b_np = np.array([-3.0, -1.0], dtype=np.float32)
        result = cp.logaddexp2(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logaddexp2(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestRint:
    def test_basic(self):
        a_np = np.array([1.2, 2.5, 3.7, -0.5, -1.5], dtype=np.float32)
        result = cp.rint(cp.array(a_np)).get()
        expected = np.rint(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_already_integer(self):
        a_np = np.array([1.0, 2.0, -3.0, 0.0], dtype=np.float32)
        result = cp.rint(cp.array(a_np)).get()
        assert_allclose(result, a_np, rtol=1e-5)

    def test_near_half(self):
        a_np = np.array([0.49999, 0.50001, 1.49999, 1.50001], dtype=np.float32)
        result = cp.rint(cp.array(a_np)).get()
        expected = np.rint(a_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestSignbit:
    def test_basic(self):
        a_np = np.array([1.0, -1.0, 0.0, -0.0, np.inf, -np.inf], dtype=np.float32)
        result = cp.signbit(cp.array(a_np)).get()
        expected = np.signbit(a_np)
        assert_array_equal(result, expected)

    def test_all_positive(self):
        a_np = np.array([0.1, 1.0, 100.0], dtype=np.float32)
        result = cp.signbit(cp.array(a_np)).get()
        expected = np.signbit(a_np)
        assert_array_equal(result, expected)

    def test_all_negative(self):
        a_np = np.array([-0.1, -1.0, -100.0], dtype=np.float32)
        result = cp.signbit(cp.array(a_np)).get()
        expected = np.signbit(a_np)
        assert_array_equal(result, expected)


class TestSinc:
    def test_basic(self):
        a_np = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        result = cp.sinc(cp.array(a_np)).get()
        expected = np.sinc(a_np)
        assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_negative_values(self):
        a_np = np.array([-0.5, -1.0, -2.0], dtype=np.float32)
        result = cp.sinc(cp.array(a_np)).get()
        expected = np.sinc(a_np)
        assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_at_zero(self):
        a_np = np.array([0.0], dtype=np.float32)
        result = cp.sinc(cp.array(a_np)).get()
        assert_allclose(result, np.array([1.0], dtype=np.float32), rtol=1e-5, atol=1e-6)


class TestCopysign:
    def test_basic(self):
        a_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        b_np = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        result = cp.copysign(cp.array(a_np), cp.array(b_np)).get()
        expected = np.copysign(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_all_positive_sign(self):
        a_np = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        b_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = cp.copysign(cp.array(a_np), cp.array(b_np)).get()
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert_allclose(result, expected, rtol=1e-5)

    def test_zero_magnitude(self):
        a_np = np.array([0.0, 0.0], dtype=np.float32)
        b_np = np.array([1.0, -1.0], dtype=np.float32)
        result = cp.copysign(cp.array(a_np), cp.array(b_np)).get()
        expected = np.copysign(a_np, b_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestDeg2rad:
    def test_basic(self):
        a_np = np.array([0.0, 90.0, 180.0, 270.0, 360.0], dtype=np.float32)
        result = cp.deg2rad(cp.array(a_np)).get()
        expected = np.deg2rad(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_negative_angles(self):
        a_np = np.array([-90.0, -180.0, -360.0], dtype=np.float32)
        result = cp.deg2rad(cp.array(a_np)).get()
        expected = np.deg2rad(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_common_angles(self):
        a_np = np.array([30.0, 45.0, 60.0], dtype=np.float32)
        result = cp.deg2rad(cp.array(a_np)).get()
        expected = np.deg2rad(a_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestHeaviside:
    def test_basic(self):
        a_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        h_np = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        result = cp.heaviside(cp.array(a_np), cp.array(h_np)).get()
        expected = np.heaviside(a_np, h_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_zero_at_zero(self):
        a_np = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        h_np = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = cp.heaviside(cp.array(a_np), cp.array(h_np)).get()
        expected = np.heaviside(a_np, h_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_one_at_zero(self):
        a_np = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        h_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = cp.heaviside(cp.array(a_np), cp.array(h_np)).get()
        expected = np.heaviside(a_np, h_np)
        assert_allclose(result, expected, rtol=1e-5)


# ====================================================================
# Complex number operations
# ====================================================================

class TestConj:
    def test_complex(self):
        a_np = np.array([1+2j, 3+4j, 5-6j], dtype=np.complex64)
        result = cp.conj(cp.array(a_np)).get()
        expected = np.conj(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_real_input(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.conj(cp.array(a_np)).get()
        expected = np.conj(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_pure_imaginary(self):
        a_np = np.array([1j, -2j, 3j], dtype=np.complex64)
        result = cp.conj(cp.array(a_np)).get()
        expected = np.conj(a_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestConjugate:
    def test_matches_conj(self):
        a_np = np.array([1+2j, 3-4j, -5+6j], dtype=np.complex64)
        a_cp = cp.array(a_np)
        conj_result = cp.conj(a_cp).get()
        conjugate_result = cp.conjugate(a_cp).get()
        assert_allclose(conj_result, conjugate_result, rtol=1e-5)

    def test_complex_basic(self):
        a_np = np.array([2+3j, -1-1j], dtype=np.complex64)
        result = cp.conjugate(cp.array(a_np)).get()
        expected = np.conjugate(a_np)
        assert_allclose(result, expected, rtol=1e-5)


class TestReal:
    def test_complex(self):
        a_np = np.array([1+2j, 3+4j, 5-6j], dtype=np.complex64)
        result = cp.real(cp.array(a_np)).get()
        expected = np.real(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_real_input(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.real(cp.array(a_np)).get()
        assert_allclose(result, a_np, rtol=1e-5)

    def test_pure_imaginary(self):
        a_np = np.array([1j, -2j, 3j], dtype=np.complex64)
        result = cp.real(cp.array(a_np)).get()
        expected = np.zeros(3, dtype=np.float32)
        assert_allclose(result, expected, rtol=1e-5)


class TestImag:
    def test_complex(self):
        a_np = np.array([1+2j, 3+4j, 5-6j], dtype=np.complex64)
        result = cp.imag(cp.array(a_np)).get()
        expected = np.imag(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_real_input(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.imag(cp.array(a_np)).get()
        expected = np.zeros(3, dtype=np.float32)
        assert_allclose(result, expected, rtol=1e-5)

    def test_pure_imaginary(self):
        a_np = np.array([1j, -2j, 3j], dtype=np.complex64)
        result = cp.imag(cp.array(a_np)).get()
        expected = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        assert_allclose(result, expected, rtol=1e-5)


class TestAngle:
    def test_basic(self):
        a_np = np.array([1+1j, 1+0j, 0+1j, -1+0j], dtype=np.complex64)
        result = cp.angle(cp.array(a_np)).get()
        expected = np.angle(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_negative_real(self):
        a_np = np.array([-1-1j, -1+1j], dtype=np.complex64)
        result = cp.angle(cp.array(a_np)).get()
        expected = np.angle(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_pure_real(self):
        a_np = np.array([1+0j, -1+0j, 2+0j], dtype=np.complex64)
        result = cp.angle(cp.array(a_np)).get()
        expected = np.angle(a_np)
        assert_allclose(result, expected, rtol=1e-5)


# ====================================================================
# Absolute value
# ====================================================================

class TestFabs:
    def test_basic(self):
        a_np = np.array([-1.0, -2.5, 3.0, 0.0, -0.0], dtype=np.float32)
        result = cp.fabs(cp.array(a_np)).get()
        expected = np.fabs(a_np)
        assert_allclose(result, expected, rtol=1e-5)

    def test_all_positive(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.fabs(cp.array(a_np)).get()
        assert_allclose(result, a_np, rtol=1e-5)

    def test_all_negative(self):
        a_np = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        result = cp.fabs(cp.array(a_np)).get()
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert_allclose(result, expected, rtol=1e-5)


# ====================================================================
# Logical operations (bool arrays)
# ====================================================================

class TestLogicalAnd:
    def test_basic(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        result = cp.logical_and(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logical_and(a_np, b_np)
        assert_array_equal(result, expected)

    def test_all_true(self):
        a_np = np.array([True, True, True])
        b_np = np.array([True, True, True])
        result = cp.logical_and(cp.array(a_np), cp.array(b_np)).get()
        expected = np.array([True, True, True])
        assert_array_equal(result, expected)

    def test_from_numeric(self):
        a_np = np.array([0, 1, 2, 0], dtype=np.float32)
        b_np = np.array([1, 0, 1, 0], dtype=np.float32)
        result = cp.logical_and(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logical_and(a_np, b_np)
        assert_array_equal(result, expected)


class TestLogicalOr:
    def test_basic(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        result = cp.logical_or(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logical_or(a_np, b_np)
        assert_array_equal(result, expected)

    def test_all_false(self):
        a_np = np.array([False, False, False])
        b_np = np.array([False, False, False])
        result = cp.logical_or(cp.array(a_np), cp.array(b_np)).get()
        expected = np.array([False, False, False])
        assert_array_equal(result, expected)

    def test_from_numeric(self):
        a_np = np.array([0, 0, 1, 0], dtype=np.float32)
        b_np = np.array([0, 1, 0, 0], dtype=np.float32)
        result = cp.logical_or(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logical_or(a_np, b_np)
        assert_array_equal(result, expected)


class TestLogicalXor:
    def test_basic(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        result = cp.logical_xor(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logical_xor(a_np, b_np)
        assert_array_equal(result, expected)

    def test_same_values(self):
        a_np = np.array([True, False, True, False])
        result = cp.logical_xor(cp.array(a_np), cp.array(a_np)).get()
        expected = np.array([False, False, False, False])
        assert_array_equal(result, expected)

    def test_from_numeric(self):
        a_np = np.array([0, 1, 1, 0], dtype=np.float32)
        b_np = np.array([1, 1, 0, 0], dtype=np.float32)
        result = cp.logical_xor(cp.array(a_np), cp.array(b_np)).get()
        expected = np.logical_xor(a_np, b_np)
        assert_array_equal(result, expected)


class TestLogicalNot:
    def test_basic(self):
        a_np = np.array([True, False, True, False])
        result = cp.logical_not(cp.array(a_np)).get()
        expected = np.logical_not(a_np)
        assert_array_equal(result, expected)

    def test_double_not(self):
        a_np = np.array([True, False, True, False])
        a_cp = cp.array(a_np)
        result = cp.logical_not(cp.logical_not(a_cp)).get()
        assert_array_equal(result, a_np)

    def test_from_numeric(self):
        a_np = np.array([0, 1, 2, 0, -1], dtype=np.float32)
        result = cp.logical_not(cp.array(a_np)).get()
        expected = np.logical_not(a_np)
        assert_array_equal(result, expected)
