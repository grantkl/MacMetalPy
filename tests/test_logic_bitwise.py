"""Tests for logic, comparison, and bitwise operations -- CuPy-parity parametrized suite.

Ref: cupy_tests/logic_tests/, cupy_tests/binary_tests/
~2,052 parametrized cases covering: logical_and, logical_or, logical_not,
logical_xor, greater, greater_equal, less, less_equal, equal, not_equal,
isneginf, isposinf, iscomplex, isreal, isscalar, array_equiv,
bitwise_and, bitwise_or, bitwise_xor, invert, left_shift, right_shift,
packbits, unpackbits, gcd, lcm.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import macmetalpy as cp
from conftest import (
    ALL_DTYPES, ALL_DTYPES_NO_BOOL, FLOAT_DTYPES, INT_DTYPES,
    NUMERIC_DTYPES,
    assert_eq, make_arg,
)


# ── Shape groups for logic tests ──────────────────────────────────
LOGIC_SHAPES = [(5,), (2, 3), (2, 3, 4)]

# Comparison ops as (cp_func, np_func) pairs for parametrized tests
COMPARISON_OPS = [
    ("greater", np.greater),
    ("greater_equal", np.greater_equal),
    ("less", np.less),
    ("less_equal", np.less_equal),
    ("equal", np.equal),
    ("not_equal", np.not_equal),
]


# ====================================================================
# Logical (logical_and, logical_or, logical_not, logical_xor)
# ====================================================================
# Ref: cupy_tests/logic_tests/test_ops.py

class TestLogicalAnd:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        b_np = make_arg(shape, dtype)
        a_cp, b_cp = cp.array(a_np), cp.array(b_np)
        assert_array_equal(cp.logical_and(a_cp, b_cp).get(),
                           np.logical_and(a_np, b_np))

    def test_mixed_truthy(self):
        a = cp.array(np.array([0, 1, 2, 0], dtype=np.float32))
        b = cp.array(np.array([1, 0, 1, 0], dtype=np.float32))
        expected = np.logical_and([0, 1, 2, 0], [1, 0, 1, 0])
        assert_array_equal(cp.logical_and(a, b).get(), expected)


class TestLogicalOr:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        b_np = make_arg(shape, dtype)
        a_cp, b_cp = cp.array(a_np), cp.array(b_np)
        assert_array_equal(cp.logical_or(a_cp, b_cp).get(),
                           np.logical_or(a_np, b_np))

    def test_mixed_truthy(self):
        a = cp.array(np.array([0, 0, 1, 0], dtype=np.float32))
        b = cp.array(np.array([0, 1, 0, 0], dtype=np.float32))
        expected = np.logical_or([0, 0, 1, 0], [0, 1, 0, 0])
        assert_array_equal(cp.logical_or(a, b).get(), expected)


class TestLogicalNot:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_array_equal(cp.logical_not(a_cp).get(),
                           np.logical_not(a_np))

    def test_zeros(self):
        a = cp.array(np.array([0, 0, 0], dtype=np.float32))
        assert_array_equal(cp.logical_not(a).get(),
                           np.logical_not([0, 0, 0]))


class TestLogicalXor:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        b_np = make_arg(shape, dtype)
        a_cp, b_cp = cp.array(a_np), cp.array(b_np)
        assert_array_equal(cp.logical_xor(a_cp, b_cp).get(),
                           np.logical_xor(a_np, b_np))


# ====================================================================
# Comparison functions (greater, greater_equal, less, less_equal,
#                       equal, not_equal)
# ====================================================================
# Ref: cupy_tests/logic_tests/test_comparison.py
#
# Big test: ALL_DTYPES_NO_BOOL x ALL_DTYPES_NO_BOOL x 6 ops x 3 shapes = 1458

class TestComparisonDtypePairs:
    @pytest.mark.parametrize("op_name,np_func", COMPARISON_OPS,
                             ids=[n for n, _ in COMPARISON_OPS])
    @pytest.mark.parametrize("x_dtype", ALL_DTYPES_NO_BOOL)
    @pytest.mark.parametrize("y_dtype", ALL_DTYPES_NO_BOOL)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_dtype_pair(self, op_name, np_func, x_dtype, y_dtype, shape):
        if (x_dtype == np.complex64 or y_dtype == np.complex64) and op_name not in ("equal", "not_equal"):
            pytest.skip("ordering comparison not supported for complex types")
        a_np = make_arg(shape, x_dtype)
        b_np = make_arg(shape, y_dtype)
        cp_func = getattr(cp, op_name)
        result = cp_func(cp.array(a_np), cp.array(b_np))
        expected = np_func(a_np, b_np)
        assert_array_equal(result.get(), expected)


class TestComparisonEdgeCases:
    @pytest.mark.parametrize("op_name,np_func", COMPARISON_OPS,
                             ids=[n for n, _ in COMPARISON_OPS])
    def test_nan(self, op_name, np_func):
        a = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float32)
        b = np.array([np.nan, 2.0, 3.0, np.nan], dtype=np.float32)
        cp_func = getattr(cp, op_name)
        result = cp_func(cp.array(a), cp.array(b))
        expected = np_func(a, b)
        assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("op_name,np_func", COMPARISON_OPS,
                             ids=[n for n, _ in COMPARISON_OPS])
    def test_inf(self, op_name, np_func):
        a = np.array([np.inf, -np.inf, 0.0, 1.0], dtype=np.float32)
        b = np.array([1.0, -np.inf, np.inf, np.inf], dtype=np.float32)
        cp_func = getattr(cp, op_name)
        result = cp_func(cp.array(a), cp.array(b))
        expected = np_func(a, b)
        assert_array_equal(result.get(), expected)


# ====================================================================
# Predicate (isneginf, isposinf, iscomplex, isreal, isscalar, array_equiv)
# ====================================================================
# Ref: cupy_tests/logic_tests/test_type_test.py

class TestIsneginf:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_special_values(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        # Inject specials
        a_flat = a_np.copy().ravel()
        if len(a_flat) >= 3:
            a_flat[0] = np.inf
            a_flat[1] = -np.inf
            a_flat[2] = np.nan
        a_np_mod = a_flat.reshape(shape)
        a_cp = cp.array(a_np_mod)
        assert_array_equal(cp.isneginf(a_cp).get(), np.isneginf(a_np_mod))


class TestIsposinf:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_special_values(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_flat = a_np.copy().ravel()
        if len(a_flat) >= 3:
            a_flat[0] = np.inf
            a_flat[1] = -np.inf
            a_flat[2] = np.nan
        a_np_mod = a_flat.reshape(shape)
        a_cp = cp.array(a_np_mod)
        assert_array_equal(cp.isposinf(a_cp).get(), np.isposinf(a_np_mod))


class TestIscomplex:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_real_array(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_array_equal(cp.iscomplex(a_cp).get(), np.iscomplex(a_np))

    def test_complex_array(self):
        a_np = np.array([1 + 2j, 3 + 0j, 0 + 0j], dtype=np.complex64)
        a_cp = cp.array(a_np)
        assert_array_equal(cp.iscomplex(a_cp).get(), np.iscomplex(a_np))


class TestIsreal:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_real_array(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_array_equal(cp.isreal(a_cp).get(), np.isreal(a_np))

    def test_complex_array(self):
        a_np = np.array([1 + 2j, 3 + 0j, 0 + 0j], dtype=np.complex64)
        a_cp = cp.array(a_np)
        assert_array_equal(cp.isreal(a_cp).get(), np.isreal(a_np))


class TestIsscalar:
    def test_python_scalar(self):
        assert cp.isscalar(3.0) is True
        assert cp.isscalar(5) is True

    def test_array_not_scalar(self):
        a = cp.array(np.array([1.0, 2.0], dtype=np.float32))
        assert cp.isscalar(a) is False

    def test_0d_array(self):
        a = cp.array(np.float32(5.0))
        # NumPy behavior: np.isscalar(np.array(5)) is False
        # Our implementation delegates to numpy
        assert cp.isscalar(a) is False


class TestArrayEquiv:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_equal(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        b_cp = cp.array(a_np.copy())
        assert cp.array_equiv(a_cp, b_cp) is True

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_not_equal(self, dtype):
        a_np = make_arg((5,), dtype)
        b_np = a_np.copy()
        b_np[0] = a_np[0] + 100
        assert cp.array_equiv(cp.array(a_np), cp.array(b_np)) is False

    def test_broadcast_equiv(self):
        a = cp.array(np.array([1, 1, 1], dtype=np.float32))
        b = cp.array(np.array([1], dtype=np.float32))
        assert cp.array_equiv(a, b) is True


# ====================================================================
# Bitwise (bitwise_and, bitwise_or, bitwise_xor, invert)
# ====================================================================
# Ref: cupy_tests/binary_tests/test_elementwise.py

class TestBitwiseAnd:
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        b_np = make_arg(shape, dtype)
        result = cp.bitwise_and(cp.array(a_np), cp.array(b_np))
        expected = np.bitwise_and(a_np.astype(np.int32), b_np.astype(np.int32))
        assert_array_equal(result.get(), expected)


class TestBitwiseOr:
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        b_np = make_arg(shape, dtype)
        result = cp.bitwise_or(cp.array(a_np), cp.array(b_np))
        expected = np.bitwise_or(a_np.astype(np.int32), b_np.astype(np.int32))
        assert_array_equal(result.get(), expected)


class TestBitwiseXor:
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        b_np = make_arg(shape, dtype)
        result = cp.bitwise_xor(cp.array(a_np), cp.array(b_np))
        expected = np.bitwise_xor(a_np.astype(np.int32), b_np.astype(np.int32))
        assert_array_equal(result.get(), expected)


class TestInvert:
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        result = cp.invert(cp.array(a_np))
        expected = np.invert(a_np.astype(np.int32))
        assert_array_equal(result.get(), expected)

    def test_specific_values(self):
        a_np = np.array([0b1100, 0b1010, 0b1111, 0b0001], dtype=np.int32)
        result = cp.invert(cp.array(a_np))
        expected = np.invert(a_np)
        assert_array_equal(result.get(), expected)


# ====================================================================
# Shift (left_shift, right_shift)
# ====================================================================
# Ref: cupy_tests/binary_tests/test_elementwise.py

class TestLeftShift:
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        shift = np.ones(shape, dtype=dtype) * 2
        result = cp.left_shift(cp.array(a_np), cp.array(shift))
        expected = np.left_shift(a_np.astype(np.int32), shift.astype(np.int32))
        assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("shift", [0, 1, 4, 8])
    def test_shift_amounts(self, shift):
        a_np = np.array([1, 2, 3, 4], dtype=np.int32)
        s_np = np.full(4, shift, dtype=np.int32)
        result = cp.left_shift(cp.array(a_np), cp.array(s_np))
        expected = np.left_shift(a_np, s_np)
        assert_array_equal(result.get(), expected)


class TestRightShift:
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    @pytest.mark.parametrize("shape", LOGIC_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        shift = np.ones(shape, dtype=dtype)
        result = cp.right_shift(cp.array(a_np), cp.array(shift))
        expected = np.right_shift(a_np.astype(np.int32), shift.astype(np.int32))
        assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("shift", [0, 1, 4, 8])
    def test_shift_amounts(self, shift):
        a_np = np.array([16, 32, 64, 128], dtype=np.int32)
        s_np = np.full(4, shift, dtype=np.int32)
        result = cp.right_shift(cp.array(a_np), cp.array(s_np))
        expected = np.right_shift(a_np, s_np)
        assert_array_equal(result.get(), expected)


# ====================================================================
# Pack (packbits, unpackbits)
# ====================================================================
# Ref: cupy_tests/binary_tests/test_packing.py

class TestPackbits:
    def test_basic(self):
        a_np = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int32)
        a_cp = cp.array(a_np)
        expected = np.packbits(a_np.astype(np.uint8))
        assert_array_equal(cp.packbits(a_cp).get(), expected.astype(np.uint16))

    def test_all_ones(self):
        a_np = np.ones(8, dtype=np.int32)
        a_cp = cp.array(a_np)
        expected = np.packbits(a_np.astype(np.uint8))
        assert_array_equal(cp.packbits(a_cp).get(), expected.astype(np.uint16))

    def test_all_zeros(self):
        a_np = np.zeros(8, dtype=np.int32)
        a_cp = cp.array(a_np)
        expected = np.packbits(a_np.astype(np.uint8))
        assert_array_equal(cp.packbits(a_cp).get(), expected.astype(np.uint16))


class TestUnpackbits:
    def test_basic(self):
        a_np = np.array([0b10101101], dtype=np.uint16)
        a_cp = cp.array(a_np)
        expected = np.unpackbits(a_np.astype(np.uint8))
        assert_array_equal(cp.unpackbits(a_cp).get(), expected.astype(np.uint16))

    def test_roundtrip(self):
        a_np = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int32)
        a_cp = cp.array(a_np)
        packed = cp.packbits(a_cp)
        unpacked = cp.unpackbits(packed)
        assert_array_equal(unpacked.get(), a_np.astype(np.uint16))


# ====================================================================
# GCD / LCM
# ====================================================================
# Ref: cupy_tests/math_tests/test_arithmetic.py

class TestGcd:
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_basic(self, dtype):
        a_np = np.array([12, 15, 20, 100], dtype=dtype)
        b_np = np.array([8, 25, 15, 75], dtype=dtype)
        result = cp.gcd(cp.array(a_np), cp.array(b_np))
        expected = np.gcd(a_np, b_np)
        assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_zero(self, dtype):
        a_np = np.array([0, 5, 10], dtype=dtype)
        b_np = np.array([5, 0, 10], dtype=dtype)
        result = cp.gcd(cp.array(a_np), cp.array(b_np))
        expected = np.gcd(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_negative(self):
        a_np = np.array([-12, 15, -20], dtype=np.int32)
        b_np = np.array([8, -25, -15], dtype=np.int32)
        result = cp.gcd(cp.array(a_np), cp.array(b_np))
        expected = np.gcd(a_np, b_np)
        assert_array_equal(result.get(), expected)


class TestLcm:
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_basic(self, dtype):
        a_np = np.array([4, 6, 8, 12], dtype=dtype)
        b_np = np.array([6, 8, 12, 16], dtype=dtype)
        result = cp.lcm(cp.array(a_np), cp.array(b_np))
        expected = np.lcm(a_np, b_np)
        assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_zero(self, dtype):
        a_np = np.array([0, 5, 10], dtype=dtype)
        b_np = np.array([5, 0, 10], dtype=dtype)
        result = cp.lcm(cp.array(a_np), cp.array(b_np))
        expected = np.lcm(a_np, b_np)
        assert_array_equal(result.get(), expected)

    def test_negative(self):
        a_np = np.array([-4, 6, -8], dtype=np.int32)
        b_np = np.array([6, -8, -12], dtype=np.int32)
        result = cp.lcm(cp.array(a_np), cp.array(b_np))
        expected = np.lcm(a_np, b_np)
        assert_array_equal(result.get(), expected)
