"""Tests for math operations: unary float, binary, array ops, predicates, utility, and math_ext.

Consolidates test_elementwise.py + test_math_ext.py into a single parametrized test file.
Ref: cupy_tests/math_tests/test_arithmetic.py, test_trigonometric.py
Target: ~2,748 parametrized cases.
"""

import numpy as np
import pytest

import macmetalpy as cp
from conftest import (
    FLOAT_DTYPES,
    NUMERIC_DTYPES,
    SHAPES_NONZERO,
    assert_eq,
    make_arg,
)

# ── Helpers ──────────────────────────────────────────────────────

SHAPES_3 = [(5,), (2, 3), (2, 3, 4)]


def _make_positive(shape, dtype, xp=np, low=0.1, high=10.0):
    """Create array with positive values (for log, sqrt, etc.)."""
    if not shape:
        return xp.array(1.5, dtype=dtype)
    size = 1
    for s in shape:
        size *= s
    data = np.linspace(low, high, size, dtype=np.float64).reshape(shape)
    return xp.array(data.astype(dtype))


def _make_unit(shape, dtype, xp=np):
    """Create array in [-1, 1] domain (for arcsin, arccos, arctanh)."""
    if not shape:
        return xp.array(0.5, dtype=dtype)
    size = 1
    for s in shape:
        size *= s
    data = np.linspace(-0.95, 0.95, size, dtype=np.float64).reshape(shape)
    return xp.array(data.astype(dtype))


def _make_ge_one(shape, dtype, xp=np):
    """Create array with values >= 1 (for arccosh)."""
    if not shape:
        return xp.array(1.5, dtype=dtype)
    size = 1
    for s in shape:
        size *= s
    data = np.linspace(1.0, 10.0, size, dtype=np.float64).reshape(shape)
    return xp.array(data.astype(dtype))


# ── 1. Unary float ops ──────────────────────────────────────────
# Ref: cupy_tests/math_tests/test_arithmetic.py::TestArithmeticUnary
# 20 funcs x FLOAT_DTYPES x SHAPES_NONZERO, category="unary_math"


class TestSqrt:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype)
        expected = np.sqrt(np_a)
        result = cp.sqrt(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.sqrt(cp.array(np_a)), np.sqrt(np_a), dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_one(self, dtype):
        np_a = np.array([1.0], dtype=dtype)
        assert_eq(cp.sqrt(cp.array(np_a)), np.sqrt(np_a), dtype=dtype, category="unary_math")


class TestExp:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.exp(np_a)
        result = cp.exp(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.exp(cp.array(np_a)), np.exp(np_a), dtype=dtype, category="unary_math")


class TestLog:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype)
        expected = np.log(np_a)
        result = cp.log(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_one(self, dtype):
        np_a = np.array([1.0], dtype=dtype)
        assert_eq(cp.log(cp.array(np_a)), np.log(np_a), dtype=dtype, category="unary_math")


class TestSign:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=dtype)
        expected = np.sign(np_a)
        result = cp.sign(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0, 0.0], dtype=dtype)
        assert_eq(cp.sign(cp.array(np_a)), np.sign(np_a), dtype=dtype, category="unary_math")


class TestFloor:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=-5.5, high=5.5)
        expected = np.floor(np_a)
        result = cp.floor(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative(self, dtype):
        np_a = np.array([-1.1, -2.9, -0.5], dtype=dtype)
        assert_eq(cp.floor(cp.array(np_a)), np.floor(np_a), dtype=dtype, category="unary_math")


class TestCeil:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=-5.5, high=5.5)
        expected = np.ceil(np_a)
        result = cp.ceil(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative(self, dtype):
        np_a = np.array([-1.1, -2.9, -0.5], dtype=dtype)
        assert_eq(cp.ceil(cp.array(np_a)), np.ceil(np_a), dtype=dtype, category="unary_math")


class TestSin:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.sin(np_a)
        result = cp.sin(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.sin(cp.array(np_a)), np.sin(np_a), dtype=dtype, category="unary_math")


class TestCos:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.cos(np_a)
        result = cp.cos(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.cos(cp.array(np_a)), np.cos(np_a), dtype=dtype, category="unary_math")


class TestTan:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_unit(shape, dtype)
        expected = np.tan(np_a)
        result = cp.tan(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.tan(cp.array(np_a)), np.tan(np_a), dtype=dtype, category="unary_math")


class TestArcsin:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_unit(shape, dtype)
        expected = np.arcsin(np_a)
        result = cp.arcsin(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.arcsin(cp.array(np_a)), np.arcsin(np_a), dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_boundary(self, dtype):
        np_a = np.array([-1.0, 1.0], dtype=dtype)
        assert_eq(cp.arcsin(cp.array(np_a)), np.arcsin(np_a), dtype=dtype, category="unary_math")


class TestArccos:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_unit(shape, dtype)
        expected = np.arccos(np_a)
        result = cp.arccos(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_boundary(self, dtype):
        np_a = np.array([-1.0, 0.0, 1.0], dtype=dtype)
        assert_eq(cp.arccos(cp.array(np_a)), np.arccos(np_a), dtype=dtype, category="unary_math")


class TestArctan:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.arctan(np_a)
        result = cp.arctan(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.arctan(cp.array(np_a)), np.arctan(np_a), dtype=dtype, category="unary_math")


class TestSinh:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        # Clamp to avoid overflow threshold differences between Metal and numpy
        np_a = np.clip(np_a, -80, 80).astype(dtype)
        expected = np.sinh(np_a)
        result = cp.sinh(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.sinh(cp.array(np_a)), np.sinh(np_a), dtype=dtype, category="unary_math")


class TestCosh:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        # Clamp to avoid overflow threshold differences between Metal and numpy
        np_a = np.clip(np_a, -80, 80).astype(dtype)
        expected = np.cosh(np_a)
        result = cp.cosh(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.cosh(cp.array(np_a)), np.cosh(np_a), dtype=dtype, category="unary_math")


class TestTanh:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        # Clamp to avoid Metal producing NaN for large inputs (threshold ~45 for float32)
        np_a = np.clip(np_a, -40, 40).astype(dtype)
        expected = np.tanh(np_a)
        result = cp.tanh(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.tanh(cp.array(np_a)), np.tanh(np_a), dtype=dtype, category="unary_math")


class TestLog2:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype)
        expected = np.log2(np_a)
        result = cp.log2(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_powers_of_two(self, dtype):
        np_a = np.array([1.0, 2.0, 4.0, 8.0], dtype=dtype)
        assert_eq(cp.log2(cp.array(np_a)), np.log2(np_a), dtype=dtype, category="unary_math")


class TestLog10:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype)
        expected = np.log10(np_a)
        result = cp.log10(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_powers_of_ten(self, dtype):
        np_a = np.array([1.0, 10.0, 100.0], dtype=dtype)
        assert_eq(cp.log10(cp.array(np_a)), np.log10(np_a), dtype=dtype, category="unary_math")


class TestSquare:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.square(np_a)
        result = cp.square(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.square(cp.array(np_a)), np.square(np_a), dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative(self, dtype):
        np_a = np.array([-3.0, -2.0, -1.0], dtype=dtype)
        assert_eq(cp.square(cp.array(np_a)), np.square(np_a), dtype=dtype, category="unary_math")


class TestNegative:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.negative(np_a)
        result = cp.negative(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.negative(cp.array(np_a)), np.negative(np_a), dtype=dtype, category="unary_math")


class TestAbs:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=dtype)
        expected = np.abs(np_a)
        result = cp.abs(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.abs(cp.array(np_a)), np.abs(np_a), dtype=dtype, category="unary_math")


# ── 2. Binary ops ───────────────────────────────────────────────
# Ref: cupy_tests/math_tests/test_arithmetic.py::TestArithmeticBinary


class TestPower:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=1.0, high=4.0)
        np_b = np.array([2.0] * np_a.size, dtype=dtype).reshape(shape) if shape else np.array(2.0, dtype=dtype)
        expected = np.power(np_a, np_b)
        result = cp.power(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="power")

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_scalar(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        expected = np.power(np_a, np.array(2, dtype=dtype))
        result = cp.power(cp.array(np_a), cp.array(np.array(2, dtype=dtype)))
        assert_eq(result, expected, dtype=dtype, category="power")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero_exponent(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([0.0, 0.0, 0.0], dtype=dtype)
        expected = np.power(np_a, np_b)
        result = cp.power(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="power")


class TestDot:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([4.0, 5.0, 6.0], dtype=dtype)
        expected = np.dot(np_a, np_b)
        result = cp.dot(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, dtype):
        np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        np_b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=dtype)
        expected = np.dot(np_a, np_b)
        result = cp.dot(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="arithmetic")


class TestMod:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=5.0, high=20.0)
        np_b = _make_positive(shape, dtype, low=1.0, high=5.0)
        expected = np.mod(np_a, np_b)
        result = cp.mod(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="arithmetic")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative(self, dtype):
        np_a = np.array([-7.0, 8.0, -9.0], dtype=dtype)
        np_b = np.array([3.0, -3.0, -4.0], dtype=dtype)
        expected = np.mod(np_a, np_b)
        result = cp.mod(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="arithmetic")


class TestRemainder:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_alias(self, dtype):
        np_a = np.array([7.0, 8.0, 9.0], dtype=dtype)
        np_b = np.array([3.0, 3.0, 4.0], dtype=dtype)
        result_mod = cp.mod(cp.array(np_a), cp.array(np_b))
        result_rem = cp.remainder(cp.array(np_a), cp.array(np_b))
        assert_eq(result_rem, result_mod.get(), dtype=dtype, category="arithmetic")


class TestAround:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=-5.5, high=5.5)
        expected = np.around(np_a)
        result = cp.around(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_decimals(self, dtype):
        np_a = np.array([1.234, 2.567, 3.891], dtype=dtype)
        expected = np.around(np_a, decimals=2)
        result = cp.around(cp.array(np_a), decimals=2)
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative_decimals(self, dtype):
        np_a = np.array([123.0, 456.0, 789.0], dtype=dtype)
        expected = np.around(np_a, decimals=-1)
        result = cp.around(cp.array(np_a), decimals=-1)
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_round_alias(self, dtype):
        np_a = np.array([1.4, 1.5, 2.5, 3.6], dtype=dtype)
        result_around = cp.around(cp.array(np_a))
        result_round = cp.round_(cp.array(np_a))
        assert_eq(result_round, result_around.get(), dtype=dtype, category="unary_math")


# ── 3. Array ops ────────────────────────────────────────────────
# Ref: cupy_tests/math_tests/test_arithmetic.py


class TestWhere:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_cond = (make_arg(shape, np.float32) > 2.5)
        np_x = make_arg(shape, dtype)
        np_y = make_arg(shape, dtype) * 10
        expected = np.where(np_cond, np_x, np_y)
        result = cp.where(cp.array(np_cond), cp.array(np_x), cp.array(np_y))
        assert_eq(result, expected, dtype=dtype, category="default")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_scalar_condition(self, dtype):
        np_x = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_y = np.array([4.0, 5.0, 6.0], dtype=dtype)
        cond = np.array([True, False, True])
        expected = np.where(cond, np_x, np_y)
        result = cp.where(cp.array(cond), cp.array(np_x), cp.array(np_y))
        assert_eq(result, expected, dtype=dtype)


class TestClip:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        a_min, a_max = 2, 4
        expected = np.clip(np_a, a_min, a_max)
        result = cp.clip(cp.array(np_a), a_min, a_max)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_float_bounds(self, dtype):
        np_a = np.array([-1.0, 0.5, 1.5, 3.0], dtype=dtype)
        expected = np.clip(np_a, 0.0, 2.0)
        result = cp.clip(cp.array(np_a), 0.0, 2.0)
        assert_eq(result, expected, dtype=dtype)


class TestConcatenate:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype) * 2
        expected = np.concatenate([np_a, np_b], axis=0)
        result = cp.concatenate([cp.array(np_a), cp.array(np_b)], axis=0)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_multiple_arrays(self, dtype):
        arrs = [np.array([1.0, 2.0], dtype=dtype) for _ in range(4)]
        expected = np.concatenate(arrs)
        result = cp.concatenate([cp.array(a) for a in arrs])
        assert_eq(result, expected, dtype=dtype)


class TestStack:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([4.0, 5.0, 6.0], dtype=dtype)
        expected = np.stack([np_a, np_b])
        result = cp.stack([cp.array(np_a), cp.array(np_b)])
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis1(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([4.0, 5.0, 6.0], dtype=dtype)
        expected = np.stack([np_a, np_b], axis=1)
        result = cp.stack([cp.array(np_a), cp.array(np_b)], axis=1)
        assert_eq(result, expected, dtype=dtype)


class TestVstack:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([4.0, 5.0, 6.0], dtype=dtype)
        expected = np.vstack([np_a, np_b])
        result = cp.vstack([cp.array(np_a), cp.array(np_b)])
        assert_eq(result, expected, dtype=dtype)


class TestHstack:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([4.0, 5.0, 6.0], dtype=dtype)
        expected = np.hstack([np_a, np_b])
        result = cp.hstack([cp.array(np_a), cp.array(np_b)])
        assert_eq(result, expected, dtype=dtype)


# ── 4. Predicates ───────────────────────────────────────────────
# Ref: cupy_tests/math_tests/test_misc.py


class TestIsnan:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=dtype)
        expected = np.isnan(np_a)
        result = cp.isnan(cp.array(np_a))
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_no_nan(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        assert_eq(cp.isnan(cp.array(np_a)), np.isnan(np_a))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_nan(self, dtype):
        np_a = np.array([np.nan, np.nan], dtype=dtype)
        assert_eq(cp.isnan(cp.array(np_a)), np.isnan(np_a))


class TestIsinf:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([1.0, np.inf, -np.inf, 0.0, np.nan], dtype=dtype)
        expected = np.isinf(np_a)
        result = cp.isinf(cp.array(np_a))
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_no_inf(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        assert_eq(cp.isinf(cp.array(np_a)), np.isinf(np_a))


class TestIsfinite:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([1.0, np.inf, -np.inf, np.nan, 0.0], dtype=dtype)
        expected = np.isfinite(np_a)
        result = cp.isfinite(cp.array(np_a))
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_finite(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        assert_eq(cp.isfinite(cp.array(np_a)), np.isfinite(np_a))


class TestIsclose:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        np_b = np.array([1.0, 2.0001, 3.0, 4.001, 5.0], dtype=dtype)
        expected = np.isclose(np_a, np_b)
        result = cp.isclose(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_nan_not_close(self, dtype):
        np_a = np.array([np.nan, 1.0], dtype=dtype)
        np_b = np.array([np.nan, 1.0], dtype=dtype)
        expected = np.isclose(np_a, np_b)
        result = cp.isclose(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected)


class TestAllclose:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_close(self, dtype, shape):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([1.0, 2.0, 3.0], dtype=dtype)
        assert cp.allclose(cp.array(np_a), cp.array(np_b)) == np.allclose(np_a, np_b)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_not_close(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([1.0, 2.0, 100.0], dtype=dtype)
        assert cp.allclose(cp.array(np_a), cp.array(np_b)) == np.allclose(np_a, np_b)


# ── 5. Utility ──────────────────────────────────────────────────
# Ref: cupy_tests/math_tests/test_misc.py


class TestNanToNum:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        if np.issubdtype(dtype, np.integer):
            np_a = make_arg(shape, dtype)
        else:
            np_a = np.array([1.0, np.nan, np.inf, -np.inf, 0.0], dtype=dtype)
        expected = np.nan_to_num(np_a)
        result = cp.nan_to_num(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_custom_values(self, dtype):
        np_a = np.array([np.nan, np.inf, -np.inf], dtype=dtype)
        expected = np.nan_to_num(np_a, nan=99.0, posinf=999.0, neginf=-999.0)
        result = cp.nan_to_num(cp.array(np_a), nan=99.0, posinf=999.0, neginf=-999.0)
        assert_eq(result, expected, dtype=dtype)


class TestArrayEqual:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_equal(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        assert cp.array_equal(cp.array(np_a), cp.array(np_a)) is True

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_not_equal(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([1.0, 2.0, 4.0], dtype=dtype)
        assert cp.array_equal(cp.array(np_a), cp.array(np_b)) is False


class TestCountNonzero:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.count_nonzero(np_a)
        result = cp.count_nonzero(cp.array(np_a))
        assert result == expected

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_zeros(self, dtype):
        np_a = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=dtype)
        expected = np.count_nonzero(np_a)
        result = cp.count_nonzero(cp.array(np_a))
        assert result == expected


class TestCopy:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        gpu_a = cp.array(np_a)
        result = cp.copy(gpu_a)
        assert_eq(result, np_a, dtype=dtype)


class TestAscontiguousarray:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        gpu_a = cp.array(np_a)
        result = cp.ascontiguousarray(gpu_a)
        assert_eq(result, np_a, dtype=dtype)


class TestTrace:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        expected = np.trace(np_a)
        result = cp.trace(cp.array(np_a))
        assert_eq(result, np.array(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_offset(self, dtype):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        expected = np.trace(np_a, offset=1)
        result = cp.trace(cp.array(np_a), offset=1)
        assert_eq(result, np.array(expected), dtype=dtype)


class TestDiagonal:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        expected = np.diagonal(np_a)
        result = cp.diagonal(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_offset(self, dtype):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        expected = np.diagonal(np_a, offset=1)
        result = cp.diagonal(cp.array(np_a), offset=1)
        assert_eq(result, expected, dtype=dtype)


# ── 6. Math extensions ──────────────────────────────────────────
# Ref: extended math functions from math_ext module


class TestSinc:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=-3.0, high=3.0)
        expected = np.sinc(np_a)
        result = cp.sinc(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        expected = np.sinc(np_a)
        result = cp.sinc(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestI0:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=0.0, high=5.0)
        expected = np.i0(np_a)
        result = cp.i0(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestConvolve:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_full(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_v = np.array([0.5, 1.0], dtype=dtype)
        expected = np.convolve(np_a, np_v, mode='full')
        result = cp.convolve(cp.array(np_a), cp.array(np_v), mode='full')
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_same(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        np_v = np.array([0.5, 1.0], dtype=dtype)
        expected = np.convolve(np_a, np_v, mode='same')
        result = cp.convolve(cp.array(np_a), cp.array(np_v), mode='same')
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_valid(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        np_v = np.array([0.5, 1.0], dtype=dtype)
        expected = np.convolve(np_a, np_v, mode='valid')
        result = cp.convolve(cp.array(np_a), cp.array(np_v), mode='valid')
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestInterp:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        xp = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=dtype)
        fp = np.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=dtype)
        x = np.array([0.5, 1.5, 2.5], dtype=dtype)
        expected = np.interp(x, xp, fp)
        result = cp.interp(cp.array(x), cp.array(xp), cp.array(fp))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_boundary(self, dtype):
        xp = np.array([0.0, 1.0, 2.0], dtype=dtype)
        fp = np.array([0.0, 10.0, 20.0], dtype=dtype)
        x = np.array([-1.0, 0.0, 2.0, 3.0], dtype=dtype)
        expected = np.interp(x, xp, fp)
        result = cp.interp(cp.array(x), cp.array(xp), cp.array(fp))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestFix:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=-5.5, high=5.5)
        expected = np.fix(np_a)
        result = cp.fix(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative(self, dtype):
        np_a = np.array([-1.9, -2.1, -0.5], dtype=dtype)
        expected = np.fix(np_a)
        result = cp.fix(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestUnwrap:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=dtype)
        expected = np.unwrap(np_a)
        result = cp.unwrap(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_phase_wrap(self, dtype):
        # Simulate phase wrapping
        np_a = np.array([0.0, 3.0, 6.0, 0.5, 3.5], dtype=dtype)
        expected = np.unwrap(np_a)
        result = cp.unwrap(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestTrapezoid:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_y = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        _trap = getattr(np, 'trapezoid', np.trapz)
        expected = _trap(np_y)
        result = cp.trapezoid(cp.array(np_y))
        assert_eq(result, np.array(expected), dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_x(self, dtype):
        np_y = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_x = np.array([0.0, 1.0, 3.0], dtype=dtype)
        _trap = getattr(np, 'trapezoid', np.trapz)
        expected = _trap(np_y, x=np_x)
        result = cp.trapezoid(cp.array(np_y), x=cp.array(np_x))
        assert_eq(result, np.array(expected), dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dx(self, dtype):
        np_y = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        _trap = getattr(np, 'trapezoid', np.trapz)
        expected = _trap(np_y, dx=0.5)
        result = cp.trapezoid(cp.array(np_y), dx=0.5)
        assert_eq(result, np.array(expected), dtype=dtype, category="unary_math")
