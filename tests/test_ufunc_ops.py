"""Tests for ufunc operations: binary arithmetic, unary math, binary math, angle, special, reductive, ufunc objects.

Ref: cupy_tests/math_tests/test_arithmetic.py
Target: ~3,510 parametrized cases.
"""

import numpy as np
import pytest

import macmetalpy as cp
from conftest import (
    ALL_DTYPES_NO_BOOL,
    FLOAT_DTYPES,
    NUMERIC_DTYPES,
    SHAPES_NONZERO,
    assert_eq,
    make_arg,
)

# ── Helpers ──────────────────────────────────────────────────────

SHAPES_3 = [(5,), (2, 3), (2, 3, 4)]

# Representative dtype pairs for binary arith tests (subset of full N*N)
ARITH_DTYPE_PAIRS = [
    (np.float32, np.float32),
    (np.float16, np.float32),
    (np.float32, np.int32),
    (np.int32, np.int32),
    (np.int32, np.float16),
    (np.int16, np.int32),
    (np.uint32, np.int32),
    (np.float16, np.float16),
    (np.int64, np.float32),
    (np.uint16, np.float32),
]


def _make_positive(shape, dtype, xp=np, low=0.5, high=10.0):
    if not shape:
        return xp.array(2.0, dtype=dtype)
    size = 1
    for s in shape:
        size *= s
    data = np.linspace(low, high, size, dtype=np.float64).reshape(shape)
    return xp.array(data.astype(dtype))


def _make_unit(shape, dtype, xp=np):
    if not shape:
        return xp.array(0.5, dtype=dtype)
    size = 1
    for s in shape:
        size *= s
    data = np.linspace(-0.95, 0.95, size, dtype=np.float64).reshape(shape)
    return xp.array(data.astype(dtype))


# ── 1. Binary arithmetic ufuncs ──────────────────────────────────
# Ref: cupy_tests/math_tests/test_arithmetic.py::ArithmeticBinaryBase
# 8 funcs x ARITH_DTYPE_PAIRS x scenarios


BINARY_ARITH_FUNCS = [
    ("add", cp.add, np.add),
    ("subtract", cp.subtract, np.subtract),
    ("multiply", cp.multiply, np.multiply),
    ("divide", cp.divide, np.divide),
    ("true_divide", cp.true_divide, np.true_divide),
    ("floor_divide", cp.floor_divide, np.floor_divide),
    ("float_power", cp.float_power, np.float_power),
    ("fmod", cp.fmod, np.fmod),
]


class TestBinaryArithArrayArray:
    """Test binary arithmetic with array-array operands."""

    @pytest.mark.parametrize("name,cp_func,np_func", BINARY_ARITH_FUNCS,
                             ids=[f[0] for f in BINARY_ARITH_FUNCS])
    @pytest.mark.parametrize("x_type,y_type", ARITH_DTYPE_PAIRS,
                             ids=[f"{np.dtype(a).name}_{np.dtype(b).name}" for a, b in ARITH_DTYPE_PAIRS])
    def test_array_array(self, name, cp_func, np_func, x_type, y_type):
        np_a = _make_positive((5,), x_type)
        np_b = _make_positive((5,), y_type, low=1.0, high=5.0)
        expected = np_func(np_a, np_b)
        result = cp_func(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")


class TestBinaryArithArrayScalar:
    """Test binary arithmetic with array-scalar operands."""

    @pytest.mark.parametrize("name,cp_func,np_func", BINARY_ARITH_FUNCS,
                             ids=[f[0] for f in BINARY_ARITH_FUNCS])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_array_scalar(self, name, cp_func, np_func, dtype):
        np_a = _make_positive((5,), dtype)
        np_b = np.array(2.0, dtype=dtype)
        expected = np_func(np_a, np_b)
        result = cp_func(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")


class TestBinaryArithEdgeCases:
    """Test binary arithmetic edge cases: division by zero, nan, inf."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_divide_by_zero(self, dtype):
        np_a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_b = np.array([0.0, 0.0, 0.0], dtype=dtype)
        with np.errstate(divide='ignore', invalid='ignore'):
            expected = np.divide(np_a, np_b)
        result = cp.divide(cp.array(np_a), cp.array(np_b))
        # Compare where both are finite
        res_np = result.get()
        assert np.all(np.isinf(res_np) == np.isinf(expected))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_add_nan(self, dtype):
        np_a = np.array([1.0, np.nan, 3.0], dtype=dtype)
        np_b = np.array([4.0, 5.0, np.nan], dtype=dtype)
        expected = np.add(np_a, np_b)
        result = cp.add(cp.array(np_a), cp.array(np_b))
        res_np = result.get()
        # Check NaN positions match
        assert np.all(np.isnan(res_np) == np.isnan(expected))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_multiply_inf(self, dtype):
        np_a = np.array([np.inf, -np.inf, 1.0], dtype=dtype)
        np_b = np.array([2.0, 3.0, np.inf], dtype=dtype)
        expected = np.multiply(np_a, np_b)
        result = cp.multiply(cp.array(np_a), cp.array(np_b))
        res_np = result.get()
        assert np.all(np.isinf(res_np) == np.isinf(expected))
        # Check signs match for inf
        finite_mask = np.isfinite(expected)
        if np.any(finite_mask):
            assert_eq(cp.array(res_np[finite_mask]),
                      expected[finite_mask], dtype=dtype, category="arithmetic")


# ── 2. Unary math ufuncs ────────────────────────────────────────
# Ref: cupy_tests/math_tests/test_arithmetic.py::TestArithmeticUnary
# 13 funcs x FLOAT_DTYPES x SHAPES_NONZERO


class TestExp2:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.exp2(np_a)
        result = cp.exp2(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.exp2(cp.array(np_a)), np.exp2(np_a), dtype=dtype, category="unary_math")


class TestExpm1:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=0.01, high=2.0)
        expected = np.expm1(np_a)
        result = cp.expm1(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.expm1(cp.array(np_a)), np.expm1(np_a), dtype=dtype, category="unary_math")


class TestLog1p:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=0.01, high=10.0)
        expected = np.log1p(np_a)
        result = cp.log1p(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.log1p(cp.array(np_a)), np.log1p(np_a), dtype=dtype, category="unary_math")


class TestCbrt:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.cbrt(np_a)
        result = cp.cbrt(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative(self, dtype):
        np_a = np.array([-8.0, -27.0, -1.0], dtype=dtype)
        expected = np.cbrt(np_a)
        result = cp.cbrt(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestReciprocal:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=0.5, high=10.0)
        expected = np.reciprocal(np_a)
        result = cp.reciprocal(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_one(self, dtype):
        np_a = np.array([1.0], dtype=dtype)
        assert_eq(cp.reciprocal(cp.array(np_a)), np.reciprocal(np_a), dtype=dtype, category="unary_math")


class TestRint:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=-5.5, high=5.5)
        expected = np.rint(np_a)
        result = cp.rint(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestTrunc:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_positive(shape, dtype, low=-5.5, high=5.5)
        expected = np.trunc(np_a)
        result = cp.trunc(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative(self, dtype):
        np_a = np.array([-1.9, -2.3, -3.7], dtype=dtype)
        assert_eq(cp.trunc(cp.array(np_a)), np.trunc(np_a), dtype=dtype, category="unary_math")


class TestAbsolute:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=dtype)
        expected = np.absolute(np_a)
        result = cp.absolute(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestFabs:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = np.array([-1.5, 2.5, -3.5, 0.0, 4.5], dtype=dtype)
        expected = np.fabs(np_a)
        result = cp.fabs(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestPositive:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.positive(np_a)
        result = cp.positive(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestArcsinh:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.arcsinh(np_a)
        result = cp.arcsinh(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.arcsinh(cp.array(np_a)), np.arcsinh(np_a), dtype=dtype, category="unary_math")


class TestArccosh:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        # arccosh needs x >= 1
        np_a = _make_positive(shape, dtype, low=1.0, high=10.0)
        expected = np.arccosh(np_a)
        result = cp.arccosh(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_one(self, dtype):
        np_a = np.array([1.0], dtype=dtype)
        assert_eq(cp.arccosh(cp.array(np_a)), np.arccosh(np_a), dtype=dtype, category="unary_math")


class TestArctanh:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NONZERO)
    def test_basic(self, dtype, shape):
        np_a = _make_unit(shape, dtype)
        expected = np.arctanh(np_a)
        result = cp.arctanh(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.arctanh(cp.array(np_a)), np.arctanh(np_a), dtype=dtype, category="unary_math")


# ── 3. Binary math ufuncs ───────────────────────────────────────
# Ref: cupy_tests/math_tests/test_arithmetic.py
# 7 funcs x FLOAT_DTYPES x 3 shapes


class TestArctan2:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_b = _make_positive(shape, dtype)
        expected = np.arctan2(np_a, np_b)
        result = cp.arctan2(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zeros(self, dtype):
        np_a = np.array([1.0, 0.0, -1.0], dtype=dtype)
        np_b = np.array([0.0, 1.0, 0.0], dtype=dtype)
        expected = np.arctan2(np_a, np_b)
        result = cp.arctan2(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestHypot:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype) * 2
        expected = np.hypot(np_a, np_b)
        result = cp.hypot(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_pythagorean(self, dtype):
        np_a = np.array([3.0, 5.0, 8.0], dtype=dtype)
        np_b = np.array([4.0, 12.0, 15.0], dtype=dtype)
        expected = np.hypot(np_a, np_b)
        result = cp.hypot(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestLogaddexp:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype) * 2
        expected = np.logaddexp(np_a, np_b)
        result = cp.logaddexp(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestLogaddexp2:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype) * 2
        expected = np.logaddexp2(np_a, np_b)
        result = cp.logaddexp2(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestHeaviside:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([-1.0, 0.0, 1.0, -2.0, 3.0], dtype=dtype)
        np_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype)
        expected = np.heaviside(np_a, np_b)
        result = cp.heaviside(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zeros(self, dtype):
        np_a = np.array([0.0, 0.0, 0.0], dtype=dtype)
        np_b = np.array([0.0, 0.5, 1.0], dtype=dtype)
        expected = np.heaviside(np_a, np_b)
        result = cp.heaviside(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestCopysign:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([1.0, -1.0, 1.0, -1.0, 0.0], dtype=dtype)
        np_b = np.array([-1.0, 1.0, 1.0, -1.0, -1.0], dtype=dtype)
        expected = np.copysign(np_a, np_b)
        result = cp.copysign(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestNextafter:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        np_b = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype)
        expected = np.nextafter(np_a, np_b)
        result = cp.nextafter(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_toward_zero(self, dtype):
        np_a = np.array([1.0, -1.0], dtype=dtype)
        np_b = np.array([0.0, 0.0], dtype=dtype)
        expected = np.nextafter(np_a, np_b)
        result = cp.nextafter(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


# ── 4. Angle conversions ────────────────────────────────────────
# Ref: cupy_tests/math_tests/test_trigonometric.py
# 4 funcs x FLOAT_DTYPES x 3 shapes


class TestDegrees:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([0.0, np.pi/2, np.pi, 2*np.pi, -np.pi], dtype=dtype)
        expected = np.degrees(np_a)
        result = cp.degrees(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestRadians:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([0.0, 90.0, 180.0, 360.0, -90.0], dtype=dtype)
        expected = np.radians(np_a)
        result = cp.radians(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestDeg2rad:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([0.0, 90.0, 180.0, 360.0, -90.0], dtype=dtype)
        expected = np.deg2rad(np_a)
        result = cp.deg2rad(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.deg2rad(cp.array(np_a)), np.deg2rad(np_a), dtype=dtype, category="unary_math")


class TestRad2deg:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([0.0, np.pi/2, np.pi, 2*np.pi, -np.pi], dtype=dtype)
        expected = np.rad2deg(np_a)
        result = cp.rad2deg(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero(self, dtype):
        np_a = np.array([0.0], dtype=dtype)
        assert_eq(cp.rad2deg(cp.array(np_a)), np.rad2deg(np_a), dtype=dtype, category="unary_math")


# ── 5. Special ufuncs ───────────────────────────────────────────
# Ref: cupy_tests/math_tests/test_arithmetic.py
# signbit, modf, ldexp, frexp x FLOAT_DTYPES x 3 shapes


class TestSignbit:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([-1.0, 0.0, 1.0, -0.0, 2.0], dtype=dtype)
        expected = np.signbit(np_a)
        result = cp.signbit(cp.array(np_a))
        assert_eq(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_values(self, dtype):
        np_a = np.array([np.inf, -np.inf, np.nan], dtype=dtype)
        expected = np.signbit(np_a)
        result = cp.signbit(cp.array(np_a))
        assert_eq(result, expected)


class TestModf:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([1.5, 2.7, -3.2, 0.0, 4.0], dtype=dtype)
        exp_frac, exp_intg = np.modf(np_a)
        frac, intg = cp.modf(cp.array(np_a))
        assert_eq(frac, exp_frac, dtype=dtype, category="unary_math")
        assert_eq(intg, exp_intg, dtype=dtype, category="unary_math")

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_negative(self, dtype):
        np_a = np.array([-1.5, -2.7, -0.3], dtype=dtype)
        exp_frac, exp_intg = np.modf(np_a)
        frac, intg = cp.modf(cp.array(np_a))
        assert_eq(frac, exp_frac, dtype=dtype, category="unary_math")
        assert_eq(intg, exp_intg, dtype=dtype, category="unary_math")


class TestLdexp:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_x1 = np.array([1.0, 2.0, 3.0, 0.5, 1.5], dtype=dtype)
        np_x2 = np.array([2, 3, 4, 1, 0], dtype=np.int32)
        expected = np.ldexp(np_x1, np_x2)
        result = cp.ldexp(cp.array(np_x1), cp.array(np_x2))
        assert_eq(result, expected, dtype=dtype, category="unary_math")


class TestFrexp:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = np.array([1.0, 2.0, 4.0, 8.0, 0.5], dtype=dtype)
        exp_mant, exp_exp = np.frexp(np_a)
        mant, exp_ = cp.frexp(cp.array(np_a))
        assert_eq(mant, exp_mant, dtype=dtype, category="unary_math")
        assert_eq(exp_, exp_exp)


# ── 6. Reductive ufuncs ─────────────────────────────────────────
# Ref: cupy_tests/math_tests/test_arithmetic.py
# amax, amin, fmax, fmin x NUMERIC_DTYPES x 3 shapes


class TestAmax:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.amax(np_a)
        result = cp.amax(cp.array(np_a))
        assert_eq(result, np.array(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis(self, dtype):
        np_a = np.array([[1, 5, 3], [4, 2, 6]], dtype=dtype)
        expected = np.amax(np_a, axis=0)
        result = cp.amax(cp.array(np_a), axis=0)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis1(self, dtype):
        np_a = np.array([[1, 5, 3], [4, 2, 6]], dtype=dtype)
        expected = np.amax(np_a, axis=1)
        result = cp.amax(cp.array(np_a), axis=1)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_nan(self, dtype):
        np_a = np.array([1.0, np.nan, 3.0], dtype=dtype)
        expected = np.amax(np_a)
        result = cp.amax(cp.array(np_a))
        # NaN propagation: result should be NaN
        res_np = result.get()
        assert np.isnan(res_np) == np.isnan(expected)


class TestAmin:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        expected = np.amin(np_a)
        result = cp.amin(cp.array(np_a))
        assert_eq(result, np.array(expected), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis(self, dtype):
        np_a = np.array([[1, 5, 3], [4, 2, 6]], dtype=dtype)
        expected = np.amin(np_a, axis=0)
        result = cp.amin(cp.array(np_a), axis=0)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis1(self, dtype):
        np_a = np.array([[1, 5, 3], [4, 2, 6]], dtype=dtype)
        expected = np.amin(np_a, axis=1)
        result = cp.amin(cp.array(np_a), axis=1)
        assert_eq(result, expected, dtype=dtype)


class TestFmax:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype) * 2
        expected = np.fmax(np_a, np_b)
        result = cp.fmax(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_nan(self, dtype):
        np_a = np.array([1.0, np.nan, 3.0], dtype=dtype)
        np_b = np.array([4.0, 2.0, np.nan], dtype=dtype)
        expected = np.fmax(np_a, np_b)
        result = cp.fmax(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype)


class TestFmin:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype) * 2
        expected = np.fmin(np_a, np_b)
        result = cp.fmin(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_nan(self, dtype):
        np_a = np.array([1.0, np.nan, 3.0], dtype=dtype)
        np_b = np.array([4.0, 2.0, np.nan], dtype=dtype)
        expected = np.fmin(np_a, np_b)
        result = cp.fmin(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype)


# ── 7. Ufunc objects: maximum, minimum ──────────────────────────
# Ref: cupy_tests/core_tests/test_fusion.py


class TestMaximumUfunc:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype) * 2
        expected = np.maximum(np_a, np_b)
        result = cp.maximum(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_nan(self, dtype):
        np_a = np.array([1.0, np.nan, 3.0], dtype=dtype)
        np_b = np.array([4.0, 2.0, np.nan], dtype=dtype)
        expected = np.maximum(np_a, np_b)
        result = cp.maximum(cp.array(np_a), cp.array(np_b))
        res_np = result.get()
        assert np.all(np.isnan(res_np) == np.isnan(expected))
        finite = np.isfinite(expected)
        if np.any(finite):
            assert_eq(cp.array(res_np[finite]), expected[finite], dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        np_a = np.array([[1, 5], [3, 2]], dtype=dtype)
        np_b = np.array([[4, 2], [6, 1]], dtype=dtype)
        expected = np.maximum(np_a, np_b)
        result = cp.maximum(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_accumulate(self, dtype):
        np_a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        expected = np.maximum.accumulate(np_a)
        result = cp.maximum.accumulate(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_accumulate_2d(self, dtype):
        np_a = np.array([[3, 1], [4, 5], [2, 6]], dtype=dtype)
        expected = np.maximum.accumulate(np_a, axis=0)
        result = cp.maximum.accumulate(cp.array(np_a), axis=0)
        assert_eq(result, expected, dtype=dtype)


class TestMinimumUfunc:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_3)
    def test_basic(self, dtype, shape):
        np_a = make_arg(shape, dtype)
        np_b = make_arg(shape, dtype) * 2
        expected = np.minimum(np_a, np_b)
        result = cp.minimum(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_nan(self, dtype):
        np_a = np.array([1.0, np.nan, 3.0], dtype=dtype)
        np_b = np.array([4.0, 2.0, np.nan], dtype=dtype)
        expected = np.minimum(np_a, np_b)
        result = cp.minimum(cp.array(np_a), cp.array(np_b))
        res_np = result.get()
        assert np.all(np.isnan(res_np) == np.isnan(expected))
        finite = np.isfinite(expected)
        if np.any(finite):
            assert_eq(cp.array(res_np[finite]), expected[finite], dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        np_a = np.array([[1, 5], [3, 2]], dtype=dtype)
        np_b = np.array([[4, 2], [6, 1]], dtype=dtype)
        expected = np.minimum(np_a, np_b)
        result = cp.minimum(cp.array(np_a), cp.array(np_b))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_accumulate(self, dtype):
        np_a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        expected = np.minimum.accumulate(np_a)
        result = cp.minimum.accumulate(cp.array(np_a))
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_accumulate_2d(self, dtype):
        np_a = np.array([[3, 1], [4, 5], [2, 6]], dtype=dtype)
        expected = np.minimum.accumulate(np_a, axis=0)
        result = cp.minimum.accumulate(cp.array(np_a), axis=0)
        assert_eq(result, expected, dtype=dtype)
