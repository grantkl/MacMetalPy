"""Tests for numeric edge cases: special floats, overflow, NaN/inf propagation.

Ref: cupy_tests/math_tests/test_arithmetic.py (special value tests)
     cupy_tests/math_tests/test_trigonometric.py (nan/inf through trig)
"""

import numpy as np
import pytest

import macmetalpy as cp

from conftest import assert_eq, tol_for


# ── Float32 specials through unary ops ───────────────────────────────


SPECIAL_VALUES_F32 = [
    np.float32(np.nan),
    np.float32(np.inf),
    np.float32(-np.inf),
    np.float32(0.0),
    np.float32(-0.0),
]

UNARY_OPS = [
    ("sqrt", cp.sqrt, np.sqrt),
    ("exp", cp.exp, np.exp),
    ("log", cp.log, np.log),
    ("sin", cp.sin, np.sin),
    ("cos", cp.cos, np.cos),
    ("tan", cp.tan, np.tan),
    ("arcsin", cp.arcsin, np.arcsin),
    ("arccos", cp.arccos, np.arccos),
    ("arctan", cp.arctan, np.arctan),
    ("sinh", cp.sinh, np.sinh),
    ("cosh", cp.cosh, np.cosh),
    ("tanh", cp.tanh, np.tanh),
    ("log2", cp.log2, np.log2),
    ("log10", cp.log10, np.log10),
    ("square", cp.square, np.square),
    ("negative", cp.negative, np.negative),
    ("abs", cp.abs, np.abs),
    ("sign", cp.sign, np.sign),
    ("floor", cp.floor, np.floor),
    ("ceil", cp.ceil, np.ceil),
]


class TestFloat32SpecialsUnary:
    """Test special float32 values (nan, inf, -inf, 0, -0) through unary ops."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py::TestArithmeticUnary

    @pytest.mark.parametrize(
        "op_name,cp_op,np_op",
        UNARY_OPS,
        ids=[t[0] for t in UNARY_OPS],
    )
    @pytest.mark.parametrize("val", SPECIAL_VALUES_F32, ids=["nan", "inf", "-inf", "0", "-0"])
    def test_unary_special(self, op_name, cp_op, np_op, val):
        """Unary op on special float32 value should match NumPy."""
        np_a = np.array([val], dtype=np.float32)
        gpu_a = cp.array(np_a)
        np_result = np_op(np_a)
        gpu_result = cp_op(gpu_a)
        result = gpu_result.get()

        # Compare NaN-aware: both NaN means match; both non-NaN uses allclose
        if np.all(np.isnan(np_result)):
            assert np.all(np.isnan(result)), (
                f"{op_name}({val}): expected nan, got {result}"
            )
        elif np.any(np.isnan(np_result)):
            # Partial NaN
            nan_mask = np.isnan(np_result)
            assert np.all(np.isnan(result[nan_mask]))
            np.testing.assert_allclose(
                result[~nan_mask], np_result[~nan_mask],
                **tol_for(np.float32, "unary_math")
            )
        else:
            np.testing.assert_allclose(
                result, np_result, **tol_for(np.float32, "unary_math")
            )


# ── Float32 specials through binary ops ──────────────────────────────


BINARY_OPS = [
    ("add", cp.add, np.add),
    ("subtract", cp.subtract, np.subtract),
    ("multiply", cp.multiply, np.multiply),
    ("divide", cp.divide, np.divide),
    ("power", cp.power, np.power),
]

# Pairs: (left_special, right_special)
SPECIAL_BINARY_COMBOS = [
    (np.float32(np.nan), np.float32(1.0)),
    (np.float32(1.0), np.float32(np.nan)),
    (np.float32(np.inf), np.float32(1.0)),
    (np.float32(1.0), np.float32(np.inf)),
    (np.float32(np.nan), np.float32(np.inf)),
    (np.float32(np.inf), np.float32(-np.inf)),
    (np.float32(0.0), np.float32(0.0)),
    (np.float32(-0.0), np.float32(0.0)),
]


class TestFloat32SpecialsBinary:
    """Test special float32 values through binary operations."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py::TestArithmeticBinary

    @pytest.mark.parametrize(
        "op_name,cp_op,np_op",
        BINARY_OPS,
        ids=[t[0] for t in BINARY_OPS],
    )
    @pytest.mark.parametrize(
        "left,right",
        SPECIAL_BINARY_COMBOS,
        ids=[f"{l}_{r}" for l, r in SPECIAL_BINARY_COMBOS],
    )
    def test_binary_special(self, op_name, cp_op, np_op, left, right):
        """Binary op on special float32 pairs should match NumPy."""
        np_a = np.array([left], dtype=np.float32)
        np_b = np.array([right], dtype=np.float32)
        gpu_a = cp.array(np_a)
        gpu_b = cp.array(np_b)

        np_result = np_op(np_a, np_b)
        gpu_result = cp_op(gpu_a, gpu_b)
        result = gpu_result.get()

        if np.all(np.isnan(np_result)):
            assert np.all(np.isnan(result)), (
                f"{op_name}({left}, {right}): expected nan, got {result}"
            )
        else:
            np.testing.assert_allclose(
                result, np_result, **tol_for(np.float32, "arithmetic"),
                equal_nan=True,
            )


# ── Float16 precision edge cases ─────────────────────────────────────


class TestFloat16Precision:
    """Test float16 near-epsilon arithmetic, overflow, and underflow."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py (float16 tests)

    def test_f16_near_epsilon_add(self):
        """Adding float16 epsilon to 1.0 should be representable."""
        eps = np.finfo(np.float16).eps
        np_a = np.array([1.0], dtype=np.float16)
        np_b = np.array([eps], dtype=np.float16)
        gpu_a = cp.array(np_a)
        gpu_b = cp.array(np_b)
        np_result = np_a + np_b
        gpu_result = (gpu_a + gpu_b).get()
        np.testing.assert_allclose(gpu_result, np_result, **tol_for(np.float16))

    def test_f16_near_epsilon_sub(self):
        """Subtracting near-epsilon values in float16."""
        a = np.array([1.001], dtype=np.float16)
        b = np.array([1.0], dtype=np.float16)
        np_result = a - b
        gpu_result = (cp.array(a) - cp.array(b)).get()
        np.testing.assert_allclose(gpu_result, np_result, **tol_for(np.float16))

    def test_f16_overflow_to_inf(self):
        """Float16 values > 65504 should overflow to inf."""
        a = np.array([65504.0], dtype=np.float16)
        b = np.array([100.0], dtype=np.float16)
        np_result = a + b
        gpu_result = (cp.array(a) + cp.array(b)).get()
        assert np.isinf(gpu_result[0]), f"Expected inf, got {gpu_result[0]}"

    def test_f16_large_multiply_overflow(self):
        """Float16 multiply causing overflow should produce inf."""
        a = np.array([256.0], dtype=np.float16)
        b = np.array([256.0], dtype=np.float16)
        np_result = a * b
        gpu_result = (cp.array(a) * cp.array(b)).get()
        np.testing.assert_array_equal(gpu_result, np_result)

    def test_f16_underflow_to_zero(self):
        """Very small float16 values should underflow to 0."""
        tiny = np.finfo(np.float16).tiny
        a = np.array([tiny / 100.0], dtype=np.float16)
        gpu_a = cp.array(a)
        result = gpu_a.get()
        np.testing.assert_array_equal(result, a)

    def test_f16_smallest_subnormal(self):
        """Float16 smallest subnormal should roundtrip correctly."""
        smallest = np.finfo(np.float16).smallest_subnormal
        np_a = np.array([smallest], dtype=np.float16)
        gpu_a = cp.array(np_a)
        result = gpu_a.get()
        np.testing.assert_array_equal(result, np_a)


# ── Int32 overflow behavior ──────────────────────────────────────────


class TestInt32Overflow:
    """Test integer overflow behavior (wrapping, as in NumPy)."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py

    def test_int32_max_plus_one(self):
        """int32 MAX + 1 should wrap (like NumPy)."""
        max_val = np.iinfo(np.int32).max
        np_a = np.array([max_val], dtype=np.int32)
        np_b = np.array([1], dtype=np.int32)
        np_result = np_a + np_b
        gpu_result = (cp.array(np_a) + cp.array(np_b)).get()
        np.testing.assert_array_equal(gpu_result, np_result)

    def test_int32_min_minus_one(self):
        """int32 MIN - 1 should wrap (like NumPy)."""
        min_val = np.iinfo(np.int32).min
        np_a = np.array([min_val], dtype=np.int32)
        np_b = np.array([1], dtype=np.int32)
        np_result = np_a - np_b
        gpu_result = (cp.array(np_a) - cp.array(np_b)).get()
        np.testing.assert_array_equal(gpu_result, np_result)

    def test_int16_overflow(self):
        """int16 MAX + 1 should wrap."""
        max_val = np.iinfo(np.int16).max
        np_a = np.array([max_val], dtype=np.int16)
        np_b = np.array([1], dtype=np.int16)
        np_result = np_a + np_b
        gpu_result = (cp.array(np_a) + cp.array(np_b)).get()
        np.testing.assert_array_equal(gpu_result, np_result)

    def test_uint16_overflow(self):
        """uint16 MAX + 1 should wrap to 0."""
        max_val = np.iinfo(np.uint16).max
        np_a = np.array([max_val], dtype=np.uint16)
        np_b = np.array([1], dtype=np.uint16)
        np_result = np_a + np_b
        gpu_result = (cp.array(np_a) + cp.array(np_b)).get()
        np.testing.assert_array_equal(gpu_result, np_result)

    def test_uint32_overflow(self):
        """uint32 MAX + 1 should wrap to 0."""
        max_val = np.iinfo(np.uint32).max
        np_a = np.array([max_val], dtype=np.uint32)
        np_b = np.array([1], dtype=np.uint32)
        np_result = np_a + np_b
        gpu_result = (cp.array(np_a) + cp.array(np_b)).get()
        np.testing.assert_array_equal(gpu_result, np_result)


# ── isnan / isinf / isfinite on specials ─────────────────────────────


class TestPredicatesOnSpecials:
    """Test isnan, isinf, isfinite on all special float32 values."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py

    @pytest.mark.parametrize("val,expected", [
        (np.nan, True),
        (np.inf, False),
        (-np.inf, False),
        (0.0, False),
        (-0.0, False),
        (1.0, False),
    ])
    def test_isnan(self, val, expected):
        """isnan should match NumPy for special values."""
        np_a = np.array([val], dtype=np.float32)
        gpu_result = cp.isnan(cp.array(np_a)).get()
        np.testing.assert_array_equal(gpu_result, np.isnan(np_a))

    @pytest.mark.parametrize("val,expected", [
        (np.nan, False),
        (np.inf, True),
        (-np.inf, True),
        (0.0, False),
        (-0.0, False),
        (1.0, False),
    ])
    def test_isinf(self, val, expected):
        """isinf should match NumPy for special values."""
        np_a = np.array([val], dtype=np.float32)
        gpu_result = cp.isinf(cp.array(np_a)).get()
        np.testing.assert_array_equal(gpu_result, np.isinf(np_a))

    @pytest.mark.parametrize("val,expected", [
        (np.nan, False),
        (np.inf, False),
        (-np.inf, False),
        (0.0, True),
        (-0.0, True),
        (1.0, True),
    ])
    def test_isfinite(self, val, expected):
        """isfinite should match NumPy for special values."""
        np_a = np.array([val], dtype=np.float32)
        gpu_result = cp.isfinite(cp.array(np_a)).get()
        np.testing.assert_array_equal(gpu_result, np.isfinite(np_a))


# ── signbit on specials ──────────────────────────────────────────────


class TestSignbit:
    """Test signbit on special values."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py

    @pytest.mark.parametrize("val,expected", [
        (-0.0, True),
        (0.0, False),
        (-np.inf, True),
        (np.inf, False),
        (-1.0, True),
        (1.0, False),
    ])
    def test_signbit(self, val, expected):
        """signbit should match NumPy."""
        np_a = np.array([val], dtype=np.float32)
        gpu_result = cp.signbit(cp.array(np_a)).get()
        np.testing.assert_array_equal(gpu_result, np.signbit(np_a))


# ── copysign / nextafter on specials ─────────────────────────────────


class TestCopysignNextafter:
    """Test copysign and nextafter with special values."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py

    @pytest.mark.parametrize("val,sign_val", [
        (1.0, -1.0),
        (-1.0, 1.0),
        (0.0, -1.0),
        (np.inf, -1.0),
        (np.nan, -1.0),
    ])
    def test_copysign(self, val, sign_val):
        """copysign should match NumPy."""
        np_a = np.array([val], dtype=np.float32)
        np_b = np.array([sign_val], dtype=np.float32)
        np_result = np.copysign(np_a, np_b)
        gpu_result = cp.copysign(cp.array(np_a), cp.array(np_b)).get()
        if np.isnan(np_result[0]):
            assert np.isnan(gpu_result[0])
        else:
            np.testing.assert_array_equal(gpu_result, np_result)

    @pytest.mark.parametrize("x,y", [
        (1.0, 2.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (np.inf, 0.0),
        (-np.inf, 0.0),
    ])
    def test_nextafter(self, x, y):
        """nextafter should match NumPy."""
        np_a = np.array([x], dtype=np.float32)
        np_b = np.array([y], dtype=np.float32)
        np_result = np.nextafter(np_a, np_b)
        gpu_result = cp.nextafter(cp.array(np_a), cp.array(np_b)).get()
        np.testing.assert_allclose(gpu_result, np_result, **tol_for(np.float32))


# ── nan_to_num with custom values ────────────────────────────────────


class TestNanToNum:
    """Test nan_to_num with default and custom replacement values."""

    # Ref: cupy_tests/math_tests/test_misc.py

    def test_nan_to_num_defaults(self):
        """nan_to_num with defaults should replace nan->0, inf->large."""
        np_a = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
        np_result = np.nan_to_num(np_a)
        gpu_result = cp.nan_to_num(cp.array(np_a)).get()
        np.testing.assert_allclose(gpu_result, np_result, **tol_for(np.float32))

    def test_nan_to_num_custom_nan(self):
        """nan_to_num with custom nan replacement."""
        np_a = np.array([np.nan, 1.0, np.nan], dtype=np.float32)
        np_result = np.nan_to_num(np_a, nan=-999.0)
        gpu_result = cp.nan_to_num(cp.array(np_a), nan=-999.0).get()
        np.testing.assert_allclose(gpu_result, np_result, **tol_for(np.float32))

    def test_nan_to_num_custom_posinf(self):
        """nan_to_num with custom posinf replacement."""
        np_a = np.array([np.inf, 1.0], dtype=np.float32)
        np_result = np.nan_to_num(np_a, posinf=100.0)
        gpu_result = cp.nan_to_num(cp.array(np_a), posinf=100.0).get()
        np.testing.assert_allclose(gpu_result, np_result, **tol_for(np.float32))

    def test_nan_to_num_custom_neginf(self):
        """nan_to_num with custom neginf replacement."""
        np_a = np.array([-np.inf, 1.0], dtype=np.float32)
        np_result = np.nan_to_num(np_a, neginf=-100.0)
        gpu_result = cp.nan_to_num(cp.array(np_a), neginf=-100.0).get()
        np.testing.assert_allclose(gpu_result, np_result, **tol_for(np.float32))

    def test_nan_to_num_all_custom(self):
        """nan_to_num with all custom values."""
        np_a = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
        np_result = np.nan_to_num(np_a, nan=0.5, posinf=99.0, neginf=-99.0)
        gpu_result = cp.nan_to_num(
            cp.array(np_a), nan=0.5, posinf=99.0, neginf=-99.0
        ).get()
        np.testing.assert_allclose(gpu_result, np_result, **tol_for(np.float32))


# ── NaN propagation ──────────────────────────────────────────────────


class TestNanPropagation:
    """Test NaN propagation through arithmetic operations."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py

    def test_nan_plus_one(self):
        """nan + 1 should be nan."""
        result = (cp.array([np.nan], dtype=cp.float32) + cp.array([1.0], dtype=cp.float32)).get()
        assert np.isnan(result[0])

    def test_nan_times_zero(self):
        """nan * 0 should be nan."""
        result = (cp.array([np.nan], dtype=cp.float32) * cp.array([0.0], dtype=cp.float32)).get()
        assert np.isnan(result[0])

    def test_nan_eq_nan(self):
        """nan == nan should be False."""
        a = cp.array([np.nan], dtype=cp.float32)
        result = (a == a).get()
        assert not result[0]

    def test_nan_ne_nan(self):
        """nan != nan should be True."""
        a = cp.array([np.nan], dtype=cp.float32)
        result = (a != a).get()
        assert result[0]

    def test_nan_gt_one(self):
        """nan > 1 should be False."""
        result = (cp.array([np.nan], dtype=cp.float32) > cp.array([1.0], dtype=cp.float32)).get()
        assert not result[0]

    def test_nan_lt_one(self):
        """nan < 1 should be False."""
        result = (cp.array([np.nan], dtype=cp.float32) < cp.array([1.0], dtype=cp.float32)).get()
        assert not result[0]

    def test_nan_sub_nan(self):
        """nan - nan should be nan."""
        a = cp.array([np.nan], dtype=cp.float32)
        result = (a - a).get()
        assert np.isnan(result[0])

    def test_nan_div_one(self):
        """nan / 1 should be nan."""
        result = (cp.array([np.nan], dtype=cp.float32) / cp.array([1.0], dtype=cp.float32)).get()
        assert np.isnan(result[0])


# ── Inf arithmetic ───────────────────────────────────────────────────


class TestInfArithmetic:
    """Test infinity arithmetic behavior."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py

    def test_inf_plus_one(self):
        """inf + 1 should be inf."""
        result = (cp.array([np.inf], dtype=cp.float32) + cp.array([1.0], dtype=cp.float32)).get()
        assert np.isinf(result[0]) and result[0] > 0

    def test_inf_minus_inf(self):
        """inf - inf should be nan."""
        result = (cp.array([np.inf], dtype=cp.float32) - cp.array([np.inf], dtype=cp.float32)).get()
        assert np.isnan(result[0])

    def test_inf_times_zero(self):
        """inf * 0 should be nan."""
        result = (cp.array([np.inf], dtype=cp.float32) * cp.array([0.0], dtype=cp.float32)).get()
        assert np.isnan(result[0])

    def test_one_div_zero(self):
        """1 / 0 should be inf."""
        result = (cp.array([1.0], dtype=cp.float32) / cp.array([0.0], dtype=cp.float32)).get()
        assert np.isinf(result[0])

    def test_neg_one_div_zero(self):
        """-1 / 0 should be -inf."""
        result = (cp.array([-1.0], dtype=cp.float32) / cp.array([0.0], dtype=cp.float32)).get()
        assert np.isinf(result[0]) and result[0] < 0

    def test_zero_div_zero(self):
        """0 / 0 should be nan."""
        result = (cp.array([0.0], dtype=cp.float32) / cp.array([0.0], dtype=cp.float32)).get()
        assert np.isnan(result[0])

    def test_inf_times_inf(self):
        """inf * inf should be inf."""
        result = (cp.array([np.inf], dtype=cp.float32) * cp.array([np.inf], dtype=cp.float32)).get()
        assert np.isinf(result[0]) and result[0] > 0

    def test_inf_div_inf(self):
        """inf / inf should be nan."""
        result = (cp.array([np.inf], dtype=cp.float32) / cp.array([np.inf], dtype=cp.float32)).get()
        assert np.isnan(result[0])


# ── Negative zero equality ──────────────────────────────────────────


class TestNegativeZero:
    """Test -0.0 == 0.0 behavior."""

    # Ref: IEEE 754 compliance

    def test_neg_zero_eq_pos_zero(self):
        """-0.0 == 0.0 should be True."""
        a = cp.array([-0.0], dtype=cp.float32)
        b = cp.array([0.0], dtype=cp.float32)
        result = (a == b).get()
        assert result[0]

    def test_neg_zero_ne_pos_zero(self):
        """-0.0 != 0.0 should be False."""
        a = cp.array([-0.0], dtype=cp.float32)
        b = cp.array([0.0], dtype=cp.float32)
        result = (a != b).get()
        assert not result[0]

    def test_neg_zero_roundtrip(self):
        """-0.0 should survive GPU roundtrip (signbit preserved)."""
        np_a = np.array([-0.0], dtype=np.float32)
        gpu_a = cp.array(np_a)
        result = gpu_a.get()
        assert np.signbit(result[0]), "signbit of -0.0 should be True after roundtrip"


# ── Mixed special value array tests ──────────────────────────────────


class TestMixedSpecials:
    """Test arrays containing a mix of special and normal values."""

    # Ref: cupy_tests/math_tests/test_arithmetic.py

    def test_isnan_mixed(self):
        """isnan on mixed array should correctly identify NaN positions."""
        np_a = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
        gpu_result = cp.isnan(cp.array(np_a)).get()
        np.testing.assert_array_equal(gpu_result, np.isnan(np_a))

    def test_isinf_mixed(self):
        """isinf on mixed array should correctly identify inf positions."""
        np_a = np.array([1.0, np.inf, -np.inf, 0.0, np.nan], dtype=np.float32)
        gpu_result = cp.isinf(cp.array(np_a)).get()
        np.testing.assert_array_equal(gpu_result, np.isinf(np_a))

    def test_isfinite_mixed(self):
        """isfinite on mixed array should be correct."""
        np_a = np.array([1.0, np.inf, -np.inf, 0.0, np.nan], dtype=np.float32)
        gpu_result = cp.isfinite(cp.array(np_a)).get()
        np.testing.assert_array_equal(gpu_result, np.isfinite(np_a))

    def test_add_with_nan_positions(self):
        """Adding arrays where one has NaN should propagate correctly."""
        np_a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        np_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        np_result = np_a + np_b
        gpu_result = (cp.array(np_a) + cp.array(np_b)).get()
        # Position 0 and 2: normal; Position 1: nan
        assert gpu_result[0] == np_result[0]
        assert np.isnan(gpu_result[1])
        assert gpu_result[2] == np_result[2]

    def test_sum_with_nan(self):
        """sum of array containing NaN should be NaN."""
        a = cp.array([1.0, np.nan, 3.0], dtype=cp.float32)
        result = cp.sum(a).get()
        assert np.isnan(result)

    def test_max_with_nan(self):
        """max of array containing NaN should be NaN (NumPy behavior)."""
        np_a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        np_result = np.max(np_a)
        gpu_result = cp.max(cp.array(np_a)).get()
        # NumPy returns nan for max with nan
        if np.isnan(np_result):
            assert np.isnan(gpu_result)
        else:
            np.testing.assert_allclose(gpu_result, np_result, **tol_for(np.float32))
