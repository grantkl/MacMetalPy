"""Tests for reduction operations — CuPy-parity parametrized suite.

Ref: cupy_tests/math_tests/test_sumprod.py, test_misc.py
~1,356 parametrized cases covering: sum, mean, max, min, any, all,
argmax, argmin, std, var, prod, median, percentile, cumsum, cumprod,
diff, ptp, quantile, average.
"""

import numpy as np
import pytest

import macmetalpy as cp
from conftest import (
    NUMERIC_DTYPES, FLOAT_DTYPES, INT_DTYPES,
    assert_eq, make_arg, tol_for,
)

# ── Shape groups for reductions ─────────────────────────────────────
REDUCE_SHAPES = [(5,), (2, 3), (2, 3, 4)]
REDUCE_SHAPES_SMALL = [(5,), (2, 3)]


# ====================================================================
# sum / mean
# ====================================================================
# Ref: cupy_tests/math_tests/test_sumprod.py::TestSumprod

class TestSum:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.sum(a_cp), np.sum(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.sum(a_cp, axis=0), np.sum(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.sum(a_cp, axis=1), np.sum(a_np, axis=1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_neg1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.sum(a_cp, axis=-1), np.sum(a_np, axis=-1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.sum(a_cp, axis=0, keepdims=True)
        expected = np.sum(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis_neg2_3d(self, dtype):
        a_np = make_arg((2, 3, 4), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.sum(a_cp, axis=-2), np.sum(a_np, axis=-2), dtype=dtype)


class TestMean:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.mean(a_cp), np.mean(a_np.astype(np.float64)).astype(np.float32),
                  dtype=np.float32)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.mean(a_cp, axis=0)
        expected = np.mean(a_np, axis=0)
        assert_eq(result, expected, dtype=np.float32)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.mean(a_cp, axis=1), np.mean(a_np, axis=1), dtype=np.float32)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_neg1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.mean(a_cp, axis=-1), np.mean(a_np, axis=-1), dtype=np.float32)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.mean(a_cp, axis=0, keepdims=True)
        expected = np.mean(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)


# ====================================================================
# max / min
# ====================================================================
# Ref: cupy_tests/math_tests/test_sumprod.py::TestCumprod

class TestMax:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.max(a_cp), np.max(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.max(a_cp, axis=0), np.max(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.max(a_cp, axis=1), np.max(a_np, axis=1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_neg1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.max(a_cp, axis=-1), np.max(a_np, axis=-1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.max(a_cp, axis=0, keepdims=True)
        expected = np.max(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)


class TestMin:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.min(a_cp), np.min(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.min(a_cp, axis=0), np.min(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.min(a_cp, axis=1), np.min(a_np, axis=1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_neg1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.min(a_cp, axis=-1), np.min(a_np, axis=-1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.min(a_cp, axis=0, keepdims=True)
        expected = np.min(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)


# ====================================================================
# std / var
# ====================================================================
# Ref: cupy_tests/math_tests/test_sumprod.py

class TestStd:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.std(a_cp), np.std(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.std(a_cp, axis=0), np.std(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.std(a_cp, axis=0, keepdims=True)
        expected = np.std(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)


class TestVar:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.var(a_cp), np.var(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.var(a_cp, axis=0), np.var(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_keepdims(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.var(a_cp, axis=0, keepdims=True)
        expected = np.var(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)


# ====================================================================
# prod
# ====================================================================
# Ref: cupy_tests/math_tests/test_sumprod.py::TestCumprod

class TestProd:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES_SMALL)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.prod(a_cp), np.prod(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.prod(a_cp, axis=0), np.prod(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_keepdims(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.prod(a_cp, axis=0, keepdims=True)
        expected = np.prod(a_np, axis=0, keepdims=True)
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_overflow_small_values(self, dtype):
        """Use small values to avoid overflow."""
        a_np = np.array([1.0, 1.5, 2.0], dtype=dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.prod(a_cp), np.prod(a_np), dtype=dtype)


# ====================================================================
# argmax / argmin
# ====================================================================
# Ref: cupy_tests/math_tests/test_sumprod.py

class TestArgmax:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES_SMALL)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.argmax(a_cp)
        expected = np.argmax(a_np)
        assert int(result) == int(expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.argmax(a_cp, axis=0)
        expected = np.argmax(a_np, axis=0)
        assert_eq(result, expected, dtype=np.int32)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_axis1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.argmax(a_cp, axis=1)
        expected = np.argmax(a_np, axis=1)
        assert_eq(result, expected, dtype=np.int32)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_ties(self, dtype):
        """When elements are tied, return the first index (like NumPy)."""
        a_np = np.array([3, 3, 1, 2], dtype=dtype)
        a_cp = cp.array(a_np)
        assert int(cp.argmax(a_cp)) == int(np.argmax(a_np))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_equal(self, dtype):
        a_np = np.array([5, 5, 5], dtype=dtype)
        a_cp = cp.array(a_np)
        assert int(cp.argmax(a_cp)) == int(np.argmax(a_np))


class TestArgmin:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES_SMALL)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.argmin(a_cp)
        expected = np.argmin(a_np)
        assert int(result) == int(expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        result = cp.argmin(a_cp, axis=0)
        expected = np.argmin(a_np, axis=0)
        assert_eq(result, expected, dtype=np.int32)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_ties(self, dtype):
        a_np = np.array([1, 1, 3, 2], dtype=dtype)
        a_cp = cp.array(a_np)
        assert int(cp.argmin(a_cp)) == int(np.argmin(a_np))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_equal(self, dtype):
        a_np = np.array([5, 5, 5], dtype=dtype)
        a_cp = cp.array(a_np)
        assert int(cp.argmin(a_cp)) == int(np.argmin(a_np))


# ====================================================================
# any / all
# ====================================================================
# Ref: cupy_tests/logic_tests/test_truth.py

class TestAny:
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_all_true(self, shape):
        a_np = np.ones(shape, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.any(a_cp), np.any(a_np))

    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_all_false(self, shape):
        a_np = np.zeros(shape, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.any(a_cp), np.any(a_np))

    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_mixed(self, shape):
        a_np = make_arg(shape, np.float32)
        a_np_mod = a_np.copy()
        a_np_mod.flat[0] = 0
        a_cp = cp.array(a_np_mod)
        assert_eq(cp.any(a_cp), np.any(a_np_mod))

    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, shape):
        a_np = np.zeros(shape, dtype=np.float32)
        a_np[0] = 1
        a_cp = cp.array(a_np)
        assert_eq(cp.any(a_cp, axis=0), np.any(a_np, axis=0))

    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis_neg1(self, shape):
        a_np = make_arg(shape, np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.any(a_cp, axis=-1), np.any(a_np, axis=-1))

    def test_bool_input(self):
        a_np = np.array([True, False, True])
        a_cp = cp.array(a_np)
        assert_eq(cp.any(a_cp), np.any(a_np))

    def test_int_input(self):
        a_np = np.array([0, 0, 1, 0], dtype=np.int32)
        a_cp = cp.array(a_np)
        assert_eq(cp.any(a_cp), np.any(a_np))


class TestAll:
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_all_true(self, shape):
        a_np = np.ones(shape, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.all(a_cp), np.all(a_np))

    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_all_false(self, shape):
        a_np = np.zeros(shape, dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.all(a_cp), np.all(a_np))

    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_mixed(self, shape):
        a_np = make_arg(shape, np.float32)
        a_np_mod = a_np.copy()
        a_np_mod.flat[0] = 0
        a_cp = cp.array(a_np_mod)
        assert_eq(cp.all(a_cp), np.all(a_np_mod))

    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, shape):
        a_np = np.ones(shape, dtype=np.float32)
        a_np[0, 0] = 0
        a_cp = cp.array(a_np)
        assert_eq(cp.all(a_cp, axis=0), np.all(a_np, axis=0))

    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis_neg1(self, shape):
        a_np = make_arg(shape, np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.all(a_cp, axis=-1), np.all(a_np, axis=-1))

    def test_bool_input(self):
        a_np = np.array([True, True, True])
        a_cp = cp.array(a_np)
        assert_eq(cp.all(a_cp), np.all(a_np))

    def test_int_input_has_zero(self):
        a_np = np.array([1, 2, 0, 3], dtype=np.int32)
        a_cp = cp.array(a_np)
        assert_eq(cp.all(a_cp), np.all(a_np))


# ====================================================================
# cumsum / cumprod
# ====================================================================
# Ref: cupy_tests/math_tests/test_sumprod.py::TestCumsum

class TestCumsum:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES_SMALL)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.cumsum(a_cp), np.cumsum(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.cumsum(a_cp, axis=0), np.cumsum(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_axis_neg1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.cumsum(a_cp, axis=-1), np.cumsum(a_np, axis=-1), dtype=dtype)


class TestCumprod:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES_SMALL)
    def test_axis_none(self, dtype, shape):
        # Use small values to avoid overflow
        a_np = np.ones(shape, dtype=dtype)
        a_np.flat[:min(a_np.size, 3)] = [1, 2, 1][:min(a_np.size, 3)]
        a_cp = cp.array(a_np)
        assert_eq(cp.cumprod(a_cp), np.cumprod(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.cumprod(a_cp, axis=0), np.cumprod(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3)])
    def test_axis_neg1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.cumprod(a_cp, axis=-1), np.cumprod(a_np, axis=-1), dtype=dtype)


# ====================================================================
# diff
# ====================================================================
# Ref: cupy_tests/math_tests/test_misc.py::TestDiff

class TestDiff:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_n1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.diff(a_cp, n=1), np.diff(a_np, n=1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(5,), (2, 3)])
    def test_n2(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.diff(a_cp, n=2), np.diff(a_np, n=2), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.diff(a_cp, axis=0), np.diff(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis_neg1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.diff(a_cp, axis=-1), np.diff(a_np, axis=-1), dtype=dtype)


# ====================================================================
# median
# ====================================================================
# Ref: cupy_tests/math_tests/test_misc.py::TestMedian

class TestMedian:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_odd_count(self, dtype):
        a_np = np.array([3, 1, 2, 5, 4], dtype=dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.median(a_cp), np.median(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_even_count(self, dtype):
        a_np = np.array([3, 1, 2, 4], dtype=dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.median(a_cp), np.median(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(5,), (2, 3)])
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.median(a_cp), np.median(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.median(a_cp, axis=0), np.median(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis_neg1(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.median(a_cp, axis=-1), np.median(a_np, axis=-1), dtype=dtype)


# ====================================================================
# percentile / quantile
# ====================================================================
# Ref: cupy_tests/math_tests/test_misc.py

class TestPercentile:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("q", [0, 25, 50, 75, 100])
    @pytest.mark.parametrize("shape", [(5,), (2, 3)])
    def test_basic(self, dtype, q, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.percentile(a_cp, q), np.percentile(a_np, q), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("q", [0, 50, 100])
    def test_axis0(self, dtype, q):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.percentile(a_cp, q, axis=0),
                  np.percentile(a_np, q, axis=0), dtype=dtype)


class TestQuantile:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("shape", [(5,), (2, 3)])
    def test_basic(self, dtype, q, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.quantile(a_cp, q), np.quantile(a_np, q), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
    def test_axis0(self, dtype, q):
        a_np = make_arg((2, 3), dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.quantile(a_cp, q, axis=0),
                  np.quantile(a_np, q, axis=0), dtype=dtype)


# ====================================================================
# ptp
# ====================================================================
# Ref: cupy_tests/math_tests/test_misc.py

class TestPtp:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_axis_none(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.ptp(a_cp), np.ptp(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.ptp(a_cp, axis=0), np.ptp(a_np, axis=0), dtype=dtype)


# ====================================================================
# average
# ====================================================================
# Ref: cupy_tests/math_tests/test_misc.py

class TestAverage:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", REDUCE_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.average(a_cp), np.average(a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis0(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.average(a_cp, axis=0), np.average(a_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_weights_1d(self, dtype):
        a_np = np.array([1, 2, 3, 4], dtype=dtype)
        w_np = np.array([4, 3, 2, 1], dtype=dtype)
        a_cp = cp.array(a_np)
        w_cp = cp.array(w_np)
        assert_eq(cp.average(a_cp, weights=w_cp),
                  np.average(a_np, weights=w_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_weights_axis(self, dtype):
        a_np = make_arg((2, 3), dtype)
        w_np = np.array([0.5, 0.3, 0.2], dtype=dtype)
        a_cp = cp.array(a_np)
        w_cp = cp.array(w_np)
        assert_eq(cp.average(a_cp, axis=1, weights=w_cp),
                  np.average(a_np, axis=1, weights=w_np), dtype=dtype)
