"""Tests for statistics, cumulative, and NaN/comparison functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp
from macmetalpy.reductions import (
    std,
    var,
    prod,
    median,
    percentile,
    cumsum,
    cumprod,
    diff,
)
from macmetalpy.math_ops import (
    isnan,
    isinf,
    isfinite,
    nan_to_num,
    isclose,
    allclose,
    array_equal,
    count_nonzero,
)


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def arr_1d():
    np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    return cp.array(np_arr), np_arr


@pytest.fixture
def arr_2d():
    np_arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    return cp.array(np_arr), np_arr


@pytest.fixture
def arr_nan():
    np_arr = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 0.0], dtype=np.float32)
    return cp.array(np_arr), np_arr


# ------------------------------------------------------------------ std


class TestStd:
    def test_std_1d_full(self, arr_1d):
        gpu, np_arr = arr_1d
        result = gpu.std()
        expected = np.std(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_std_2d_full(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.std()
        expected = np.std(np_arr)
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_std_axis0(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.std(axis=0)
        expected = np.std(np_arr, axis=0)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_std_axis1(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.std(axis=1)
        expected = np.std(np_arr, axis=1)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_std_keepdims(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.std(keepdims=True)
        expected = np.std(np_arr, keepdims=True)
        assert result.get().shape == expected.shape
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_std_module_level(self, arr_1d):
        gpu, np_arr = arr_1d
        result = std(gpu)
        expected = np.std(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ------------------------------------------------------------------ var


class TestVar:
    def test_var_1d_full(self, arr_1d):
        gpu, np_arr = arr_1d
        result = gpu.var()
        expected = np.var(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_var_2d_full(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.var()
        expected = np.var(np_arr)
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_var_axis0(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.var(axis=0)
        expected = np.var(np_arr, axis=0)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_var_axis1(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.var(axis=1)
        expected = np.var(np_arr, axis=1)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_var_keepdims(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.var(keepdims=True)
        expected = np.var(np_arr, keepdims=True)
        assert result.get().shape == expected.shape
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_var_module_level(self, arr_1d):
        gpu, np_arr = arr_1d
        result = var(gpu)
        expected = np.var(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ------------------------------------------------------------------ prod


class TestProd:
    def test_prod_1d(self, arr_1d):
        gpu, np_arr = arr_1d
        result = gpu.prod()
        expected = np.prod(np_arr)
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_prod_2d_full(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.prod()
        expected = np.prod(np_arr)
        assert_allclose(float(result.get()), float(expected), rtol=1e-3)

    def test_prod_axis0(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.prod(axis=0)
        expected = np.prod(np_arr, axis=0)
        assert_allclose(result.get(), expected, rtol=1e-3)

    def test_prod_axis1(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.prod(axis=1)
        expected = np.prod(np_arr, axis=1)
        assert_allclose(result.get(), expected, rtol=1e-3)

    def test_prod_keepdims(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.prod(keepdims=True)
        expected = np.prod(np_arr, keepdims=True)
        assert result.get().shape == expected.shape

    def test_prod_module_level(self, arr_1d):
        gpu, np_arr = arr_1d
        result = prod(gpu)
        expected = np.prod(np_arr)
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)


# ------------------------------------------------------------------ median


class TestMedian:
    def test_median_1d(self, arr_1d):
        gpu, np_arr = arr_1d
        result = median(gpu)
        expected = np.median(np_arr)
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_median_2d_full(self, arr_2d):
        gpu, np_arr = arr_2d
        result = median(gpu)
        expected = np.median(np_arr)
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_median_axis0(self, arr_2d):
        gpu, np_arr = arr_2d
        result = median(gpu, axis=0)
        expected = np.median(np_arr, axis=0)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_median_axis1(self, arr_2d):
        gpu, np_arr = arr_2d
        result = median(gpu, axis=1)
        expected = np.median(np_arr, axis=1)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_median_even_count(self):
        np_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = median(gpu)
        expected = np.median(np_arr)
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)


# ------------------------------------------------------------------ percentile


class TestPercentile:
    def test_percentile_50(self, arr_1d):
        gpu, np_arr = arr_1d
        result = percentile(gpu, 50)
        expected = np.percentile(np_arr, 50)
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_percentile_25_75(self, arr_1d):
        gpu, np_arr = arr_1d
        for q in [25, 75]:
            result = percentile(gpu, q)
            expected = np.percentile(np_arr, q)
            assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_percentile_axis0(self, arr_2d):
        gpu, np_arr = arr_2d
        result = percentile(gpu, 50, axis=0)
        expected = np.percentile(np_arr, 50, axis=0)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_percentile_0_and_100(self, arr_1d):
        gpu, np_arr = arr_1d
        assert_allclose(float(percentile(gpu, 0).get()), float(np.percentile(np_arr, 0)), rtol=1e-5)
        assert_allclose(float(percentile(gpu, 100).get()), float(np.percentile(np_arr, 100)), rtol=1e-5)


# ------------------------------------------------------------------ cumsum


class TestCumsum:
    def test_cumsum_1d(self, arr_1d):
        gpu, np_arr = arr_1d
        result = gpu.cumsum()
        expected = np.cumsum(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_cumsum_2d_flat(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.cumsum()
        expected = np.cumsum(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_cumsum_axis0(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.cumsum(axis=0)
        expected = np.cumsum(np_arr, axis=0)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_cumsum_axis1(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.cumsum(axis=1)
        expected = np.cumsum(np_arr, axis=1)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_cumsum_module_level(self, arr_1d):
        gpu, np_arr = arr_1d
        result = cumsum(gpu)
        expected = np.cumsum(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ------------------------------------------------------------------ cumprod


class TestCumprod:
    def test_cumprod_1d(self, arr_1d):
        gpu, np_arr = arr_1d
        result = gpu.cumprod()
        expected = np.cumprod(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_cumprod_2d_flat(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.cumprod()
        expected = np.cumprod(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-3)

    def test_cumprod_axis0(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.cumprod(axis=0)
        expected = np.cumprod(np_arr, axis=0)
        assert_allclose(result.get(), expected, rtol=1e-3)

    def test_cumprod_axis1(self, arr_2d):
        gpu, np_arr = arr_2d
        result = gpu.cumprod(axis=1)
        expected = np.cumprod(np_arr, axis=1)
        assert_allclose(result.get(), expected, rtol=1e-3)

    def test_cumprod_module_level(self, arr_1d):
        gpu, np_arr = arr_1d
        result = cumprod(gpu)
        expected = np.cumprod(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ------------------------------------------------------------------ diff


class TestDiff:
    def test_diff_1d(self, arr_1d):
        gpu, np_arr = arr_1d
        result = diff(gpu)
        expected = np.diff(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_diff_n2(self, arr_1d):
        gpu, np_arr = arr_1d
        result = diff(gpu, n=2)
        expected = np.diff(np_arr, n=2)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_diff_2d_axis0(self, arr_2d):
        gpu, np_arr = arr_2d
        result = diff(gpu, axis=0)
        expected = np.diff(np_arr, axis=0)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_diff_2d_axis1(self, arr_2d):
        gpu, np_arr = arr_2d
        result = diff(gpu, axis=1)
        expected = np.diff(np_arr, axis=1)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ------------------------------------------------------------------ isnan / isinf / isfinite


class TestNanInfChecks:
    def test_isnan(self, arr_nan):
        gpu, np_arr = arr_nan
        result = isnan(gpu)
        expected = np.isnan(np_arr)
        assert_array_equal(result.get(), expected)

    def test_isinf(self, arr_nan):
        gpu, np_arr = arr_nan
        result = isinf(gpu)
        expected = np.isinf(np_arr)
        assert_array_equal(result.get(), expected)

    def test_isfinite(self, arr_nan):
        gpu, np_arr = arr_nan
        result = isfinite(gpu)
        expected = np.isfinite(np_arr)
        assert_array_equal(result.get(), expected)

    def test_isnan_no_nans(self, arr_1d):
        gpu, np_arr = arr_1d
        result = isnan(gpu)
        expected = np.isnan(np_arr)
        assert_array_equal(result.get(), expected)


# ------------------------------------------------------------------ nan_to_num


class TestNanToNum:
    def test_nan_to_num_defaults(self, arr_nan):
        gpu, np_arr = arr_nan
        result = nan_to_num(gpu)
        expected = np.nan_to_num(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_nan_to_num_custom(self):
        np_arr = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = nan_to_num(gpu, nan=-1.0, posinf=999.0, neginf=-999.0)
        expected = np.nan_to_num(np_arr, nan=-1.0, posinf=999.0, neginf=-999.0)
        assert_allclose(result.get(), expected, rtol=1e-5)


# ------------------------------------------------------------------ isclose / allclose


class TestCloseComparisons:
    def test_isclose_exact(self, arr_1d):
        gpu, np_arr = arr_1d
        gpu2 = cp.array(np_arr.copy())
        result = isclose(gpu, gpu2)
        expected = np.isclose(np_arr, np_arr.copy())
        assert_array_equal(result.get(), expected)

    def test_isclose_with_tolerance(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.array([1.0, 2.00001, 3.1], dtype=np.float32)
        a_gpu = cp.array(a_np)
        b_gpu = cp.array(b_np)
        result = isclose(a_gpu, b_gpu, atol=1e-3)
        expected = np.isclose(a_np, b_np, atol=1e-3)
        assert_array_equal(result.get(), expected)

    def test_allclose_true(self, arr_1d):
        gpu, np_arr = arr_1d
        gpu2 = cp.array(np_arr.copy())
        assert allclose(gpu, gpu2) is True

    def test_allclose_false(self):
        a_gpu = cp.array(np.array([1.0, 2.0], dtype=np.float32))
        b_gpu = cp.array(np.array([1.0, 3.0], dtype=np.float32))
        assert allclose(a_gpu, b_gpu) is False


# ------------------------------------------------------------------ array_equal


class TestArrayEqual:
    def test_equal_arrays(self, arr_1d):
        gpu, np_arr = arr_1d
        gpu2 = cp.array(np_arr.copy())
        assert array_equal(gpu, gpu2) is True

    def test_unequal_arrays(self):
        a_gpu = cp.array(np.array([1.0, 2.0], dtype=np.float32))
        b_gpu = cp.array(np.array([1.0, 3.0], dtype=np.float32))
        assert array_equal(a_gpu, b_gpu) is False


# ------------------------------------------------------------------ count_nonzero


class TestCountNonzero:
    def test_count_nonzero_1d(self):
        np_arr = np.array([0.0, 1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = count_nonzero(gpu)
        expected = np.count_nonzero(np_arr)
        assert result == expected

    def test_count_nonzero_all(self, arr_1d):
        gpu, np_arr = arr_1d
        result = count_nonzero(gpu)
        expected = np.count_nonzero(np_arr)
        assert result == expected

    def test_count_nonzero_axis(self, arr_2d):
        gpu, np_arr = arr_2d
        result = count_nonzero(gpu, axis=0)
        expected = np.count_nonzero(np_arr, axis=0)
        assert_array_equal(result.get(), expected)

    def test_count_nonzero_returns_int(self, arr_1d):
        gpu, _ = arr_1d
        result = count_nonzero(gpu)
        assert isinstance(result, int)


# ------------------------------------------------------------------ plain-array input (non-ndarray)


class TestPlainArrayInput:
    """Module-level functions should accept plain lists/numpy arrays."""

    def test_std_from_list(self):
        result = std([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_median_from_numpy(self):
        np_arr = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        result = median(np_arr)
        expected = np.median(np_arr)
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_isnan_from_list(self):
        result = isnan([1.0, float("nan"), 3.0])
        expected = np.isnan([1.0, float("nan"), 3.0])
        assert_array_equal(result.get(), expected)

    def test_count_nonzero_from_list(self):
        result = count_nonzero([0, 1, 0, 3])
        expected = np.count_nonzero([0, 1, 0, 3])
        assert result == expected
