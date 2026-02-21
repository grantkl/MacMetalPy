"""Tests for NaN-aware reductions, histograms, statistics, and diff extensions.

Consolidates test_nan_stats.py.
Ref: cupy_tests/math_tests/test_misc.py, NumPy test_nanfunctions.py
Target: ~306 parametrized cases.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp

from conftest import FLOAT_DTYPES, assert_eq, tol_for


# ── data helpers ──────────────────────────────────────────────────────────

SHAPES_NAN = [(5,), (2, 3), (2, 3, 4)]


def _make_nan_array(shape, dtype, seed=42):
    """Array with ~30% NaN values."""
    rng = np.random.RandomState(seed)
    size = 1
    for s in shape:
        size *= s
    data = rng.randn(size).astype(dtype).reshape(shape)
    mask = rng.random(size).reshape(shape) < 0.3
    data[mask] = np.nan
    return data


def _make_all_nan(shape, dtype):
    return np.full(shape, np.nan, dtype=dtype)


def _make_no_nan(shape, dtype, seed=42):
    rng = np.random.RandomState(seed)
    size = 1
    for s in shape:
        size *= s
    return (rng.randn(size).astype(dtype).reshape(shape) + 2.0).astype(dtype)


# ======================================================================
# NaN reductions
# ======================================================================
# Ref: cupy_tests/math_tests/test_misc.py::TestNanSum etc.

class TestNanSum:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nansum(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nansum(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_no_nan(self, dtype, shape):
        np_arr = _make_no_nan(shape, dtype)
        result = cp.nansum(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nansum(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_nan(self, dtype):
        np_arr = _make_all_nan((5,), dtype)
        result = cp.nansum(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nansum(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nansum(cp.array(np_arr), axis=0)
        npt.assert_allclose(result.get(), np.nansum(np_arr, axis=0), **tol_for(dtype))


class TestNanProd:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nanprod(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanprod(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_all_nan(self, dtype):
        np_arr = _make_all_nan((5,), dtype)
        result = cp.nanprod(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanprod(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nanprod(cp.array(np_arr), axis=1)
        npt.assert_allclose(result.get(), np.nanprod(np_arr, axis=1), **tol_for(dtype))


class TestNanMax:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nanmax(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanmax(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_no_nan(self, dtype):
        np_arr = _make_no_nan((5,), dtype)
        result = cp.nanmax(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanmax(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nanmax(cp.array(np_arr), axis=0)
        npt.assert_allclose(result.get(), np.nanmax(np_arr, axis=0), **tol_for(dtype))


class TestNanMin:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nanmin(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanmin(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nanmin(cp.array(np_arr), axis=1)
        npt.assert_allclose(result.get(), np.nanmin(np_arr, axis=1), **tol_for(dtype))


class TestNanMean:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nanmean(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanmean(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_no_nan(self, dtype):
        np_arr = _make_no_nan((2, 3), dtype)
        result = cp.nanmean(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanmean(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nanmean(cp.array(np_arr), axis=0)
        npt.assert_allclose(result.get(), np.nanmean(np_arr, axis=0), **tol_for(dtype))


class TestNanMedian:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nanmedian(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanmedian(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_no_nan(self, dtype):
        np_arr = _make_no_nan((5,), dtype)
        result = cp.nanmedian(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanmedian(np_arr), **tol_for(dtype))


class TestNanStd:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nanstd(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanstd(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nanstd(cp.array(np_arr), axis=0)
        npt.assert_allclose(result.get(), np.nanstd(np_arr, axis=0), **tol_for(dtype))


class TestNanVar:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nanvar(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nanvar(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nanvar(cp.array(np_arr), axis=1)
        npt.assert_allclose(result.get(), np.nanvar(np_arr, axis=1), **tol_for(dtype))


class TestNanArgmax:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nanargmax(cp.array(np_arr))
        expected = np.nanargmax(np_arr)
        if isinstance(result, int):
            assert result == int(expected)
        else:
            assert int(result.get()) == int(expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nanargmax(cp.array(np_arr), axis=1)
        expected = np.nanargmax(np_arr, axis=1)
        npt.assert_array_equal(result.get(), expected)


class TestNanArgmin:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nanargmin(cp.array(np_arr))
        expected = np.nanargmin(np_arr)
        if isinstance(result, int):
            assert result == int(expected)
        else:
            assert int(result.get()) == int(expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nanargmin(cp.array(np_arr), axis=0)
        expected = np.nanargmin(np_arr, axis=0)
        npt.assert_array_equal(result.get(), expected)


class TestNanCumsum:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nancumsum(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nancumsum(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis(self, dtype):
        np_arr = _make_nan_array((2, 3), dtype)
        result = cp.nancumsum(cp.array(np_arr), axis=1)
        npt.assert_allclose(result.get(), np.nancumsum(np_arr, axis=1), **tol_for(dtype))


class TestNanCumprod:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_NAN)
    def test_basic(self, dtype, shape):
        np_arr = _make_nan_array(shape, dtype)
        result = cp.nancumprod(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.nancumprod(np_arr), **tol_for(dtype))


# ======================================================================
# Histograms
# ======================================================================
# Ref: NumPy test_histograms.py

class TestHistogram:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        hist, edges = cp.histogram(cp.array(np_arr))
        np_hist, np_edges = np.histogram(np_arr)
        npt.assert_array_equal(hist.get(), np_hist)
        npt.assert_allclose(edges.get(), np_edges, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_bins(self, dtype):
        np_arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        hist, edges = cp.histogram(cp.array(np_arr), bins=3)
        np_hist, np_edges = np.histogram(np_arr, bins=3)
        npt.assert_array_equal(hist.get(), np_hist)
        npt.assert_allclose(edges.get(), np_edges, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_range(self, dtype):
        rng = np.random.RandomState(42)
        np_arr = rng.randn(100).astype(dtype)
        hist, edges = cp.histogram(cp.array(np_arr), bins=10)
        np_hist, np_edges = np.histogram(np_arr, bins=10)
        npt.assert_array_equal(hist.get(), np_hist)


class TestHistogram2d:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_x = np.array([1, 2, 3, 4], dtype=dtype)
        np_y = np.array([4, 3, 2, 1], dtype=dtype)
        H, xedges, yedges = cp.histogram2d(cp.array(np_x), cp.array(np_y))
        np_H, np_xe, np_ye = np.histogram2d(np_x, np_y)
        npt.assert_array_equal(H.get(), np_H)
        npt.assert_allclose(xedges.get(), np_xe, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_bins(self, dtype):
        rng = np.random.RandomState(42)
        np_x = rng.randn(50).astype(dtype)
        np_y = rng.randn(50).astype(dtype)
        H, xedges, yedges = cp.histogram2d(cp.array(np_x), cp.array(np_y), bins=5)
        np_H, np_xe, np_ye = np.histogram2d(np_x, np_y, bins=5)
        npt.assert_array_equal(H.get(), np_H)


class TestHistogramdd:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_sample = np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype)
        H, edges = cp.histogramdd(cp.array(np_sample))
        np_H, np_edges = np.histogramdd(np_sample)
        npt.assert_array_equal(H.get(), np_H)
        for e, ne in zip(edges, np_edges):
            npt.assert_allclose(e.get(), ne, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_bins(self, dtype):
        rng = np.random.RandomState(42)
        np_sample = rng.randn(30, 3).astype(dtype)
        H, edges = cp.histogramdd(cp.array(np_sample), bins=3)
        np_H, np_edges = np.histogramdd(np_sample, bins=3)
        npt.assert_array_equal(H.get(), np_H)


# ======================================================================
# Bincount / Digitize
# ======================================================================

class TestBincount:
    def test_basic(self):
        np_x = np.array([0, 1, 1, 2, 3, 3, 3], dtype=np.int32)
        result = cp.bincount(cp.array(np_x))
        npt.assert_array_equal(result.get(), np.bincount(np_x))

    def test_with_weights(self):
        np_x = np.array([0, 1, 1, 2], dtype=np.int32)
        np_w = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        result = cp.bincount(cp.array(np_x), weights=cp.array(np_w))
        npt.assert_allclose(result.get(), np.bincount(np_x, weights=np_w), rtol=1e-5)

    def test_empty(self):
        np_x = np.array([], dtype=np.int32)
        result = cp.bincount(cp.array(np_x))
        npt.assert_array_equal(result.get(), np.bincount(np_x))


class TestDigitize:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_x = np.array([0.5, 1.5, 2.5, 3.5], dtype=dtype)
        np_bins = np.array([1.0, 2.0, 3.0], dtype=dtype)
        result = cp.digitize(cp.array(np_x), cp.array(np_bins))
        npt.assert_array_equal(result.get(), np.digitize(np_x, np_bins))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_right(self, dtype):
        np_x = np.array([1.0, 2.0, 3.0], dtype=dtype)
        np_bins = np.array([1.0, 2.0, 3.0], dtype=dtype)
        result = cp.digitize(cp.array(np_x), cp.array(np_bins), right=True)
        npt.assert_array_equal(result.get(), np.digitize(np_x, np_bins, right=True))


# ======================================================================
# Corrcoef / Cov / Correlate
# ======================================================================

class TestCorrcoef:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_single(self, dtype):
        np_x = np.array([1, 2, 3, 4], dtype=dtype)
        result = cp.corrcoef(cp.array(np_x))
        npt.assert_allclose(result.get(), np.corrcoef(np_x), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_xy(self, dtype):
        np_x = np.array([1, 2, 3], dtype=dtype)
        np_y = np.array([4, 5, 6], dtype=dtype)
        result = cp.corrcoef(cp.array(np_x), cp.array(np_y))
        npt.assert_allclose(result.get(), np.corrcoef(np_x, np_y), **tol_for(dtype))


class TestCov:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_single(self, dtype):
        np_m = np.array([1, 2, 3, 4], dtype=dtype)
        result = cp.cov(cp.array(np_m))
        npt.assert_allclose(result.get(), np.cov(np_m), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_xy(self, dtype):
        np_x = np.array([1, 2, 3], dtype=dtype)
        np_y = np.array([4, 5, 6], dtype=dtype)
        result = cp.cov(cp.array(np_x), cp.array(np_y))
        npt.assert_allclose(result.get(), np.cov(np_x, np_y), **tol_for(dtype))


class TestCorrelate:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_valid(self, dtype):
        a_np = np.array([1, 2, 3, 4, 5], dtype=dtype)
        v_np = np.array([1, 1], dtype=dtype)
        result = cp.correlate(cp.array(a_np), cp.array(v_np))
        npt.assert_allclose(result.get(), np.correlate(a_np, v_np, mode="valid"), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_full(self, dtype):
        a_np = np.array([1, 2, 3], dtype=dtype)
        v_np = np.array([1, 2], dtype=dtype)
        result = cp.correlate(cp.array(a_np), cp.array(v_np), mode="full")
        npt.assert_allclose(result.get(), np.correlate(a_np, v_np, mode="full"), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_same(self, dtype):
        a_np = np.array([1, 2, 3, 4], dtype=dtype)
        v_np = np.array([1, 2], dtype=dtype)
        result = cp.correlate(cp.array(a_np), cp.array(v_np), mode="same")
        npt.assert_allclose(result.get(), np.correlate(a_np, v_np, mode="same"), **tol_for(dtype))


# ======================================================================
# Ediff1d / Gradient
# ======================================================================

class TestEdiff1d:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([1, 2, 4, 7, 11], dtype=dtype)
        result = cp.ediff1d(cp.array(np_arr))
        npt.assert_allclose(result.get(), np.ediff1d(np_arr), **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_endpoints(self, dtype):
        np_arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = cp.ediff1d(
            cp.array(np_arr),
            to_end=cp.array(np.array([99], dtype=dtype)),
            to_begin=cp.array(np.array([-1], dtype=dtype)),
        )
        expected = np.ediff1d(
            np_arr,
            to_end=np.array([99], dtype=dtype),
            to_begin=np.array([-1], dtype=dtype),
        )
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


class TestGradient:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1d(self, dtype):
        np_arr = np.array([1, 2, 4, 7, 11], dtype=dtype)
        result = cp.gradient(cp.array(np_arr))
        expected = np.gradient(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, dtype):
        np_arr = np.arange(12, dtype=dtype).reshape(3, 4)
        results = cp.gradient(cp.array(np_arr))
        expected = np.gradient(np_arr)
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            npt.assert_allclose(r.get(), e, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_spacing(self, dtype):
        np_arr = np.array([1, 2, 4, 8], dtype=dtype)
        result = cp.gradient(cp.array(np_arr), cp.array(np.array(2.0, dtype=dtype)))
        expected = np.gradient(np_arr, 2.0)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))
