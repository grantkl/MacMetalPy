"""Tests for NaN-aware reductions, extended statistics, histograms, and diff extensions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import macmetalpy as cp
from macmetalpy.nan_ops import (
    nansum,
    nanprod,
    nancumsum,
    nancumprod,
    nanmax,
    nanmin,
    nanmean,
    nanmedian,
    nanstd,
    nanvar,
    nanargmax,
    nanargmin,
    corrcoef,
    correlate,
    cov,
    histogram,
    histogram2d,
    histogramdd,
    bincount,
    digitize,
    ediff1d,
    gradient,
)
from macmetalpy.reductions import ptp, quantile, average


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def nan_1d():
    np_arr = np.array([1.0, np.nan, 3.0, 4.0, np.nan], dtype=np.float32)
    return cp.array(np_arr), np_arr


@pytest.fixture
def nan_2d():
    np_arr = np.array(
        [[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32
    )
    return cp.array(np_arr), np_arr


@pytest.fixture
def clean_1d():
    np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    return cp.array(np_arr), np_arr


@pytest.fixture
def clean_2d():
    np_arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    return cp.array(np_arr), np_arr


# ================================================================== NaN reductions
class TestNanSum:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nansum(gpu)
        assert_allclose(result.get(), np.nansum(np_arr), rtol=1e-5)

    def test_2d_axis(self, nan_2d):
        gpu, np_arr = nan_2d
        result = nansum(gpu, axis=0)
        assert_allclose(result.get(), np.nansum(np_arr, axis=0), rtol=1e-5)

    def test_2d_full(self, nan_2d):
        gpu, np_arr = nan_2d
        result = nansum(gpu)
        assert_allclose(result.get(), np.nansum(np_arr), rtol=1e-5)


class TestNanProd:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nanprod(gpu)
        assert_allclose(result.get(), np.nanprod(np_arr), rtol=1e-5)

    def test_2d_axis(self, nan_2d):
        gpu, np_arr = nan_2d
        result = nanprod(gpu, axis=1)
        assert_allclose(result.get(), np.nanprod(np_arr, axis=1), rtol=1e-5)


class TestNanCumsum:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nancumsum(gpu)
        assert_allclose(result.get(), np.nancumsum(np_arr), rtol=1e-5)

    def test_2d_axis(self, nan_2d):
        gpu, np_arr = nan_2d
        result = nancumsum(gpu, axis=1)
        assert_allclose(result.get(), np.nancumsum(np_arr, axis=1), rtol=1e-5)


class TestNanCumprod:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nancumprod(gpu)
        assert_allclose(result.get(), np.nancumprod(np_arr), rtol=1e-5)


class TestNanMax:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nanmax(gpu)
        assert_allclose(result.get(), np.nanmax(np_arr), rtol=1e-5)

    def test_2d_axis(self, nan_2d):
        gpu, np_arr = nan_2d
        result = nanmax(gpu, axis=0)
        assert_allclose(result.get(), np.nanmax(np_arr, axis=0), rtol=1e-5)


class TestNanMin:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nanmin(gpu)
        assert_allclose(result.get(), np.nanmin(np_arr), rtol=1e-5)

    def test_2d_axis(self, nan_2d):
        gpu, np_arr = nan_2d
        result = nanmin(gpu, axis=1)
        assert_allclose(result.get(), np.nanmin(np_arr, axis=1), rtol=1e-5)


class TestNanMean:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nanmean(gpu)
        assert_allclose(result.get(), np.nanmean(np_arr), rtol=1e-5)

    def test_2d_axis(self, nan_2d):
        gpu, np_arr = nan_2d
        result = nanmean(gpu, axis=0)
        assert_allclose(result.get(), np.nanmean(np_arr, axis=0), rtol=1e-5)


class TestNanMedian:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nanmedian(gpu)
        assert_allclose(result.get(), np.nanmedian(np_arr), rtol=1e-5)


class TestNanStd:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nanstd(gpu)
        assert_allclose(result.get(), np.nanstd(np_arr), rtol=1e-5)


class TestNanVar:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nanvar(gpu)
        assert_allclose(result.get(), np.nanvar(np_arr), rtol=1e-5)


class TestNanArgmax:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nanargmax(gpu)
        assert result == int(np.nanargmax(np_arr))
        assert isinstance(result, int)

    def test_2d_axis(self, nan_2d):
        gpu, np_arr = nan_2d
        result = nanargmax(gpu, axis=1)
        assert_allclose(result.get(), np.nanargmax(np_arr, axis=1))


class TestNanArgmin:
    def test_1d(self, nan_1d):
        gpu, np_arr = nan_1d
        result = nanargmin(gpu)
        assert result == int(np.nanargmin(np_arr))
        assert isinstance(result, int)

    def test_2d_axis(self, nan_2d):
        gpu, np_arr = nan_2d
        result = nanargmin(gpu, axis=0)
        assert_allclose(result.get(), np.nanargmin(np_arr, axis=0))


# ================================================================== Extended stats
class TestPtp:
    def test_1d(self, clean_1d):
        gpu, np_arr = clean_1d
        result = ptp(gpu)
        assert_allclose(result.get(), np.ptp(np_arr), rtol=1e-5)

    def test_2d_axis(self, clean_2d):
        gpu, np_arr = clean_2d
        result = ptp(gpu, axis=0)
        assert_allclose(result.get(), np.ptp(np_arr, axis=0), rtol=1e-5)


class TestQuantile:
    def test_scalar_q(self, clean_1d):
        gpu, np_arr = clean_1d
        result = quantile(gpu, 0.5)
        assert_allclose(result.get(), np.quantile(np_arr, 0.5), rtol=1e-5)

    def test_array_q(self, clean_1d):
        gpu, np_arr = clean_1d
        result = quantile(gpu, [0.25, 0.75])
        assert_allclose(result.get(), np.quantile(np_arr, [0.25, 0.75]), rtol=1e-5)


class TestAverage:
    def test_unweighted(self, clean_1d):
        gpu, np_arr = clean_1d
        result = average(gpu)
        assert_allclose(result.get(), np.average(np_arr), rtol=1e-5)

    def test_weighted(self, clean_1d):
        gpu, np_arr = clean_1d
        w_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        w_gpu = cp.array(w_np)
        result = average(gpu, weights=w_gpu)
        assert_allclose(result.get(), np.average(np_arr, weights=w_np), rtol=1e-5)


class TestCorrcoef:
    def test_single(self):
        np_x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        gpu_x = cp.array(np_x)
        result = corrcoef(gpu_x)
        assert_allclose(result.get(), np.corrcoef(np_x), rtol=1e-5)

    def test_xy(self):
        np_x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np_y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = corrcoef(cp.array(np_x), cp.array(np_y))
        assert_allclose(result.get(), np.corrcoef(np_x, np_y), rtol=1e-5)


class TestCorrelate:
    def test_valid(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        v_np = np.array([1.0, 1.0], dtype=np.float32)
        result = correlate(cp.array(a_np), cp.array(v_np))
        assert_allclose(result.get(), np.correlate(a_np, v_np, mode="valid"), rtol=1e-5)

    def test_full(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_np = np.array([1.0, 2.0], dtype=np.float32)
        result = correlate(cp.array(a_np), cp.array(v_np), mode="full")
        assert_allclose(result.get(), np.correlate(a_np, v_np, mode="full"), rtol=1e-5)


class TestCov:
    def test_single(self):
        np_m = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = cov(cp.array(np_m))
        assert_allclose(result.get(), np.cov(np_m), rtol=1e-5)

    def test_xy(self):
        np_x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np_y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = cov(cp.array(np_x), cp.array(np_y))
        assert_allclose(result.get(), np.cov(np_x, np_y), rtol=1e-5)


# ================================================================== Histograms
class TestHistogram:
    def test_basic(self, clean_1d):
        gpu, np_arr = clean_1d
        hist, edges = histogram(gpu)
        np_hist, np_edges = np.histogram(np_arr)
        assert_allclose(hist.get(), np_hist)
        assert_allclose(edges.get(), np_edges, rtol=1e-5)


class TestHistogram2d:
    def test_basic(self):
        np_x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np_y = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        H, xedges, yedges = histogram2d(cp.array(np_x), cp.array(np_y))
        np_H, np_xe, np_ye = np.histogram2d(np_x, np_y)
        assert_allclose(H.get(), np_H)
        assert_allclose(xedges.get(), np_xe, rtol=1e-5)
        assert_allclose(yedges.get(), np_ye, rtol=1e-5)


class TestHistogramdd:
    def test_basic(self):
        np_sample = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        gpu_sample = cp.array(np_sample)
        H, edges = histogramdd(gpu_sample)
        np_H, np_edges = np.histogramdd(np_sample)
        assert_allclose(H.get(), np_H)
        for e, ne in zip(edges, np_edges):
            assert_allclose(e.get(), ne, rtol=1e-5)


class TestBincount:
    def test_basic(self):
        np_x = np.array([0, 1, 1, 2, 3, 3, 3], dtype=np.int32)
        result = bincount(cp.array(np_x))
        assert_allclose(result.get(), np.bincount(np_x))

    def test_with_weights(self):
        np_x = np.array([0, 1, 1, 2], dtype=np.int32)
        np_w = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        result = bincount(cp.array(np_x), weights=cp.array(np_w))
        assert_allclose(result.get(), np.bincount(np_x, weights=np_w), rtol=1e-5)


class TestDigitize:
    def test_basic(self):
        np_x = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        np_bins = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = digitize(cp.array(np_x), cp.array(np_bins))
        assert_allclose(result.get(), np.digitize(np_x, np_bins))


# ================================================================== Diff extensions
class TestEdiff1d:
    def test_basic(self, clean_1d):
        gpu, np_arr = clean_1d
        result = ediff1d(gpu)
        assert_allclose(result.get(), np.ediff1d(np_arr), rtol=1e-5)

    def test_with_endpoints(self, clean_1d):
        gpu, np_arr = clean_1d
        result = ediff1d(gpu, to_end=cp.array(np.array([99.0], dtype=np.float32)),
                         to_begin=cp.array(np.array([-1.0], dtype=np.float32)))
        expected = np.ediff1d(np_arr, to_end=np.array([99.0], dtype=np.float32),
                              to_begin=np.array([-1.0], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestGradient:
    def test_1d(self, clean_1d):
        gpu, np_arr = clean_1d
        result = gradient(gpu)
        expected = np.gradient(np_arr)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_2d(self, clean_2d):
        gpu, np_arr = clean_2d
        results = gradient(gpu)
        expected = np.gradient(np_arr)
        assert len(results) == len(expected)
        for r, e in zip(results, expected):
            assert_allclose(r.get(), e, rtol=1e-5)
