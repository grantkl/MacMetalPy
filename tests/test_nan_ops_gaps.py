"""Tests for nan_ops enhancements: nanpercentile, nanquantile, histogram_bin_edges.

Covers:
- nanpercentile with NaN values, axis, out
- nanquantile with NaN values, axis, out
- histogram_bin_edges with various bins/range
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp

from conftest import FLOAT_DTYPES, tol_for


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


def _make_no_nan(shape, dtype, seed=42):
    rng = np.random.RandomState(seed)
    size = 1
    for s in shape:
        size *= s
    return (rng.randn(size).astype(dtype).reshape(shape) + 2.0).astype(dtype)


# ======================================================================
# nanpercentile
# ======================================================================

class TestNanpercentile:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("q", [0, 25, 50, 75, 100])
    def test_basic(self, dtype, q):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nanpercentile(cp.array(np_arr), q)
        expected = np.nanpercentile(np_arr, q)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_no_nan(self, dtype):
        np_arr = _make_no_nan((10,), dtype)
        result = cp.nanpercentile(cp.array(np_arr), 50)
        expected = np.nanpercentile(np_arr, 50)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis0(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanpercentile(cp.array(np_arr), 50, axis=0)
        expected = np.nanpercentile(np_arr, 50, axis=0)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis1(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanpercentile(cp.array(np_arr), 75, axis=1)
        expected = np.nanpercentile(np_arr, 75, axis=1)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_multiple_q(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nanpercentile(cp.array(np_arr), [25, 50, 75])
        expected = np.nanpercentile(np_arr, [25, 50, 75])
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanpercentile(cp.array(np_arr), 50, axis=0, keepdims=True)
        expected = np.nanpercentile(np_arr, 50, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


# ======================================================================
# nanquantile
# ======================================================================

class TestNanquantile:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_basic(self, dtype, q):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nanquantile(cp.array(np_arr), q)
        expected = np.nanquantile(np_arr, q)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_no_nan(self, dtype):
        np_arr = _make_no_nan((10,), dtype)
        result = cp.nanquantile(cp.array(np_arr), 0.5)
        expected = np.nanquantile(np_arr, 0.5)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis0(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanquantile(cp.array(np_arr), 0.5, axis=0)
        expected = np.nanquantile(np_arr, 0.5, axis=0)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axis1(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanquantile(cp.array(np_arr), 0.75, axis=1)
        expected = np.nanquantile(np_arr, 0.75, axis=1)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_multiple_q(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nanquantile(cp.array(np_arr), [0.25, 0.5, 0.75])
        expected = np.nanquantile(np_arr, [0.25, 0.5, 0.75])
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanquantile(cp.array(np_arr), 0.5, axis=0, keepdims=True)
        expected = np.nanquantile(np_arr, 0.5, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


# ======================================================================
# histogram_bin_edges
# ======================================================================

class TestHistogramBinEdges:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = cp.histogram_bin_edges(cp.array(np_arr))
        expected = np.histogram_bin_edges(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_custom_bins(self, dtype):
        np_arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = cp.histogram_bin_edges(cp.array(np_arr), bins=3)
        expected = np.histogram_bin_edges(np_arr, bins=3)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_range(self, dtype):
        np_arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = cp.histogram_bin_edges(cp.array(np_arr), bins=5, range=(0, 10))
        expected = np.histogram_bin_edges(np_arr, bins=5, range=(0, 10))
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_random_data(self, dtype):
        rng = np.random.RandomState(42)
        np_arr = rng.randn(100).astype(dtype)
        result = cp.histogram_bin_edges(cp.array(np_arr), bins=20)
        expected = np.histogram_bin_edges(np_arr, bins=20)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_string_bins(self, dtype):
        """Test with string bin selection method."""
        np_arr = np.arange(100, dtype=dtype)
        result = cp.histogram_bin_edges(cp.array(np_arr), bins='auto')
        expected = np.histogram_bin_edges(np_arr, bins='auto')
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))
