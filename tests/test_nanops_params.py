"""Tests for nan_ops parameter gaps (HIGH + MEDIUM severity).

Covers missing parameters: dtype, keepdims, out, ddof, initial, where,
rowvar, bias, fweights, aweights for all nan reduction and stats functions.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp

from conftest import FLOAT_DTYPES, tol_for


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
# nansum — dtype, keepdims, out, initial, where
# ======================================================================

class TestNansumParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nansum(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nansum(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nansum(cp.array(np_arr), dtype=np.float32)
        expected = np.nansum(np_arr, dtype=np.float32)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.float32)
        result = cp.nansum(cp.array(np_arr), axis=0, out=out)
        expected = np.nansum(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_initial(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nansum(cp.array(np_arr), initial=100.0)
        expected = np.nansum(np_arr, initial=100.0)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_where(self):
        np_arr = _make_no_nan((10,), np.float32)
        mask = np.array([True, False] * 5)
        result = cp.nansum(cp.array(np_arr), where=cp.array(mask), initial=0.0)
        expected = np.nansum(np_arr, where=mask, initial=0.0)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_default_behavior_unchanged(self):
        """Existing behavior must not change."""
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nansum(cp.array(np_arr))
        expected = np.nansum(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# nanprod — dtype, keepdims, out, initial, where
# ======================================================================

class TestNanprodParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanprod(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nanprod(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nanprod(cp.array(np_arr), dtype=np.float32)
        expected = np.nanprod(np_arr, dtype=np.float32)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.float32)
        result = cp.nanprod(cp.array(np_arr), axis=0, out=out)
        expected = np.nanprod(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_initial(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanprod(cp.array(np_arr), initial=2.0)
        expected = np.nanprod(np_arr, initial=2.0)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_where(self):
        np_arr = _make_no_nan((10,), np.float32)
        mask = np.array([True, False] * 5)
        result = cp.nanprod(cp.array(np_arr), where=cp.array(mask), initial=1.0)
        expected = np.nanprod(np_arr, where=mask, initial=1.0)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanprod(cp.array(np_arr))
        expected = np.nanprod(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# nanmax — keepdims, out, initial, where
# ======================================================================

class TestNanmaxParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanmax(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nanmax(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.float32)
        result = cp.nanmax(cp.array(np_arr), axis=0, out=out)
        expected = np.nanmax(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_initial(self):
        np_arr = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        result = cp.nanmax(cp.array(np_arr), initial=-999.0)
        expected = np.nanmax(np_arr, initial=-999.0)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_where(self):
        np_arr = _make_no_nan((10,), np.float32)
        mask = np.array([True, False] * 5)
        result = cp.nanmax(cp.array(np_arr), where=cp.array(mask), initial=-np.inf)
        expected = np.nanmax(np_arr, where=mask, initial=-np.inf)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanmax(cp.array(np_arr))
        expected = np.nanmax(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# nanmin — keepdims, out, initial, where
# ======================================================================

class TestNanminParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanmin(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nanmin(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.float32)
        result = cp.nanmin(cp.array(np_arr), axis=0, out=out)
        expected = np.nanmin(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_initial(self):
        np_arr = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        result = cp.nanmin(cp.array(np_arr), initial=999.0)
        expected = np.nanmin(np_arr, initial=999.0)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_where(self):
        np_arr = _make_no_nan((10,), np.float32)
        mask = np.array([True, False] * 5)
        result = cp.nanmin(cp.array(np_arr), where=cp.array(mask), initial=np.inf)
        expected = np.nanmin(np_arr, where=mask, initial=np.inf)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanmin(cp.array(np_arr))
        expected = np.nanmin(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# nanmean — dtype, keepdims, out, where
# ======================================================================

class TestNanmeanParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanmean(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nanmean(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nanmean(cp.array(np_arr), dtype=np.float32)
        expected = np.nanmean(np_arr, dtype=np.float32)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.float32)
        result = cp.nanmean(cp.array(np_arr), axis=0, out=out)
        expected = np.nanmean(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_where(self):
        np_arr = _make_no_nan((3, 4), np.float32)
        mask = np.ones((3, 4), dtype=bool)
        mask[0, :] = False
        result = cp.nanmean(cp.array(np_arr), axis=0, where=cp.array(mask))
        expected = np.nanmean(np_arr, axis=0, where=mask)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanmean(cp.array(np_arr))
        expected = np.nanmean(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# nanmedian — keepdims, out
# ======================================================================

class TestNanmedianParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanmedian(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nanmedian(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.float32)
        result = cp.nanmedian(cp.array(np_arr), axis=0, out=out)
        expected = np.nanmedian(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanmedian(cp.array(np_arr))
        expected = np.nanmedian(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# nanstd — ddof, dtype, keepdims, out, where
# ======================================================================

class TestNanstdParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanstd(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nanstd(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_ddof(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanstd(cp.array(np_arr), ddof=1)
        expected = np.nanstd(np_arr, ddof=1)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nanstd(cp.array(np_arr), dtype=np.float32)
        expected = np.nanstd(np_arr, dtype=np.float32)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.float32)
        result = cp.nanstd(cp.array(np_arr), axis=0, out=out)
        expected = np.nanstd(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_where(self):
        np_arr = _make_no_nan((3, 4), np.float32)
        mask = np.ones((3, 4), dtype=bool)
        mask[0, :] = False
        result = cp.nanstd(cp.array(np_arr), axis=0, where=cp.array(mask))
        expected = np.nanstd(np_arr, axis=0, where=mask)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanstd(cp.array(np_arr))
        expected = np.nanstd(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# nanvar — ddof, dtype, keepdims, out, where
# ======================================================================

class TestNanvarParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanvar(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nanvar(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_ddof(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanvar(cp.array(np_arr), ddof=1)
        expected = np.nanvar(np_arr, ddof=1)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nanvar(cp.array(np_arr), dtype=np.float32)
        expected = np.nanvar(np_arr, dtype=np.float32)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.float32)
        result = cp.nanvar(cp.array(np_arr), axis=0, out=out)
        expected = np.nanvar(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_where(self):
        np_arr = _make_no_nan((3, 4), np.float32)
        mask = np.ones((3, 4), dtype=bool)
        mask[0, :] = False
        result = cp.nanvar(cp.array(np_arr), axis=0, where=cp.array(mask))
        expected = np.nanvar(np_arr, axis=0, where=mask)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanvar(cp.array(np_arr))
        expected = np.nanvar(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# nanargmax — keepdims, out
# ======================================================================

class TestNanargmaxParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanargmax(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nanargmax(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.int64)
        result = cp.nanargmax(cp.array(np_arr), axis=0, out=out)
        expected = np.nanargmax(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, atol=0, rtol=0)

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanargmax(cp.array(np_arr))
        expected = np.nanargmax(np_arr)
        assert result == expected


# ======================================================================
# nanargmin — keepdims, out
# ======================================================================

class TestNanargminParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_keepdims(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        result = cp.nanargmin(cp.array(np_arr), axis=0, keepdims=True)
        expected = np.nanargmin(np_arr, axis=0, keepdims=True)
        assert result.shape == expected.shape
        npt.assert_allclose(result.get(), expected, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((3, 4), dtype)
        out = cp.zeros((4,), dtype=np.int64)
        result = cp.nanargmin(cp.array(np_arr), axis=0, out=out)
        expected = np.nanargmin(np_arr, axis=0)
        assert result is out
        npt.assert_allclose(result.get(), expected, atol=0, rtol=0)

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nanargmin(cp.array(np_arr))
        expected = np.nanargmin(np_arr)
        assert result == expected


# ======================================================================
# nancumsum — dtype, out
# ======================================================================

class TestNancumsumParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nancumsum(cp.array(np_arr), dtype=np.float32)
        expected = np.nancumsum(np_arr, dtype=np.float32)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        out = cp.zeros((10,), dtype=np.float32)
        result = cp.nancumsum(cp.array(np_arr), out=out)
        expected = np.nancumsum(np_arr)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nancumsum(cp.array(np_arr))
        expected = np.nancumsum(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# nancumprod — dtype, out
# ======================================================================

class TestNancumprodParams:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_dtype(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        result = cp.nancumprod(cp.array(np_arr), dtype=np.float32)
        expected = np.nancumprod(np_arr, dtype=np.float32)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        np_arr = _make_nan_array((10,), dtype)
        out = cp.zeros((10,), dtype=np.float32)
        result = cp.nancumprod(cp.array(np_arr), out=out)
        expected = np.nancumprod(np_arr)
        assert result is out
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    def test_default_behavior_unchanged(self):
        np_arr = _make_nan_array((10,), np.float32)
        result = cp.nancumprod(cp.array(np_arr))
        expected = np.nancumprod(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# corrcoef — rowvar, bias, ddof, dtype
# ======================================================================

class TestCorrcoefParams:
    def test_rowvar_true(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.corrcoef(cp.array(np_arr), rowvar=True)
        expected = np.corrcoef(np_arr, rowvar=True)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_rowvar_false(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.corrcoef(cp.array(np_arr), rowvar=False)
        expected = np.corrcoef(np_arr, rowvar=False)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_dtype(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.corrcoef(cp.array(np_arr), dtype=np.float32)
        expected = np.corrcoef(np_arr, dtype=np.float32)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_default_behavior_unchanged(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.corrcoef(cp.array(np_arr))
        expected = np.corrcoef(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))


# ======================================================================
# cov — rowvar, bias, fweights, aweights, dtype
# ======================================================================

class TestCovParams:
    def test_rowvar_true(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.cov(cp.array(np_arr), rowvar=True)
        expected = np.cov(np_arr, rowvar=True)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_rowvar_false(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.cov(cp.array(np_arr), rowvar=False)
        expected = np.cov(np_arr, rowvar=False)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_bias(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.cov(cp.array(np_arr), bias=True)
        expected = np.cov(np_arr, bias=True)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_fweights(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        fw = np.array([1, 2, 3], dtype=np.int64)
        result = cp.cov(cp.array(np_arr), fweights=cp.array(fw))
        expected = np.cov(np_arr, fweights=fw)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_aweights(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        aw = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        result = cp.cov(cp.array(np_arr), aweights=cp.array(aw))
        expected = np.cov(np_arr, aweights=aw)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_dtype(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.cov(cp.array(np_arr), dtype=np.float32)
        expected = np.cov(np_arr, dtype=np.float32)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))

    def test_default_behavior_unchanged(self):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.cov(cp.array(np_arr))
        expected = np.cov(np_arr)
        npt.assert_allclose(result.get(), expected, **tol_for(np.float32))
