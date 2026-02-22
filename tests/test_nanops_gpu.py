"""Tests verifying GPU-accelerated NaN operations."""
import numpy as np
import numpy.testing as npt
import pytest
import macmetalpy as mp

class TestNanOpsGPU:
    """Test NaN operations run on GPU and match NumPy."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_nansum(self, dtype):
        np_a = np.array([1, np.nan, 3, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nansum(gpu_a)
        expected = np.nansum(np_a)
        npt.assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_nansum_axis(self, dtype):
        np_a = np.array([[1, np.nan, 3], [np.nan, 5, 6]], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nansum(gpu_a, axis=0)
        expected = np.nansum(np_a, axis=0)
        npt.assert_allclose(result.get(), expected, rtol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_nanprod(self, dtype):
        np_a = np.array([1, np.nan, 3, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nanprod(gpu_a)
        expected = np.nanprod(np_a)
        npt.assert_allclose(float(result.get()), float(expected), rtol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nanmax(self, dtype):
        np_a = np.array([1, np.nan, 3, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nanmax(gpu_a)
        expected = np.nanmax(np_a)
        npt.assert_allclose(float(result.get()), float(expected))

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nanmin(self, dtype):
        np_a = np.array([1, np.nan, 3, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nanmin(gpu_a)
        expected = np.nanmin(np_a)
        npt.assert_allclose(float(result.get()), float(expected))

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nanmean(self, dtype):
        np_a = np.array([1, np.nan, 3, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nanmean(gpu_a)
        expected = np.nanmean(np_a)
        npt.assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nancumsum(self, dtype):
        np_a = np.array([1, np.nan, 3, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nancumsum(gpu_a)
        expected = np.nancumsum(np_a)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nancumprod(self, dtype):
        np_a = np.array([1, np.nan, 3, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nancumprod(gpu_a)
        expected = np.nancumprod(np_a)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nanargmax(self, dtype):
        np_a = np.array([1, np.nan, 5, np.nan, 3], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nanargmax(gpu_a)
        expected = np.nanargmax(np_a)
        assert int(result) == int(expected)

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nanargmin(self, dtype):
        np_a = np.array([3, np.nan, 1, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nanargmin(gpu_a)
        expected = np.nanargmin(np_a)
        assert int(result) == int(expected)

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nanstd(self, dtype):
        np_a = np.array([1, np.nan, 3, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nanstd(gpu_a)
        expected = np.nanstd(np_a)
        npt.assert_allclose(float(result.get()), float(expected), rtol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nanvar(self, dtype):
        np_a = np.array([1, np.nan, 3, np.nan, 5], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nanvar(gpu_a)
        expected = np.nanvar(np_a)
        npt.assert_allclose(float(result.get()), float(expected), rtol=1e-4)

    def test_nansum_no_nan(self):
        """No NaN values - should behave like regular sum."""
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.nansum(gpu_a)
        expected = np.nansum(np_a)
        npt.assert_allclose(float(result.get()), float(expected))

    def test_nansum_all_nan(self):
        """All NaN values - should return 0."""
        np_a = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.nansum(gpu_a)
        expected = np.nansum(np_a)
        npt.assert_allclose(float(result.get()), float(expected))

    def test_nansum_int(self):
        """Integer input - no NaNs possible, should work like sum."""
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        gpu_a = mp.array(np_a)
        result = mp.nansum(gpu_a)
        expected = np.nansum(np_a)
        assert int(result.get()) == int(expected)

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_nanmean_axis(self, dtype):
        np_a = np.array([[1, np.nan, 3], [np.nan, 5, 6]], dtype=dtype)
        gpu_a = mp.array(np_a)
        result = mp.nanmean(gpu_a, axis=1)
        expected = np.nanmean(np_a, axis=1)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)
