"""Tests verifying GPU-accelerated math operations."""
import numpy as np
import numpy.testing as npt
import pytest
import macmetalpy as mp


class TestMathOpsGPU:
    def test_around_decimals0(self):
        np_a = np.array([1.4, 2.5, 3.6, -1.4, -2.5], dtype=np.float32)
        result = mp.around(mp.array(np_a))
        npt.assert_allclose(result.get(), np.around(np_a), rtol=1e-5)

    def test_around_decimals2(self):
        np_a = np.array([1.456, 2.789, 3.123], dtype=np.float32)
        result = mp.around(mp.array(np_a), decimals=2)
        npt.assert_allclose(result.get(), np.around(np_a, decimals=2), rtol=1e-4)

    def test_nan_to_num(self):
        np_a = np.array([1, np.nan, np.inf, -np.inf, 3], dtype=np.float32)
        result = mp.nan_to_num(mp.array(np_a))
        expected = np.nan_to_num(np_a)
        npt.assert_allclose(result.get(), expected)

    def test_nan_to_num_custom(self):
        np_a = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
        result = mp.nan_to_num(mp.array(np_a), nan=-1.0, posinf=999.0, neginf=-999.0)
        expected = np.nan_to_num(np_a, nan=-1.0, posinf=999.0, neginf=-999.0)
        npt.assert_allclose(result.get(), expected)

    def test_copy(self):
        np_a = np.array([1, 2, 3], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.copy(gpu_a)
        npt.assert_allclose(result.get(), np_a)
        # Verify it's a true copy (independent data)
        if result._buffer is not None or gpu_a._buffer is not None:
            assert result._buffer is not gpu_a._buffer
        else:
            assert result._np_data is not gpu_a._np_data

    def test_count_nonzero(self):
        np_a = np.array([0, 1, 0, 3, 0, 5], dtype=np.float32)
        result = mp.count_nonzero(mp.array(np_a))
        assert result == np.count_nonzero(np_a)

    def test_isclose(self):
        a = mp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = mp.array(np.array([1.0, 2.00001, 3.1], dtype=np.float32))
        result = mp.isclose(a, b)
        expected = np.isclose(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.00001, 3.1]))
        npt.assert_array_equal(result.get(), expected)

    def test_allclose(self):
        a = mp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = mp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert mp.allclose(a, b) == True

    def test_array_equal(self):
        a = mp.array(np.array([1, 2, 3], dtype=np.int32))
        b = mp.array(np.array([1, 2, 3], dtype=np.int32))
        assert mp.array_equal(a, b) == True

    def test_trace_2d(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = mp.trace(mp.array(np_a))
        expected = np.trace(np_a)
        npt.assert_allclose(float(result.get()), float(expected))

    def test_diagonal_2d(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = mp.diagonal(mp.array(np_a))
        expected = np.diagonal(np_a)
        npt.assert_allclose(result.get(), expected)

    def test_concatenate_1d(self):
        a = mp.array(np.array([1, 2, 3], dtype=np.float32))
        b = mp.array(np.array([4, 5, 6], dtype=np.float32))
        result = mp.concatenate([a, b])
        expected = np.concatenate([np.array([1, 2, 3]), np.array([4, 5, 6])])
        npt.assert_allclose(result.get(), expected)
