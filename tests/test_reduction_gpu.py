"""Tests verifying GPU-accelerated reduction operations."""
import numpy as np
import numpy.testing as npt
import pytest
import macmetalpy as mp

class TestReductionGPU:
    def test_std_ddof1(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.std(gpu_a, ddof=1)
        expected = np.std(np_a, ddof=1)
        npt.assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_std_ddof1_axis(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.std(gpu_a, axis=1, ddof=1)
        expected = np.std(np_a, axis=1, ddof=1)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_var_ddof1(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.var(gpu_a, ddof=1)
        expected = np.var(np_a, ddof=1)
        npt.assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_ptp(self):
        np_a = np.array([1, 5, 3, 2, 4], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.ptp(gpu_a)
        expected = np.ptp(np_a)
        npt.assert_allclose(float(result.get()), float(expected))

    def test_ptp_axis(self):
        np_a = np.array([[1, 5, 3], [2, 4, 6]], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.ptp(gpu_a, axis=1)
        expected = np.ptp(np_a, axis=1)
        npt.assert_allclose(result.get(), expected)

    def test_diff_1d(self):
        np_a = np.array([1, 3, 6, 10], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.diff(gpu_a)
        expected = np.diff(np_a)
        npt.assert_allclose(result.get(), expected)

    def test_diff_2d(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        gpu_a = mp.array(np_a)
        result = mp.diff(gpu_a, axis=0)
        expected = np.diff(np_a, axis=0)
        npt.assert_allclose(result.get(), expected)

    def test_divmod_gpu(self):
        np_a = np.array([7, 8, 9], dtype=np.float32)
        np_b = np.array([3, 3, 3], dtype=np.float32)
        gpu_a, gpu_b = mp.array(np_a), mp.array(np_b)
        q, r = mp.divmod(gpu_a, gpu_b)
        eq, er = np.divmod(np_a, np_b)
        npt.assert_allclose(q.get(), eq, rtol=1e-5)
        npt.assert_allclose(r.get(), er, rtol=1e-5)

    def test_apply_where_sum(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mask = np.array([True, False, True, False, True])
        gpu_a = mp.array(np_a)
        gpu_mask = mp.array(mask)
        result = mp.sum(gpu_a, where=gpu_mask)
        expected = np.sum(np_a, where=mask)
        npt.assert_allclose(float(result.get()), float(expected))
