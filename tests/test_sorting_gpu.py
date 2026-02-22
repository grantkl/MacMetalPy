"""Tests verifying GPU-accelerated sorting and indexing operations."""
import numpy as np
import numpy.testing as npt
import pytest
import macmetalpy as mp

class TestSortingGPU:
    def test_sort_1d_float(self):
        np_a = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=np.float32)
        result = mp.sort(mp.array(np_a))
        npt.assert_allclose(result.get(), np.sort(np_a))

    def test_sort_1d_int(self):
        np_a = np.array([5, 3, 8, 1, 9, 2], dtype=np.int32)
        result = mp.sort(mp.array(np_a))
        npt.assert_array_equal(result.get(), np.sort(np_a))

    def test_argsort_1d(self):
        np_a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        result = mp.argsort(mp.array(np_a))
        expected = np.argsort(np_a)
        # Verify that applying indices gives sorted result
        sorted_result = np_a[result.get()]
        npt.assert_allclose(sorted_result, np.sort(np_a))

    def test_sort_already_sorted(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        result = mp.sort(mp.array(np_a))
        npt.assert_allclose(result.get(), np_a)

    def test_sort_reverse(self):
        np_a = np.array([5, 4, 3, 2, 1], dtype=np.float32)
        result = mp.sort(mp.array(np_a))
        npt.assert_allclose(result.get(), np.sort(np_a))

    def test_sort_single_element(self):
        np_a = np.array([42], dtype=np.float32)
        result = mp.sort(mp.array(np_a))
        npt.assert_allclose(result.get(), np_a)

class TestIndexingGPU:
    def test_take_1d(self):
        np_a = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        indices = np.array([0, 2, 4], dtype=np.int32)
        result = mp.take(mp.array(np_a), mp.array(indices))
        npt.assert_allclose(result.get(), np.take(np_a, indices))

    def test_take_wrap(self):
        np_a = np.array([10, 20, 30], dtype=np.float32)
        indices = np.array([0, 5, -1], dtype=np.int32)
        result = mp.take(mp.array(np_a), mp.array(indices), mode='wrap')
        npt.assert_allclose(result.get(), np.take(np_a, indices, mode='wrap'))

    def test_fill_diagonal(self):
        np_a = np.zeros((3, 3), dtype=np.float32)
        gpu_a = mp.array(np_a.copy())
        mp.fill_diagonal(gpu_a, 5.0)
        np.fill_diagonal(np_a, 5.0)
        npt.assert_allclose(gpu_a.get(), np_a)

    def test_searchsorted(self):
        sorted_arr = mp.array(np.array([1, 3, 5, 7, 9], dtype=np.float32))
        values = mp.array(np.array([2, 4, 6], dtype=np.float32))
        result = mp.searchsorted(sorted_arr, values)
        expected = np.searchsorted(np.array([1, 3, 5, 7, 9]), np.array([2, 4, 6]))
        npt.assert_array_equal(result.get(), expected)

    def test_flatnonzero(self):
        np_a = np.array([0, 1, 0, 3, 0], dtype=np.float32)
        result = mp.flatnonzero(mp.array(np_a))
        expected = np.flatnonzero(np_a)
        npt.assert_array_equal(result.get(), expected)
