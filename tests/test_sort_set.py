"""Tests for extended sorting functions and set operations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp
from macmetalpy.sorting import (
    lexsort, partition, argpartition, msort, sort_complex,
)
from macmetalpy.set_ops import (
    union1d, intersect1d, setdiff1d, setxor1d, in1d, isin,
)


# ------------------------------------------------------------------ Sorting


class TestLexsort:
    def test_lexsort_two_keys_int32(self):
        k1 = cp.array(np.array([1, 2, 1, 2], dtype=np.int32))
        k2 = cp.array(np.array([10, 20, 20, 10], dtype=np.int32))
        result = lexsort((k1, k2))
        expected = np.lexsort((np.array([1, 2, 1, 2], dtype=np.int32),
                               np.array([10, 20, 20, 10], dtype=np.int32)))
        assert_array_equal(result.get(), expected)

    def test_lexsort_two_keys_float32(self):
        k1 = cp.array(np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32))
        k2 = cp.array(np.array([10.0, 20.0, 20.0, 10.0], dtype=np.float32))
        result = lexsort((k1, k2))
        expected = np.lexsort((np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32),
                               np.array([10.0, 20.0, 20.0, 10.0], dtype=np.float32)))
        assert_array_equal(result.get(), expected)

    def test_lexsort_single_key(self):
        k = cp.array(np.array([3, 1, 2], dtype=np.int32))
        result = lexsort((k,))
        expected = np.lexsort((np.array([3, 1, 2], dtype=np.int32),))
        assert_array_equal(result.get(), expected)

    def test_lexsort_from_lists(self):
        result = lexsort(([1, 2, 1], [10, 20, 10]))
        expected = np.lexsort(([1, 2, 1], [10, 20, 10]))
        assert_array_equal(result.get(), expected)


class TestPartition:
    def test_partition_1d_int32(self):
        np_arr = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32)
        gpu = cp.array(np_arr)
        result = partition(gpu, 3)
        expected = np.partition(np_arr, 3)
        # After partition, element at kth position equals the sorted element
        assert result.get()[3] == expected[3]

    def test_partition_1d_float32(self):
        np_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = partition(gpu, 2)
        expected = np.partition(np_arr, 2)
        assert result.get()[2] == expected[2]

    def test_partition_2d(self):
        np_arr = np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int32)
        gpu = cp.array(np_arr)
        result = partition(gpu, 1, axis=1)
        expected = np.partition(np_arr, 1, axis=1)
        assert_array_equal(result.get()[:, 1], expected[:, 1])

    def test_partition_from_list(self):
        result = partition([5.0, 3.0, 1.0, 4.0, 2.0], 2)
        expected = np.partition(np.array([5.0, 3.0, 1.0, 4.0, 2.0]), 2)
        assert result.get()[2] == expected[2]


class TestArgpartition:
    def test_argpartition_1d_int32(self):
        np_arr = np.array([3, 1, 4, 1, 5, 9], dtype=np.int32)
        gpu = cp.array(np_arr)
        result = argpartition(gpu, 2)
        expected = np.argpartition(np_arr, 2)
        # The element at position kth should be the same as sorted
        assert np_arr[result.get()[2]] == np_arr[expected[2]]

    def test_argpartition_1d_float32(self):
        np_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = argpartition(gpu, 2)
        expected = np.argpartition(np_arr, 2)
        assert np_arr[result.get()[2]] == np_arr[expected[2]]

    def test_argpartition_2d(self):
        np_arr = np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int32)
        gpu = cp.array(np_arr)
        result = argpartition(gpu, 1, axis=1)
        expected = np.argpartition(np_arr, 1, axis=1)
        # kth element indices should point to same values
        for row in range(np_arr.shape[0]):
            assert np_arr[row, result.get()[row, 1]] == np_arr[row, expected[row, 1]]

    def test_argpartition_from_list(self):
        result = argpartition([5.0, 3.0, 1.0, 4.0, 2.0], 2)
        arr = np.array([5.0, 3.0, 1.0, 4.0, 2.0])
        expected = np.argpartition(arr, 2)
        assert arr[result.get()[2]] == arr[expected[2]]


class TestMsort:
    def test_msort_1d_int32(self):
        np_arr = np.array([3, 1, 4, 1, 5], dtype=np.int32)
        gpu = cp.array(np_arr)
        result = msort(gpu)
        expected = np.sort(np_arr, axis=0)
        assert_array_equal(result.get(), expected)

    def test_msort_1d_float32(self):
        np_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = msort(gpu)
        expected = np.sort(np_arr, axis=0)
        assert_array_equal(result.get(), expected)

    def test_msort_2d(self):
        np_arr = np.array([[3, 1], [2, 4]], dtype=np.int32)
        gpu = cp.array(np_arr)
        result = msort(gpu)
        expected = np.sort(np_arr, axis=0)
        assert_array_equal(result.get(), expected)

    def test_msort_from_list(self):
        result = msort([5.0, 3.0, 1.0, 4.0, 2.0])
        expected = np.sort(np.array([5.0, 3.0, 1.0, 4.0, 2.0]), axis=0)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestSortComplex:
    def test_sort_complex_real_int32(self):
        np_arr = np.array([3, 1, 4, 1, 5], dtype=np.int32)
        gpu = cp.array(np_arr)
        result = sort_complex(gpu)
        # Metal lacks complex support; sort_complex returns sorted real parts as float32
        expected = np.sort_complex(np_arr).real.astype(np.float32)
        assert_array_equal(result.get(), expected)

    def test_sort_complex_real_float32(self):
        np_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = sort_complex(gpu)
        expected = np.sort_complex(np_arr).real.astype(np.float32)
        assert_array_equal(result.get(), expected)

    def test_sort_complex_preserves_order(self):
        # Verify the sort order matches numpy's sort_complex on real values
        np_arr = np.array([5.0, 2.0, 8.0, 1.0, 3.0], dtype=np.float32)
        gpu = cp.array(np_arr)
        result = sort_complex(gpu)
        expected = np.sort(np_arr)
        assert_array_equal(result.get(), expected)

    def test_sort_complex_from_list(self):
        result = sort_complex([3.0, 1.0, 2.0])
        expected = np.sort_complex(np.array([3.0, 1.0, 2.0])).real.astype(np.float32)
        assert_array_equal(result.get(), expected)


# ------------------------------------------------------------------ Set operations


class TestUnion1d:
    def test_union1d_int32(self):
        a = cp.array(np.array([1, 2, 3], dtype=np.int32))
        b = cp.array(np.array([3, 4, 5], dtype=np.int32))
        result = union1d(a, b)
        expected = np.union1d(np.array([1, 2, 3], dtype=np.int32),
                              np.array([3, 4, 5], dtype=np.int32))
        assert_array_equal(result.get(), expected)

    def test_union1d_float32(self):
        a = cp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = cp.array(np.array([3.0, 4.0, 5.0], dtype=np.float32))
        result = union1d(a, b)
        expected = np.union1d(np.array([1.0, 2.0, 3.0], dtype=np.float32),
                              np.array([3.0, 4.0, 5.0], dtype=np.float32))
        assert_array_equal(result.get(), expected)

    def test_union1d_no_overlap(self):
        a = cp.array(np.array([1, 2], dtype=np.int32))
        b = cp.array(np.array([3, 4], dtype=np.int32))
        result = union1d(a, b)
        expected = np.union1d(np.array([1, 2]), np.array([3, 4]))
        assert_array_equal(result.get(), expected)

    def test_union1d_from_list(self):
        result = union1d([1, 2, 3], [3, 4, 5])
        expected = np.union1d([1, 2, 3], [3, 4, 5])
        assert_array_equal(result.get(), expected)


class TestIntersect1d:
    def test_intersect1d_int32(self):
        a = cp.array(np.array([1, 2, 3, 4], dtype=np.int32))
        b = cp.array(np.array([3, 4, 5, 6], dtype=np.int32))
        result = intersect1d(a, b)
        expected = np.intersect1d(np.array([1, 2, 3, 4], dtype=np.int32),
                                  np.array([3, 4, 5, 6], dtype=np.int32))
        assert_array_equal(result.get(), expected)

    def test_intersect1d_float32(self):
        a = cp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = cp.array(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        result = intersect1d(a, b)
        expected = np.intersect1d(np.array([1.0, 2.0, 3.0], dtype=np.float32),
                                  np.array([2.0, 3.0, 4.0], dtype=np.float32))
        assert_array_equal(result.get(), expected)

    def test_intersect1d_return_indices(self):
        a = cp.array(np.array([1, 2, 3, 4], dtype=np.int32))
        b = cp.array(np.array([3, 4, 5, 6], dtype=np.int32))
        result, idx1, idx2 = intersect1d(a, b, return_indices=True)
        exp_r, exp_i1, exp_i2 = np.intersect1d(
            np.array([1, 2, 3, 4], dtype=np.int32),
            np.array([3, 4, 5, 6], dtype=np.int32),
            return_indices=True)
        assert_array_equal(result.get(), exp_r)
        assert_array_equal(idx1.get(), exp_i1)
        assert_array_equal(idx2.get(), exp_i2)

    def test_intersect1d_no_overlap(self):
        a = cp.array(np.array([1, 2], dtype=np.int32))
        b = cp.array(np.array([3, 4], dtype=np.int32))
        result = intersect1d(a, b)
        assert len(result.get()) == 0


class TestSetdiff1d:
    def test_setdiff1d_int32(self):
        a = cp.array(np.array([1, 2, 3, 4], dtype=np.int32))
        b = cp.array(np.array([3, 4, 5], dtype=np.int32))
        result = setdiff1d(a, b)
        expected = np.setdiff1d(np.array([1, 2, 3, 4], dtype=np.int32),
                                np.array([3, 4, 5], dtype=np.int32))
        assert_array_equal(result.get(), expected)

    def test_setdiff1d_float32(self):
        a = cp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = cp.array(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        result = setdiff1d(a, b)
        expected = np.setdiff1d(np.array([1.0, 2.0, 3.0], dtype=np.float32),
                                np.array([2.0, 3.0, 4.0], dtype=np.float32))
        assert_array_equal(result.get(), expected)

    def test_setdiff1d_all_common(self):
        a = cp.array(np.array([1, 2, 3], dtype=np.int32))
        b = cp.array(np.array([1, 2, 3], dtype=np.int32))
        result = setdiff1d(a, b)
        assert len(result.get()) == 0

    def test_setdiff1d_from_list(self):
        result = setdiff1d([1, 2, 3, 4], [3, 4, 5])
        expected = np.setdiff1d([1, 2, 3, 4], [3, 4, 5])
        assert_array_equal(result.get(), expected)


class TestSetxor1d:
    def test_setxor1d_int32(self):
        a = cp.array(np.array([1, 2, 3], dtype=np.int32))
        b = cp.array(np.array([3, 4, 5], dtype=np.int32))
        result = setxor1d(a, b)
        expected = np.setxor1d(np.array([1, 2, 3], dtype=np.int32),
                               np.array([3, 4, 5], dtype=np.int32))
        assert_array_equal(result.get(), expected)

    def test_setxor1d_float32(self):
        a = cp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = cp.array(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        result = setxor1d(a, b)
        expected = np.setxor1d(np.array([1.0, 2.0, 3.0], dtype=np.float32),
                               np.array([2.0, 3.0, 4.0], dtype=np.float32))
        assert_array_equal(result.get(), expected)

    def test_setxor1d_no_overlap(self):
        a = cp.array(np.array([1, 2], dtype=np.int32))
        b = cp.array(np.array([3, 4], dtype=np.int32))
        result = setxor1d(a, b)
        expected = np.setxor1d(np.array([1, 2]), np.array([3, 4]))
        assert_array_equal(result.get(), expected)

    def test_setxor1d_from_list(self):
        result = setxor1d([1, 2, 3], [3, 4, 5])
        expected = np.setxor1d([1, 2, 3], [3, 4, 5])
        assert_array_equal(result.get(), expected)


class TestIn1d:
    def test_in1d_int32(self):
        a = cp.array(np.array([1, 2, 3, 4, 5], dtype=np.int32))
        b = cp.array(np.array([2, 4], dtype=np.int32))
        result = in1d(a, b)
        expected = np.in1d(np.array([1, 2, 3, 4, 5], dtype=np.int32),
                           np.array([2, 4], dtype=np.int32))
        assert_array_equal(result.get(), expected)

    def test_in1d_float32(self):
        a = cp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = cp.array(np.array([2.0, 3.0], dtype=np.float32))
        result = in1d(a, b)
        expected = np.in1d(np.array([1.0, 2.0, 3.0], dtype=np.float32),
                           np.array([2.0, 3.0], dtype=np.float32))
        assert_array_equal(result.get(), expected)

    def test_in1d_invert(self):
        a = cp.array(np.array([1, 2, 3, 4, 5], dtype=np.int32))
        b = cp.array(np.array([2, 4], dtype=np.int32))
        result = in1d(a, b, invert=True)
        expected = np.in1d(np.array([1, 2, 3, 4, 5], dtype=np.int32),
                           np.array([2, 4], dtype=np.int32), invert=True)
        assert_array_equal(result.get(), expected)

    def test_in1d_from_list(self):
        result = in1d([1, 2, 3, 4], [2, 4])
        expected = np.in1d([1, 2, 3, 4], [2, 4])
        assert_array_equal(result.get(), expected)


class TestIsin:
    def test_isin_int32(self):
        a = cp.array(np.array([1, 2, 3, 4, 5], dtype=np.int32))
        b = cp.array(np.array([2, 4], dtype=np.int32))
        result = isin(a, b)
        expected = np.isin(np.array([1, 2, 3, 4, 5], dtype=np.int32),
                           np.array([2, 4], dtype=np.int32))
        assert_array_equal(result.get(), expected)

    def test_isin_float32(self):
        a = cp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = cp.array(np.array([2.0, 3.0], dtype=np.float32))
        result = isin(a, b)
        expected = np.isin(np.array([1.0, 2.0, 3.0], dtype=np.float32),
                           np.array([2.0, 3.0], dtype=np.float32))
        assert_array_equal(result.get(), expected)

    def test_isin_invert(self):
        a = cp.array(np.array([1, 2, 3, 4, 5], dtype=np.int32))
        b = cp.array(np.array([2, 4], dtype=np.int32))
        result = isin(a, b, invert=True)
        expected = np.isin(np.array([1, 2, 3, 4, 5], dtype=np.int32),
                           np.array([2, 4], dtype=np.int32), invert=True)
        assert_array_equal(result.get(), expected)

    def test_isin_from_list(self):
        result = isin([1, 2, 3, 4], [2, 4])
        expected = np.isin([1, 2, 3, 4], [2, 4])
        assert_array_equal(result.get(), expected)

    def test_isin_2d(self):
        a = cp.array(np.array([[1, 2], [3, 4]], dtype=np.int32))
        b = cp.array(np.array([2, 4], dtype=np.int32))
        result = isin(a, b)
        expected = np.isin(np.array([[1, 2], [3, 4]], dtype=np.int32),
                           np.array([2, 4], dtype=np.int32))
        assert_array_equal(result.get(), expected)
