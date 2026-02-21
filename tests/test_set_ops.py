"""Tests for set operations.

Separated from test_sort_set.py.
Ref: cupy_tests/sorting_tests/test_search.py, NumPy test_setops.py
Target: ~208 parametrized cases.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp

from conftest import NUMERIC_DTYPES


# ======================================================================
# union1d
# ======================================================================
# Ref: cupy_tests/sorting_tests/test_search.py

class TestUnion1d:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([3, 4, 5], dtype=dtype)
        result = cp.union1d(cp.array(a), cp.array(b))
        expected = np.union1d(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_disjoint(self, dtype):
        a = np.array([1, 2], dtype=dtype)
        b = np.array([3, 4], dtype=dtype)
        result = cp.union1d(cp.array(a), cp.array(b))
        expected = np.union1d(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_identical(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        result = cp.union1d(cp.array(a), cp.array(a.copy()))
        expected = np.union1d(a, a)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_empty(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([], dtype=dtype)
        result = cp.union1d(cp.array(a), cp.array(b))
        expected = np.union1d(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_from_list(self, dtype):
        result = cp.union1d([1, 2, 3], [3, 4, 5])
        expected = np.union1d([1, 2, 3], [3, 4, 5])
        npt.assert_array_equal(result.get(), expected)


# ======================================================================
# intersect1d
# ======================================================================

class TestIntersect1d:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a = np.array([1, 2, 3, 4], dtype=dtype)
        b = np.array([3, 4, 5, 6], dtype=dtype)
        result = cp.intersect1d(cp.array(a), cp.array(b))
        expected = np.intersect1d(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_disjoint(self, dtype):
        a = np.array([1, 2], dtype=dtype)
        b = np.array([3, 4], dtype=dtype)
        result = cp.intersect1d(cp.array(a), cp.array(b))
        assert len(result.get()) == 0

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_identical(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        result = cp.intersect1d(cp.array(a), cp.array(a.copy()))
        expected = np.intersect1d(a, a)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_empty(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([], dtype=dtype)
        result = cp.intersect1d(cp.array(a), cp.array(b))
        assert len(result.get()) == 0

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_return_indices(self, dtype):
        a = np.array([1, 2, 3, 4], dtype=dtype)
        b = np.array([3, 4, 5, 6], dtype=dtype)
        result, idx1, idx2 = cp.intersect1d(cp.array(a), cp.array(b), return_indices=True)
        exp_r, exp_i1, exp_i2 = np.intersect1d(a, b, return_indices=True)
        npt.assert_array_equal(result.get(), exp_r)
        npt.assert_array_equal(idx1.get(), exp_i1)
        npt.assert_array_equal(idx2.get(), exp_i2)


# ======================================================================
# setdiff1d
# ======================================================================

class TestSetdiff1d:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a = np.array([1, 2, 3, 4], dtype=dtype)
        b = np.array([3, 4, 5], dtype=dtype)
        result = cp.setdiff1d(cp.array(a), cp.array(b))
        expected = np.setdiff1d(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_disjoint(self, dtype):
        a = np.array([1, 2], dtype=dtype)
        b = np.array([3, 4], dtype=dtype)
        result = cp.setdiff1d(cp.array(a), cp.array(b))
        expected = np.setdiff1d(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_identical(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        result = cp.setdiff1d(cp.array(a), cp.array(a.copy()))
        assert len(result.get()) == 0

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_empty_b(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([], dtype=dtype)
        result = cp.setdiff1d(cp.array(a), cp.array(b))
        expected = np.setdiff1d(a, b)
        npt.assert_array_equal(result.get(), expected)


# ======================================================================
# setxor1d
# ======================================================================

class TestSetxor1d:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([3, 4, 5], dtype=dtype)
        result = cp.setxor1d(cp.array(a), cp.array(b))
        expected = np.setxor1d(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_disjoint(self, dtype):
        a = np.array([1, 2], dtype=dtype)
        b = np.array([3, 4], dtype=dtype)
        result = cp.setxor1d(cp.array(a), cp.array(b))
        expected = np.setxor1d(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_identical(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        result = cp.setxor1d(cp.array(a), cp.array(a.copy()))
        assert len(result.get()) == 0

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_empty(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([], dtype=dtype)
        result = cp.setxor1d(cp.array(a), cp.array(b))
        expected = np.setxor1d(a, b)
        npt.assert_array_equal(result.get(), expected)


# ======================================================================
# in1d
# ======================================================================

class TestIn1d:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a = np.array([1, 2, 3, 4, 5], dtype=dtype)
        b = np.array([2, 4], dtype=dtype)
        result = cp.in1d(cp.array(a), cp.array(b))
        expected = np.in1d(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_invert(self, dtype):
        a = np.array([1, 2, 3, 4, 5], dtype=dtype)
        b = np.array([2, 4], dtype=dtype)
        result = cp.in1d(cp.array(a), cp.array(b), invert=True)
        expected = np.in1d(a, b, invert=True)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_assume_unique(self, dtype):
        a = np.array([1, 2, 3, 4, 5], dtype=dtype)
        b = np.array([2, 4], dtype=dtype)
        result = cp.in1d(cp.array(a), cp.array(b), assume_unique=True)
        expected = np.in1d(a, b, assume_unique=True)
        npt.assert_array_equal(result.get(), expected)


# ======================================================================
# isin
# ======================================================================

class TestIsin:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a = np.array([1, 2, 3, 4, 5], dtype=dtype)
        b = np.array([2, 4], dtype=dtype)
        result = cp.isin(cp.array(a), cp.array(b))
        expected = np.isin(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_invert(self, dtype):
        a = np.array([1, 2, 3, 4, 5], dtype=dtype)
        b = np.array([2, 4], dtype=dtype)
        result = cp.isin(cp.array(a), cp.array(b), invert=True)
        expected = np.isin(a, b, invert=True)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        a = np.array([[1, 2], [3, 4]], dtype=dtype)
        b = np.array([2, 4], dtype=dtype)
        result = cp.isin(cp.array(a), cp.array(b))
        expected = np.isin(a, b)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_assume_unique(self, dtype):
        a = np.array([1, 2, 3, 4, 5], dtype=dtype)
        b = np.array([2, 4], dtype=dtype)
        result = cp.isin(cp.array(a), cp.array(b), assume_unique=True)
        expected = np.isin(a, b, assume_unique=True)
        npt.assert_array_equal(result.get(), expected)
