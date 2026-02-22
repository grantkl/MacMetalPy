"""Tests for sorting functions.

Consolidates sorting parts from test_sort_set.py.
Ref: cupy_tests/sorting_tests/
Target: ~314 parametrized cases.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp

from conftest import NUMERIC_DTYPES, FLOAT_DTYPES, assert_eq, tol_for


SHAPES_SORT = [(5,), (2, 3), (2, 3, 4)]


# ======================================================================
# sort
# ======================================================================
# Ref: cupy_tests/sorting_tests/test_sort.py

class TestSort:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_SORT)
    def test_basic(self, dtype, shape):
        rng = np.random.RandomState(42)
        size = 1
        for s in shape:
            size *= s
        np_arr = rng.randint(0, 100, size=size).astype(dtype).reshape(shape)
        result = cp.sort(cp.array(np_arr))
        expected = np.sort(np_arr)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis_0(self, dtype, shape):
        rng = np.random.RandomState(42)
        size = 1
        for s in shape:
            size *= s
        np_arr = rng.randint(0, 100, size=size).astype(dtype).reshape(shape)
        result = cp.sort(cp.array(np_arr), axis=0)
        expected = np.sort(np_arr, axis=0)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_already_sorted(self, dtype):
        np_arr = np.arange(10, dtype=dtype)
        result = cp.sort(cp.array(np_arr))
        npt.assert_array_equal(result.get(), np_arr)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_reverse(self, dtype):
        np_arr = np.arange(10, dtype=dtype)[::-1].copy()
        result = cp.sort(cp.array(np_arr))
        npt.assert_array_equal(result.get(), np.sort(np_arr))


# ======================================================================
# argsort
# ======================================================================

class TestArgsort:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES_SORT)
    def test_basic(self, dtype, shape):
        rng = np.random.RandomState(42)
        size = 1
        for s in shape:
            size *= s
        np_arr = rng.randint(0, 100, size=size).astype(dtype).reshape(shape)
        result = cp.argsort(cp.array(np_arr))
        expected = np.argsort(np_arr)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
    def test_axis_0(self, dtype, shape):
        rng = np.random.RandomState(42)
        size = 1
        for s in shape:
            size *= s
        np_arr = rng.randint(0, 100, size=size).astype(dtype).reshape(shape)
        result = cp.argsort(cp.array(np_arr), axis=0)
        expected = np.argsort(np_arr, axis=0)
        npt.assert_array_equal(result.get(), expected)


# ======================================================================
# unique
# ======================================================================

class TestUnique:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([3, 1, 2, 1, 3, 2], dtype=dtype)
        result = cp.unique(cp.array(np_arr))
        expected = np.unique(np_arr)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_same(self, dtype):
        np_arr = np.array([5, 5, 5, 5], dtype=dtype)
        result = cp.unique(cp.array(np_arr))
        npt.assert_array_equal(result.get(), np.array([5], dtype=dtype))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_unique(self, dtype):
        np_arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = cp.unique(cp.array(np_arr))
        npt.assert_array_equal(result.get(), np_arr)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_empty(self, dtype):
        np_arr = np.array([], dtype=dtype)
        result = cp.unique(cp.array(np_arr))
        assert result.get().shape == (0,)


# ======================================================================
# searchsorted
# ======================================================================

class TestSearchsorted:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_left(self, dtype):
        a_np = np.array([1, 3, 5, 7], dtype=dtype)
        v_np = np.array([0, 2, 4, 6, 8], dtype=dtype)
        result = cp.searchsorted(cp.array(a_np), cp.array(v_np))
        expected = np.searchsorted(a_np, v_np)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_right(self, dtype):
        a_np = np.array([1, 3, 5, 7], dtype=dtype)
        v_np = np.array([1, 3, 5, 7], dtype=dtype)
        result = cp.searchsorted(cp.array(a_np), cp.array(v_np), side='right')
        expected = np.searchsorted(a_np, v_np, side='right')
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_scalar(self, dtype):
        a_np = np.array([1, 3, 5], dtype=dtype)
        v = np.array(3, dtype=dtype)
        result = cp.searchsorted(cp.array(a_np), cp.array(v))
        expected = np.searchsorted(a_np, v)
        npt.assert_array_equal(result.get(), expected)


# ======================================================================
# lexsort
# ======================================================================

class TestLexsort:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_two_keys(self, dtype):
        k1 = np.array([1, 2, 1, 2], dtype=dtype)
        k2 = np.array([10, 20, 20, 10], dtype=dtype)
        result = cp.lexsort((cp.array(k1), cp.array(k2)))
        expected = np.lexsort((k1, k2))
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_three_keys(self, dtype):
        k1 = np.array([1, 2, 1, 2, 1, 2], dtype=dtype)
        k2 = np.array([10, 10, 20, 20, 10, 20], dtype=dtype)
        k3 = np.array([1, 1, 1, 1, 2, 2], dtype=dtype)
        result = cp.lexsort((cp.array(k1), cp.array(k2), cp.array(k3)))
        expected = np.lexsort((k1, k2, k3))
        npt.assert_array_equal(result.get(), expected)


# ======================================================================
# partition / argpartition
# ======================================================================

class TestPartition:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        result = cp.partition(cp.array(np_arr), 3)
        expected = np.partition(np_arr, 3)
        # Element at kth position should match
        assert result.get()[3] == expected[3]

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_kth_0(self, dtype):
        np_arr = np.array([5, 3, 1, 4, 2], dtype=dtype)
        result = cp.partition(cp.array(np_arr), 0)
        expected = np.partition(np_arr, 0)
        assert result.get()[0] == expected[0]

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d_axis(self, dtype):
        np_arr = np.array([[3, 1, 2], [6, 4, 5]], dtype=dtype)
        result = cp.partition(cp.array(np_arr), 1, axis=1)
        expected = np.partition(np_arr, 1, axis=1)
        npt.assert_array_equal(result.get()[:, 1], expected[:, 1])


class TestArgpartition:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([3, 1, 4, 1, 5, 9], dtype=dtype)
        result = cp.argpartition(cp.array(np_arr), 2)
        expected = np.argpartition(np_arr, 2)
        assert np_arr[result.get()[2]] == np_arr[expected[2]]

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d_axis(self, dtype):
        np_arr = np.array([[3, 1, 2], [6, 4, 5]], dtype=dtype)
        result = cp.argpartition(cp.array(np_arr), 1, axis=1)
        expected = np.argpartition(np_arr, 1, axis=1)
        for row in range(np_arr.shape[0]):
            assert np_arr[row, result.get()[row, 1]] == np_arr[row, expected[row, 1]]


# ======================================================================
# msort
# ======================================================================

class TestMsort:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_1d(self, dtype):
        from macmetalpy.sorting import msort
        np_arr = np.array([3, 1, 4, 1, 5], dtype=dtype)
        result = msort(cp.array(np_arr))
        npt.assert_array_equal(result.get(), np.sort(np_arr, axis=0))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        from macmetalpy.sorting import msort
        np_arr = np.array([[3, 1], [2, 4]], dtype=dtype)
        result = msort(cp.array(np_arr))
        npt.assert_array_equal(result.get(), np.sort(np_arr, axis=0))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_from_list(self, dtype):
        from macmetalpy.sorting import msort
        data = [5, 3, 1, 4, 2]
        result = msort(data)
        expected = np.sort(np.array(data), axis=0)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


# ======================================================================
# sort_complex
# ======================================================================

class TestSortComplex:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        np_arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=dtype)
        result = cp.sort_complex(cp.array(np_arr))
        expected = np.sort_complex(np_arr).real.astype(np.float32)
        npt.assert_array_equal(result.get(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_preserves_order(self, dtype):
        np_arr = np.array([5.0, 2.0, 8.0, 1.0, 3.0], dtype=dtype)
        result = cp.sort_complex(cp.array(np_arr))
        npt.assert_array_equal(result.get(), np.sort(np_arr))
