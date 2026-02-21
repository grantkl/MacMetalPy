"""Tests for indexing module functions (take, put, diag_indices, etc.)."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import macmetalpy as cp
from macmetalpy.indexing import (
    take,
    take_along_axis,
    put,
    put_along_axis,
    putmask,
    place,
    choose,
    compress,
    select,
    extract,
    diag_indices,
    diag_indices_from,
    tril_indices,
    tril_indices_from,
    triu_indices,
    triu_indices_from,
    ravel_multi_index,
    unravel_index,
    fill_diagonal,
    nonzero,
    flatnonzero,
    argwhere,
    ix_,
)


# ------------------------------------------------------------------ Take / Put


class TestTake:
    def test_take_1d(self):
        a = cp.array(np.array([4, 3, 5, 7, 6, 8], dtype=np.float32))
        indices = cp.array(np.array([0, 1, 4]))
        result = take(a, indices)
        expected = np.take(np.array([4, 3, 5, 7, 6, 8], dtype=np.float32), [0, 1, 4])
        assert_array_equal(result.get(), expected)

    def test_take_axis(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        a = cp.array(np_a)
        indices = np.array([0, 2])
        result = take(a, indices, axis=1)
        expected = np.take(np_a, indices, axis=1)
        assert_array_equal(result.get(), expected)

    def test_take_plain_indices(self):
        np_a = np.array([10, 20, 30, 40], dtype=np.float32)
        a = cp.array(np_a)
        result = take(a, [1, 3])
        expected = np.take(np_a, [1, 3])
        assert_array_equal(result.get(), expected)


class TestTakeAlongAxis:
    def test_basic(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        np_idx = np.array([[0], [2], [1]])
        a = cp.array(np_a)
        idx = cp.array(np_idx)
        result = take_along_axis(a, idx, axis=1)
        expected = np.take_along_axis(np_a, np_idx, axis=1)
        assert_array_equal(result.get(), expected)


class TestPut:
    def test_put_basic(self):
        np_a = np.arange(5, dtype=np.float32)
        a = cp.array(np_a.copy())
        put(a, [0, 2], [-44, -55])
        np.put(np_a, [0, 2], [-44, -55])
        assert_array_equal(a.get(), np_a)

    def test_put_returns_none(self):
        a = cp.array(np.arange(5, dtype=np.float32))
        result = put(a, [0], [99])
        assert result is None

    def test_put_ndarray_indices(self):
        np_a = np.arange(5, dtype=np.float32)
        a = cp.array(np_a.copy())
        ind = cp.array(np.array([1, 3]))
        put(a, ind, [10, 30])
        np.put(np_a, [1, 3], [10, 30])
        assert_array_equal(a.get(), np_a)


class TestPutAlongAxis:
    def test_basic(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        np_idx = np.array([[0], [2], [1]])
        np_vals = np.array([[-99], [-99], [-99]], dtype=np.float32)
        a = cp.array(np_a.copy())
        idx = cp.array(np_idx)
        vals = cp.array(np_vals)
        put_along_axis(a, idx, vals, axis=1)
        np.put_along_axis(np_a, np_idx, np_vals, axis=1)
        assert_array_equal(a.get(), np_a)


class TestPutmask:
    def test_basic(self):
        np_a = np.arange(6, dtype=np.float32)
        a = cp.array(np_a.copy())
        mask = cp.array(np.array([True, False, True, False, True, False]))
        putmask(a, mask, [-1, -2, -3])
        np.putmask(np_a, [True, False, True, False, True, False], [-1, -2, -3])
        assert_array_equal(a.get(), np_a)

    def test_returns_none(self):
        a = cp.array(np.arange(3, dtype=np.float32))
        result = putmask(a, cp.array(np.array([True, False, True])), [0])
        assert result is None


class TestPlace:
    def test_basic(self):
        np_a = np.arange(6, dtype=np.float32)
        a = cp.array(np_a.copy())
        mask = cp.array(np.array([True, False, True, False, True, False]))
        place(a, mask, [-1, -2, -3])
        np.place(np_a, [True, False, True, False, True, False], [-1, -2, -3])
        assert_array_equal(a.get(), np_a)

    def test_returns_none(self):
        a = cp.array(np.arange(3, dtype=np.float32))
        result = place(a, cp.array(np.array([True, False, True])), [10, 20])
        assert result is None


# ------------------------------------------------------------------ Selection


class TestChoose:
    def test_basic(self):
        np_a = np.array([0, 1, 2, 1])
        choices = [
            cp.array(np.array([0, 1, 2, 3], dtype=np.float32)),
            cp.array(np.array([10, 11, 12, 13], dtype=np.float32)),
            cp.array(np.array([20, 21, 22, 23], dtype=np.float32)),
        ]
        a = cp.array(np_a)
        result = choose(a, choices)
        np_choices = [c.get() for c in choices]
        expected = np.choose(np_a, np_choices)
        assert_array_equal(result.get(), expected)


class TestCompress:
    def test_basic(self):
        np_a = np.arange(6, dtype=np.float32)
        a = cp.array(np_a)
        cond = cp.array(np.array([True, False, True, False, True, False]))
        result = compress(cond, a)
        expected = np.compress([True, False, True, False, True, False], np_a)
        assert_array_equal(result.get(), expected)

    def test_with_axis(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        a = cp.array(np_a)
        cond = np.array([True, False, True])
        result = compress(cond, a, axis=0)
        expected = np.compress(cond, np_a, axis=0)
        assert_array_equal(result.get(), expected)


class TestSelect:
    def test_basic(self):
        np_x = np.arange(10, dtype=np.float32)
        x = cp.array(np_x)
        condlist = [x.get() < 3, x.get() > 5]
        condlist_cp = [cp.array(c) for c in condlist]
        choicelist = [
            cp.array(np.full(10, -1, dtype=np.float32)),
            cp.array(np.full(10, 1, dtype=np.float32)),
        ]
        result = select(condlist_cp, choicelist, default=0)
        expected = np.select(condlist, [c.get() for c in choicelist], default=0)
        assert_array_equal(result.get(), expected)


class TestExtract:
    def test_basic(self):
        np_a = np.arange(6, dtype=np.float32)
        a = cp.array(np_a)
        cond = cp.array(np.array([True, False, True, False, True, False]))
        result = extract(cond, a)
        expected = np.extract([True, False, True, False, True, False], np_a)
        assert_array_equal(result.get(), expected)


# ------------------------------------------------------------------ Index arrays


class TestDiagIndices:
    def test_basic(self):
        result = diag_indices(3)
        expected = np.diag_indices(3)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_ndim3(self):
        result = diag_indices(3, ndim=3)
        expected = np.diag_indices(3, ndim=3)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestDiagIndicesFrom:
    def test_basic(self):
        np_a = np.arange(16, dtype=np.float32).reshape(4, 4)
        a = cp.array(np_a)
        result = diag_indices_from(a)
        expected = np.diag_indices_from(np_a)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestTrilIndices:
    def test_basic(self):
        result = tril_indices(3)
        expected = np.tril_indices(3)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_with_k(self):
        result = tril_indices(4, k=1)
        expected = np.tril_indices(4, k=1)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_with_m(self):
        result = tril_indices(3, m=4)
        expected = np.tril_indices(3, m=4)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestTrilIndicesFrom:
    def test_basic(self):
        np_a = np.arange(16, dtype=np.float32).reshape(4, 4)
        a = cp.array(np_a)
        result = tril_indices_from(a)
        expected = np.tril_indices_from(np_a)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestTriuIndices:
    def test_basic(self):
        result = triu_indices(3)
        expected = np.triu_indices(3)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_with_k(self):
        result = triu_indices(4, k=1)
        expected = np.triu_indices(4, k=1)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_with_m(self):
        result = triu_indices(3, m=5)
        expected = np.triu_indices(3, m=5)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestTriuIndicesFrom:
    def test_basic(self):
        np_a = np.arange(16, dtype=np.float32).reshape(4, 4)
        a = cp.array(np_a)
        result = triu_indices_from(a)
        expected = np.triu_indices_from(np_a)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_with_k(self):
        np_a = np.arange(9, dtype=np.float32).reshape(3, 3)
        a = cp.array(np_a)
        result = triu_indices_from(a, k=1)
        expected = np.triu_indices_from(np_a, k=1)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


# ------------------------------------------------------------------ Conversion


class TestRavelMultiIndex:
    def test_basic(self):
        multi = [cp.array(np.array([0, 1, 2])), cp.array(np.array([1, 2, 0]))]
        result = ravel_multi_index(multi, (3, 3))
        expected = np.ravel_multi_index([[0, 1, 2], [1, 2, 0]], (3, 3))
        assert_array_equal(result.get(), expected)


class TestUnravelIndex:
    def test_basic(self):
        indices = cp.array(np.array([1, 5, 8]))
        result = unravel_index(indices, (3, 3))
        expected = np.unravel_index([1, 5, 8], (3, 3))
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_plain_indices(self):
        result = unravel_index(np.array([2, 7]), (3, 4))
        expected = np.unravel_index([2, 7], (3, 4))
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


# ------------------------------------------------------------------ Fill diagonal


class TestFillDiagonal:
    def test_basic(self):
        np_a = np.zeros((3, 3), dtype=np.float32)
        a = cp.array(np_a.copy())
        fill_diagonal(a, 5.0)
        np.fill_diagonal(np_a, 5.0)
        assert_array_equal(a.get(), np_a)

    def test_returns_none(self):
        a = cp.array(np.zeros((3, 3), dtype=np.float32))
        result = fill_diagonal(a, 1.0)
        assert result is None

    def test_rectangular(self):
        np_a = np.zeros((3, 5), dtype=np.float32)
        a = cp.array(np_a.copy())
        fill_diagonal(a, 9.0)
        np.fill_diagonal(np_a, 9.0)
        assert_array_equal(a.get(), np_a)


# ------------------------------------------------------------------ Search


class TestNonzero:
    def test_basic(self):
        np_a = np.array([0, 1, 0, 3, 0, 5], dtype=np.float32)
        a = cp.array(np_a)
        result = nonzero(a)
        expected = np.nonzero(np_a)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_2d(self):
        np_a = np.array([[0, 1], [2, 0]], dtype=np.float32)
        a = cp.array(np_a)
        result = nonzero(a)
        expected = np.nonzero(np_a)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestFlatnonzero:
    def test_basic(self):
        np_a = np.array([0, 1, 0, 3, 0, 5], dtype=np.float32)
        a = cp.array(np_a)
        result = flatnonzero(a)
        expected = np.flatnonzero(np_a)
        assert_array_equal(result.get(), expected)


class TestArgwhere:
    def test_basic(self):
        np_a = np.array([0, 1, 0, 3, 0, 5], dtype=np.float32)
        a = cp.array(np_a)
        result = argwhere(a)
        expected = np.argwhere(np_a)
        assert_array_equal(result.get(), expected)

    def test_2d(self):
        np_a = np.array([[0, 1], [2, 0]], dtype=np.float32)
        a = cp.array(np_a)
        result = argwhere(a)
        expected = np.argwhere(np_a)
        assert_array_equal(result.get(), expected)


# ------------------------------------------------------------------ Advanced


class TestIx:
    def test_basic(self):
        a = cp.array(np.array([0, 1]))
        b = cp.array(np.array([2, 4]))
        result = ix_(a, b)
        expected = np.ix_([0, 1], [2, 4])
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_plain_arrays(self):
        result = ix_(np.array([0, 2]), np.array([1, 3]))
        expected = np.ix_([0, 2], [1, 3])
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)
