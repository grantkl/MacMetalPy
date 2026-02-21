"""Tests for indexing routines -- CuPy-parity parametrized suite.

Ref: cupy_tests/indexing_tests/
~696 parametrized cases covering: take, take_along_axis, put,
put_along_axis, putmask, place, choose, compress, select, extract,
diag_indices, diag_indices_from, tril_indices, tril_indices_from,
triu_indices, triu_indices_from, ravel_multi_index, unravel_index,
fill_diagonal, nonzero, flatnonzero, argwhere, ix_.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import macmetalpy as cp
from conftest import (
    NUMERIC_DTYPES, FLOAT_DTYPES,
    assert_eq, make_arg,
)


# ── Shape groups for indexing ──────────────────────────────────────
INDEX_SHAPES = [(5,), (2, 3), (2, 3, 4)]
INDEX_SHAPES_SMALL = [(5,), (2, 3)]


# ====================================================================
# take
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_generate.py

class TestTake:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", INDEX_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        indices = np.array([0, 1])
        a_cp = cp.array(a_np)
        assert_eq(cp.take(a_cp, indices), np.take(a_np, indices), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis0(self, dtype):
        a_np = make_arg((3, 4), dtype)
        a_cp = cp.array(a_np)
        indices = np.array([0, 2])
        assert_eq(cp.take(a_cp, indices, axis=0),
                  np.take(a_np, indices, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis1(self, dtype):
        a_np = make_arg((3, 4), dtype)
        a_cp = cp.array(a_np)
        indices = np.array([1, 3])
        assert_eq(cp.take(a_cp, indices, axis=1),
                  np.take(a_np, indices, axis=1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_negative_idx(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        indices = np.array([-1, -2])
        assert_eq(cp.take(a_cp, indices), np.take(a_np, indices), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_ndarray_indices(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        idx = cp.array(np.array([0, 2, 4]))
        assert_eq(cp.take(a_cp, idx), np.take(a_np, [0, 2, 4]), dtype=dtype)


# ====================================================================
# take_along_axis
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_generate.py

class TestTakeAlongAxis:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis0(self, dtype):
        a_np = make_arg((3, 4), dtype)
        idx_np = np.array([[0], [2], [1]])
        a_cp = cp.array(a_np)
        idx_cp = cp.array(idx_np)
        assert_eq(cp.take_along_axis(a_cp, idx_cp, axis=1),
                  np.take_along_axis(a_np, idx_np, axis=1), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_axis1(self, dtype):
        a_np = make_arg((3, 4), dtype)
        idx_np = np.array([[0, 1, 2, 0]])  # indices must be in [0, 3) for axis=0
        a_cp = cp.array(a_np)
        idx_cp = cp.array(idx_np)
        assert_eq(cp.take_along_axis(a_cp, idx_cp, axis=0),
                  np.take_along_axis(a_np, idx_np, axis=0), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_argmax_usage(self, dtype):
        """Common pattern: use argmax indices with take_along_axis."""
        a_np = make_arg((3, 4), dtype)
        a_cp = cp.array(a_np)
        idx_np = np.argmax(a_np, axis=1, keepdims=True)
        idx_cp = cp.array(idx_np)
        assert_eq(cp.take_along_axis(a_cp, idx_cp, axis=1),
                  np.take_along_axis(a_np, idx_np, axis=1), dtype=dtype)


# ====================================================================
# put / put_along_axis
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_insert.py

class TestPut:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((5,), dtype).copy()
        a_cp = cp.array(a_np.copy())
        cp.put(a_cp, [0, 2], [-10, -20])
        np.put(a_np, [0, 2], [-10, -20])
        assert_eq(a_cp, a_np, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_returns_none(self, dtype):
        a_cp = cp.array(make_arg((5,), dtype))
        result = cp.put(a_cp, [0], [99])
        assert result is None

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_ndarray_indices(self, dtype):
        a_np = make_arg((5,), dtype).copy()
        a_cp = cp.array(a_np.copy())
        ind = cp.array(np.array([1, 3]))
        cp.put(a_cp, ind, [10, 30])
        np.put(a_np, [1, 3], [10, 30])
        assert_eq(a_cp, a_np, dtype=dtype)


class TestPutAlongAxis:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((3, 4), dtype).copy()
        idx_np = np.array([[0], [2], [1]])
        vals_np = np.array([[-99], [-99], [-99]], dtype=dtype)
        a_cp = cp.array(a_np.copy())
        idx_cp = cp.array(idx_np)
        vals_cp = cp.array(vals_np)
        cp.put_along_axis(a_cp, idx_cp, vals_cp, axis=1)
        np.put_along_axis(a_np, idx_np, vals_np, axis=1)
        assert_eq(a_cp, a_np, dtype=dtype)


# ====================================================================
# putmask / place
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_insert.py

class TestPutmask:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((6,), dtype).copy()
        mask = np.array([True, False, True, False, True, False])
        a_cp = cp.array(a_np.copy())
        mask_cp = cp.array(mask)
        cp.putmask(a_cp, mask_cp, [-1, -2, -3])
        np.putmask(a_np, mask, [-1, -2, -3])
        assert_eq(a_cp, a_np, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_returns_none(self, dtype):
        a_cp = cp.array(make_arg((3,), dtype))
        result = cp.putmask(a_cp, cp.array(np.array([True, False, True])), [0])
        assert result is None

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        a_np = make_arg((2, 3), dtype).copy()
        mask = a_np > 3
        a_cp = cp.array(a_np.copy())
        mask_cp = cp.array(mask)
        cp.putmask(a_cp, mask_cp, [-1])
        np.putmask(a_np, mask, [-1])
        assert_eq(a_cp, a_np, dtype=dtype)


class TestPlace:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((6,), dtype).copy()
        mask = np.array([True, False, True, False, True, False])
        a_cp = cp.array(a_np.copy())
        mask_cp = cp.array(mask)
        cp.place(a_cp, mask_cp, [-1, -2, -3])
        np.place(a_np, mask, [-1, -2, -3])
        assert_eq(a_cp, a_np, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_returns_none(self, dtype):
        a_cp = cp.array(make_arg((3,), dtype))
        result = cp.place(a_cp, cp.array(np.array([True, False, True])), [10, 20])
        assert result is None


# ====================================================================
# choose / compress / select / extract
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_generate.py

class TestChoose:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        idx_np = np.array([0, 1, 2, 1])
        c0 = make_arg((4,), dtype)
        c1 = make_arg((4,), dtype) + 10
        c2 = make_arg((4,), dtype) + 20
        choices_np = [c0, c1, c2]
        choices_cp = [cp.array(c) for c in choices_np]
        idx_cp = cp.array(idx_np)
        result = cp.choose(idx_cp, choices_cp)
        expected = np.choose(idx_np, choices_np)
        assert_eq(result, expected, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_two_choices(self, dtype):
        idx_np = np.array([0, 1, 0, 1])
        c0 = make_arg((4,), dtype)
        c1 = make_arg((4,), dtype) + 10
        choices_cp = [cp.array(c0), cp.array(c1)]
        result = cp.choose(cp.array(idx_np), choices_cp)
        expected = np.choose(idx_np, [c0, c1])
        assert_eq(result, expected, dtype=dtype)


class TestCompress:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((6,), dtype)
        cond = np.array([True, False, True, False, True, False])
        a_cp = cp.array(a_np)
        assert_eq(cp.compress(cond, a_cp), np.compress(cond, a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_with_axis(self, dtype):
        a_np = make_arg((3, 4), dtype)
        cond = np.array([True, False, True])
        a_cp = cp.array(a_np)
        assert_eq(cp.compress(cond, a_cp, axis=0),
                  np.compress(cond, a_np, axis=0), dtype=dtype)


class TestSelect:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((10,), dtype)
        condlist = [a_np < 3, a_np > 7]
        c0 = np.full(10, -1, dtype=dtype)
        c1 = np.full(10, 1, dtype=dtype)
        condlist_cp = [cp.array(c) for c in condlist]
        choicelist_cp = [cp.array(c0), cp.array(c1)]
        result = cp.select(condlist_cp, choicelist_cp, default=0)
        expected = np.select(condlist, [c0, c1], default=0)
        assert_eq(result, expected, dtype=dtype)


class TestExtract:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_basic(self, dtype):
        a_np = make_arg((6,), dtype)
        cond = np.array([True, False, True, False, True, False])
        a_cp = cp.array(a_np)
        cond_cp = cp.array(cond)
        assert_eq(cp.extract(cond_cp, a_cp),
                  np.extract(cond, a_np), dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_2d(self, dtype):
        a_np = make_arg((2, 3), dtype)
        cond = a_np > 3
        a_cp = cp.array(a_np)
        cond_cp = cp.array(cond)
        assert_eq(cp.extract(cond_cp, a_cp),
                  np.extract(cond, a_np), dtype=dtype)


# ====================================================================
# diag_indices / tril_indices / triu_indices + _from
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_indexing.py

class TestDiagIndices:
    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_basic(self, n):
        result = cp.diag_indices(n)
        expected = np.diag_indices(n)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_ndim3(self):
        result = cp.diag_indices(3, ndim=3)
        expected = np.diag_indices(3, ndim=3)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestDiagIndicesFrom:
    @pytest.mark.parametrize("n", [3, 4])
    def test_basic(self, n):
        a_np = np.arange(n * n, dtype=np.float32).reshape(n, n)
        a_cp = cp.array(a_np)
        result = cp.diag_indices_from(a_cp)
        expected = np.diag_indices_from(a_np)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestTrilIndices:
    @pytest.mark.parametrize("n", [3, 4])
    def test_basic(self, n):
        result = cp.tril_indices(n)
        expected = np.tril_indices(n)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_with_k(self, k):
        result = cp.tril_indices(4, k=k)
        expected = np.tril_indices(4, k=k)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_with_m(self):
        result = cp.tril_indices(3, m=4)
        expected = np.tril_indices(3, m=4)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestTrilIndicesFrom:
    @pytest.mark.parametrize("n", [3, 4])
    def test_basic(self, n):
        a_np = np.zeros((n, n), dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.tril_indices_from(a_cp)
        expected = np.tril_indices_from(a_np)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_with_k(self, k):
        a_np = np.zeros((4, 4), dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.tril_indices_from(a_cp, k=k)
        expected = np.tril_indices_from(a_np, k=k)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestTriuIndices:
    @pytest.mark.parametrize("n", [3, 4])
    def test_basic(self, n):
        result = cp.triu_indices(n)
        expected = np.triu_indices(n)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_with_k(self, k):
        result = cp.triu_indices(4, k=k)
        expected = np.triu_indices(4, k=k)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_with_m(self):
        result = cp.triu_indices(3, m=5)
        expected = np.triu_indices(3, m=5)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestTriuIndicesFrom:
    @pytest.mark.parametrize("n", [3, 4])
    def test_basic(self, n):
        a_np = np.zeros((n, n), dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.triu_indices_from(a_cp)
        expected = np.triu_indices_from(a_np)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_with_k(self, k):
        a_np = np.zeros((4, 4), dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.triu_indices_from(a_cp, k=k)
        expected = np.triu_indices_from(a_np, k=k)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


# ====================================================================
# ravel_multi_index / unravel_index
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_indexing.py

class TestRavelMultiIndex:
    def test_2d(self):
        multi = [cp.array(np.array([0, 1, 2])), cp.array(np.array([1, 2, 0]))]
        result = cp.ravel_multi_index(multi, (3, 3))
        expected = np.ravel_multi_index([[0, 1, 2], [1, 2, 0]], (3, 3))
        assert_array_equal(result.get(), expected)

    def test_3d(self):
        multi = [
            cp.array(np.array([0, 1])),
            cp.array(np.array([1, 0])),
            cp.array(np.array([2, 1])),
        ]
        result = cp.ravel_multi_index(multi, (2, 3, 4))
        expected = np.ravel_multi_index([[0, 1], [1, 0], [2, 1]], (2, 3, 4))
        assert_array_equal(result.get(), expected)

    def test_plain_lists(self):
        result = cp.ravel_multi_index([np.array([0, 1]), np.array([2, 0])], (3, 3))
        expected = np.ravel_multi_index([[0, 1], [2, 0]], (3, 3))
        assert_array_equal(result.get(), expected)


class TestUnravelIndex:
    def test_2d(self):
        indices = cp.array(np.array([1, 5, 8]))
        result = cp.unravel_index(indices, (3, 3))
        expected = np.unravel_index([1, 5, 8], (3, 3))
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_3d(self):
        indices = cp.array(np.array([0, 5, 23]))
        result = cp.unravel_index(indices, (2, 3, 4))
        expected = np.unravel_index([0, 5, 23], (2, 3, 4))
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_plain_indices(self):
        result = cp.unravel_index(np.array([2, 7]), (3, 4))
        expected = np.unravel_index([2, 7], (3, 4))
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


# ====================================================================
# fill_diagonal
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_indexing.py

class TestFillDiagonal:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_square(self, dtype):
        a_np = np.zeros((3, 3), dtype=dtype)
        a_cp = cp.array(a_np.copy())
        cp.fill_diagonal(a_cp, 5)
        np.fill_diagonal(a_np, 5)
        assert_eq(a_cp, a_np, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_rectangular(self, dtype):
        a_np = np.zeros((3, 5), dtype=dtype)
        a_cp = cp.array(a_np.copy())
        cp.fill_diagonal(a_cp, 9)
        np.fill_diagonal(a_np, 9)
        assert_eq(a_cp, a_np, dtype=dtype)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_returns_none(self, dtype):
        a_cp = cp.zeros((3, 3), dtype=dtype)
        result = cp.fill_diagonal(a_cp, 1)
        assert result is None


# ====================================================================
# nonzero / flatnonzero / argwhere
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_generate.py

class TestNonzero:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", INDEX_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_np_mod = a_np.copy()
        # Ensure some zeros
        a_np_mod.flat[0] = 0
        a_cp = cp.array(a_np_mod)
        result = cp.nonzero(a_cp)
        expected = np.nonzero(a_np_mod)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_zero(self, dtype):
        a_np = np.zeros(5, dtype=dtype)
        a_cp = cp.array(a_np)
        result = cp.nonzero(a_cp)
        expected = np.nonzero(a_np)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_nonzero(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        result = cp.nonzero(a_cp)
        expected = np.nonzero(a_np)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)


class TestFlatnonzero:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", INDEX_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_np_mod = a_np.copy()
        a_np_mod.flat[0] = 0
        a_cp = cp.array(a_np_mod)
        assert_array_equal(cp.flatnonzero(a_cp).get(),
                           np.flatnonzero(a_np_mod))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_zero(self, dtype):
        a_np = np.zeros(5, dtype=dtype)
        a_cp = cp.array(a_np)
        assert_array_equal(cp.flatnonzero(a_cp).get(), np.flatnonzero(a_np))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_nonzero(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        assert_array_equal(cp.flatnonzero(a_cp).get(), np.flatnonzero(a_np))


class TestArgwhere:
    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    @pytest.mark.parametrize("shape", INDEX_SHAPES)
    def test_basic(self, dtype, shape):
        a_np = make_arg(shape, dtype)
        a_np_mod = a_np.copy()
        a_np_mod.flat[0] = 0
        a_cp = cp.array(a_np_mod)
        assert_array_equal(cp.argwhere(a_cp).get(), np.argwhere(a_np_mod))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_zero(self, dtype):
        a_np = np.zeros(5, dtype=dtype)
        a_cp = cp.array(a_np)
        assert_array_equal(cp.argwhere(a_cp).get(), np.argwhere(a_np))

    @pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
    def test_all_nonzero(self, dtype):
        a_np = make_arg((5,), dtype)
        a_cp = cp.array(a_np)
        assert_array_equal(cp.argwhere(a_cp).get(), np.argwhere(a_np))


# ====================================================================
# ix_
# ====================================================================
# Ref: cupy_tests/indexing_tests/test_generate.py

class TestIx:
    def test_2d(self):
        a = cp.array(np.array([0, 1]))
        b = cp.array(np.array([2, 4]))
        result = cp.ix_(a, b)
        expected = np.ix_([0, 1], [2, 4])
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_3d(self):
        a = cp.array(np.array([0, 1]))
        b = cp.array(np.array([0, 2]))
        c = cp.array(np.array([1, 3]))
        result = cp.ix_(a, b, c)
        expected = np.ix_([0, 1], [0, 2], [1, 3])
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_plain_arrays(self):
        result = cp.ix_(np.array([0, 2]), np.array([1, 3]))
        expected = np.ix_([0, 2], [1, 3])
        for r, e in zip(result, expected):
            assert_array_equal(r.get(), e)

    def test_single_arg(self):
        result = cp.ix_(np.array([0, 1, 2]))
        expected = np.ix_([0, 1, 2])
        assert len(result) == 1
        assert_array_equal(result[0].get(), expected[0])
