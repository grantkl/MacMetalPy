"""Tests for untested top-level APIs: reductions, sorting, and indexing.

Covers: amax, amin, ptp, nancumprod, nancumsum, count_nonzero,
sort_complex, take, take_along_axis, put_along_axis, place, putmask,
select, extract, compress, choose, fill_diagonal, diagonal, diagflat, trace.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp


# ====================================================================
# amax / amin
# ====================================================================

class TestAmax:
    def test_1d(self):
        np_a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.amax(cp_a).get(), np.amax(np_a), rtol=1e-5)

    def test_2d_axis0(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.amax(cp_a, axis=0).get(), np.amax(np_a, axis=0), rtol=1e-5)

    def test_2d_axis1_keepdims(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        result = cp.amax(cp_a, axis=1, keepdims=True)
        expected = np.amax(np_a, axis=1, keepdims=True)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)
        assert result.shape == expected.shape


class TestAmin:
    def test_1d(self):
        np_a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.amin(cp_a).get(), np.amin(np_a), rtol=1e-5)

    def test_2d_axis0(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.amin(cp_a, axis=0).get(), np.amin(np_a, axis=0), rtol=1e-5)

    def test_2d_axis1_keepdims(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        result = cp.amin(cp_a, axis=1, keepdims=True)
        expected = np.amin(np_a, axis=1, keepdims=True)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)
        assert result.shape == expected.shape


# ====================================================================
# ptp
# ====================================================================

class TestPtp:
    def test_1d(self):
        np_a = np.array([3.0, 1.0, 7.0, 2.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.ptp(cp_a).get(), np.ptp(np_a), rtol=1e-5)

    def test_2d_axis0(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.ptp(cp_a, axis=0).get(), np.ptp(np_a, axis=0), rtol=1e-5)

    def test_2d_axis1(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.ptp(cp_a, axis=1).get(), np.ptp(np_a, axis=1), rtol=1e-5)


# ====================================================================
# nancumsum / nancumprod
# ====================================================================

class TestNancumsum:
    def test_no_nan(self):
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.nancumsum(cp_a).get(), np.nancumsum(np_a), rtol=1e-5)

    def test_with_nan(self):
        np_a = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.nancumsum(cp_a).get(), np.nancumsum(np_a), rtol=1e-5)

    def test_2d_axis0_with_nan(self):
        np_a = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(
            cp.nancumsum(cp_a, axis=0).get(),
            np.nancumsum(np_a, axis=0),
            rtol=1e-5,
        )


class TestNancumprod:
    def test_no_nan(self):
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.nancumprod(cp_a).get(), np.nancumprod(np_a), rtol=1e-5)

    def test_with_nan(self):
        np_a = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.nancumprod(cp_a).get(), np.nancumprod(np_a), rtol=1e-5)

    def test_2d_axis1_with_nan(self):
        np_a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(
            cp.nancumprod(cp_a, axis=1).get(),
            np.nancumprod(np_a, axis=1),
            rtol=1e-5,
        )


# ====================================================================
# count_nonzero
# ====================================================================

class TestCountNonzero:
    def test_1d(self):
        np_a = np.array([0, 1, 0, 3, 0, 5], dtype=np.float32)
        cp_a = cp.array(np_a)
        assert cp.count_nonzero(cp_a) == np.count_nonzero(np_a)

    def test_2d_axis0(self):
        np_a = np.array([[0, 1, 0], [3, 0, 5]], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_array_equal(
            cp.count_nonzero(cp_a, axis=0).get(),
            np.count_nonzero(np_a, axis=0),
        )

    def test_all_zero(self):
        np_a = np.zeros(5, dtype=np.float32)
        cp_a = cp.array(np_a)
        assert cp.count_nonzero(cp_a) == 0


# ====================================================================
# sort_complex
# ====================================================================

class TestSortComplex:
    def test_real_input(self):
        np_a = np.array([5.0, 2.0, 8.0, 1.0, 3.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        # sort_complex returns real part as float32 for non-complex input
        result = cp.sort_complex(cp_a).get()
        expected = np.sort(np_a)  # real input just gets sorted
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_real_input_negative(self):
        np_a = np.array([-3.0, 1.0, -1.0, 5.0, 0.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        result = cp.sort_complex(cp_a).get()
        expected = np.sort(np_a)
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_integer_input(self):
        np_a = np.array([4, 2, 5, 1, 3], dtype=np.int32)
        cp_a = cp.array(np_a)
        result = cp.sort_complex(cp_a).get()
        expected = np.sort(np_a).astype(np.float32)
        npt.assert_allclose(result, expected, rtol=1e-5)


# ====================================================================
# take
# ====================================================================

class TestTakeTopLevel:
    def test_1d(self):
        np_a = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        cp_a = cp.array(np_a)
        indices = np.array([0, 2, 4])
        npt.assert_array_equal(cp.take(cp_a, indices).get(), np.take(np_a, indices))

    def test_2d_axis0(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        indices = np.array([0, 2])
        npt.assert_array_equal(
            cp.take(cp_a, indices, axis=0).get(),
            np.take(np_a, indices, axis=0),
        )

    def test_2d_axis1(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        indices = np.array([1, 3])
        npt.assert_array_equal(
            cp.take(cp_a, indices, axis=1).get(),
            np.take(np_a, indices, axis=1),
        )


# ====================================================================
# take_along_axis
# ====================================================================

class TestTakeAlongAxis:
    def test_1d(self):
        np_a = np.array([10, 20, 30, 40], dtype=np.float32)
        np_idx = np.array([3, 0, 2, 1], dtype=np.intp)
        cp_a = cp.array(np_a)
        cp_idx = cp.array(np_idx)
        npt.assert_array_equal(
            cp.take_along_axis(cp_a, cp_idx, axis=0).get(),
            np.take_along_axis(np_a, np_idx, axis=0),
        )

    def test_2d_axis1(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        np_idx = np.argsort(np_a, axis=1).astype(np.intp)
        cp_a = cp.array(np_a)
        cp_idx = cp.array(np_idx)
        npt.assert_array_equal(
            cp.take_along_axis(cp_a, cp_idx, axis=1).get(),
            np.take_along_axis(np_a, np_idx, axis=1),
        )

    def test_argmax_gather(self):
        np_a = np.array([[5.0, 1.0, 3.0], [2.0, 8.0, 4.0]], dtype=np.float32)
        np_idx = np.argmax(np_a, axis=1, keepdims=True).astype(np.intp)
        cp_a = cp.array(np_a)
        cp_idx = cp.array(np_idx)
        npt.assert_array_equal(
            cp.take_along_axis(cp_a, cp_idx, axis=1).get(),
            np.take_along_axis(np_a, np_idx, axis=1),
        )


# ====================================================================
# put_along_axis
# ====================================================================

class TestPutAlongAxis:
    def test_2d_axis1(self):
        np_a = np.zeros((2, 4), dtype=np.float32)
        np_idx = np.array([[1, 3], [0, 2]], dtype=np.intp)
        np_vals = np.array([[10, 20], [30, 40]], dtype=np.float32)

        cp_a = cp.array(np_a.copy())
        cp_idx = cp.array(np_idx)
        cp_vals = cp.array(np_vals)

        np.put_along_axis(np_a, np_idx, np_vals, axis=1)
        cp.put_along_axis(cp_a, cp_idx, cp_vals, axis=1)
        npt.assert_array_equal(cp_a.get(), np_a)

    def test_2d_axis0(self):
        np_a = np.zeros((3, 2), dtype=np.float32)
        np_idx = np.array([[0], [2]], dtype=np.intp)
        np_vals = np.array([[5], [7]], dtype=np.float32)

        cp_a = cp.array(np_a.copy())
        cp_idx = cp.array(np_idx)
        cp_vals = cp.array(np_vals)

        np.put_along_axis(np_a, np_idx, np_vals, axis=0)
        cp.put_along_axis(cp_a, cp_idx, cp_vals, axis=0)
        npt.assert_array_equal(cp_a.get(), np_a)


# ====================================================================
# place
# ====================================================================

class TestPlace:
    def test_basic(self):
        np_a = np.arange(6, dtype=np.float32)
        cp_a = cp.array(np_a.copy())
        mask = np.array([False, False, True, True, False, True])
        vals = np.array([10, 20, 30], dtype=np.float32)

        np.place(np_a, mask, vals)
        cp.place(cp_a, mask, vals)
        npt.assert_array_equal(cp_a.get(), np_a)

    def test_cycling_values(self):
        np_a = np.arange(6, dtype=np.float32)
        cp_a = cp.array(np_a.copy())
        mask = np.array([True, True, True, True, True, True])
        vals = np.array([99, 88], dtype=np.float32)

        np.place(np_a, mask, vals)
        cp.place(cp_a, mask, vals)
        npt.assert_array_equal(cp_a.get(), np_a)


# ====================================================================
# putmask
# ====================================================================

class TestPutmask:
    def test_basic(self):
        np_a = np.arange(5, dtype=np.float32)
        cp_a = cp.array(np_a.copy())
        mask = np.array([True, False, True, False, True])
        vals = np.array([10, 20, 30], dtype=np.float32)

        np.putmask(np_a, mask, vals)
        cp.putmask(cp_a, mask, vals)
        npt.assert_array_equal(cp_a.get(), np_a)

    def test_scalar_value(self):
        np_a = np.arange(5, dtype=np.float32)
        cp_a = cp.array(np_a.copy())
        mask = np.array([False, True, False, True, False])

        np.putmask(np_a, mask, 99.0)
        cp.putmask(cp_a, mask, 99.0)
        npt.assert_array_equal(cp_a.get(), np_a)

    def test_all_false(self):
        np_a = np.arange(4, dtype=np.float32)
        cp_a = cp.array(np_a.copy())
        mask = np.array([False, False, False, False])

        np.putmask(np_a, mask, 99.0)
        cp.putmask(cp_a, mask, 99.0)
        npt.assert_array_equal(cp_a.get(), np_a)


# ====================================================================
# select
# ====================================================================

class TestSelect:
    def test_basic(self):
        np_a = np.arange(10, dtype=np.float32)
        cp_a = cp.array(np_a)
        condlist = [np_a < 3, np_a > 7]
        choicelist = [np_a * 10, np_a * 100]

        cp_condlist = [cp_a < 3, cp_a > 7]
        cp_choicelist = [cp_a * 10, cp_a * 100]

        expected = np.select(condlist, choicelist, default=0)
        result = cp.select(cp_condlist, cp_choicelist, default=0)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_with_default(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        cp_a = cp.array(np_a)
        condlist = [np_a < 2, np_a > 4]
        choicelist = [np.full(5, -1, dtype=np.float32), np.full(5, 99, dtype=np.float32)]

        cp_condlist = [cp_a < 2, cp_a > 4]
        cp_choicelist = [cp.array(np.full(5, -1, dtype=np.float32)),
                         cp.array(np.full(5, 99, dtype=np.float32))]

        expected = np.select(condlist, choicelist, default=42)
        result = cp.select(cp_condlist, cp_choicelist, default=42)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


# ====================================================================
# extract
# ====================================================================

class TestExtract:
    def test_basic(self):
        np_a = np.arange(10, dtype=np.float32)
        cp_a = cp.array(np_a)
        cond = np_a > 5
        cp_cond = cp_a > 5
        npt.assert_array_equal(
            cp.extract(cp_cond, cp_a).get(),
            np.extract(cond, np_a),
        )

    def test_none_match(self):
        np_a = np.arange(5, dtype=np.float32)
        cp_a = cp.array(np_a)
        cond = np_a > 100
        cp_cond = cp_a > 100
        result = cp.extract(cp_cond, cp_a)
        expected = np.extract(cond, np_a)
        assert result.shape == expected.shape

    def test_2d(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        cond = np_a % 3 == 0
        cp_cond = cp.array(cond)
        npt.assert_array_equal(
            cp.extract(cp_cond, cp_a).get(),
            np.extract(cond, np_a),
        )


# ====================================================================
# compress
# ====================================================================

class TestCompress:
    def test_1d(self):
        np_a = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        cp_a = cp.array(np_a)
        cond = [True, False, True, False, True]
        npt.assert_array_equal(
            cp.compress(cond, cp_a).get(),
            np.compress(cond, np_a),
        )

    def test_2d_axis0(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        cond = [True, False, True]
        npt.assert_array_equal(
            cp.compress(cond, cp_a, axis=0).get(),
            np.compress(cond, np_a, axis=0),
        )

    def test_2d_axis1(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        cond = [False, True, True, False]
        npt.assert_array_equal(
            cp.compress(cond, cp_a, axis=1).get(),
            np.compress(cond, np_a, axis=1),
        )


# ====================================================================
# choose
# ====================================================================

class TestChoose:
    def test_basic(self):
        choices = [
            np.array([0, 1, 2, 3], dtype=np.float32),
            np.array([10, 11, 12, 13], dtype=np.float32),
            np.array([20, 21, 22, 23], dtype=np.float32),
        ]
        np_idx = np.array([0, 1, 2, 1])
        cp_idx = cp.array(np_idx)
        cp_choices = [cp.array(c) for c in choices]

        npt.assert_array_equal(
            cp.choose(cp_idx, cp_choices).get(),
            np.choose(np_idx, choices),
        )

    def test_2d_index(self):
        choices = [
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            np.array([[10, 11], [12, 13]], dtype=np.float32),
        ]
        np_idx = np.array([[0, 1], [1, 0]])
        cp_idx = cp.array(np_idx)
        cp_choices = [cp.array(c) for c in choices]

        npt.assert_array_equal(
            cp.choose(cp_idx, cp_choices).get(),
            np.choose(np_idx, choices),
        )


# ====================================================================
# fill_diagonal
# ====================================================================

class TestFillDiagonal:
    def test_2d_scalar(self):
        np_a = np.zeros((4, 4), dtype=np.float32)
        cp_a = cp.array(np_a.copy())

        np.fill_diagonal(np_a, 5.0)
        cp.fill_diagonal(cp_a, 5.0)
        npt.assert_array_equal(cp_a.get(), np_a)

    def test_2d_array(self):
        np_a = np.zeros((3, 3), dtype=np.float32)
        cp_a = cp.array(np_a.copy())

        np.fill_diagonal(np_a, [1, 2, 3])
        cp.fill_diagonal(cp_a, [1, 2, 3])
        npt.assert_array_equal(cp_a.get(), np_a)

    def test_nonsquare(self):
        np_a = np.zeros((3, 5), dtype=np.float32)
        cp_a = cp.array(np_a.copy())

        np.fill_diagonal(np_a, 9.0)
        cp.fill_diagonal(cp_a, 9.0)
        npt.assert_array_equal(cp_a.get(), np_a)


# ====================================================================
# diagonal
# ====================================================================

class TestDiagonal:
    def test_2d_main(self):
        np_a = np.arange(16, dtype=np.float32).reshape(4, 4)
        cp_a = cp.array(np_a)
        npt.assert_array_equal(cp.diagonal(cp_a).get(), np.diagonal(np_a))

    def test_2d_offset(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_array_equal(
            cp.diagonal(cp_a, offset=1).get(),
            np.diagonal(np_a, offset=1),
        )

    def test_2d_negative_offset(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_array_equal(
            cp.diagonal(cp_a, offset=-1).get(),
            np.diagonal(np_a, offset=-1),
        )


# ====================================================================
# diagflat
# ====================================================================

class TestDiagflat:
    def test_1d(self):
        np_a = np.array([1, 2, 3], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_array_equal(cp.diagflat(cp_a).get(), np.diagflat(np_a))

    def test_offset(self):
        np_a = np.array([1, 2], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_array_equal(
            cp.diagflat(cp_a, k=1).get(),
            np.diagflat(np_a, k=1),
        )

    def test_negative_offset(self):
        np_a = np.array([10, 20], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_array_equal(
            cp.diagflat(cp_a, k=-1).get(),
            np.diagflat(np_a, k=-1),
        )


# ====================================================================
# trace
# ====================================================================

class TestTrace:
    def test_2d_basic(self):
        np_a = np.arange(16, dtype=np.float32).reshape(4, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.trace(cp_a).get(), np.trace(np_a), rtol=1e-5)

    def test_2d_offset(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(
            cp.trace(cp_a, offset=1).get(),
            np.trace(np_a, offset=1),
            rtol=1e-5,
        )

    def test_nonsquare(self):
        np_a = np.arange(6, dtype=np.float32).reshape(2, 3)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.trace(cp_a).get(), np.trace(np_a), rtol=1e-5)


# ====================================================================
# max / min (top-level, not amax/amin)
# ====================================================================

class TestMax:
    def test_1d(self):
        np_a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.max(cp_a).get(), np.max(np_a), rtol=1e-5)

    def test_2d_axis0(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.max(cp_a, axis=0).get(), np.max(np_a, axis=0), rtol=1e-5)

    def test_keepdims(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        result = cp.max(cp_a, axis=1, keepdims=True)
        expected = np.max(np_a, axis=1, keepdims=True)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)
        assert result.shape == expected.shape


class TestMin:
    def test_1d(self):
        np_a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.min(cp_a).get(), np.min(np_a), rtol=1e-5)

    def test_2d_axis1(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.min(cp_a, axis=1).get(), np.min(np_a, axis=1), rtol=1e-5)

    def test_keepdims(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        result = cp.min(cp_a, axis=0, keepdims=True)
        expected = np.min(np_a, axis=0, keepdims=True)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)
        assert result.shape == expected.shape


# ====================================================================
# any / all
# ====================================================================

class TestAny:
    def test_true(self):
        np_a = np.array([0, 0, 1, 0], dtype=np.float32)
        cp_a = cp.array(np_a)
        assert bool(cp.any(cp_a)) == bool(np.any(np_a))

    def test_false(self):
        np_a = np.zeros(5, dtype=np.float32)
        cp_a = cp.array(np_a)
        assert bool(cp.any(cp_a)) == bool(np.any(np_a))

    def test_2d_axis(self):
        np_a = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_array_equal(
            cp.any(cp_a, axis=0).get().astype(bool),
            np.any(np_a, axis=0),
        )


class TestAll:
    def test_true(self):
        np_a = np.array([1, 2, 3, 4], dtype=np.float32)
        cp_a = cp.array(np_a)
        assert bool(cp.all(cp_a)) == bool(np.all(np_a))

    def test_false(self):
        np_a = np.array([1, 0, 3, 4], dtype=np.float32)
        cp_a = cp.array(np_a)
        assert bool(cp.all(cp_a)) == bool(np.all(np_a))

    def test_2d_axis(self):
        np_a = np.array([[1, 1], [1, 0], [1, 1]], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_array_equal(
            cp.all(cp_a, axis=0).get().astype(bool),
            np.all(np_a, axis=0),
        )


# ====================================================================
# logical_and / logical_or / logical_xor / logical_not
# ====================================================================

class TestLogicalAnd:
    def test_basic(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        npt.assert_array_equal(
            cp.logical_and(cp.array(a), cp.array(b)).get().astype(bool),
            np.logical_and(a, b),
        )

    def test_numeric(self):
        a = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32)
        npt.assert_array_equal(
            cp.logical_and(cp.array(a), cp.array(b)).get().astype(bool),
            np.logical_and(a, b),
        )


class TestLogicalOr:
    def test_basic(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        npt.assert_array_equal(
            cp.logical_or(cp.array(a), cp.array(b)).get().astype(bool),
            np.logical_or(a, b),
        )

    def test_numeric(self):
        a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 2.0, 0.0], dtype=np.float32)
        npt.assert_array_equal(
            cp.logical_or(cp.array(a), cp.array(b)).get().astype(bool),
            np.logical_or(a, b),
        )


class TestLogicalXor:
    def test_basic(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        npt.assert_array_equal(
            cp.logical_xor(cp.array(a), cp.array(b)).get().astype(bool),
            np.logical_xor(a, b),
        )

    def test_numeric(self):
        a = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 2.0, 0.0], dtype=np.float32)
        npt.assert_array_equal(
            cp.logical_xor(cp.array(a), cp.array(b)).get().astype(bool),
            np.logical_xor(a, b),
        )


class TestLogicalNot:
    def test_bool(self):
        a = np.array([True, False, True, False])
        npt.assert_array_equal(
            cp.logical_not(cp.array(a)).get().astype(bool),
            np.logical_not(a),
        )

    def test_numeric(self):
        a = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
        npt.assert_array_equal(
            cp.logical_not(cp.array(a)).get().astype(bool),
            np.logical_not(a),
        )


# ====================================================================
# median
# ====================================================================

class TestMedian:
    def test_1d_odd(self):
        np_a = np.array([7.0, 1.0, 3.0, 5.0, 9.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.median(cp_a).get(), np.median(np_a), rtol=1e-5)

    def test_1d_even(self):
        np_a = np.array([4.0, 2.0, 8.0, 6.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.median(cp_a).get(), np.median(np_a), rtol=1e-5)

    def test_2d_axis0(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.median(cp_a, axis=0).get(), np.median(np_a, axis=0), rtol=1e-5)


# ====================================================================
# average
# ====================================================================

class TestAverage:
    def test_unweighted(self):
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.average(cp_a).get(), np.average(np_a), rtol=1e-5)

    def test_weighted(self):
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np_w = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        cp_w = cp.array(np_w)
        npt.assert_allclose(
            cp.average(cp_a, weights=cp_w).get(),
            np.average(np_a, weights=np_w),
            rtol=1e-5,
        )

    def test_2d_axis1(self):
        np_a = np.arange(6, dtype=np.float32).reshape(2, 3)
        cp_a = cp.array(np_a)
        npt.assert_allclose(
            cp.average(cp_a, axis=1).get(),
            np.average(np_a, axis=1),
            rtol=1e-5,
        )


# ====================================================================
# percentile
# ====================================================================

class TestPercentile:
    def test_50th(self):
        np_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.percentile(cp_a, 50).get(), np.percentile(np_a, 50), rtol=1e-5)

    def test_multiple(self):
        np_a = np.arange(10, dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(
            cp.percentile(cp_a, [25, 50, 75]).get(),
            np.percentile(np_a, [25, 50, 75]),
            rtol=1e-5,
        )

    def test_2d_axis0(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(
            cp.percentile(cp_a, 75, axis=0).get(),
            np.percentile(np_a, 75, axis=0),
            rtol=1e-5,
        )


# ====================================================================
# quantile
# ====================================================================

class TestQuantile:
    def test_basic(self):
        np_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(cp.quantile(cp_a, 0.5).get(), np.quantile(np_a, 0.5), rtol=1e-5)

    def test_multiple(self):
        np_a = np.arange(10, dtype=np.float32)
        cp_a = cp.array(np_a)
        npt.assert_allclose(
            cp.quantile(cp_a, [0.25, 0.5, 0.75]).get(),
            np.quantile(np_a, [0.25, 0.5, 0.75]),
            rtol=1e-5,
        )

    def test_2d_axis1(self):
        np_a = np.arange(12, dtype=np.float32).reshape(3, 4)
        cp_a = cp.array(np_a)
        npt.assert_allclose(
            cp.quantile(cp_a, 0.25, axis=1).get(),
            np.quantile(np_a, 0.25, axis=1),
            rtol=1e-5,
        )


# ====================================================================
# put
# ====================================================================

class TestPut:
    def test_basic(self):
        np_a = np.arange(5, dtype=np.float32)
        cp_a = cp.array(np_a.copy())
        np.put(np_a, [0, 2, 4], [10, 20, 30])
        cp.put(cp_a, [0, 2, 4], [10, 20, 30])
        npt.assert_array_equal(cp_a.get(), np_a)

    def test_wrap_mode(self):
        np_a = np.zeros(5, dtype=np.float32)
        cp_a = cp.array(np_a.copy())
        np.put(np_a, [5, 6], [99, 88], mode='wrap')
        cp.put(cp_a, [5, 6], [99, 88], mode='wrap')
        npt.assert_array_equal(cp_a.get(), np_a)

    def test_clip_mode(self):
        np_a = np.zeros(5, dtype=np.float32)
        cp_a = cp.array(np_a.copy())
        np.put(np_a, [-1, 10], [77, 66], mode='clip')
        cp.put(cp_a, [-1, 10], [77, 66], mode='clip')
        npt.assert_array_equal(cp_a.get(), np_a)


# ====================================================================
# searchsorted
# ====================================================================

class TestSearchsorted:
    def test_left(self):
        np_a = np.array([1, 3, 5, 7, 9], dtype=np.float32)
        cp_a = cp.array(np_a)
        v = np.array([2, 4, 6], dtype=np.float32)
        npt.assert_array_equal(
            cp.searchsorted(cp_a, cp.array(v)).get(),
            np.searchsorted(np_a, v),
        )

    def test_right(self):
        np_a = np.array([1, 3, 5, 7, 9], dtype=np.float32)
        cp_a = cp.array(np_a)
        v = np.array([3, 5, 7], dtype=np.float32)
        npt.assert_array_equal(
            cp.searchsorted(cp_a, cp.array(v), side='right').get(),
            np.searchsorted(np_a, v, side='right'),
        )

    def test_duplicates(self):
        np_a = np.array([1, 2, 2, 3, 3, 3, 4], dtype=np.float32)
        cp_a = cp.array(np_a)
        v = np.array([2, 3], dtype=np.float32)
        npt.assert_array_equal(
            cp.searchsorted(cp_a, cp.array(v)).get(),
            np.searchsorted(np_a, v),
        )
