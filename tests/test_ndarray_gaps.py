"""Tests for missing ndarray methods, properties, and operators.

TDD: tests written first, then implementation in ndarray.py.
"""

import math
import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from conftest import (
    ALL_DTYPES,
    ALL_DTYPES_NO_BOOL,
    FLOAT_DTYPES,
    INT_DTYPES,
    NUMERIC_DTYPES,
    SHAPES_ALL,
    SHAPES_NONZERO,
    assert_eq,
    make_arg,
    tol_for,
)


# =====================================================================
# 1. Properties: real, imag, flat, base
# =====================================================================


class TestPropertyReal:
    """real property: non-complex returns self, complex returns real part."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=lambda d: np.dtype(d).name)
    def test_real_non_complex(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        result = ga.real
        assert_eq(result, np_a.real, dtype=dtype)

    def test_real_complex(self):
        np_a = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        ga = cp.array(np_a)
        result = ga.real
        expected = np_a.real
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_real_scalar(self):
        np_a = np.array(3.0, dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.real
        assert_eq(result, np_a.real)


class TestPropertyImag:
    """imag property: non-complex returns zeros, complex returns imag part."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=lambda d: np.dtype(d).name)
    def test_imag_non_complex(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        result = ga.imag
        expected = np.zeros_like(np_a)
        assert_eq(result, expected)

    def test_imag_complex(self):
        np_a = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        ga = cp.array(np_a)
        result = ga.imag
        expected = np_a.imag
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestPropertyFlat:
    """flat property: returns an iterator that yields scalars."""

    def test_flat_1d(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        flat_values = list(ga.flat)
        expected = list(np_a.flat)
        assert len(flat_values) == len(expected)
        for v, e in zip(flat_values, expected):
            assert float(v) == pytest.approx(float(e), rel=1e-5)

    def test_flat_2d(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        ga = cp.array(np_a)
        flat_values = list(ga.flat)
        expected = list(np_a.flat)
        assert len(flat_values) == len(expected)
        for v, e in zip(flat_values, expected):
            assert int(v) == int(e)


class TestPropertyBase:
    """base property: expose _base."""

    def test_base_original(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        assert ga.base is None

    def test_base_view(self):
        np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ga = cp.array(np_a)
        view = ga.reshape(4)
        assert view.base is ga


# =====================================================================
# 2. Methods: item, tolist, fill, round, clip, conj, diagonal, trace,
#    repeat, take, put, choose, compress, searchsorted, nonzero, sort,
#    argsort, argmax, argmin, ptp, partition, argpartition, tobytes, view
# =====================================================================


class TestItem:
    """item(): extract Python scalar."""

    def test_item_0d(self):
        np_a = np.array(42.0, dtype=np.float32)
        ga = cp.array(np_a)
        assert ga.item() == pytest.approx(42.0)

    def test_item_1d(self):
        np_a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        ga = cp.array(np_a)
        assert ga.item(1) == pytest.approx(20.0)

    def test_item_2d(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        ga = cp.array(np_a)
        assert ga.item(1, 0) == 3

    def test_item_error_multi_element(self):
        ga = cp.array(np.array([1.0, 2.0], dtype=np.float32))
        with pytest.raises((ValueError, IndexError)):
            ga.item()


class TestTolist:
    """tolist(): convert to nested Python list."""

    def test_tolist_1d(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.tolist()
        expected = np_a.tolist()
        assert result == pytest.approx(expected, rel=1e-5)

    def test_tolist_2d(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        ga = cp.array(np_a)
        assert ga.tolist() == np_a.tolist()

    def test_tolist_0d(self):
        np_a = np.array(5, dtype=np.int32)
        ga = cp.array(np_a)
        assert ga.tolist() == np_a.tolist()


class TestFill:
    """fill(value): fill array in-place."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=lambda d: np.dtype(d).name)
    def test_fill_basic(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        ga.fill(42)
        expected = np.full((5,), 42, dtype=dtype)
        assert_eq(ga, expected, dtype=dtype)

    def test_fill_2d(self):
        ga = cp.zeros((2, 3), dtype=np.float32)
        ga.fill(7.0)
        expected = np.full((2, 3), 7.0, dtype=np.float32)
        assert_eq(ga, expected)


class TestRound:
    """round(decimals=0): round elements."""

    def test_round_default(self):
        np_a = np.array([1.5, 2.3, 3.7], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.round()
        expected = np.around(np_a)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_round_decimals(self):
        np_a = np.array([1.555, 2.345, 3.789], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.round(decimals=2)
        expected = np.around(np_a, decimals=2)
        npt.assert_allclose(result.get(), expected, rtol=1e-4)


class TestClipMethod:
    """clip(min, max): clip values."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=lambda d: np.dtype(d).name)
    def test_clip_basic(self, dtype):
        np_a = make_arg((5,), dtype)
        ga = cp.array(np_a)
        result = ga.clip(2, 4)
        expected = np.clip(np_a, 2, 4)
        assert_eq(result, expected, dtype=dtype)


class TestConj:
    """conj(): complex conjugate."""

    def test_conj_complex(self):
        np_a = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
        ga = cp.array(np_a)
        result = ga.conj()
        expected = np.conj(np_a)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_conj_real(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.conj()
        assert_eq(result, np_a)


class TestDiagonalMethod:
    """diagonal(offset=0)."""

    def test_diagonal_basic(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.diagonal()
        expected = np.diagonal(np_a)
        assert_eq(result, expected)

    def test_diagonal_offset(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.diagonal(offset=1)
        expected = np.diagonal(np_a, offset=1)
        assert_eq(result, expected)


class TestTraceMethod:
    """trace(offset=0)."""

    def test_trace_basic(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.trace()
        expected = np.trace(np_a)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_trace_offset(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.trace(offset=1)
        expected = np.trace(np_a, offset=1)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestRepeatMethod:
    """repeat(repeats, axis=None)."""

    def test_repeat_flat(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.repeat(2)
        expected = np.repeat(np_a, 2)
        assert_eq(result, expected)

    def test_repeat_axis(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.repeat(2, axis=0)
        expected = np.repeat(np_a, 2, axis=0)
        assert_eq(result, expected)


class TestTakeMethod:
    """take(indices, axis=None)."""

    def test_take_flat(self):
        np_a = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.take([0, 2, 4])
        expected = np.take(np_a, [0, 2, 4])
        assert_eq(result, expected)

    def test_take_axis(self):
        np_a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.take([0, 2], axis=0)
        expected = np.take(np_a, [0, 2], axis=0)
        assert_eq(result, expected)


class TestPutMethod:
    """put(indices, values): in-place put."""

    def test_put_basic(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        ga = cp.array(np_a.copy())
        ga.put([0, 2], [99, 88])
        np.put(np_a, [0, 2], [99, 88])
        assert_eq(ga, np_a)


class TestChooseMethod:
    """choose(choices)."""

    def test_choose_basic(self):
        np_a = np.array([0, 1, 2, 1], dtype=np.int32)
        choices = [np.array([0, 1, 2, 3], dtype=np.int32),
                   np.array([10, 11, 12, 13], dtype=np.int32),
                   np.array([20, 21, 22, 23], dtype=np.int32)]
        ga = cp.array(np_a)
        g_choices = [cp.array(c) for c in choices]
        result = ga.choose(g_choices)
        expected = np.choose(np_a, choices)
        assert_eq(result, expected)


class TestCompressMethod:
    """compress(condition, axis=None)."""

    def test_compress_basic(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        cond = [True, False, True, False, True]
        ga = cp.array(np_a)
        result = ga.compress(cond)
        expected = np.compress(cond, np_a)
        assert_eq(result, expected)

    def test_compress_axis(self):
        np_a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        cond = [True, False, True]
        ga = cp.array(np_a)
        result = ga.compress(cond, axis=0)
        expected = np.compress(cond, np_a, axis=0)
        assert_eq(result, expected)


class TestSearchsortedMethod:
    """searchsorted(v, side='left')."""

    def test_searchsorted_left(self):
        np_a = np.array([1, 3, 5, 7, 9], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.searchsorted(4.0)
        expected = np.searchsorted(np_a, 4.0)
        assert_eq(result, expected)

    def test_searchsorted_right(self):
        np_a = np.array([1, 3, 5, 7, 9], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.searchsorted(5.0, side='right')
        expected = np.searchsorted(np_a, 5.0, side='right')
        assert_eq(result, expected)


class TestNonzeroMethod:
    """nonzero()."""

    def test_nonzero_1d(self):
        np_a = np.array([0, 1, 0, 2, 0], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.nonzero()
        expected = np.nonzero(np_a)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            npt.assert_array_equal(r.get(), e)

    def test_nonzero_2d(self):
        np_a = np.array([[0, 1], [2, 0]], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.nonzero()
        expected = np.nonzero(np_a)
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            npt.assert_array_equal(r.get(), e)


class TestSortMethod:
    """sort(axis=-1): in-place sort."""

    def test_sort_1d(self):
        np_a = np.array([3, 1, 4, 1, 5], dtype=np.float32)
        ga = cp.array(np_a.copy())
        ga.sort()
        np_a.sort()
        assert_eq(ga, np_a)

    def test_sort_2d_axis(self):
        np_a = np.array([[3, 1], [4, 2]], dtype=np.int32)
        ga = cp.array(np_a.copy())
        ga.sort(axis=0)
        np_a.sort(axis=0)
        assert_eq(ga, np_a)


class TestArgsortMethod:
    """argsort(axis=-1)."""

    def test_argsort_1d(self):
        np_a = np.array([3, 1, 4, 1, 5], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.argsort()
        expected = np.argsort(np_a)
        npt.assert_array_equal(result.get(), expected)

    def test_argsort_axis0(self):
        np_a = np.array([[3, 1], [1, 4]], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.argsort(axis=0)
        expected = np.argsort(np_a, axis=0)
        npt.assert_array_equal(result.get(), expected)


class TestArgmaxMethod:
    """argmax(axis=None)."""

    def test_argmax_flat(self):
        np_a = np.array([3, 1, 4, 1, 5], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.argmax()
        expected = np.argmax(np_a)
        assert int(result) == int(expected)

    def test_argmax_axis(self):
        np_a = np.array([[3, 1], [1, 4]], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.argmax(axis=0)
        expected = np.argmax(np_a, axis=0)
        npt.assert_array_equal(result.get(), expected)


class TestArgminMethod:
    """argmin(axis=None)."""

    def test_argmin_flat(self):
        np_a = np.array([3, 1, 4, 1, 5], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.argmin()
        expected = np.argmin(np_a)
        assert int(result) == int(expected)

    def test_argmin_axis(self):
        np_a = np.array([[3, 1], [1, 4]], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.argmin(axis=0)
        expected = np.argmin(np_a, axis=0)
        npt.assert_array_equal(result.get(), expected)


class TestPtpMethod:
    """ptp(axis=None): peak-to-peak."""

    def test_ptp_flat(self):
        np_a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.ptp()
        expected = np.ptp(np_a)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_ptp_axis(self):
        np_a = np.array([[3, 1], [1, 4]], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.ptp(axis=0)
        expected = np.ptp(np_a, axis=0)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestPartitionMethod:
    """partition(kth, axis=-1): in-place partition."""

    def test_partition_basic(self):
        np_a = np.array([3, 4, 2, 1, 5], dtype=np.float32)
        ga = cp.array(np_a.copy())
        ga.partition(2)
        # After partition(2), element at index 2 should be the same as sorted[2]
        result = ga.get()
        sorted_a = np.sort(np_a)
        assert result[2] == pytest.approx(sorted_a[2])
        # Elements before kth should be <= kth element
        assert all(result[:2] <= result[2])
        # Elements after kth should be >= kth element
        assert all(result[3:] >= result[2])


class TestArgpartitionMethod:
    """argpartition(kth, axis=-1)."""

    def test_argpartition_basic(self):
        np_a = np.array([3, 4, 2, 1, 5], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.argpartition(2)
        np_result = np.argpartition(np_a, 2)
        # Element at kth position should be same as sorted[kth]
        assert np_a[result.get()[2]] == pytest.approx(np_a[np_result[2]])


class TestTobytes:
    """tobytes(): return raw bytes."""

    def test_tobytes_float32(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.tobytes()
        expected = np_a.tobytes()
        assert result == expected

    def test_tobytes_int32(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.tobytes()
        expected = np_a.tobytes()
        assert result == expected


class TestView:
    """view(dtype): reinterpret buffer with different dtype."""

    def test_view_float32_to_int32(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.view(np.int32)
        expected = np_a.view(np.int32)
        npt.assert_array_equal(result.get(), expected)

    def test_view_int32_to_float32(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.view(np.float32)
        expected = np_a.view(np.float32)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


# =====================================================================
# 3. Operators: __bool__, __pos__, __complex__, __index__, __contains__,
#    floordiv, mod, lshift, rshift, iand, ixor, rmatmul, imatmul, divmod
# =====================================================================


class TestBoolOperator:
    """__bool__(): 0-d arrays return bool, multi-element raises."""

    def test_bool_true(self):
        ga = cp.array(np.array(1.0, dtype=np.float32))
        assert bool(ga) is True

    def test_bool_false(self):
        ga = cp.array(np.array(0.0, dtype=np.float32))
        assert bool(ga) is False

    def test_bool_multi_element_error(self):
        ga = cp.array(np.array([1.0, 2.0], dtype=np.float32))
        with pytest.raises((ValueError, TypeError)):
            bool(ga)


class TestPosOperator:
    """__pos__(): unary positive, return copy."""

    def test_pos_float(self):
        np_a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = +ga
        assert_eq(result, +np_a)

    def test_pos_int(self):
        np_a = np.array([1, -2, 3], dtype=np.int32)
        ga = cp.array(np_a)
        result = +ga
        assert_eq(result, +np_a)


class TestComplexOperator:
    """__complex__(): 0-d arrays return complex."""

    def test_complex_0d(self):
        ga = cp.array(np.array(3.0, dtype=np.float32))
        assert complex(ga) == 3.0 + 0j

    def test_complex_multi_error(self):
        ga = cp.array(np.array([1.0, 2.0], dtype=np.float32))
        with pytest.raises(TypeError):
            complex(ga)


class TestIndexOperator:
    """__index__(): 0-d integer arrays return int for use as index."""

    def test_index_int(self):
        ga = cp.array(np.array(5, dtype=np.int32))
        result = ga.__index__()
        assert result == 5
        assert isinstance(result, int)

    def test_index_float_error(self):
        ga = cp.array(np.array(5.0, dtype=np.float32))
        with pytest.raises(TypeError):
            ga.__index__()

    def test_index_multi_error(self):
        ga = cp.array(np.array([1, 2], dtype=np.int32))
        with pytest.raises(TypeError):
            ga.__index__()


class TestContainsOperator:
    """__contains__(item): membership test."""

    def test_contains_present(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ga = cp.array(np_a)
        assert 3 in ga

    def test_contains_absent(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ga = cp.array(np_a)
        assert 10 not in ga


class TestFloorDivOperator:
    """// operator: __floordiv__, __rfloordiv__, __ifloordiv__."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=lambda d: np.dtype(d).name)
    def test_floordiv_basic(self, dtype):
        np_a = make_arg((5,), dtype)
        np_b = np.where(make_arg((5,), dtype) == 0, np.ones(5, dtype=dtype), make_arg((5,), dtype))
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = ga // gb
        expected = np_a // np_b
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    def test_rfloordiv(self):
        np_a = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = 10.0 // ga
        expected = 10.0 // np_a
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_ifloordiv(self):
        np_a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        ga = cp.array(np_a.copy())
        ga //= 3.0
        np_a //= 3.0
        npt.assert_allclose(ga.get(), np_a, rtol=1e-5)


class TestModOperator:
    """% operator: __mod__, __rmod__, __imod__."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=lambda d: np.dtype(d).name)
    def test_mod_basic(self, dtype):
        np_a = make_arg((5,), dtype)
        np_b = np.where(make_arg((5,), dtype) == 0, np.ones(5, dtype=dtype), make_arg((5,), dtype))
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = ga % gb
        expected = np_a % np_b
        assert_eq(result, expected, dtype=expected.dtype, category="arithmetic")

    def test_rmod(self):
        np_a = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = 10.0 % ga
        expected = 10.0 % np_a
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_imod(self):
        np_a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        ga = cp.array(np_a.copy())
        ga %= 7.0
        np_a %= 7.0
        npt.assert_allclose(ga.get(), np_a, rtol=1e-5)


class TestShiftOperators:
    """<< and >> operators."""

    def test_lshift(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a)
        gb = cp.array(np.array([1, 2, 3], dtype=np.int32))
        result = ga << gb
        expected = np_a << np.array([1, 2, 3], dtype=np.int32)
        npt.assert_array_equal(result.get(), expected)

    def test_rshift(self):
        np_a = np.array([8, 16, 32], dtype=np.int32)
        ga = cp.array(np_a)
        gb = cp.array(np.array([1, 2, 3], dtype=np.int32))
        result = ga >> gb
        expected = np_a >> np.array([1, 2, 3], dtype=np.int32)
        npt.assert_array_equal(result.get(), expected)

    def test_rlshift(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a)
        result = 1 << ga
        expected = 1 << np_a
        npt.assert_array_equal(result.get(), expected)

    def test_rrshift(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a)
        result = 64 >> ga
        expected = 64 >> np_a
        npt.assert_array_equal(result.get(), expected)


class TestInplaceBoolOps:
    """In-place &= and ^=."""

    def test_iand(self):
        np_a = np.array([True, False, True, False], dtype=np.bool_)
        np_b = np.array([True, True, False, False], dtype=np.bool_)
        ga = cp.array(np_a.copy())
        ga &= cp.array(np_b)
        expected = np_a & np_b
        npt.assert_array_equal(ga.get(), expected)

    def test_ixor(self):
        np_a = np.array([True, False, True, False], dtype=np.bool_)
        np_b = np.array([True, True, False, False], dtype=np.bool_)
        ga = cp.array(np_a.copy())
        ga ^= cp.array(np_b)
        expected = np_a ^ np_b
        npt.assert_array_equal(ga.get(), expected)


class TestRmatmul:
    """__rmatmul__: reverse @ operator."""

    def test_rmatmul(self):
        np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np_b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        # ga.__rmatmul__(gb) should be gb @ ga
        result = ga.__rmatmul__(gb)
        expected = np_b @ np_a
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestImatmul:
    """__imatmul__: in-place @=."""

    def test_imatmul(self):
        np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np_b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        ga = cp.array(np_a.copy())
        ga @= cp.array(np_b)
        expected = np_a @ np_b
        npt.assert_allclose(ga.get(), expected, rtol=1e-5)


class TestDivmod:
    """__divmod__: divmod() support."""

    def test_divmod_basic(self):
        np_a = np.array([7.0, 13.0, 20.0], dtype=np.float32)
        np_b = np.array([3.0, 5.0, 7.0], dtype=np.float32)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        q, r = divmod(ga, gb)
        eq, er = divmod(np_a, np_b)
        npt.assert_allclose(q.get(), eq, rtol=1e-5)
        npt.assert_allclose(r.get(), er, rtol=1e-5)


class TestXorOperator:
    """__xor__ operator."""

    def test_xor(self):
        np_a = np.array([True, False, True, False], dtype=np.bool_)
        np_b = np.array([True, True, False, False], dtype=np.bool_)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = ga ^ gb
        expected = np_a ^ np_b
        npt.assert_array_equal(result.get(), expected)
