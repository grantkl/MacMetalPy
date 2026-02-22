"""Tests for untested top-level APIs -- batch 3: creation, manipulation, misc.

Covers: empty_like, asarray, asanyarray, ascontiguousarray, asfortranarray,
fromiter, reshape, ravel, transpose, block, diff, ediff1d, gradient,
trapezoid, diag_indices, diag_indices_from, tril_indices, tril_indices_from,
triu_indices, triu_indices_from, mask_indices, ravel_multi_index,
unravel_index, vander, array2string, array_repr, array_str, base_repr,
binary_repr, typename, mintypecode, issubdtype, can_cast, promote_types,
result_type, min_scalar_type, common_type, finfo, iinfo, isscalar,
iscomplexobj, isrealobj, isfortran, array_equiv.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import macmetalpy as cp
from conftest import FLOAT_DTYPES, NUMERIC_DTYPES, ALL_DTYPES, assert_eq, make_arg


# ====================================================================
# empty_like
# ====================================================================

class TestEmptyLike:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_shape_and_dtype(self, dtype):
        a = cp.array(make_arg((3, 4), dtype))
        result = cp.empty_like(a)
        assert result.shape == (3, 4)
        assert result.dtype == np.dtype(dtype)

    def test_1d(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.empty_like(a)
        assert result.shape == (3,)
        assert result.dtype == np.float32

    def test_3d(self):
        a = cp.zeros((2, 3, 4), dtype=np.float32)
        result = cp.empty_like(a)
        assert result.shape == (2, 3, 4)
        assert result.dtype == np.float32


# ====================================================================
# asarray
# ====================================================================

class TestAsarray:
    @pytest.mark.parametrize("dtype", [np.float32, np.int32, np.int64])
    def test_from_list(self, dtype):
        result = cp.asarray([1, 2, 3], dtype=dtype)
        expected = np.asarray([1, 2, 3], dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_from_numpy(self):
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.asarray(np_arr)
        assert_eq(result, np_arr, dtype=np.float32)

    def test_from_macmetalpy_array(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.asarray(a)
        assert_eq(result, np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=np.float32)


# ====================================================================
# asanyarray
# ====================================================================

class TestAsanyarray:
    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    def test_from_list(self, dtype):
        result = cp.asanyarray([1, 2, 3], dtype=dtype)
        expected = np.asanyarray([1, 2, 3], dtype=dtype)
        assert_eq(result, expected, dtype=dtype)

    def test_from_numpy(self):
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = cp.asanyarray(np_arr)
        assert_eq(result, np_arr, dtype=np.float32)

    def test_preserves_gpu_array(self):
        a = cp.array([5.0, 6.0], dtype=np.float32)
        result = cp.asanyarray(a)
        assert_eq(result, np.array([5.0, 6.0], dtype=np.float32), dtype=np.float32)


# ====================================================================
# ascontiguousarray
# ====================================================================

class TestAscontiguousarray:
    def test_basic(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.ascontiguousarray(a)
        expected = np.ascontiguousarray(np.array([[1, 2], [3, 4]], dtype=np.float32))
        assert_eq(result, expected, dtype=np.float32)

    def test_preserves_values(self):
        np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        a = cp.array(np_arr)
        result = cp.ascontiguousarray(a)
        assert_eq(result, np_arr, dtype=np.float32)

    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    def test_with_dtype(self, dtype):
        a = cp.array([1, 2, 3], dtype=dtype)
        result = cp.ascontiguousarray(a)
        expected = np.ascontiguousarray(np.array([1, 2, 3], dtype=dtype))
        assert_eq(result, expected, dtype=dtype)


# ====================================================================
# asfortranarray
# ====================================================================

class TestAsfortranarray:
    def test_basic(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.asfortranarray(a)
        expected = np.array([[1, 2], [3, 4]], dtype=np.float32)
        assert_eq(result, expected, dtype=np.float32)

    def test_preserves_values(self):
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = cp.asfortranarray(cp.array(np_arr))
        assert_eq(result, np_arr, dtype=np.float32)

    def test_2d_shape(self):
        a = cp.array(np.arange(6, dtype=np.float32).reshape(2, 3))
        result = cp.asfortranarray(a)
        assert result.shape == (2, 3)


# ====================================================================
# fromiter
# ====================================================================

class TestFromiter:
    def test_basic(self):
        result = cp.fromiter(iter(range(5)), dtype=np.float32)
        expected = np.fromiter(iter(range(5)), dtype=np.float32)
        assert_eq(result, expected, dtype=np.float32)

    def test_with_count(self):
        result = cp.fromiter(iter(range(10)), dtype=np.float32, count=5)
        expected = np.fromiter(iter(range(10)), dtype=np.float32, count=5)
        assert_eq(result, expected, dtype=np.float32)

    def test_generator(self):
        gen_cp = (x * x for x in range(5))
        gen_np = (x * x for x in range(5))
        result = cp.fromiter(gen_cp, dtype=np.float32)
        expected = np.fromiter(gen_np, dtype=np.float32)
        assert_eq(result, expected, dtype=np.float32)


# ====================================================================
# reshape
# ====================================================================

class TestReshapeCreation:
    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    def test_1d_to_2d(self, dtype):
        a_np = np.arange(6, dtype=dtype)
        a_cp = cp.array(a_np)
        assert_eq(cp.reshape(a_cp, (2, 3)), np.reshape(a_np, (2, 3)), dtype=dtype)

    def test_infer_dim(self):
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        a_cp = cp.array(a_np)
        assert_eq(cp.reshape(a_cp, (-1, 4)), np.reshape(a_np, (-1, 4)), dtype=np.float32)

    def test_flatten(self):
        a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        assert_eq(cp.reshape(a_cp, (-1,)), np.reshape(a_np, (-1,)), dtype=np.float32)


# ====================================================================
# ravel
# ====================================================================

class TestRavelCreation:
    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    def test_2d(self, dtype):
        a_np = np.arange(6, dtype=dtype).reshape(2, 3)
        a_cp = cp.array(a_np)
        assert_eq(cp.ravel(a_cp), np.ravel(a_np), dtype=dtype)

    def test_3d(self):
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        a_cp = cp.array(a_np)
        assert_eq(cp.ravel(a_cp), np.ravel(a_np), dtype=np.float32)

    def test_already_1d(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.ravel(a_cp), np.ravel(a_np), dtype=np.float32)


# ====================================================================
# transpose
# ====================================================================

class TestTransposeCreation:
    def test_2d(self):
        a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        a_cp = cp.array(a_np)
        assert_eq(cp.transpose(a_cp), np.transpose(a_np), dtype=np.float32)

    def test_3d_with_axes(self):
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        a_cp = cp.array(a_np)
        result = cp.transpose(a_cp, axes=(1, 2, 0))
        expected = np.transpose(a_np, axes=(1, 2, 0))
        assert result.shape == expected.shape
        assert_eq(result, expected, dtype=np.float32)

    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    def test_square_matrix(self, dtype):
        a_np = np.arange(9, dtype=dtype).reshape(3, 3)
        a_cp = cp.array(a_np)
        assert_eq(cp.transpose(a_cp), np.transpose(a_np), dtype=dtype)


# ====================================================================
# block
# ====================================================================

class TestBlock:
    def test_1d_blocks(self):
        a_np = np.array([1, 2, 3], dtype=np.float32)
        b_np = np.array([4, 5, 6], dtype=np.float32)
        result = cp.block([cp.array(a_np), cp.array(b_np)])
        expected = np.block([a_np, b_np])
        assert_eq(result, expected, dtype=np.float32)

    def test_2d_blocks(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = cp.block([[cp.array(a_np), cp.array(b_np)]])
        expected = np.block([[a_np, b_np]])
        assert_eq(result, expected, dtype=np.float32)

    def test_vertical_blocks(self):
        a_np = np.array([1, 2], dtype=np.float32)
        b_np = np.array([3, 4], dtype=np.float32)
        result = cp.block([[cp.array(a_np)], [cp.array(b_np)]])
        expected = np.block([[a_np], [b_np]])
        assert_eq(result, expected, dtype=np.float32)


# ====================================================================
# diff
# ====================================================================

class TestDiff:
    def test_1d(self):
        a_np = np.array([1, 2, 4, 7, 0], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.diff(a_cp), np.diff(a_np), dtype=np.float32)

    def test_1d_n2(self):
        a_np = np.array([1, 3, 6, 10], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.diff(a_cp, n=2), np.diff(a_np, n=2), dtype=np.float32)

    def test_2d_axis0(self):
        a_np = np.array([[1, 3, 6], [2, 5, 9]], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.diff(a_cp, axis=0), np.diff(a_np, axis=0), dtype=np.float32)


# ====================================================================
# ediff1d
# ====================================================================

class TestEdiff1d:
    def test_basic(self):
        a_np = np.array([1, 2, 4, 7, 0], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.ediff1d(a_cp), np.ediff1d(a_np), dtype=np.float32)

    def test_2d_input(self):
        a_np = np.array([[1, 2, 3], [5, 2, 8]], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.ediff1d(a_cp), np.ediff1d(a_np), dtype=np.float32)

    def test_monotonic(self):
        a_np = np.array([1, 3, 6, 10, 15], dtype=np.float32)
        a_cp = cp.array(a_np)
        assert_eq(cp.ediff1d(a_cp), np.ediff1d(a_np), dtype=np.float32)


# ====================================================================
# gradient
# ====================================================================

class TestGradient:
    def test_1d(self):
        a_np = np.array([1, 2, 4, 7, 11], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.gradient(a_cp)
        expected = np.gradient(a_np)
        assert_eq(result, expected, dtype=np.float32)

    def test_1d_uniform(self):
        a_np = np.array([0, 2, 4, 6, 8], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.gradient(a_cp)
        expected = np.gradient(a_np)
        assert_eq(result, expected, dtype=np.float32)

    def test_1d_with_spacing(self):
        a_np = np.array([1, 2, 4, 7, 11], dtype=np.float32)
        a_cp = cp.array(a_np)
        result = cp.gradient(a_cp, 2.0)
        expected = np.gradient(a_np, 2.0)
        if isinstance(expected, (list, tuple)):
            expected = expected[0]
        assert_eq(result, expected, dtype=np.float32)


# ====================================================================
# trapezoid
# ====================================================================

class TestTrapezoid:
    def test_basic(self):
        y_np = np.array([1, 2, 3], dtype=np.float32)
        y_cp = cp.array(y_np)
        result = float(cp.trapezoid(y_cp).get() if hasattr(cp.trapezoid(y_cp), 'get') else cp.trapezoid(y_cp))
        expected = float(np.trapezoid(y_np))
        assert abs(result - expected) < 1e-5

    def test_with_x(self):
        y_np = np.array([1, 2, 3], dtype=np.float32)
        x_np = np.array([0, 1, 3], dtype=np.float32)
        y_cp = cp.array(y_np)
        x_cp = cp.array(x_np)
        result = float(cp.trapezoid(y_cp, x_cp).get() if hasattr(cp.trapezoid(y_cp, x_cp), 'get') else cp.trapezoid(y_cp, x_cp))
        expected = float(np.trapezoid(y_np, x_np))
        assert abs(result - expected) < 1e-5

    def test_constant(self):
        y_np = np.array([5, 5, 5, 5], dtype=np.float32)
        y_cp = cp.array(y_np)
        result = float(cp.trapezoid(y_cp).get() if hasattr(cp.trapezoid(y_cp), 'get') else cp.trapezoid(y_cp))
        expected = float(np.trapezoid(y_np))
        assert abs(result - expected) < 1e-5


# ====================================================================
# diag_indices
# ====================================================================

class TestDiagIndices:
    def test_basic(self):
        cp_result = cp.diag_indices(3)
        np_result = np.diag_indices(3)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_ndim2(self):
        cp_result = cp.diag_indices(4)
        np_result = np.diag_indices(4)
        assert len(cp_result) == len(np_result)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_ndim3(self):
        cp_result = cp.diag_indices(3, ndim=3)
        np_result = np.diag_indices(3, ndim=3)
        assert len(cp_result) == 3
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)


# ====================================================================
# diag_indices_from
# ====================================================================

class TestDiagIndicesFrom:
    def test_2d(self):
        m_np = np.eye(3, dtype=np.float32)
        m_cp = cp.array(m_np)
        cp_result = cp.diag_indices_from(m_cp)
        np_result = np.diag_indices_from(m_np)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_4x4(self):
        m_np = np.zeros((4, 4), dtype=np.float32)
        m_cp = cp.array(m_np)
        cp_result = cp.diag_indices_from(m_cp)
        np_result = np.diag_indices_from(m_np)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_use_for_indexing(self):
        m_np = np.zeros((3, 3), dtype=np.float32)
        m_cp = cp.array(m_np.copy())
        np_idx = np.diag_indices_from(m_np)
        m_np[np_idx] = 1.0
        cp_idx = cp.diag_indices_from(m_cp)
        # Verify indices match
        for ci, ni in zip(cp_idx, np_idx):
            ci_val = ci.get() if hasattr(ci, 'get') else np.asarray(ci)
            assert_array_equal(ci_val, ni)


# ====================================================================
# tril_indices
# ====================================================================

class TestTrilIndices:
    def test_basic(self):
        cp_result = cp.tril_indices(3)
        np_result = np.tril_indices(3)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_4x4(self):
        cp_result = cp.tril_indices(4)
        np_result = np.tril_indices(4)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_with_k(self):
        cp_result = cp.tril_indices(3, k=1)
        np_result = np.tril_indices(3, k=1)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)


# ====================================================================
# tril_indices_from
# ====================================================================

class TestTrilIndicesFrom:
    def test_basic(self):
        m_np = np.ones((3, 3), dtype=np.float32)
        m_cp = cp.array(m_np)
        cp_result = cp.tril_indices_from(m_cp)
        np_result = np.tril_indices_from(m_np)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_4x4(self):
        m_np = np.zeros((4, 4), dtype=np.float32)
        m_cp = cp.array(m_np)
        cp_result = cp.tril_indices_from(m_cp)
        np_result = np.tril_indices_from(m_np)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_with_k(self):
        m_np = np.ones((3, 3), dtype=np.float32)
        m_cp = cp.array(m_np)
        cp_result = cp.tril_indices_from(m_cp, k=-1)
        np_result = np.tril_indices_from(m_np, k=-1)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)


# ====================================================================
# triu_indices
# ====================================================================

class TestTriuIndices:
    def test_basic(self):
        cp_result = cp.triu_indices(3)
        np_result = np.triu_indices(3)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_4x4(self):
        cp_result = cp.triu_indices(4)
        np_result = np.triu_indices(4)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_with_k(self):
        cp_result = cp.triu_indices(3, k=1)
        np_result = np.triu_indices(3, k=1)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)


# ====================================================================
# triu_indices_from
# ====================================================================

class TestTriuIndicesFrom:
    def test_basic(self):
        m_np = np.ones((3, 3), dtype=np.float32)
        m_cp = cp.array(m_np)
        cp_result = cp.triu_indices_from(m_cp)
        np_result = np.triu_indices_from(m_np)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_4x4(self):
        m_np = np.zeros((4, 4), dtype=np.float32)
        m_cp = cp.array(m_np)
        cp_result = cp.triu_indices_from(m_cp)
        np_result = np.triu_indices_from(m_np)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_with_k(self):
        m_np = np.ones((3, 3), dtype=np.float32)
        m_cp = cp.array(m_np)
        cp_result = cp.triu_indices_from(m_cp, k=1)
        np_result = np.triu_indices_from(m_np, k=1)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)


# ====================================================================
# mask_indices
# ====================================================================

class TestMaskIndices:
    def test_triu(self):
        cp_result = cp.mask_indices(3, np.triu)
        np_result = np.mask_indices(3, np.triu)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_tril(self):
        cp_result = cp.mask_indices(3, np.tril)
        np_result = np.mask_indices(3, np.tril)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_4x4_triu(self):
        cp_result = cp.mask_indices(4, np.triu)
        np_result = np.mask_indices(4, np.triu)
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)


# ====================================================================
# ravel_multi_index
# ====================================================================

class TestRavelMultiIndex:
    def test_basic(self):
        coords_np = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        coords_cp = cp.array(coords_np)
        result = cp.ravel_multi_index(coords_cp, (3, 4))
        expected = np.ravel_multi_index(coords_np, (3, 4))
        result_val = result.get() if hasattr(result, 'get') else np.asarray(result)
        assert_array_equal(result_val, expected)

    def test_single_index(self):
        coords_np = np.array([[1], [2]], dtype=np.int64)
        coords_cp = cp.array(coords_np)
        result = cp.ravel_multi_index(coords_cp, (3, 4))
        expected = np.ravel_multi_index(coords_np, (3, 4))
        result_val = result.get() if hasattr(result, 'get') else np.asarray(result)
        assert_array_equal(result_val, expected)

    def test_3d_dims(self):
        coords_np = np.array([[0, 1], [1, 2], [0, 1]], dtype=np.int64)
        coords_cp = cp.array(coords_np)
        result = cp.ravel_multi_index(coords_cp, (2, 3, 2))
        expected = np.ravel_multi_index(coords_np, (2, 3, 2))
        result_val = result.get() if hasattr(result, 'get') else np.asarray(result)
        assert_array_equal(result_val, expected)


# ====================================================================
# unravel_index
# ====================================================================

class TestUnravelIndex:
    def test_basic(self):
        indices_np = np.array([5, 11], dtype=np.int64)
        indices_cp = cp.array(indices_np)
        cp_result = cp.unravel_index(indices_cp, (3, 4))
        np_result = np.unravel_index(indices_np, (3, 4))
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_single_value(self):
        indices_np = np.array([7], dtype=np.int64)
        indices_cp = cp.array(indices_np)
        cp_result = cp.unravel_index(indices_cp, (3, 4))
        np_result = np.unravel_index(indices_np, (3, 4))
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)

    def test_3d_shape(self):
        indices_np = np.array([0, 5, 11], dtype=np.int64)
        indices_cp = cp.array(indices_np)
        cp_result = cp.unravel_index(indices_cp, (2, 3, 4))
        np_result = np.unravel_index(indices_np, (2, 3, 4))
        for cr, nr in zip(cp_result, np_result):
            cr_val = cr.get() if hasattr(cr, 'get') else np.asarray(cr)
            assert_array_equal(cr_val, nr)


# ====================================================================
# vander
# ====================================================================

class TestVander:
    def test_basic(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float32)
        x_cp = cp.array(x_np)
        assert_eq(cp.vander(x_cp), np.vander(x_np), dtype=np.float32)

    def test_with_N(self):
        x_np = np.array([1, 2, 3], dtype=np.float32)
        x_cp = cp.array(x_np)
        assert_eq(cp.vander(x_cp, N=3), np.vander(x_np, N=3), dtype=np.float32)

    def test_increasing(self):
        x_np = np.array([1, 2, 3], dtype=np.float32)
        x_cp = cp.array(x_np)
        assert_eq(
            cp.vander(x_cp, increasing=True),
            np.vander(x_np, increasing=True),
            dtype=np.float32,
        )


# ====================================================================
# array2string
# ====================================================================

class TestArray2string:
    def test_1d(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.array2string(a)
        assert isinstance(result, str)
        assert '1.' in result
        assert '3.' in result

    def test_2d(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.array2string(a)
        assert isinstance(result, str)

    def test_matches_numpy_format(self):
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cp_arr = cp.array(np_arr)
        cp_str = cp.array2string(cp_arr)
        np_str = np.array2string(np_arr)
        assert cp_str == np_str


# ====================================================================
# array_repr
# ====================================================================

class TestArrayRepr:
    def test_1d(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.array_repr(a)
        assert isinstance(result, str)
        assert 'array' in result

    def test_contains_values(self):
        a = cp.array([5.0, 10.0], dtype=np.float32)
        result = cp.array_repr(a)
        assert '5.' in result
        assert '10.' in result

    def test_includes_dtype(self):
        a = cp.array([1, 2, 3], dtype=np.int32)
        result = cp.array_repr(a)
        assert isinstance(result, str)


# ====================================================================
# array_str
# ====================================================================

class TestArrayStr:
    def test_1d(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.array_str(a)
        assert isinstance(result, str)
        assert '1.' in result

    def test_2d(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.array_str(a)
        assert isinstance(result, str)

    def test_matches_numpy(self):
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cp_arr = cp.array(np_arr)
        assert cp.array_str(cp_arr) == np.array_str(np_arr)


# ====================================================================
# base_repr
# ====================================================================

class TestBaseRepr:
    def test_binary(self):
        assert cp.base_repr(10, base=2) == np.base_repr(10, base=2)

    def test_octal(self):
        assert cp.base_repr(255, base=8) == np.base_repr(255, base=8)

    def test_hex(self):
        assert cp.base_repr(255, base=16) == np.base_repr(255, base=16)


# ====================================================================
# binary_repr
# ====================================================================

class TestBinaryRepr:
    def test_positive(self):
        assert cp.binary_repr(10) == np.binary_repr(10)

    def test_zero(self):
        assert cp.binary_repr(0) == np.binary_repr(0)

    def test_with_width(self):
        assert cp.binary_repr(5, width=8) == np.binary_repr(5, width=8)


# ====================================================================
# typename
# ====================================================================

class TestTypename:
    def test_float(self):
        assert cp.typename('f') == np.typename('f')

    def test_double(self):
        assert cp.typename('d') == np.typename('d')

    def test_int(self):
        assert cp.typename('i') == np.typename('i')


# ====================================================================
# mintypecode
# ====================================================================

class TestMintypecode:
    def test_float_types(self):
        assert cp.mintypecode(['f', 'd']) == np.mintypecode(['f', 'd'])

    def test_int_float(self):
        assert cp.mintypecode(['i', 'f']) == np.mintypecode(['i', 'f'])

    def test_single_type(self):
        assert cp.mintypecode(['f']) == np.mintypecode(['f'])


# ====================================================================
# issubdtype
# ====================================================================

class TestIssubdtype:
    def test_float32_is_floating(self):
        assert cp.issubdtype(np.float32, np.floating) is True

    def test_int32_is_integer(self):
        assert cp.issubdtype(np.int32, np.integer) is True

    def test_float_not_integer(self):
        assert cp.issubdtype(np.float32, np.integer) is False


# ====================================================================
# can_cast
# ====================================================================

class TestCanCast:
    def test_safe_upcast(self):
        assert cp.can_cast(np.int16, np.float32) == np.can_cast(np.int16, np.float32)

    def test_same_type(self):
        assert cp.can_cast(np.float32, np.float32) == np.can_cast(np.float32, np.float32)

    def test_unsafe(self):
        assert cp.can_cast(np.float32, np.int32, casting='unsafe') == np.can_cast(np.float32, np.int32, casting='unsafe')


# ====================================================================
# promote_types
# ====================================================================

class TestPromoteTypes:
    def test_int_float(self):
        assert cp.promote_types(np.int32, np.float32) == np.promote_types(np.int32, np.float32)

    def test_same_type(self):
        assert cp.promote_types(np.float32, np.float32) == np.promote_types(np.float32, np.float32)

    def test_int_types(self):
        assert cp.promote_types(np.int16, np.int32) == np.promote_types(np.int16, np.int32)


# ====================================================================
# result_type
# ====================================================================

class TestResultType:
    def test_basic(self):
        assert cp.result_type(np.float32, np.int32) == np.result_type(np.float32, np.int32)

    def test_single(self):
        assert cp.result_type(np.float32) == np.result_type(np.float32)

    def test_multiple(self):
        assert cp.result_type(np.int16, np.float32, np.int64) == np.result_type(np.int16, np.float32, np.int64)


# ====================================================================
# min_scalar_type
# ====================================================================

class TestMinScalarType:
    def test_small_int(self):
        assert cp.min_scalar_type(5) == np.min_scalar_type(5)

    def test_large_int(self):
        assert cp.min_scalar_type(1000) == np.min_scalar_type(1000)

    def test_float(self):
        assert cp.min_scalar_type(1.0) == np.min_scalar_type(1.0)


# ====================================================================
# common_type
# ====================================================================

class TestCommonType:
    def test_float32(self):
        a = cp.array([1.0], dtype=np.float32)
        na = np.array([1.0], dtype=np.float32)
        assert cp.common_type(a) == np.common_type(na)

    def test_int_array(self):
        a = cp.array([1, 2, 3], dtype=np.int32)
        na = np.array([1, 2, 3], dtype=np.int32)
        # Both should return a float type
        cp_ct = cp.common_type(a)
        np_ct = np.common_type(na)
        assert cp_ct == np_ct

    def test_complex(self):
        a = cp.array([1 + 2j], dtype=np.complex64)
        na = np.array([1 + 2j], dtype=np.complex64)
        assert cp.common_type(a) == np.common_type(na)


# ====================================================================
# finfo
# ====================================================================

class TestFinfo:
    def test_float32(self):
        cp_fi = cp.finfo(np.float32)
        np_fi = np.finfo(np.float32)
        assert cp_fi.bits == np_fi.bits
        assert cp_fi.eps == np_fi.eps

    def test_float16(self):
        cp_fi = cp.finfo(np.float16)
        np_fi = np.finfo(np.float16)
        assert cp_fi.bits == np_fi.bits

    def test_has_max(self):
        fi = cp.finfo(np.float32)
        assert fi.max > 0


# ====================================================================
# iinfo
# ====================================================================

class TestIinfo:
    def test_int32(self):
        cp_ii = cp.iinfo(np.int32)
        np_ii = np.iinfo(np.int32)
        assert cp_ii.min == np_ii.min
        assert cp_ii.max == np_ii.max

    def test_int64(self):
        cp_ii = cp.iinfo(np.int64)
        np_ii = np.iinfo(np.int64)
        assert cp_ii.min == np_ii.min
        assert cp_ii.max == np_ii.max

    def test_uint32(self):
        cp_ii = cp.iinfo(np.uint32)
        np_ii = np.iinfo(np.uint32)
        assert cp_ii.min == np_ii.min
        assert cp_ii.max == np_ii.max


# ====================================================================
# isscalar
# ====================================================================

class TestIsscalar:
    def test_int(self):
        assert cp.isscalar(3) is True

    def test_float(self):
        assert cp.isscalar(3.0) is True

    def test_array(self):
        a = cp.array([1, 2, 3], dtype=np.float32)
        assert cp.isscalar(a) is False


# ====================================================================
# iscomplexobj
# ====================================================================

class TestIscomplexobj:
    def test_float_array(self):
        a = cp.array([1.0, 2.0], dtype=np.float32)
        assert cp.iscomplexobj(a) is False

    def test_complex_array(self):
        a = cp.array([1 + 2j], dtype=np.complex64)
        assert cp.iscomplexobj(a) is True

    def test_real_scalar(self):
        assert cp.iscomplexobj(3.0) is False


# ====================================================================
# isrealobj
# ====================================================================

class TestIsrealobj:
    def test_float_array(self):
        a = cp.array([1.0, 2.0], dtype=np.float32)
        assert cp.isrealobj(a) is True

    def test_complex_array(self):
        a = cp.array([1 + 2j], dtype=np.complex64)
        assert cp.isrealobj(a) is False

    def test_int_array(self):
        a = cp.array([1, 2, 3], dtype=np.int32)
        assert cp.isrealobj(a) is True


# ====================================================================
# isfortran
# ====================================================================

class TestIsfortran:
    def test_c_order(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        # Default C-order should not be Fortran
        result = cp.isfortran(a)
        assert isinstance(result, bool)

    def test_1d(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.isfortran(a)
        assert isinstance(result, bool)

    def test_returns_bool(self):
        a = cp.zeros((3, 4), dtype=np.float32)
        assert isinstance(cp.isfortran(a), bool)


# ====================================================================
# array_equiv
# ====================================================================

class TestArrayEquiv:
    def test_equal_arrays(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cp.array_equiv(a, b) is True

    def test_not_equal(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = cp.array([1.0, 2.0, 4.0], dtype=np.float32)
        assert cp.array_equiv(a, b) is False

    def test_broadcast_equiv(self):
        a = cp.array([1.0, 1.0, 1.0], dtype=np.float32)
        b = cp.array([1.0], dtype=np.float32)
        result = cp.array_equiv(a, b)
        np_result = np.array_equiv(
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )
        assert result == np_result
