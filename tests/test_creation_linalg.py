"""Tests for creation functions (diag, identity, tri, triu, tril, logspace,
meshgrid, indices), linear algebra module, and utility functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp
from macmetalpy.creation import (
    diag, identity, tri, triu, tril, logspace, meshgrid, indices,
)
from macmetalpy import linalg
from macmetalpy.math_ops import copy, ascontiguousarray, trace, diagonal


# ======================================================================
# Creation functions
# ======================================================================

class TestDiag:
    def test_diag_from_1d(self):
        """Construct a diagonal matrix from a 1-D array."""
        v = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = diag(v)
        expected = np.diag([1.0, 2.0, 3.0]).astype(np.float32)
        assert result.shape == (3, 3)
        assert_array_equal(result.get(), expected)

    def test_diag_from_2d(self):
        """Extract diagonal from a 2-D array."""
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = diag(m)
        expected = np.array([1, 5, 9], dtype=np.float32)
        assert result.shape == (3,)
        assert_array_equal(result.get(), expected)

    def test_diag_offset_positive(self):
        """Construct diagonal with positive offset."""
        v = cp.array([1.0, 2.0], dtype=np.float32)
        result = diag(v, k=1)
        expected = np.diag([1.0, 2.0], k=1).astype(np.float32)
        assert_array_equal(result.get(), expected)

    def test_diag_offset_negative(self):
        """Construct diagonal with negative offset."""
        v = cp.array([1.0, 2.0], dtype=np.float32)
        result = diag(v, k=-1)
        expected = np.diag([1.0, 2.0], k=-1).astype(np.float32)
        assert_array_equal(result.get(), expected)

    def test_diag_from_numpy_array(self):
        """Accepts plain numpy arrays."""
        v = np.array([4.0, 5.0], dtype=np.float32)
        result = diag(v)
        expected = np.diag(v)
        assert_array_equal(result.get(), expected)


class TestIdentity:
    def test_identity_3(self):
        result = identity(3)
        expected = np.eye(3, dtype=np.float32)
        assert result.shape == (3, 3)
        assert_array_equal(result.get(), expected)

    def test_identity_dtype(self):
        result = identity(4, dtype=np.int32)
        expected = np.eye(4, dtype=np.int32)
        assert result.dtype == np.int32
        assert_array_equal(result.get(), expected)


class TestTri:
    def test_tri_square(self):
        result = tri(3)
        expected = np.tri(3, dtype=np.float32)
        assert result.shape == (3, 3)
        assert_array_equal(result.get(), expected)

    def test_tri_rect(self):
        result = tri(3, 4)
        expected = np.tri(3, 4, dtype=np.float32)
        assert result.shape == (3, 4)
        assert_array_equal(result.get(), expected)

    def test_tri_with_offset(self):
        result = tri(4, 4, k=1)
        expected = np.tri(4, 4, k=1, dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_tri_dtype(self):
        result = tri(3, dtype=np.int32)
        assert result.dtype == np.int32


class TestTriu:
    def test_triu_basic(self):
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = triu(m)
        expected = np.triu(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
        assert_array_equal(result.get(), expected)

    def test_triu_offset(self):
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = triu(m, k=1)
        expected = np.triu(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), k=1)
        assert_array_equal(result.get(), expected)

    def test_triu_from_numpy(self):
        m = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = triu(m)
        expected = np.triu(m)
        assert_array_equal(result.get(), expected)


class TestTril:
    def test_tril_basic(self):
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = tril(m)
        expected = np.tril(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
        assert_array_equal(result.get(), expected)

    def test_tril_offset(self):
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = tril(m, k=-1)
        expected = np.tril(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), k=-1)
        assert_array_equal(result.get(), expected)


class TestLogspace:
    def test_logspace_basic(self):
        result = logspace(0, 2, num=5)
        expected = np.logspace(0, 2, num=5).astype(np.float32)
        assert result.shape == (5,)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_logspace_dtype(self):
        result = logspace(0, 1, num=3, dtype=np.float32)
        assert result.dtype == np.float32

    def test_logspace_default_num(self):
        result = logspace(0, 1)
        assert result.shape == (50,)


class TestMeshgrid:
    def test_meshgrid_2d(self):
        x = cp.array([1, 2, 3], dtype=np.float32)
        y = cp.array([4, 5], dtype=np.float32)
        X, Y = meshgrid(x, y)
        X_np, Y_np = np.meshgrid([1, 2, 3], [4, 5])
        assert_allclose(X.get(), X_np.astype(np.float32), rtol=1e-5)
        assert_allclose(Y.get(), Y_np.astype(np.float32), rtol=1e-5)

    def test_meshgrid_indexing_ij(self):
        x = cp.array([1, 2], dtype=np.float32)
        y = cp.array([3, 4, 5], dtype=np.float32)
        X, Y = meshgrid(x, y, indexing='ij')
        X_np, Y_np = np.meshgrid([1, 2], [3, 4, 5], indexing='ij')
        assert_allclose(X.get(), X_np.astype(np.float32), rtol=1e-5)
        assert_allclose(Y.get(), Y_np.astype(np.float32), rtol=1e-5)

    def test_meshgrid_numpy_input(self):
        x = np.array([1, 2], dtype=np.float32)
        y = np.array([3, 4], dtype=np.float32)
        X, Y = meshgrid(x, y)
        X_np, Y_np = np.meshgrid(x, y)
        assert_allclose(X.get(), X_np, rtol=1e-5)
        assert_allclose(Y.get(), Y_np, rtol=1e-5)


class TestIndices:
    def test_indices_2d(self):
        result = indices((2, 3))
        expected = np.indices((2, 3))
        assert result.shape == expected.shape
        assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)

    def test_indices_3d(self):
        result = indices((2, 3, 4))
        expected = np.indices((2, 3, 4))
        assert result.shape == expected.shape


# ======================================================================
# Linear algebra
# ======================================================================

class TestLinalgNorm:
    def test_vector_norm(self):
        x = cp.array([3.0, 4.0], dtype=np.float32)
        result = linalg.norm(x)
        assert_allclose(float(result), 5.0, rtol=1e-5)

    def test_matrix_frobenius_norm(self):
        m = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = linalg.norm(m)
        expected = np.linalg.norm(np.array([[1, 2], [3, 4]], dtype=np.float32))
        assert_allclose(float(result), expected, rtol=1e-5)

    def test_norm_axis(self):
        m = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = linalg.norm(m, axis=1)
        expected = np.linalg.norm(np.array([[1, 2], [3, 4]], dtype=np.float32), axis=1)
        assert_allclose(result.get(), expected, rtol=1e-5)


class TestLinalgInv:
    def test_inv_2x2(self):
        m = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = linalg.inv(m)
        expected = np.linalg.inv(np.array([[1, 2], [3, 4]], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-4, atol=1e-6)

    def test_inv_identity(self):
        m = cp.eye(3, dtype=np.float32)
        result = linalg.inv(m)
        expected = np.eye(3, dtype=np.float32)
        assert_allclose(result.get(), expected, rtol=1e-5, atol=1e-6)


class TestLinalgDet:
    def test_det_2x2(self):
        m = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = linalg.det(m)
        expected = np.linalg.det(np.array([[1, 2], [3, 4]], dtype=np.float32))
        assert_allclose(float(result), expected, rtol=1e-5)

    def test_det_identity(self):
        m = cp.eye(4, dtype=np.float32)
        result = linalg.det(m)
        assert_allclose(float(result), 1.0, rtol=1e-5)


class TestLinalgSolve:
    def test_solve_simple(self):
        a = cp.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        b = cp.array([9.0, 8.0], dtype=np.float32)
        result = linalg.solve(a, b)
        expected = np.linalg.solve(
            np.array([[3, 1], [1, 2]], dtype=np.float32),
            np.array([9, 8], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-4, atol=1e-6)

    def test_solve_identity(self):
        a = cp.eye(3, dtype=np.float32)
        b = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = linalg.solve(a, b)
        assert_allclose(result.get(), [1.0, 2.0, 3.0], rtol=1e-5)


class TestLinalgEigh:
    def test_eigh_symmetric(self):
        m_np = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        m = cp.array(m_np)
        w, v = linalg.eigh(m)
        w_np, v_np = np.linalg.eigh(m_np)
        assert_allclose(w.get(), w_np, rtol=1e-4, atol=1e-6)
        # Eigenvectors may differ in sign, check via absolute values
        assert_allclose(np.abs(v.get()), np.abs(v_np), rtol=1e-4, atol=1e-6)


class TestLinalgSVD:
    def test_svd_basic(self):
        m_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        m = cp.array(m_np)
        u, s, vh = linalg.svd(m)
        u_np, s_np, vh_np = np.linalg.svd(m_np)
        assert_allclose(s.get(), s_np, rtol=1e-4, atol=1e-6)
        # Reconstruct via reduced SVD and compare
        u_r, s_r, vh_r = linalg.svd(m, full_matrices=False)
        reconstructed = u_r.get() @ np.diag(s_r.get()) @ vh_r.get()
        assert_allclose(reconstructed, m_np, rtol=1e-4, atol=1e-5)

    def test_svd_square(self):
        m_np = np.array([[4.0, 0.0], [3.0, -5.0]], dtype=np.float32)
        m = cp.array(m_np)
        u, s, vh = linalg.svd(m)
        s_np = np.linalg.svd(m_np, compute_uv=False)
        assert_allclose(s.get(), s_np, rtol=1e-4, atol=1e-6)


class TestLinalgCholesky:
    def test_cholesky_spd(self):
        # Symmetric positive-definite matrix
        m_np = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        m = cp.array(m_np)
        L = linalg.cholesky(m)
        L_np = np.linalg.cholesky(m_np)
        assert_allclose(L.get(), L_np, rtol=1e-4, atol=1e-6)

    def test_cholesky_reconstruction(self):
        m_np = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        m = cp.array(m_np)
        L = linalg.cholesky(m)
        reconstructed = L.get() @ L.get().T
        assert_allclose(reconstructed, m_np, rtol=1e-4, atol=1e-5)


# ======================================================================
# Utility functions
# ======================================================================

class TestCopy:
    def test_copy_values(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = copy(a)
        assert_array_equal(b.get(), a.get())

    def test_copy_is_independent(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = copy(a)
        # Modify the copy; original should not change
        b_np = b.get()
        b_np[0] = 99.0
        assert a.get()[0] == 1.0


class TestAscontiguousarray:
    def test_ascontiguousarray_basic(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = ascontiguousarray(a)
        assert_array_equal(b.get(), a.get())

    def test_ascontiguousarray_from_numpy(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = ascontiguousarray(a_np)
        assert_array_equal(b.get(), a_np)


class TestTrace:
    def test_trace_basic(self):
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = trace(m)
        assert_allclose(float(result), 15.0, rtol=1e-5)

    def test_trace_offset(self):
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = trace(m, offset=1)
        expected = np.trace(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), offset=1)
        assert_allclose(float(result), expected, rtol=1e-5)


class TestDiagonal:
    def test_diagonal_basic(self):
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = diagonal(m)
        expected = np.array([1, 5, 9], dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_diagonal_offset(self):
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = diagonal(m, offset=1)
        expected = np.array([2, 6], dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_diagonal_negative_offset(self):
        m = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = diagonal(m, offset=-1)
        expected = np.array([4, 8], dtype=np.float32)
        assert_array_equal(result.get(), expected)
