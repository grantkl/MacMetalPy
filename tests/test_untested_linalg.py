"""Tests for untested linalg APIs: cond, eig, eigvals, matrix_power, multi_dot,
tensorinv, tensorsolve.

Verifies macmetalpy results match numpy with float32 tolerance.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy import linalg as la


# ── helpers ────────────────────────────────────────────────────────────────

LINALG_TOL = dict(rtol=1e-3, atol=1e-3)


def _well_conditioned(n=4, seed=42):
    """Return a well-conditioned n x n float32 matrix (A^T A + I)."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n).astype(np.float32)
    return A.T @ A + np.eye(n, dtype=np.float32)


def _symmetric(n=4, seed=42):
    """Return a symmetric float32 matrix."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n).astype(np.float32)
    return (A + A.T) / 2


# ======================================================================
# cond
# ======================================================================

class TestCond:
    def test_identity(self):
        A = np.eye(4, dtype=np.float32)
        result = la.cond(cp.array(A))
        expected = np.linalg.cond(A)
        npt.assert_allclose(float(result), float(expected), **LINALG_TOL)

    def test_well_conditioned(self):
        A = _well_conditioned(4)
        result = la.cond(cp.array(A))
        expected = np.linalg.cond(A)
        npt.assert_allclose(float(result), float(expected), rtol=1e-2, atol=1e-2)

    def test_ord_1(self):
        A = _well_conditioned(4)
        result = la.cond(cp.array(A), p=1)
        expected = np.linalg.cond(A, p=1)
        npt.assert_allclose(float(result), float(expected), rtol=1e-2, atol=1e-2)

    def test_ord_inf(self):
        A = _well_conditioned(4)
        result = la.cond(cp.array(A), p=np.inf)
        expected = np.linalg.cond(A, p=np.inf)
        npt.assert_allclose(float(result), float(expected), rtol=1e-2, atol=1e-2)

    def test_ord_fro(self):
        A = _well_conditioned(4)
        result = la.cond(cp.array(A), p='fro')
        expected = np.linalg.cond(A, p='fro')
        npt.assert_allclose(float(result), float(expected), rtol=1e-2, atol=1e-2)

    def test_diagonal(self):
        A = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
        result = la.cond(cp.array(A))
        expected = np.linalg.cond(A)
        npt.assert_allclose(float(result), float(expected), **LINALG_TOL)


# ======================================================================
# eig
# ======================================================================

class TestEig:
    def test_identity(self):
        A = np.eye(4, dtype=np.float32)
        w, v = la.eig(cp.array(A))
        w_np, v_np = np.linalg.eig(A)
        # Eigenvalues should all be 1
        npt.assert_allclose(sorted(np.abs(w.get())), sorted(np.abs(w_np)), **LINALG_TOL)

    def test_diagonal(self):
        A = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
        w, v = la.eig(cp.array(A))
        w_np, _ = np.linalg.eig(A)
        # Eigenvalues of a diagonal matrix are the diagonal entries
        npt.assert_allclose(sorted(np.abs(w.get())), sorted(np.abs(w_np)), **LINALG_TOL)

    def test_symmetric(self):
        A = _symmetric(4)
        w, v = la.eig(cp.array(A))
        # Verify A @ v = v @ diag(w) — reconstruction test
        Av = np.array(A) @ v.get()
        vw = v.get() @ np.diag(w.get())
        npt.assert_allclose(np.abs(Av), np.abs(vw), rtol=1e-2, atol=1e-2)

    def test_output_shapes(self):
        n = 4
        A = _well_conditioned(n)
        w, v = la.eig(cp.array(A))
        assert w.shape == (n,)
        assert v.shape == (n, n)

    def test_2x2(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        w, v = la.eig(cp.array(A))
        w_np, _ = np.linalg.eig(A)
        npt.assert_allclose(sorted(np.abs(w.get())), sorted(np.abs(w_np)), **LINALG_TOL)


# ======================================================================
# eigvals
# ======================================================================

class TestEigvals:
    def test_identity(self):
        A = np.eye(4, dtype=np.float32)
        result = la.eigvals(cp.array(A))
        expected = np.linalg.eigvals(A)
        npt.assert_allclose(sorted(np.abs(result.get())), sorted(np.abs(expected)), **LINALG_TOL)

    def test_diagonal(self):
        A = np.diag([2.0, 5.0, 7.0, 11.0]).astype(np.float32)
        result = la.eigvals(cp.array(A))
        expected = np.linalg.eigvals(A)
        npt.assert_allclose(sorted(np.abs(result.get())), sorted(np.abs(expected)), **LINALG_TOL)

    def test_shape(self):
        A = _well_conditioned(8)
        result = la.eigvals(cp.array(A))
        assert result.shape == (8,)

    def test_symmetric(self):
        A = _symmetric(4)
        result = la.eigvals(cp.array(A))
        expected = np.linalg.eigvals(A)
        npt.assert_allclose(sorted(np.abs(result.get())), sorted(np.abs(expected)), **LINALG_TOL)

    def test_matches_eig(self):
        A = _well_conditioned(4)
        eigvals_result = la.eigvals(cp.array(A))
        eig_w, _ = la.eig(cp.array(A))
        npt.assert_allclose(
            sorted(np.abs(eigvals_result.get())),
            sorted(np.abs(eig_w.get())),
            **LINALG_TOL,
        )


# ======================================================================
# matrix_power
# ======================================================================

class TestMatrixPower:
    def test_power_0(self):
        A = _well_conditioned(4)
        result = la.matrix_power(cp.array(A), 0)
        expected = np.linalg.matrix_power(A, 0)
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)

    def test_power_1(self):
        A = _well_conditioned(4)
        result = la.matrix_power(cp.array(A), 1)
        expected = np.linalg.matrix_power(A, 1)
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)

    def test_power_2(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = la.matrix_power(cp.array(A), 2)
        expected = np.linalg.matrix_power(A, 2)
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)

    def test_power_3(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = la.matrix_power(cp.array(A), 3)
        expected = np.linalg.matrix_power(A, 3)
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)

    def test_power_negative(self):
        A = _well_conditioned(4)
        result = la.matrix_power(cp.array(A), -1)
        expected = np.linalg.matrix_power(A, -1)
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)

    def test_power_negative_2(self):
        A = _well_conditioned(4)
        result = la.matrix_power(cp.array(A), -2)
        expected = np.linalg.matrix_power(A, -2)
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)

    def test_identity_power(self):
        I = np.eye(4, dtype=np.float32)
        result = la.matrix_power(cp.array(I), 10)
        npt.assert_allclose(result.get(), I, **LINALG_TOL)

    def test_output_shape(self):
        A = np.ones((4, 4), dtype=np.float32)
        result = la.matrix_power(cp.array(A), 2)
        assert result.shape == (4, 4)


# ======================================================================
# multi_dot
# ======================================================================

class TestMultiDot:
    def test_two_matrices(self):
        A = np.arange(6, dtype=np.float32).reshape(2, 3)
        B = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = la.multi_dot([cp.array(A), cp.array(B)])
        expected = np.linalg.multi_dot([A, B])
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)

    def test_three_matrices(self):
        rng = np.random.RandomState(42)
        A = rng.randn(2, 4).astype(np.float32)
        B = rng.randn(4, 3).astype(np.float32)
        C = rng.randn(3, 5).astype(np.float32)
        result = la.multi_dot([cp.array(A), cp.array(B), cp.array(C)])
        expected = np.linalg.multi_dot([A, B, C])
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)

    def test_four_matrices(self):
        rng = np.random.RandomState(99)
        A = rng.randn(3, 4).astype(np.float32)
        B = rng.randn(4, 2).astype(np.float32)
        C = rng.randn(2, 5).astype(np.float32)
        D = rng.randn(5, 3).astype(np.float32)
        result = la.multi_dot([cp.array(A), cp.array(B), cp.array(C), cp.array(D)])
        expected = np.linalg.multi_dot([A, B, C, D])
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)

    def test_vector_matrix_vector(self):
        rng = np.random.RandomState(42)
        v1 = rng.randn(4).astype(np.float32)
        M = rng.randn(4, 4).astype(np.float32)
        v2 = rng.randn(4).astype(np.float32)
        result = la.multi_dot([cp.array(v1), cp.array(M), cp.array(v2)])
        expected = np.linalg.multi_dot([v1, M, v2])
        npt.assert_allclose(float(result), float(expected), rtol=1e-2, atol=1e-2)

    def test_output_shape(self):
        A = np.ones((2, 3), dtype=np.float32)
        B = np.ones((3, 4), dtype=np.float32)
        result = la.multi_dot([cp.array(A), cp.array(B)])
        assert result.shape == (2, 4)


# ======================================================================
# tensorinv
# ======================================================================

class TestTensorinv:
    def test_ind2_identity_like(self):
        a_np = np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)
        result = la.tensorinv(cp.array(a_np))
        expected = np.linalg.tensorinv(a_np)
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)

    def test_ind1(self):
        a_np = np.eye(4, dtype=np.float32).reshape(4, 2, 2)
        result = la.tensorinv(cp.array(a_np), ind=1)
        expected = np.linalg.tensorinv(a_np, ind=1)
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)

    def test_output_shape_ind2(self):
        a_np = np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)
        result = la.tensorinv(cp.array(a_np), ind=2)
        assert result.shape == (2, 2, 2, 2)

    def test_output_shape_ind1(self):
        a_np = np.eye(4, dtype=np.float32).reshape(4, 2, 2)
        result = la.tensorinv(cp.array(a_np), ind=1)
        assert result.shape == (2, 2, 4)

    def test_roundtrip(self):
        # tensorinv(tensorinv(A)) should give back A for an invertible tensor
        a_np = np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)
        inv1 = la.tensorinv(cp.array(a_np))
        inv2 = la.tensorinv(inv1)
        npt.assert_allclose(inv2.get(), a_np, **LINALG_TOL)


# ======================================================================
# tensorsolve
# ======================================================================

class TestTensorsolve:
    def test_2d_identity(self):
        a_np = np.eye(4, dtype=np.float32)
        b_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = la.tensorsolve(cp.array(a_np), cp.array(b_np))
        expected = np.linalg.tensorsolve(a_np, b_np)
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)

    def test_4d_identity(self):
        a_np = np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)
        b_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = la.tensorsolve(cp.array(a_np), cp.array(b_np))
        expected = np.linalg.tensorsolve(a_np, b_np)
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)

    def test_output_shape(self):
        a_np = np.eye(4, dtype=np.float32).reshape(2, 2, 2, 2)
        b_np = np.ones((2, 2), dtype=np.float32)
        result = la.tensorsolve(cp.array(a_np), cp.array(b_np))
        assert result.shape == (2, 2)

    def test_simple_system(self):
        # Simple 2x2 system
        a_np = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        b_np = np.array([5.0, 7.0], dtype=np.float32)
        result = la.tensorsolve(cp.array(a_np), cp.array(b_np))
        expected = np.linalg.tensorsolve(a_np, b_np)
        npt.assert_allclose(result.get(), expected, **LINALG_TOL)
