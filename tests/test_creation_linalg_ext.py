"""Tests for extended creation and linalg functions (Task #7)."""

import numpy as np
import numpy.testing as npt
import pytest

from macmetalpy.creation import fromfunction, diagflat, vander, asanyarray
from macmetalpy.creation import array as mmp_array
import macmetalpy.linalg as la
from macmetalpy.linalg_top import vdot, inner, outer, tensordot, einsum, kron, matmul, cross
from macmetalpy.ndarray import ndarray


# ── creation functions ──────────────────────────────────────────────────

class TestFromfunction:
    def test_basic(self):
        result = fromfunction(lambda i, j: i + j, (3, 3), dtype=float)
        assert isinstance(result, ndarray)
        expected = np.fromfunction(lambda i, j: i + j, (3, 3), dtype=float)
        npt.assert_allclose(result.get(), expected, atol=1e-6)

    def test_1d(self):
        result = fromfunction(lambda i: i * 2, (5,), dtype=float)
        expected = np.fromfunction(lambda i: i * 2, (5,), dtype=float)
        npt.assert_allclose(result.get(), expected, atol=1e-6)


class TestDiagflat:
    def test_basic(self):
        v = mmp_array([1, 2, 3], dtype=np.float32)
        result = diagflat(v)
        assert isinstance(result, ndarray)
        expected = np.diagflat([1, 2, 3])
        npt.assert_allclose(result.get(), expected, atol=1e-6)

    def test_with_k(self):
        result = diagflat([1, 2], k=1)
        expected = np.diagflat([1, 2], k=1)
        npt.assert_allclose(result.get(), expected, atol=1e-6)

    def test_negative_k(self):
        result = diagflat([1, 2], k=-1)
        expected = np.diagflat([1, 2], k=-1)
        npt.assert_allclose(result.get(), expected, atol=1e-6)


class TestVander:
    def test_basic(self):
        x = mmp_array([1, 2, 3, 5], dtype=np.float32)
        result = vander(x)
        expected = np.vander([1, 2, 3, 5])
        npt.assert_allclose(result.get(), expected, atol=1e-3)

    def test_increasing(self):
        result = vander([1, 2, 3], increasing=True)
        expected = np.vander([1, 2, 3], increasing=True)
        npt.assert_allclose(result.get(), expected, atol=1e-6)

    def test_with_N(self):
        result = vander([1, 2, 3], N=2)
        expected = np.vander([1, 2, 3], N=2)
        npt.assert_allclose(result.get(), expected, atol=1e-6)


class TestAsanyarray:
    def test_ndarray_passthrough(self):
        a = mmp_array([1.0, 2.0, 3.0])
        result = asanyarray(a)
        assert isinstance(result, ndarray)
        npt.assert_allclose(result.get(), a.get(), atol=1e-6)

    def test_list_input(self):
        result = asanyarray([1, 2, 3])
        assert isinstance(result, ndarray)

    def test_with_dtype(self):
        result = asanyarray([1, 2, 3], dtype=np.float32)
        assert result.dtype == np.float32


# ── linalg functions ────────────────────────────────────────────────────

def _well_conditioned_float32(n=3):
    """Return a well-conditioned n x n float32 matrix."""
    np.random.seed(42)
    A = np.random.randn(n, n).astype(np.float32)
    # Make well-conditioned: A^T A + I
    return (A.T @ A + np.eye(n, dtype=np.float32))


class TestMatrixPower:
    def test_square(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        result = la.matrix_power(a, 2)
        expected = np.linalg.matrix_power(A, 2)
        npt.assert_allclose(result.get(), expected, atol=1e-3)

    def test_zero_power(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        result = la.matrix_power(a, 0)
        npt.assert_allclose(result.get(), np.eye(3, dtype=np.float32), atol=1e-6)


class TestQR:
    def test_reduced(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        q, r = la.qr(a)
        assert isinstance(q, ndarray)
        assert isinstance(r, ndarray)
        # Q @ R should reconstruct A
        reconstructed = q.get() @ r.get()
        npt.assert_allclose(reconstructed, A, atol=1e-4)

    def test_mode_r(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        r = la.qr(a, mode='r')
        assert isinstance(r, ndarray)


class TestEig:
    def test_basic(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        w, v = la.eig(a)
        assert isinstance(w, ndarray)
        assert isinstance(v, ndarray)
        # Check eigenvalue equation: A @ v[:, i] ~ w[i] * v[:, i]
        w_np, v_np = w.get(), v.get()
        for i in range(len(w_np)):
            lhs = A @ v_np[:, i]
            rhs = w_np[i] * v_np[:, i]
            npt.assert_allclose(lhs, rhs, atol=1e-3)


class TestEigvals:
    def test_basic(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        w = la.eigvals(a)
        expected = np.linalg.eigvals(A)
        npt.assert_allclose(sorted(np.abs(w.get())), sorted(np.abs(expected)), atol=1e-3)


class TestEigvalsh:
    def test_symmetric(self):
        A = _well_conditioned_float32()
        # A is already symmetric (A^T A + I)
        a = mmp_array(A)
        w = la.eigvalsh(a)
        expected = np.linalg.eigvalsh(A)
        npt.assert_allclose(sorted(w.get()), sorted(expected), atol=1e-3)


class TestCond:
    def test_basic(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        result = la.cond(a)
        expected = np.linalg.cond(A)
        npt.assert_allclose(float(result.get()), expected, rtol=1e-3)


class TestMatrixRank:
    def test_full_rank(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        result = la.matrix_rank(a)
        assert int(result.get()) == 3

    def test_rank_deficient(self):
        A = np.array([[1, 2], [2, 4]], dtype=np.float32)
        a = mmp_array(A)
        result = la.matrix_rank(a)
        assert int(result.get()) == 1


class TestSlogdet:
    def test_basic(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        sign, logdet = la.slogdet(a)
        exp_sign, exp_logdet = np.linalg.slogdet(A)
        npt.assert_allclose(float(sign.get()), exp_sign, atol=1e-6)
        npt.assert_allclose(float(logdet.get()), exp_logdet, atol=1e-3)


class TestLstsq:
    def test_overdetermined(self):
        A = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float32)
        b = np.array([1, 2, 2], dtype=np.float32)
        a_m = mmp_array(A)
        b_m = mmp_array(b)
        x, residuals, rank, sv = la.lstsq(a_m, b_m, rcond=None)
        x_np, _, rank_np, _ = np.linalg.lstsq(A, b, rcond=None)
        npt.assert_allclose(x.get(), x_np, atol=1e-4)
        assert rank == int(rank_np)


class TestPinv:
    def test_basic(self):
        A = _well_conditioned_float32()
        a = mmp_array(A)
        result = la.pinv(a)
        expected = np.linalg.pinv(A)
        npt.assert_allclose(result.get(), expected, atol=1e-3)


# ── top-level linalg functions ──────────────────────────────────────────

class TestVdot:
    def test_basic(self):
        a = mmp_array([1, 2, 3], dtype=np.float32)
        b = mmp_array([4, 5, 6], dtype=np.float32)
        result = vdot(a, b)
        npt.assert_allclose(result.get(), np.vdot([1, 2, 3], [4, 5, 6]), atol=1e-6)


class TestInner:
    def test_1d(self):
        a = mmp_array([1, 2, 3], dtype=np.float32)
        b = mmp_array([4, 5, 6], dtype=np.float32)
        result = inner(a, b)
        npt.assert_allclose(result.get(), np.inner([1, 2, 3], [4, 5, 6]), atol=1e-6)


class TestOuter:
    def test_basic(self):
        a = mmp_array([1, 2, 3], dtype=np.float32)
        b = mmp_array([4, 5], dtype=np.float32)
        result = outer(a, b)
        expected = np.outer([1, 2, 3], [4, 5])
        npt.assert_allclose(result.get(), expected, atol=1e-6)


class TestTensordot:
    def test_basic(self):
        a = mmp_array(np.arange(6, dtype=np.float32).reshape(2, 3))
        b = mmp_array(np.arange(12, dtype=np.float32).reshape(3, 4))
        result = tensordot(a, b, axes=1)
        expected = np.tensordot(np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4), axes=1)
        npt.assert_allclose(result.get(), expected, atol=1e-4)


class TestEinsum:
    def test_matrix_multiply(self):
        a = mmp_array(np.arange(6, dtype=np.float32).reshape(2, 3))
        b = mmp_array(np.arange(12, dtype=np.float32).reshape(3, 4))
        result = einsum('ij,jk->ik', a, b)
        expected = np.einsum('ij,jk->ik', np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4))
        npt.assert_allclose(result.get(), expected, atol=1e-4)

    def test_trace(self):
        a = mmp_array(np.eye(3, dtype=np.float32))
        result = einsum('ii', a)
        npt.assert_allclose(result.get(), 3.0, atol=1e-6)


class TestKron:
    def test_basic(self):
        a = mmp_array([[1, 0], [0, 1]], dtype=np.float32)
        b = mmp_array([[1, 2], [3, 4]], dtype=np.float32)
        result = kron(a, b)
        expected = np.kron([[1, 0], [0, 1]], [[1, 2], [3, 4]])
        npt.assert_allclose(result.get(), expected, atol=1e-6)


class TestMatmul:
    def test_basic(self):
        a = mmp_array(np.arange(6, dtype=np.float32).reshape(2, 3))
        b = mmp_array(np.arange(12, dtype=np.float32).reshape(3, 4))
        result = matmul(a, b)
        expected = np.matmul(np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4))
        npt.assert_allclose(result.get(), expected, atol=1e-4)


class TestCross:
    def test_3d(self):
        a = mmp_array([1, 2, 3], dtype=np.float32)
        b = mmp_array([4, 5, 6], dtype=np.float32)
        result = cross(a, b)
        expected = np.cross([1, 2, 3], [4, 5, 6])
        npt.assert_allclose(result.get(), expected, atol=1e-6)
