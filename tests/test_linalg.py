"""Tests for linear algebra functions — cp.linalg.* and top-level linalg helpers.

Consolidates test_creation_linalg.py + test_creation_linalg_ext.py.
FLOAT_DTYPES only for linalg (Metal has no float64).

Ref: cupy_tests/linalg_tests/
Target: ~182 parametrized cases.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy import linalg as la
from macmetalpy.linalg_top import vdot, inner, outer, tensordot, einsum, kron, matmul, cross

from conftest import FLOAT_DTYPES, assert_eq, tol_for


@pytest.fixture(autouse=True)
def _skip_float16(request):
    """numpy.linalg does not support float16; skip those parametrizations."""
    dtype = request.node.callspec.params.get("dtype") if hasattr(request.node, "callspec") else None
    if dtype == np.float16:
        pytest.skip("numpy.linalg does not support float16")


# ── helpers ────────────────────────────────────────────────────────────────

def _well_conditioned(n=3, dtype=np.float32):
    """Return a well-conditioned n x n float32 matrix (A^T A + I)."""
    rng = np.random.RandomState(42)
    A = rng.randn(n, n).astype(dtype)
    return (A.T @ A + np.eye(n, dtype=dtype))


def _spd(n=3, dtype=np.float32):
    """Symmetric positive-definite matrix."""
    return _well_conditioned(n, dtype)


# ======================================================================
# cp.linalg module functions
# ======================================================================

# ── norm ──────────────────────────────────────────────────────────────────
# Ref: cupy_tests/linalg_tests/test_norms.py

class TestNorm:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_vector_norm_default(self, dtype):
        a_np = np.array([3.0, 4.0], dtype=dtype)
        a = cp.array(a_np)
        result = la.norm(a)
        expected = np.linalg.norm(a_np)
        npt.assert_allclose(float(result), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("ord", [1, 2, np.inf])
    def test_vector_norm_ord(self, dtype, ord):
        a_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=dtype)
        a = cp.array(a_np)
        result = la.norm(a, ord=ord)
        expected = np.linalg.norm(a_np, ord=ord)
        npt.assert_allclose(float(result), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_matrix_frobenius(self, dtype):
        m_np = np.array([[1, 2], [3, 4]], dtype=dtype)
        m = cp.array(m_np)
        result = la.norm(m)
        expected = np.linalg.norm(m_np)
        npt.assert_allclose(float(result), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("ord", [1, np.inf, 'fro'])
    def test_matrix_norm_ord(self, dtype, ord):
        m_np = np.array([[1, 2], [3, 4]], dtype=dtype)
        m = cp.array(m_np)
        result = la.norm(m, ord=ord)
        expected = np.linalg.norm(m_np, ord=ord)
        npt.assert_allclose(float(result), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_norm_axis(self, dtype):
        m_np = np.array([[1, 2], [3, 4]], dtype=dtype)
        m = cp.array(m_np)
        result = la.norm(m, axis=1)
        expected = np.linalg.norm(m_np, axis=1)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


# ── inv ───────────────────────────────────────────────────────────────────
# Ref: cupy_tests/linalg_tests/test_solve.py

class TestInv:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        m_np = np.array([[1, 2], [3, 4]], dtype=dtype)
        result = la.inv(cp.array(m_np))
        expected = np.linalg.inv(m_np)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1x1(self, dtype):
        m_np = np.array([[5.0]], dtype=dtype)
        result = la.inv(cp.array(m_np))
        expected = np.linalg.inv(m_np)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_identity(self, dtype):
        m = cp.eye(3, dtype=dtype)
        result = la.inv(m)
        npt.assert_allclose(result.get(), np.eye(3, dtype=dtype), **tol_for(dtype))

    def test_singular_error(self):
        m = cp.array([[1, 2], [2, 4]], dtype=np.float32)
        with pytest.raises(np.linalg.LinAlgError):
            la.inv(m)

    def test_non_square_error(self):
        m = cp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            la.inv(m)


# ── det ───────────────────────────────────────────────────────────────────

class TestDet:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        m_np = np.array([[1, 2], [3, 4]], dtype=dtype)
        result = la.det(cp.array(m_np))
        expected = np.linalg.det(m_np)
        npt.assert_allclose(float(result), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_identity_equals_one(self, dtype):
        result = la.det(cp.eye(4, dtype=dtype))
        npt.assert_allclose(float(result), 1.0, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_singular_equals_zero(self, dtype):
        m_np = np.array([[1, 2], [2, 4]], dtype=dtype)
        result = la.det(cp.array(m_np))
        npt.assert_allclose(float(result), 0.0, atol=1e-4)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1x1(self, dtype):
        m_np = np.array([[7.0]], dtype=dtype)
        result = la.det(cp.array(m_np))
        npt.assert_allclose(float(result), 7.0, **tol_for(dtype))


# ── solve ─────────────────────────────────────────────────────────────────

class TestSolve:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        a_np = np.array([[3, 1], [1, 2]], dtype=dtype)
        b_np = np.array([9, 8], dtype=dtype)
        result = la.solve(cp.array(a_np), cp.array(b_np))
        expected = np.linalg.solve(a_np, b_np)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_identity(self, dtype):
        a = cp.eye(3, dtype=dtype)
        b_np = np.array([1, 2, 3], dtype=dtype)
        result = la.solve(a, cp.array(b_np))
        npt.assert_allclose(result.get(), b_np, **tol_for(dtype))

    def test_singular_error(self):
        a = cp.array([[1, 2], [2, 4]], dtype=np.float32)
        b = cp.array([1, 2], dtype=np.float32)
        with pytest.raises(np.linalg.LinAlgError):
            la.solve(a, b)


# ── eigh ──────────────────────────────────────────────────────────────────
# Ref: cupy_tests/linalg_tests/test_eigenvalue.py

class TestEigh:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        m_np = np.array([[2, 1], [1, 3]], dtype=dtype)
        w, v = la.eigh(cp.array(m_np))
        w_np, v_np = np.linalg.eigh(m_np)
        npt.assert_allclose(w.get(), w_np, rtol=1e-3, atol=1e-4)
        npt.assert_allclose(np.abs(v.get()), np.abs(v_np), rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_identity(self, dtype):
        m = cp.eye(3, dtype=dtype)
        w, v = la.eigh(m)
        npt.assert_allclose(w.get(), np.ones(3), rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_diagonal(self, dtype):
        m_np = np.diag([1.0, 4.0, 9.0]).astype(dtype)
        w, v = la.eigh(cp.array(m_np))
        npt.assert_allclose(sorted(w.get()), [1.0, 4.0, 9.0], rtol=1e-3, atol=1e-4)


# ── svd ───────────────────────────────────────────────────────────────────
# Ref: cupy_tests/linalg_tests/test_decomposition.py

class TestSVD:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        m_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype)
        u, s, vh = la.svd(cp.array(m_np))
        s_np = np.linalg.svd(m_np, compute_uv=False)
        npt.assert_allclose(s.get(), s_np, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_full_matrices_false(self, dtype):
        m_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype)
        u, s, vh = la.svd(cp.array(m_np), full_matrices=False)
        reconstructed = u.get() @ np.diag(s.get()) @ vh.get()
        npt.assert_allclose(reconstructed, m_np, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_full_matrices_true(self, dtype):
        m_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype)
        u, s, vh = la.svd(cp.array(m_np), full_matrices=True)
        assert u.shape == (3, 3)
        assert vh.shape == (2, 2)


# ── cholesky ──────────────────────────────────────────────────────────────

class TestCholesky:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        m_np = np.array([[4, 2], [2, 3]], dtype=dtype)
        L = la.cholesky(cp.array(m_np))
        L_np = np.linalg.cholesky(m_np)
        npt.assert_allclose(L.get(), L_np, rtol=1e-3, atol=1e-4)

    def test_non_pd_error(self):
        m = cp.array([[-1, 0], [0, -1]], dtype=np.float32)
        with pytest.raises(np.linalg.LinAlgError):
            la.cholesky(m)


# ── matrix_power ──────────────────────────────────────────────────────────

class TestMatrixPower:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_powers(self, dtype, n):
        A = _well_conditioned(3, dtype)
        result = la.matrix_power(cp.array(A), n)
        expected = np.linalg.matrix_power(A, n)
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_inverse_power(self, dtype):
        A = _well_conditioned(3, dtype)
        result = la.matrix_power(cp.array(A), -1)
        expected = np.linalg.matrix_power(A, -1)
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)


# ── qr ────────────────────────────────────────────────────────────────────

class TestQR:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduced(self, dtype):
        A = _well_conditioned(3, dtype)
        q, r = la.qr(cp.array(A))
        reconstructed = q.get() @ r.get()
        npt.assert_allclose(reconstructed, A, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_tall(self, dtype):
        rng = np.random.RandomState(42)
        A = rng.randn(5, 3).astype(dtype)
        q, r = la.qr(cp.array(A))
        reconstructed = q.get() @ r.get()
        npt.assert_allclose(reconstructed, A, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_wide(self, dtype):
        rng = np.random.RandomState(42)
        A = rng.randn(3, 5).astype(dtype)
        q, r = la.qr(cp.array(A))
        reconstructed = q.get() @ r.get()
        npt.assert_allclose(reconstructed, A, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mode_r(self, dtype):
        A = _well_conditioned(3, dtype)
        r = la.qr(cp.array(A), mode='r')
        assert r.shape[0] <= 3


# ── eig, eigvals, eigvalsh ────────────────────────────────────────────────

class TestEig:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        A = _well_conditioned(3, dtype)
        w, v = la.eig(cp.array(A))
        w_np, v_np = w.get(), v.get()
        for i in range(len(w_np)):
            lhs = A @ v_np[:, i]
            rhs = w_np[i] * v_np[:, i]
            npt.assert_allclose(lhs, rhs, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_diagonal(self, dtype):
        A = np.diag([1.0, 2.0, 3.0]).astype(dtype)
        w, v = la.eig(cp.array(A))
        npt.assert_allclose(sorted(np.abs(w.get())), [1.0, 2.0, 3.0], rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_symmetric(self, dtype):
        A = _spd(3, dtype)
        w, v = la.eig(cp.array(A))
        # Symmetric matrix: eigenvalues should be real
        assert np.all(np.abs(np.imag(w.get())) < 1e-3)


class TestEigvals:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        A = _well_conditioned(3, dtype)
        w = la.eigvals(cp.array(A))
        expected = np.linalg.eigvals(A)
        npt.assert_allclose(sorted(np.abs(w.get())), sorted(np.abs(expected)), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_diagonal(self, dtype):
        A = np.diag([2.0, 5.0, 8.0]).astype(dtype)
        w = la.eigvals(cp.array(A))
        npt.assert_allclose(sorted(np.abs(w.get())), [2.0, 5.0, 8.0], rtol=1e-3, atol=1e-3)


class TestEigvalsh:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_symmetric(self, dtype):
        A = _spd(3, dtype)
        w = la.eigvalsh(cp.array(A))
        expected = np.linalg.eigvalsh(A)
        npt.assert_allclose(sorted(w.get()), sorted(expected), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_diagonal(self, dtype):
        A = np.diag([1.0, 4.0, 9.0]).astype(dtype)
        w = la.eigvalsh(cp.array(A))
        npt.assert_allclose(sorted(w.get()), [1.0, 4.0, 9.0], rtol=1e-3, atol=1e-3)


# ── cond, matrix_rank, slogdet ────────────────────────────────────────────

class TestCond:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        A = _well_conditioned(3, dtype)
        result = la.cond(cp.array(A))
        expected = np.linalg.cond(A)
        npt.assert_allclose(float(result.get()), expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_identity(self, dtype):
        result = la.cond(cp.eye(3, dtype=dtype))
        npt.assert_allclose(float(result.get()), 1.0, rtol=1e-2)


class TestMatrixRank:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_full_rank(self, dtype):
        A = _well_conditioned(3, dtype)
        result = la.matrix_rank(cp.array(A))
        assert int(result.get()) == 3

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_rank_deficient(self, dtype):
        A = np.array([[1, 2], [2, 4]], dtype=dtype)
        result = la.matrix_rank(cp.array(A))
        assert int(result.get()) == 1

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_singular(self, dtype):
        A = np.zeros((3, 3), dtype=dtype)
        result = la.matrix_rank(cp.array(A))
        assert int(result.get()) == 0


class TestSlogdet:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        A = _well_conditioned(3, dtype)
        sign, logdet = la.slogdet(cp.array(A))
        exp_sign, exp_logdet = np.linalg.slogdet(A)
        npt.assert_allclose(float(sign.get()), exp_sign, atol=1e-4)
        npt.assert_allclose(float(logdet.get()), exp_logdet, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_singular(self, dtype):
        A = np.array([[1, 2], [2, 4]], dtype=dtype)
        sign, logdet = la.slogdet(cp.array(A))
        assert float(sign.get()) == 0.0 or np.isinf(float(logdet.get()))


# ── lstsq, pinv ──────────────────────────────────────────────────────────

class TestLstsq:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_overdetermined(self, dtype):
        A = np.array([[1, 1], [1, 2], [1, 3]], dtype=dtype)
        b = np.array([1, 2, 2], dtype=dtype)
        x, residuals, rank, sv = la.lstsq(cp.array(A), cp.array(b), rcond=None)
        x_np, _, rank_np, _ = np.linalg.lstsq(A, b, rcond=None)
        npt.assert_allclose(x.get(), x_np, rtol=1e-2, atol=1e-3)
        assert rank == int(rank_np)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_underdetermined(self, dtype):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        b = np.array([1, 2], dtype=dtype)
        x, residuals, rank, sv = la.lstsq(cp.array(A), cp.array(b), rcond=None)
        x_np, _, rank_np, _ = np.linalg.lstsq(A, b, rcond=None)
        npt.assert_allclose(x.get(), x_np, rtol=1e-2, atol=1e-3)


class TestPinv:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        A = _well_conditioned(3, dtype)
        result = la.pinv(cp.array(A))
        expected = np.linalg.pinv(A)
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_non_square(self, dtype):
        rng = np.random.RandomState(42)
        A = rng.randn(4, 2).astype(dtype)
        result = la.pinv(cp.array(A))
        expected = np.linalg.pinv(A)
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)


# ======================================================================
# Top-level linalg functions
# ======================================================================
# Ref: cupy_tests/linalg_tests/test_product.py

class TestVdot:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        a = cp.array([1, 2, 3], dtype=dtype)
        b = cp.array([4, 5, 6], dtype=dtype)
        result = vdot(a, b)
        expected = np.vdot(np.array([1, 2, 3], dtype=dtype),
                           np.array([4, 5, 6], dtype=dtype))
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, dtype):
        a_np = np.arange(6, dtype=dtype).reshape(2, 3)
        b_np = np.arange(6, 12, dtype=dtype).reshape(2, 3)
        result = vdot(cp.array(a_np), cp.array(b_np))
        expected = np.vdot(a_np, b_np)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


class TestInner:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1d(self, dtype):
        a = cp.array([1, 2, 3], dtype=dtype)
        b = cp.array([4, 5, 6], dtype=dtype)
        result = inner(a, b)
        expected = np.inner(np.array([1, 2, 3], dtype=dtype),
                            np.array([4, 5, 6], dtype=dtype))
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, dtype):
        a_np = np.arange(6, dtype=dtype).reshape(2, 3)
        b_np = np.arange(3, dtype=dtype)
        result = inner(cp.array(a_np), cp.array(b_np))
        expected = np.inner(a_np, b_np)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


class TestOuter:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        a = cp.array([1, 2, 3], dtype=dtype)
        b = cp.array([4, 5], dtype=dtype)
        result = outer(a, b)
        expected = np.outer(np.array([1, 2, 3], dtype=dtype),
                            np.array([4, 5], dtype=dtype))
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large(self, dtype):
        a_np = np.arange(10, dtype=dtype)
        b_np = np.arange(5, dtype=dtype)
        result = outer(cp.array(a_np), cp.array(b_np))
        expected = np.outer(a_np, b_np)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


class TestTensordot:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        a_np = np.arange(6, dtype=dtype).reshape(2, 3)
        b_np = np.arange(12, dtype=dtype).reshape(3, 4)
        result = tensordot(cp.array(a_np), cp.array(b_np), axes=1)
        expected = np.tensordot(a_np, b_np, axes=1)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_axes_tuple(self, dtype):
        a_np = np.arange(24, dtype=dtype).reshape(2, 3, 4)
        b_np = np.arange(12, dtype=dtype).reshape(3, 4)
        result = tensordot(cp.array(a_np), cp.array(b_np), axes=([1, 2], [0, 1]))
        expected = np.tensordot(a_np, b_np, axes=([1, 2], [0, 1]))
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)


class TestEinsum:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_matrix_multiply(self, dtype):
        a_np = np.arange(6, dtype=dtype).reshape(2, 3)
        b_np = np.arange(12, dtype=dtype).reshape(3, 4)
        result = einsum('ij,jk->ik', cp.array(a_np), cp.array(b_np))
        expected = np.einsum('ij,jk->ik', a_np, b_np)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_trace(self, dtype):
        a = cp.eye(3, dtype=dtype)
        result = einsum('ii', a)
        npt.assert_allclose(result.get(), 3.0, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_diagonal(self, dtype):
        a_np = np.arange(9, dtype=dtype).reshape(3, 3)
        result = einsum('ii->i', cp.array(a_np))
        expected = np.einsum('ii->i', a_np)
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


class TestKron:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        a = cp.array([[1, 0], [0, 1]], dtype=dtype)
        b = cp.array([[1, 2], [3, 4]], dtype=dtype)
        result = kron(a, b)
        expected = np.kron(np.array([[1, 0], [0, 1]], dtype=dtype),
                           np.array([[1, 2], [3, 4]], dtype=dtype))
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1d(self, dtype):
        a = cp.array([1, 2], dtype=dtype)
        b = cp.array([3, 4, 5], dtype=dtype)
        result = kron(a, b)
        expected = np.kron(np.array([1, 2], dtype=dtype),
                           np.array([3, 4, 5], dtype=dtype))
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))


class TestMatmul:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        a_np = np.arange(6, dtype=dtype).reshape(2, 3)
        b_np = np.arange(12, dtype=dtype).reshape(3, 4)
        result = matmul(cp.array(a_np), cp.array(b_np))
        expected = np.matmul(a_np, b_np)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_vector_matrix(self, dtype):
        a_np = np.array([1, 2, 3], dtype=dtype)
        b_np = np.arange(12, dtype=dtype).reshape(3, 4)
        result = matmul(cp.array(a_np), cp.array(b_np))
        expected = np.matmul(a_np, b_np)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_square(self, dtype):
        a_np = np.arange(9, dtype=dtype).reshape(3, 3)
        result = matmul(cp.array(a_np), cp.array(a_np))
        expected = np.matmul(a_np, a_np)
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)


class TestCross:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_3d(self, dtype):
        a = cp.array([1, 2, 3], dtype=dtype)
        b = cp.array([4, 5, 6], dtype=dtype)
        result = cross(a, b)
        expected = np.cross(np.array([1, 2, 3], dtype=dtype),
                            np.array([4, 5, 6], dtype=dtype))
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, dtype):
        a = cp.array([1, 2], dtype=dtype)
        b = cp.array([3, 4], dtype=dtype)
        result = cross(a, b)
        expected = np.cross(np.array([1, 2], dtype=dtype),
                            np.array([3, 4], dtype=dtype))
        npt.assert_allclose(result.get(), expected, **tol_for(dtype))
