"""Tests for linalg module gap-filling: multi_dot, tensorsolve, tensorinv, LinAlgError.

TDD: these tests are written FIRST, then the implementation in linalg.py.
"""

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy import linalg as la

from conftest import FLOAT_DTYPES, tol_for


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


# ======================================================================
# LinAlgError
# ======================================================================

class TestLinAlgError:
    def test_is_exception(self):
        assert issubclass(la.LinAlgError, Exception)

    def test_can_raise(self):
        with pytest.raises(la.LinAlgError):
            raise la.LinAlgError("test error")

    def test_message(self):
        try:
            raise la.LinAlgError("singular matrix")
        except la.LinAlgError as e:
            assert str(e) == "singular matrix"


# ======================================================================
# multi_dot
# ======================================================================

class TestMultiDot:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_two_matrices(self, dtype):
        A = np.arange(6, dtype=dtype).reshape(2, 3)
        B = np.arange(12, dtype=dtype).reshape(3, 4)
        result = la.multi_dot([cp.array(A), cp.array(B)])
        expected = np.linalg.multi_dot([A, B])
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_three_matrices(self, dtype):
        A = np.arange(6, dtype=dtype).reshape(2, 3)
        B = np.arange(12, dtype=dtype).reshape(3, 4)
        C = np.arange(8, dtype=dtype).reshape(4, 2)
        result = la.multi_dot([cp.array(A), cp.array(B), cp.array(C)])
        expected = np.linalg.multi_dot([A, B, C])
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_four_matrices(self, dtype):
        rng = np.random.RandomState(42)
        A = rng.randn(2, 3).astype(dtype)
        B = rng.randn(3, 4).astype(dtype)
        C = rng.randn(4, 5).astype(dtype)
        D = rng.randn(5, 2).astype(dtype)
        result = la.multi_dot([cp.array(A), cp.array(B), cp.array(C), cp.array(D)])
        expected = np.linalg.multi_dot([A, B, C, D])
        npt.assert_allclose(result.get(), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_with_out(self, dtype):
        A = np.arange(6, dtype=dtype).reshape(2, 3)
        B = np.arange(12, dtype=dtype).reshape(3, 4)
        out = cp.zeros((2, 4), dtype=dtype)
        result = la.multi_dot([cp.array(A), cp.array(B)], out=out)
        expected = np.linalg.multi_dot([A, B])
        npt.assert_allclose(out.get(), expected, rtol=1e-3, atol=1e-3)


# ======================================================================
# tensorsolve
# ======================================================================

class TestTensorsolve:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic(self, dtype):
        # Create a simple tensorsolve-compatible problem
        a_np = np.eye(2, dtype=dtype).reshape(2, 2)
        b_np = np.array([1.0, 2.0], dtype=dtype)
        result = la.tensorsolve(cp.array(a_np), cp.array(b_np))
        expected = np.linalg.tensorsolve(a_np, b_np)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2x2_identity(self, dtype):
        a_np = np.eye(4, dtype=dtype).reshape(2, 2, 2, 2)
        b_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        result = la.tensorsolve(cp.array(a_np), cp.array(b_np))
        expected = np.linalg.tensorsolve(a_np, b_np)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-3)


# ======================================================================
# tensorinv
# ======================================================================

class TestTensorinv:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_ind1(self, dtype):
        # shape (4, 2, 2) with ind=1: first 1 dims product = 4, rest product = 4
        a_np = np.eye(4, dtype=dtype).reshape(4, 2, 2)
        result = la.tensorinv(cp.array(a_np), ind=1)
        expected = np.linalg.tensorinv(a_np, ind=1)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_identity_like(self, dtype):
        a_np = np.eye(4, dtype=dtype).reshape(2, 2, 2, 2)
        result = la.tensorinv(cp.array(a_np))
        expected = np.linalg.tensorinv(a_np)
        npt.assert_allclose(result.get(), expected, rtol=1e-3, atol=1e-3)
