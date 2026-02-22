"""Tests for param gaps in linalg_top.py, set_ops.py, and linalg.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp
from macmetalpy.linalg_top import outer, einsum
from macmetalpy.set_ops import in1d, isin
from macmetalpy import linalg as la


# ---------------------------------------------------------------------------
# linalg_top.py: outer — add out=None
# ---------------------------------------------------------------------------
class TestOuterOut:
    def test_outer_default_no_out(self):
        """Existing behavior: out=None returns result normally."""
        a = cp.array([1.0, 2.0, 3.0])
        b = cp.array([4.0, 5.0])
        result = outer(a, b)
        expected = np.outer([1.0, 2.0, 3.0], [4.0, 5.0])
        assert_allclose(result.get(), expected)

    def test_outer_with_out(self):
        """out= is filled and returned."""
        a = cp.array([1.0, 2.0, 3.0])
        b = cp.array([4.0, 5.0])
        out = cp.zeros((3, 2))
        result = outer(a, b, out=out)
        expected = np.outer([1.0, 2.0, 3.0], [4.0, 5.0])
        assert result is out
        assert_allclose(out.get(), expected)


# ---------------------------------------------------------------------------
# linalg_top.py: einsum — add out=None, optimize=False
# ---------------------------------------------------------------------------
class TestEinsumParams:
    def test_einsum_default(self):
        """Existing behavior: basic einsum works without extra params."""
        a = cp.array([[1.0, 2.0], [3.0, 4.0]])
        result = einsum("ii", a)
        expected = np.einsum("ii", np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert_allclose(result.get(), expected)

    def test_einsum_with_out(self):
        """out= is filled and returned."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]])
        a = cp.array(a_np)
        b = cp.array(b_np)
        expected = np.einsum("ij,jk->ik", a_np, b_np)
        out = cp.zeros((2, 2))
        result = einsum("ij,jk->ik", a, b, out=out)
        assert result is out
        assert_allclose(out.get(), expected)

    def test_einsum_with_optimize(self):
        """optimize= is forwarded to np.einsum."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]])
        a = cp.array(a_np)
        b = cp.array(b_np)
        expected = np.einsum("ij,jk->ik", a_np, b_np, optimize=True)
        result = einsum("ij,jk->ik", a, b, optimize=True)
        assert_allclose(result.get(), expected)

    def test_einsum_out_and_optimize(self):
        """Both out= and optimize= together."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]])
        a = cp.array(a_np)
        b = cp.array(b_np)
        expected = np.einsum("ij,jk->ik", a_np, b_np, optimize=True)
        out = cp.zeros((2, 2))
        result = einsum("ij,jk->ik", a, b, out=out, optimize=True)
        assert result is out
        assert_allclose(out.get(), expected)


# ---------------------------------------------------------------------------
# set_ops.py: in1d — add kind=None
# ---------------------------------------------------------------------------
class TestIn1dKind:
    def test_in1d_default(self):
        """Existing behavior: kind not specified."""
        ar1 = cp.array([1, 2, 3, 4, 5])
        ar2 = cp.array([2, 4, 6])
        result = in1d(ar1, ar2)
        expected = np.isin([1, 2, 3, 4, 5], [2, 4, 6])
        assert_array_equal(result.get(), expected)

    def test_in1d_with_kind_none(self):
        """Explicitly passing kind=None gives same result."""
        ar1 = cp.array([1, 2, 3, 4, 5])
        ar2 = cp.array([2, 4, 6])
        result = in1d(ar1, ar2, kind=None)
        expected = np.isin([1, 2, 3, 4, 5], [2, 4, 6])
        assert_array_equal(result.get(), expected)

    def test_in1d_with_kind_sort(self):
        """kind='sort' forwarded to numpy."""
        ar1 = cp.array([1, 2, 3, 4, 5])
        ar2 = cp.array([2, 4, 6])
        result = in1d(ar1, ar2, kind="sort")
        expected = np.isin([1, 2, 3, 4, 5], [2, 4, 6], kind="sort")
        assert_array_equal(result.get(), expected)

    def test_in1d_with_invert_and_kind(self):
        """kind combined with invert."""
        ar1 = cp.array([1, 2, 3, 4, 5])
        ar2 = cp.array([2, 4, 6])
        result = in1d(ar1, ar2, invert=True, kind="sort")
        expected = np.isin([1, 2, 3, 4, 5], [2, 4, 6], invert=True, kind="sort")
        assert_array_equal(result.get(), expected)


# ---------------------------------------------------------------------------
# set_ops.py: isin — add kind=None
# ---------------------------------------------------------------------------
class TestIsinKind:
    def test_isin_default(self):
        """Existing behavior: kind not specified."""
        elem = cp.array([1, 2, 3, 4, 5])
        test = cp.array([2, 4, 6])
        result = isin(elem, test)
        expected = np.isin([1, 2, 3, 4, 5], [2, 4, 6])
        assert_array_equal(result.get(), expected)

    def test_isin_with_kind_none(self):
        """Explicitly passing kind=None gives same result."""
        elem = cp.array([1, 2, 3, 4, 5])
        test = cp.array([2, 4, 6])
        result = isin(elem, test, kind=None)
        expected = np.isin([1, 2, 3, 4, 5], [2, 4, 6])
        assert_array_equal(result.get(), expected)

    def test_isin_with_kind_sort(self):
        """kind='sort' forwarded to numpy."""
        elem = cp.array([1, 2, 3, 4, 5])
        test = cp.array([2, 4, 6])
        result = isin(elem, test, kind="sort")
        expected = np.isin([1, 2, 3, 4, 5], [2, 4, 6], kind="sort")
        assert_array_equal(result.get(), expected)

    def test_isin_with_invert_and_kind(self):
        """kind combined with invert."""
        elem = cp.array([[1, 2], [3, 4]])
        test = cp.array([2, 4, 6])
        result = isin(elem, test, invert=True, kind="sort")
        expected = np.isin([[1, 2], [3, 4]], [2, 4, 6], invert=True, kind="sort")
        assert_array_equal(result.get(), expected)


# ---------------------------------------------------------------------------
# linalg.py: svd — add compute_uv=True, hermitian=False
# ---------------------------------------------------------------------------
class TestSvdParams:
    def test_svd_default(self):
        """Existing behavior: returns (u, s, vh)."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        a = cp.array(a_np)
        u, s, vh = la.svd(a)
        u_np, s_np, vh_np = np.linalg.svd(a_np)
        assert_allclose(s.get(), s_np, rtol=1e-5)

    def test_svd_compute_uv_false(self):
        """compute_uv=False returns only singular values."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        a = cp.array(a_np)
        s = la.svd(a, compute_uv=False)
        s_np = np.linalg.svd(a_np, compute_uv=False)
        assert_allclose(s.get(), s_np, rtol=1e-5)

    def test_svd_hermitian(self):
        """hermitian=True forwarded to numpy."""
        a_np = np.array([[4.0, 2.0], [2.0, 3.0]])
        a = cp.array(a_np)
        u, s, vh = la.svd(a, hermitian=True)
        u_np, s_np, vh_np = np.linalg.svd(a_np, hermitian=True)
        assert_allclose(s.get(), s_np, rtol=1e-5)


# ---------------------------------------------------------------------------
# linalg.py: norm — add keepdims=False
# ---------------------------------------------------------------------------
class TestNormKeepdims:
    def test_norm_default(self):
        """Existing behavior: no keepdims."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = cp.array(a_np)
        result = la.norm(a)
        expected = np.linalg.norm(a_np)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_norm_keepdims_false(self):
        """keepdims=False same as default."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = cp.array(a_np)
        result = la.norm(a, axis=1, keepdims=False)
        expected = np.linalg.norm(a_np, axis=1, keepdims=False)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_norm_keepdims_true(self):
        """keepdims=True preserves reduced dimensions."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = cp.array(a_np)
        result = la.norm(a, axis=1, keepdims=True)
        expected = np.linalg.norm(a_np, axis=1, keepdims=True)
        assert_allclose(result.get(), expected, rtol=1e-5)
        assert result.shape == expected.shape


# ---------------------------------------------------------------------------
# linalg.py: eigh — add UPLO='L'
# ---------------------------------------------------------------------------
class TestEighUplo:
    def test_eigh_default(self):
        """Existing behavior: UPLO not specified (defaults to 'L')."""
        a_np = np.array([[4.0, 2.0], [2.0, 3.0]])
        a = cp.array(a_np)
        w, v = la.eigh(a)
        w_np, v_np = np.linalg.eigh(a_np)
        assert_allclose(w.get(), w_np, rtol=1e-5)

    def test_eigh_uplo_L(self):
        """Explicit UPLO='L' same as default."""
        a_np = np.array([[4.0, 2.0], [2.0, 3.0]])
        a = cp.array(a_np)
        w, v = la.eigh(a, UPLO='L')
        w_np, v_np = np.linalg.eigh(a_np, UPLO='L')
        assert_allclose(w.get(), w_np, rtol=1e-5)

    def test_eigh_uplo_U(self):
        """UPLO='U' uses upper triangle."""
        # Asymmetric matrix to test UPLO really differs
        a_np = np.array([[4.0, 1.0], [2.0, 3.0]])
        a = cp.array(a_np)
        w, v = la.eigh(a, UPLO='U')
        w_np, v_np = np.linalg.eigh(a_np, UPLO='U')
        assert_allclose(w.get(), w_np, rtol=1e-5)


# ---------------------------------------------------------------------------
# linalg.py: eigvalsh — add UPLO='L'
# ---------------------------------------------------------------------------
class TestEigvalshUplo:
    def test_eigvalsh_default(self):
        """Existing behavior: UPLO not specified."""
        a_np = np.array([[4.0, 2.0], [2.0, 3.0]])
        a = cp.array(a_np)
        result = la.eigvalsh(a)
        expected = np.linalg.eigvalsh(a_np)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_eigvalsh_uplo_L(self):
        """Explicit UPLO='L'."""
        a_np = np.array([[4.0, 2.0], [2.0, 3.0]])
        a = cp.array(a_np)
        result = la.eigvalsh(a, UPLO='L')
        expected = np.linalg.eigvalsh(a_np, UPLO='L')
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_eigvalsh_uplo_U(self):
        """UPLO='U' uses upper triangle."""
        a_np = np.array([[4.0, 1.0], [2.0, 3.0]])
        a = cp.array(a_np)
        result = la.eigvalsh(a, UPLO='U')
        expected = np.linalg.eigvalsh(a_np, UPLO='U')
        assert_allclose(result.get(), expected, rtol=1e-5)
