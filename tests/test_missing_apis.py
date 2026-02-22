"""Tests for NumPy 2 APIs that are NOT yet implemented in macmetalpy.

Every test is marked xfail so the suite stays green while serving as a
living TODO list.  When an API is implemented the corresponding test(s)
will start passing and pytest will report them as XPASS — at that point
remove the xfail marker.
"""

import math
import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import macmetalpy as cp

xfail = pytest.mark.xfail(reason="Not yet implemented")


# ======================================================================
# Top-level math / elementwise functions (NumPy 2 names)
# ======================================================================

class TestTopLevelMathNewNames:
    """NumPy 2 added C99/IEEE-754 aliases for trig functions."""

    def test_acos(self):
        a = cp.array([1.0, 0.5, 0.0], dtype=np.float32)
        result = cp.acos(a)
        expected = np.arccos(np.array([1.0, 0.5, 0.0], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_acosh(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.acosh(a)
        expected = np.arccosh(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_asin(self):
        a = cp.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = cp.asin(a)
        expected = np.arcsin(np.array([0.0, 0.5, 1.0], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_asinh(self):
        a = cp.array([0.0, 1.0, 2.0], dtype=np.float32)
        result = cp.asinh(a)
        expected = np.arcsinh(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_atan(self):
        a = cp.array([0.0, 1.0, -1.0], dtype=np.float32)
        result = cp.atan(a)
        expected = np.arctan(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_atan2(self):
        y = cp.array([1.0, -1.0, 0.0], dtype=np.float32)
        x = cp.array([0.0, 1.0, -1.0], dtype=np.float32)
        result = cp.atan2(y, x)
        expected = np.arctan2(
            np.array([1.0, -1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, -1.0], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_atanh(self):
        a = cp.array([0.0, 0.5, -0.5], dtype=np.float32)
        result = cp.atanh(a)
        expected = np.arctanh(np.array([0.0, 0.5, -0.5], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_pow(self):
        a = cp.array([2.0, 3.0, 4.0], dtype=np.float32)
        b = cp.array([3.0, 2.0, 0.5], dtype=np.float32)
        result = cp.pow(a, b)
        expected = np.power(
            np.array([2.0, 3.0, 4.0], dtype=np.float32),
            np.array([3.0, 2.0, 0.5], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-5)


# ======================================================================
# Top-level array manipulation (NumPy 2 new functions)
# ======================================================================

class TestTopLevelArrayManip:
    """NumPy 2 added new array manipulation functions."""

    def test_astype(self):
        a = cp.array([1.0, 2.5, 3.7], dtype=np.float32)
        result = cp.astype(a, np.int32)
        expected = np.astype(np.array([1.0, 2.5, 3.7], dtype=np.float32), np.int32)
        assert_array_equal(result.get(), expected)

    def test_matrix_transpose(self):
        a = cp.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        result = cp.matrix_transpose(a)
        expected = np.matrix_transpose(
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        )
        assert_array_equal(result.get(), expected)

    def test_permute_dims(self):
        a = cp.zeros((2, 3, 4), dtype=np.float32)
        result = cp.permute_dims(a, (2, 0, 1))
        assert result.shape == (4, 2, 3)

    def test_unstack(self):
        a = cp.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        result = cp.unstack(a, axis=0)
        assert len(result) == 3
        assert_array_equal(result[0].get(), np.array([1, 2], dtype=np.float32))

    def test_row_stack(self):
        a = cp.array([1, 2, 3], dtype=np.float32)
        b = cp.array([4, 5, 6], dtype=np.float32)
        result = cp.row_stack([a, b])
        expected = np.row_stack([
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        ])
        assert_array_equal(result.get(), expected)


# ======================================================================
# Top-level cumulative/bitwise (NumPy 2)
# ======================================================================

class TestTopLevelCumulativeBitwise:

    def test_cumulative_sum(self):
        a = cp.array([1, 2, 3, 4], dtype=np.float32)
        result = cp.cumulative_sum(a)
        expected = np.array([1, 3, 6, 10], dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_cumulative_prod(self):
        a = cp.array([1, 2, 3, 4], dtype=np.float32)
        result = cp.cumulative_prod(a)
        expected = np.array([1, 2, 6, 24], dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_bitwise_count(self):
        a = cp.array([0, 1, 7, 255], dtype=np.uint32)
        result = cp.bitwise_count(a)
        expected = np.array([0, 1, 3, 8], dtype=np.int32)
        # bitwise_count returns popcount
        assert_array_equal(result.get(), expected)


# ======================================================================
# Top-level unique_* (NumPy 2 structured returns)
# ======================================================================

class TestTopLevelUniqueVariants:

    def test_unique_all(self):
        a = cp.array([3, 1, 2, 1, 3], dtype=np.int32)
        result = cp.unique_all(a)
        # Returns NamedTuple with values, indices, inverse_indices, counts
        assert hasattr(result, "values")
        assert hasattr(result, "counts")

    def test_unique_counts(self):
        a = cp.array([3, 1, 2, 1, 3], dtype=np.int32)
        result = cp.unique_counts(a)
        assert hasattr(result, "values")
        assert hasattr(result, "counts")

    def test_unique_inverse(self):
        a = cp.array([3, 1, 2, 1, 3], dtype=np.int32)
        result = cp.unique_inverse(a)
        assert hasattr(result, "values")
        assert hasattr(result, "inverse_indices")

    def test_unique_values(self):
        a = cp.array([3, 1, 2, 1, 3], dtype=np.int32)
        result = cp.unique_values(a)
        expected = np.array([1, 2, 3], dtype=np.int32)
        assert_array_equal(result.get(), expected)


# ======================================================================
# Top-level linear algebra shortcuts (NumPy 2)
# ======================================================================

class TestTopLevelLinalgShortcuts:

    def test_matvec(self):
        m = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        v = cp.array([1, 0], dtype=np.float32)
        result = cp.matvec(m, v)
        expected = np.array([1, 3], dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_vecmat(self):
        v = cp.array([1, 0], dtype=np.float32)
        m = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.vecmat(v, m)
        expected = np.array([1, 2], dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_vecdot(self):
        a = cp.array([1, 2, 3], dtype=np.float32)
        b = cp.array([4, 5, 6], dtype=np.float32)
        result = cp.vecdot(a, b)
        expected = np.float32(32.0)
        assert_allclose(result.get(), expected)


# ======================================================================
# Top-level I/O functions
# ======================================================================

class TestTopLevelIO:

    def test_loadtxt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1.0 2.0 3.0\n4.0 5.0 6.0\n")
            fname = f.name
        try:
            result = cp.loadtxt(fname)
            expected = np.loadtxt(fname)
            assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)
        finally:
            os.unlink(fname)

    def test_savetxt(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            fname = f.name
        try:
            cp.savetxt(fname, a)
            loaded = np.loadtxt(fname)
            assert_allclose(loaded, np.array([[1.0, 2.0], [3.0, 4.0]]), rtol=1e-5)
        finally:
            os.unlink(fname)

    def test_fromfile(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            data.tofile(f)
            fname = f.name
        try:
            result = cp.fromfile(fname, dtype=np.float32)
            assert_array_equal(result.get(), data)
        finally:
            os.unlink(fname)

    def test_genfromtxt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("1,2,3\n4,5,6\n")
            fname = f.name
        try:
            result = cp.genfromtxt(fname, delimiter=",")
            expected = np.genfromtxt(fname, delimiter=",").astype(np.float32)
            assert_allclose(result.get(), expected, rtol=1e-5)
        finally:
            os.unlink(fname)

    def test_fromregex(self):
        import re
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("val=1.0\nval=2.0\nval=3.0\n")
            fname = f.name
        try:
            result = cp.fromregex(
                fname,
                r"val=([.\d]+)",
                [("value", np.float32)],
            )
            assert len(result) == 3
        finally:
            os.unlink(fname)

    def test_from_dlpack(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.from_dlpack(a)
        assert_array_equal(result.get(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


# ======================================================================
# Top-level polynomial functions
# ======================================================================

class TestTopLevelPoly:

    def test_poly(self):
        roots_arr = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = cp.poly(roots_arr)
        expected = np.poly([1.0, 2.0, 3.0]).astype(np.float32)
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_polyval(self):
        p = cp.array([1.0, -3.0, 2.0], dtype=np.float32)  # x^2 - 3x + 2
        x = cp.array([0.0, 1.0, 2.0], dtype=np.float32)
        result = cp.polyval(p, x)
        expected = np.polyval(
            np.array([1.0, -3.0, 2.0], dtype=np.float32),
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_polyfit(self):
        x = cp.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y = cp.array([0.0, 1.0, 4.0, 9.0], dtype=np.float32)
        result = cp.polyfit(x, y, 2)
        expected = np.polyfit(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 1.0, 4.0, 9.0]),
            2,
        ).astype(np.float32)
        assert_allclose(result.get(), expected, rtol=1e-4)

    def test_polyadd(self):
        a = cp.array([1.0, 2.0], dtype=np.float32)
        b = cp.array([3.0, 4.0, 5.0], dtype=np.float32)
        result = cp.polyadd(a, b)
        expected = np.polyadd(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0, 5.0], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_polysub(self):
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = cp.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = cp.polysub(a, b)
        expected = np.polysub(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_polymul(self):
        a = cp.array([1.0, 1.0], dtype=np.float32)  # x + 1
        b = cp.array([1.0, -1.0], dtype=np.float32)  # x - 1
        result = cp.polymul(a, b)
        expected = np.polymul(
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_polydiv(self):
        # (x^2 - 1) / (x + 1) = (x - 1) remainder 0
        a = cp.array([1.0, 0.0, -1.0], dtype=np.float32)
        b = cp.array([1.0, 1.0], dtype=np.float32)
        q, r = cp.polydiv(a, b)
        eq, er = np.polydiv(
            np.array([1.0, 0.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )
        assert_allclose(q.get(), eq, rtol=1e-5)
        assert_allclose(r.get(), er, atol=1e-5)

    def test_polyder(self):
        p = cp.array([3.0, 2.0, 1.0], dtype=np.float32)  # 3x^2 + 2x + 1
        result = cp.polyder(p)
        expected = np.polyder(np.array([3.0, 2.0, 1.0], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_polyint(self):
        p = cp.array([6.0, 2.0], dtype=np.float32)  # 6x + 2
        result = cp.polyint(p)
        expected = np.polyint(np.array([6.0, 2.0], dtype=np.float32))
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_roots(self):
        p = cp.array([1.0, -3.0, 2.0], dtype=np.float32)  # x^2 - 3x + 2
        result = cp.roots(p)
        expected = np.sort(np.roots([1.0, -3.0, 2.0]).real).astype(np.float32)
        result_sorted = np.sort(result.get().real)
        assert_allclose(result_sorted, expected, rtol=1e-4)


# ======================================================================
# Top-level utility / config / info functions
# ======================================================================

class TestTopLevelUtility:

    def test_frompyfunc(self):
        def my_add(x, y):
            return x + y
        ufunc = cp.frompyfunc(my_add, 2, 1)
        result = ufunc(
            cp.array([1.0, 2.0], dtype=np.float32),
            cp.array([3.0, 4.0], dtype=np.float32),
        )
        assert_allclose(result.get(), np.array([4.0, 6.0], dtype=np.float32))

    def test_require(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.require(a, dtype=np.float32, requirements="C")
        assert result.dtype == np.float32

    def test_iterable(self):
        assert cp.iterable(cp.array([1, 2, 3])) is True
        assert cp.iterable(1) is False

    def test_may_share_memory(self):
        a = cp.array([1, 2, 3, 4], dtype=np.float32)
        b = a[:2]
        assert cp.may_share_memory(a, b) is True

    def test_shares_memory(self):
        a = cp.array([1, 2, 3, 4], dtype=np.float32)
        b = a[:2]
        assert cp.shares_memory(a, b) is True

    def test_isdtype(self):
        assert cp.isdtype(np.dtype("float32"), "real floating") is True
        assert cp.isdtype(np.dtype("int32"), "real floating") is False

    def test_info(self):
        # np.info prints information; just verify it doesn't crash
        cp.info(cp.array)

    def test_show_config(self):
        cp.show_config()

    def test_show_runtime(self):
        cp.show_runtime()


# ======================================================================
# Top-level print/buffer configuration
# ======================================================================

class TestTopLevelPrintConfig:

    def test_get_printoptions(self):
        opts = cp.get_printoptions()
        assert isinstance(opts, dict)
        assert "precision" in opts

    def test_set_printoptions(self):
        cp.set_printoptions(precision=4)
        opts = cp.get_printoptions()
        assert opts["precision"] == 4

    def test_printoptions_context(self):
        with cp.printoptions(precision=2):
            opts = cp.get_printoptions()
            assert opts["precision"] == 2

    def test_getbufsize(self):
        size = cp.getbufsize()
        assert isinstance(size, int)
        assert size > 0

    def test_setbufsize(self):
        old = cp.setbufsize(8192)
        assert isinstance(old, int)

    def test_geterr(self):
        err = cp.geterr()
        assert isinstance(err, dict)

    def test_seterr(self):
        old = cp.seterr(divide="ignore")
        assert isinstance(old, dict)

    def test_geterrcall(self):
        cb = cp.geterrcall()
        # Default is None
        assert cb is None or callable(cb)

    def test_seterrcall(self):
        def handler(err, flag):
            pass
        old = cp.seterrcall(handler)
        assert old is None or callable(old)

    def test_get_include(self):
        path = cp.get_include()
        assert isinstance(path, str)


# ======================================================================
# Top-level classes
# ======================================================================

class TestTopLevelClasses:

    def test_errstate(self):
        with cp.errstate(divide="ignore"):
            pass  # Should not raise

    def test_ndenumerate(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.int32)
        items = list(cp.ndenumerate(a))
        assert len(items) == 4
        assert items[0] == ((0, 0), 1)

    def test_ndindex(self):
        indices = list(cp.ndindex(2, 3))
        assert len(indices) == 6
        assert indices[0] == (0, 0)
        assert indices[-1] == (1, 2)

    def test_poly1d(self):
        p = cp.poly1d([1, -3, 2])  # x^2 - 3x + 2
        assert p(0) == 2
        assert p(1) == 0
        assert p(2) == 0


# ======================================================================
# Top-level constants
# ======================================================================

class TestTopLevelConstants:

    def test_false_(self):
        assert cp.False_ == np.False_
        assert cp.False_.dtype == np.bool_

    def test_true_(self):
        assert cp.True_ == np.True_
        assert cp.True_.dtype == np.bool_

    def test_scalar_type(self):
        assert isinstance(cp.ScalarType, (list, tuple))
        assert len(cp.ScalarType) > 0

    def test_index_exp(self):
        idx = cp.index_exp[2:5, :3]
        assert isinstance(idx, tuple)

    def test_little_endian(self):
        assert isinstance(cp.little_endian, bool)
        assert cp.little_endian == np.little_endian

    def test_typecodes(self):
        assert isinstance(cp.typecodes, dict)
        assert "Float" in cp.typecodes


# ======================================================================
# linalg functions (NumPy 2 additions and standard gaps)
# ======================================================================

class TestLinalgFunctions:

    def test_linalg_cross(self):
        a = cp.array([1, 0, 0], dtype=np.float32)
        b = cp.array([0, 1, 0], dtype=np.float32)
        result = cp.linalg.cross(a, b)
        expected = np.array([0, 0, 1], dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_linalg_diagonal(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.linalg.diagonal(a)
        expected = np.array([1, 4], dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_linalg_matmul(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = cp.array([[5, 6], [7, 8]], dtype=np.float32)
        result = cp.linalg.matmul(a, b)
        expected = np.matmul(
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32),
        )
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_linalg_matrix_norm(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.linalg.matrix_norm(a)
        expected = np.linalg.norm(
            np.array([[1, 2], [3, 4]], dtype=np.float32)
        )
        assert_allclose(float(result.get()), float(expected), rtol=1e-5)

    def test_linalg_matrix_transpose(self):
        a = cp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = cp.linalg.matrix_transpose(a)
        expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
        assert_array_equal(result.get(), expected)

    def test_linalg_outer(self):
        a = cp.array([1, 2, 3], dtype=np.float32)
        b = cp.array([4, 5], dtype=np.float32)
        result = cp.linalg.outer(a, b)
        expected = np.outer(
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5], dtype=np.float32),
        )
        assert_array_equal(result.get(), expected)

    def test_linalg_svdvals(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.linalg.svdvals(a)
        expected = np.linalg.svd(
            np.array([[1, 2], [3, 4]], dtype=np.float32), compute_uv=False
        )
        assert_allclose(result.get(), expected.astype(np.float32), rtol=1e-5)

    def test_linalg_tensordot(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        b = cp.array([[5, 6], [7, 8]], dtype=np.float32)
        result = cp.linalg.tensordot(a, b, axes=1)
        expected = np.tensordot(
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32),
            axes=1,
        )
        assert_allclose(result.get(), expected, rtol=1e-5)

    def test_linalg_trace(self):
        a = cp.array([[1, 2], [3, 4]], dtype=np.float32)
        result = cp.linalg.trace(a)
        assert_allclose(float(result.get()), 5.0)

    def test_linalg_vecdot(self):
        a = cp.array([1, 2, 3], dtype=np.float32)
        b = cp.array([4, 5, 6], dtype=np.float32)
        result = cp.linalg.vecdot(a, b)
        assert_allclose(float(result.get()), 32.0)

    def test_linalg_vector_norm(self):
        a = cp.array([3, 4], dtype=np.float32)
        result = cp.linalg.vector_norm(a)
        assert_allclose(float(result.get()), 5.0, rtol=1e-5)


# ======================================================================
# random functions
# ======================================================================

class TestRandomFunctions:

    def test_random_bytes(self):
        result = cp.random.bytes(16)
        assert isinstance(result, bytes)
        assert len(result) == 16

    def test_random_get_state(self):
        state = cp.random.get_state()
        assert isinstance(state, (dict, tuple))

    def test_random_set_state(self):
        cp.random.seed(42)
        state = cp.random.get_state()
        a = cp.random.rand(5)
        cp.random.set_state(state)
        b = cp.random.rand(5)
        assert_array_equal(a.get(), b.get())


# ======================================================================
# random classes
# ======================================================================

class TestRandomClasses:

    def test_bit_generator(self):
        # BitGenerator is an abstract base class and cannot be instantiated;
        # verify that macmetalpy re-exports the same class as numpy.
        assert cp.random.BitGenerator is np.random.BitGenerator

    def test_mt19937(self):
        bg = cp.random.MT19937(seed=42)
        assert bg is not None

    def test_pcg64(self):
        bg = cp.random.PCG64(seed=42)
        assert bg is not None

    def test_pcg64dxsm(self):
        bg = cp.random.PCG64DXSM(seed=42)
        assert bg is not None

    def test_philox(self):
        bg = cp.random.Philox(seed=42)
        assert bg is not None

    def test_random_state(self):
        rs = cp.random.RandomState(seed=42)
        result = rs.rand(5)
        assert result.shape == (5,)

    def test_sfc64(self):
        bg = cp.random.SFC64(seed=42)
        assert bg is not None

    def test_seed_sequence(self):
        ss = cp.random.SeedSequence(42)
        assert ss is not None
