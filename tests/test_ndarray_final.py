"""Tests for missing ndarray methods and operators (task #2).

TDD: tests written first, then implementation in ndarray.py.
"""

import copy
import os
import pickle
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from conftest import assert_eq, make_arg


# =====================================================================
# 1. User-facing methods
# =====================================================================


class TestConjugate:
    """conjugate() — alias for conj()."""

    def test_conjugate_complex(self):
        np_a = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
        ga = cp.array(np_a)
        result = ga.conjugate()
        expected = np.conjugate(np_a)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_conjugate_real(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.conjugate()
        npt.assert_allclose(result.get(), np_a, rtol=1e-5)


class TestDot:
    """dot(self, b) — delegate to np.dot."""

    def test_dot_1d(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ga = cp.array(a)
        gb = cp.array(b)
        result = ga.dot(gb)
        expected = np.dot(a, b)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_dot_2d(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        ga = cp.array(a)
        gb = cp.array(b)
        result = ga.dot(gb)
        expected = np.dot(a, b)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)

    def test_dot_with_numpy_array(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ga = cp.array(a)
        result = ga.dot(b)
        expected = np.dot(a, b)
        npt.assert_allclose(result.get(), expected, rtol=1e-5)


class TestSwapaxes:
    """swapaxes(axis1, axis2) — delegate to np.swapaxes."""

    def test_swapaxes_2d(self):
        np_a = np.arange(6, dtype=np.float32).reshape(2, 3)
        ga = cp.array(np_a)
        result = ga.swapaxes(0, 1)
        expected = np.swapaxes(np_a, 0, 1)
        npt.assert_array_equal(result.get(), expected)

    def test_swapaxes_3d(self):
        np_a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        ga = cp.array(np_a)
        result = ga.swapaxes(0, 2)
        expected = np.swapaxes(np_a, 0, 2)
        npt.assert_array_equal(result.get(), expected)


class TestByteswap:
    """byteswap() — delegate to numpy byteswap."""

    def test_byteswap_int32(self):
        np_a = np.array([1, 256, 65536], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.byteswap()
        expected = np_a.byteswap()
        npt.assert_array_equal(result.get(), expected)

    def test_byteswap_returns_new_array(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.byteswap()
        # Original should be unchanged
        npt.assert_array_equal(ga.get(), np_a)


class TestItemset:
    """itemset(*args) — set a scalar value in-place."""

    def test_itemset_flat_index(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a.copy())
        ga.itemset(1, 99.0)
        np_a.flat[1] = 99.0
        npt.assert_array_equal(ga.get(), np_a)

    def test_itemset_multi_index(self):
        np_a = np.arange(6, dtype=np.float32).reshape(2, 3)
        ga = cp.array(np_a.copy())
        ga.itemset((1, 2), 42.0)
        np_a[(1, 2)] = 42.0
        npt.assert_array_equal(ga.get(), np_a)


class TestDump:
    """dump(file) — pickle array to file."""

    def test_dump_load(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            fname = f.name
        try:
            ga.dump(fname)
            loaded = np.load(fname, allow_pickle=True)
            npt.assert_array_equal(loaded, np_a)
        finally:
            os.unlink(fname)


class TestDumps:
    """dumps() — return pickled bytes."""

    def test_dumps_roundtrip(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        s = ga.dumps()
        loaded = pickle.loads(s)
        npt.assert_array_equal(loaded, np_a)


class TestTofile:
    """tofile(fid, sep, format) — write to file."""

    def test_tofile_binary(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            fname = f.name
        try:
            ga.tofile(fname)
            loaded = np.fromfile(fname, dtype=np.float32)
            npt.assert_array_equal(loaded, np_a)
        finally:
            os.unlink(fname)

    def test_tofile_text(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as f:
            fname = f.name
        try:
            ga.tofile(fname, sep=",", format="%.1f")
            with open(fname) as f:
                content = f.read()
            assert "1.0" in content
            assert "2.0" in content
            assert "3.0" in content
        finally:
            os.unlink(fname)


class TestTostring:
    """tostring(order) — deprecated alias for tobytes."""

    def test_tostring_matches_tobytes(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        assert ga.tostring() == ga.tobytes()

    def test_tostring_with_order(self):
        np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.tostring(order='C')
        expected = np_a.tobytes(order='C')
        assert result == expected


class TestNewbyteorder:
    """newbyteorder(new_order) — change byte order."""

    def test_newbyteorder(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        result = ga.newbyteorder('S')
        # GPU always stores in native byte order, so values should match
        # the byte-reinterpreted result converted back to native order
        new_dt = np_a.dtype.newbyteorder('S')
        expected = np_a.view(new_dt).astype(new_dt.newbyteorder('='))
        npt.assert_array_equal(result.get(), expected)


class TestGetfield:
    """getfield(dtype, offset) — return a field of the array."""

    def test_getfield_basic(self):
        np_a = np.array([256 + 1, 256 + 2], dtype=np.int32)
        ga = cp.array(np_a)
        result = ga.getfield(np.int16, 0)
        expected = np_a.getfield(np.int16, 0)
        npt.assert_array_equal(result.get(), expected)


class TestSetfield:
    """setfield(val, dtype, offset) — set a field of the array."""

    def test_setfield_basic(self):
        np_a = np.array([0, 0], dtype=np.int32)
        ga = cp.array(np_a.copy())
        np_a.setfield(np.array([1, 2], dtype=np.int16), np.int16, 0)
        ga.setfield(np.array([1, 2], dtype=np.int16), np.int16, 0)
        npt.assert_array_equal(ga.get(), np_a)


class TestSetflags:
    """setflags() — no-op on GPU array."""

    def test_setflags_no_error(self):
        np_a = np.array([1.0, 2.0], dtype=np.float32)
        ga = cp.array(np_a)
        # Should not raise
        ga.setflags(write=True, align=True, uic=True)


class TestResize:
    """resize(new_shape) — change shape in-place."""

    def test_resize_grow(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a.copy())
        np_ref = np_a.copy()
        np_ref.resize((5,), refcheck=False)
        ga.resize((5,))
        npt.assert_array_equal(ga.get(), np_ref)

    def test_resize_shrink(self):
        np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ga = cp.array(np_a.copy())
        np_ref = np_a.copy()
        np_ref.resize((2,), refcheck=False)
        ga.resize((2,))
        npt.assert_array_equal(ga.get(), np_ref)

    def test_resize_2d(self):
        np_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        ga = cp.array(np_a.copy())
        np_ref = np_a.copy()
        np_ref.resize((2, 3), refcheck=False)
        ga.resize((2, 3))
        npt.assert_array_equal(ga.get(), np_ref)
        assert ga.shape == (2, 3)


# =====================================================================
# 2. Operators
# =====================================================================


class TestIlshift:
    """__ilshift__ — in-place left shift."""

    def test_ilshift_basic(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a.copy())
        ga <<= 2
        expected = np_a << 2
        npt.assert_array_equal(ga.get(), expected)

    def test_ilshift_array(self):
        np_a = np.array([1, 2, 4], dtype=np.int32)
        np_b = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a.copy())
        gb = cp.array(np_b)
        ga <<= gb
        expected = np_a << np_b
        npt.assert_array_equal(ga.get(), expected)


class TestIrshift:
    """__irshift__ — in-place right shift."""

    def test_irshift_basic(self):
        np_a = np.array([8, 16, 32], dtype=np.int32)
        ga = cp.array(np_a.copy())
        ga >>= 2
        expected = np_a >> 2
        npt.assert_array_equal(ga.get(), expected)

    def test_irshift_array(self):
        np_a = np.array([8, 16, 32], dtype=np.int32)
        np_b = np.array([1, 2, 3], dtype=np.int32)
        ga = cp.array(np_a.copy())
        gb = cp.array(np_b)
        ga >>= gb
        expected = np_a >> np_b
        npt.assert_array_equal(ga.get(), expected)


class TestRand:
    """__rand__ — reflected bitwise AND."""

    def test_rand_direct_call(self):
        np_a = np.array([True, False, True], dtype=np.bool_)
        np_b = np.array([True, True, False], dtype=np.bool_)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        # Direct call to __rand__
        result = ga.__rand__(gb)
        expected = np_b & np_a
        npt.assert_array_equal(result.get(), expected)

    def test_rand_both_gpu(self):
        ga = cp.array(np.array([True, False, True], dtype=np.bool_))
        gb = cp.array(np.array([True, True, False], dtype=np.bool_))
        result = ga & gb
        expected = np.array([True, False, False], dtype=np.bool_)
        npt.assert_array_equal(result.get(), expected)


class TestRor:
    """__ror__ — reflected bitwise OR."""

    def test_ror_direct_call(self):
        np_a = np.array([True, False, True], dtype=np.bool_)
        np_b = np.array([False, True, False], dtype=np.bool_)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = ga.__ror__(gb)
        expected = np_b | np_a
        npt.assert_array_equal(result.get(), expected)

    def test_ror_both_gpu(self):
        ga = cp.array(np.array([True, False, True], dtype=np.bool_))
        gb = cp.array(np.array([False, True, False], dtype=np.bool_))
        result = ga | gb
        expected = np.array([True, True, True], dtype=np.bool_)
        npt.assert_array_equal(result.get(), expected)


class TestRxor:
    """__rxor__ — reflected bitwise XOR."""

    def test_rxor_direct_call(self):
        np_a = np.array([True, False, True], dtype=np.bool_)
        np_b = np.array([True, True, False], dtype=np.bool_)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        result = ga.__rxor__(gb)
        expected = np_b ^ np_a
        npt.assert_array_equal(result.get(), expected)

    def test_rxor_both_gpu(self):
        ga = cp.array(np.array([True, False, True], dtype=np.bool_))
        gb = cp.array(np.array([True, True, False], dtype=np.bool_))
        result = ga ^ gb
        expected = np.array([False, True, True], dtype=np.bool_)
        npt.assert_array_equal(result.get(), expected)


class TestRdivmod:
    """__rdivmod__ — reflected divmod."""

    def test_rdivmod(self):
        np_a = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        np_b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        ga = cp.array(np_a)
        gb = cp.array(np_b)
        # gb.__divmod__ doesn't exist or returns NotImplemented,
        # so divmod(np_b_scalar, ga) calls ga.__rdivmod__
        # Test via direct call:
        q, r = ga.__rdivmod__(gb)
        eq, er = divmod(np_b, np_a)
        npt.assert_allclose(q.get(), eq, rtol=1e-5)
        npt.assert_allclose(r.get(), er, rtol=1e-5)


class TestIter:
    """__iter__ — iterate over first axis."""

    def test_iter_1d(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        values = [x.item() for x in ga]
        expected = [float(x) for x in np_a]
        assert values == expected

    def test_iter_2d(self):
        np_a = np.arange(6, dtype=np.float32).reshape(2, 3)
        ga = cp.array(np_a)
        rows = list(ga)
        assert len(rows) == 2
        npt.assert_array_equal(rows[0].get(), np_a[0])
        npt.assert_array_equal(rows[1].get(), np_a[1])


class TestCopy:
    """__copy__ and __deepcopy__ — support copy module."""

    def test_copy(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        gc = copy.copy(ga)
        npt.assert_array_equal(gc.get(), np_a)
        # Verify it's a separate copy (modify original, check copy unchanged)
        ga.itemset(0, 99.0)
        assert gc.get()[0] == 1.0

    def test_deepcopy(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ga = cp.array(np_a)
        gc = copy.deepcopy(ga)
        npt.assert_array_equal(gc.get(), np_a)
        ga.itemset(0, 99.0)
        assert gc.get()[0] == 1.0
