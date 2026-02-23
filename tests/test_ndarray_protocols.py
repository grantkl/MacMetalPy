"""Tests for ndarray properties and protocol dunders."""
import numpy as np
import macmetalpy as mp
import pytest


class TestProperties:
    def test_flags(self):
        a = mp.array([1, 2, 3])
        f = a.flags
        assert hasattr(f, 'c_contiguous') or f['C_CONTIGUOUS']

    def test_data(self):
        a = mp.array([1, 2, 3])
        d = a.data
        assert d is not None

    def test_device(self):
        a = mp.array([1, 2, 3])
        assert a.device == "cpu"

    def test_mT(self):
        a = mp.array([[1, 2], [3, 4]])
        result = a.mT
        expected = np.array([[1, 2], [3, 4]]).mT
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_mT_higher_dim(self):
        a = mp.array(np.arange(24).reshape(2, 3, 4))
        result = a.mT
        expected = np.arange(24).reshape(2, 3, 4).mT
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_ctypes(self):
        a = mp.array([1, 2, 3])
        c = a.ctypes
        assert c is not None

    def test_to_device(self):
        a = mp.array([1, 2, 3])
        b = a.to_device("cpu")
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_to_device_invalid(self):
        a = mp.array([1, 2, 3])
        with pytest.raises((ValueError, TypeError)):
            a.to_device("cuda")


class TestDunders:
    def test_array(self):
        a = mp.array([1, 2, 3])
        result = np.asarray(a)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_array_with_dtype(self):
        a = mp.array([1, 2, 3])
        result = a.__array__(dtype=np.float64)
        assert result.dtype == np.float64

    def test_array_priority(self):
        a = mp.array([1, 2, 3])
        assert hasattr(a, '__array_priority__')
        assert a.__array_priority__ >= 1.0

    def test_array_interface(self):
        a = mp.array([1, 2, 3])
        iface = a.__array_interface__
        assert isinstance(iface, dict)
        assert 'shape' in iface

    def test_array_struct(self):
        a = mp.array([1, 2, 3])
        s = a.__array_struct__
        assert s is not None

    def test_array_ufunc(self):
        a = mp.array([1, 2, 3])
        assert hasattr(a, '__array_ufunc__')

    def test_array_function(self):
        a = mp.array([1, 2, 3])
        assert hasattr(a, '__array_function__')

    def test_delitem_raises(self):
        a = mp.array([1, 2, 3])
        with pytest.raises(ValueError):
            del a[0]

    def test_dlpack(self):
        a = mp.array([1.0, 2.0, 3.0])
        assert hasattr(a, '__dlpack__')
        assert hasattr(a, '__dlpack_device__')
        dev = a.__dlpack_device__()
        assert dev == (1, 0)

    def test_class_getitem(self):
        # Should not raise
        result = mp.ndarray[np.float64]
        assert result is not None

    def test_array_wrap(self):
        a = mp.array([1, 2, 3])
        assert hasattr(a, '__array_wrap__')

    def test_array_finalize(self):
        a = mp.array([1, 2, 3])
        # Should be a no-op, not raise
        a.__array_finalize__(None)

    def test_array_namespace(self):
        a = mp.array([1, 2, 3])
        ns = a.__array_namespace__()
        assert hasattr(ns, 'array')

    def test_setstate(self):
        a = mp.array([1, 2, 3])
        assert hasattr(a, '__setstate__')
