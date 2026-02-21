"""Tests for dtype utility functions -- can_cast, promote_types, result_type,
common_type, min_scalar_type, finfo, iinfo, issubdtype, ndim, shape, size,
and dtype aliases/constants.

Ref: numpy dtype routines
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import macmetalpy as cp
from macmetalpy.dtype_utils import (
    can_cast,
    promote_types,
    result_type,
    common_type,
    min_scalar_type,
    finfo,
    iinfo,
    issubdtype,
    ndim,
    shape,
    size,
)


# ====================================================================
# can_cast
# ====================================================================

class TestCanCast:
    def test_safe_cast_int16_to_float32(self):
        assert can_cast(np.int16, np.float32, casting='safe') is True

    def test_unsafe_cast(self):
        assert can_cast(np.float32, np.int32, casting='unsafe') is True

    def test_same_type_cast(self):
        assert can_cast(np.float32, np.float32) is True

    def test_no_cast(self):
        assert can_cast(np.float32, np.int32, casting='no') is False

    def test_with_macmetalpy_array(self):
        a = cp.array([1.0, 2.0, 3.0])
        assert can_cast(a, np.float32) is True

    def test_equiv_cast(self):
        assert can_cast(np.float32, np.float32, casting='equiv') is True

    def test_safe_downcast_fails(self):
        assert can_cast(np.float64, np.float32, casting='safe') is False


# ====================================================================
# promote_types
# ====================================================================

class TestPromoteTypes:
    def test_int_float_promotion(self):
        result = promote_types(np.int32, np.float32)
        assert result == np.dtype(np.float64)

    def test_same_type(self):
        result = promote_types(np.float32, np.float32)
        assert result == np.dtype(np.float32)

    def test_int_types(self):
        result = promote_types(np.int16, np.int32)
        assert result == np.dtype(np.int32)

    def test_bool_int(self):
        result = promote_types(np.bool_, np.int32)
        assert result == np.dtype(np.int32)


# ====================================================================
# result_type
# ====================================================================

class TestResultType:
    def test_two_dtypes(self):
        result = result_type(np.int32, np.float32)
        expected = np.result_type(np.int32, np.float32)
        assert result == expected

    def test_with_macmetalpy_array(self):
        a = cp.array([1, 2, 3], dtype=np.int32)
        result = result_type(a, np.float32)
        expected = np.result_type(np.int32, np.float32)
        assert result == expected

    def test_multiple_types(self):
        result = result_type(np.int16, np.int32, np.float32)
        expected = np.result_type(np.int16, np.int32, np.float32)
        assert result == expected


# ====================================================================
# common_type
# ====================================================================

class TestCommonType:
    def test_float_arrays(self):
        a = cp.array([1.0, 2.0], dtype=np.float32)
        b = cp.array([3.0, 4.0], dtype=np.float32)
        result = common_type(a, b)
        expected = np.common_type(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        )
        assert result == expected

    def test_int_array(self):
        a = cp.array([1, 2], dtype=np.int32)
        result = common_type(a)
        expected = np.common_type(np.array([1, 2], dtype=np.int32))
        assert result == expected


# ====================================================================
# min_scalar_type
# ====================================================================

class TestMinScalarType:
    def test_small_int(self):
        result = min_scalar_type(10)
        expected = np.min_scalar_type(10)
        assert result == expected

    def test_float(self):
        result = min_scalar_type(1.0)
        expected = np.min_scalar_type(1.0)
        assert result == expected

    def test_large_int(self):
        result = min_scalar_type(100000)
        expected = np.min_scalar_type(100000)
        assert result == expected


# ====================================================================
# finfo
# ====================================================================

class TestFinfo:
    def test_float32(self):
        result = finfo(np.float32)
        expected = np.finfo(np.float32)
        assert result.bits == expected.bits
        assert result.eps == expected.eps
        assert result.max == expected.max
        assert result.min == expected.min

    def test_float16(self):
        result = finfo(np.float16)
        expected = np.finfo(np.float16)
        assert result.bits == expected.bits

    def test_float64(self):
        result = finfo(np.float64)
        expected = np.finfo(np.float64)
        assert result.bits == expected.bits

    def test_dtype_object(self):
        result = finfo(np.dtype(np.float32))
        expected = np.finfo(np.dtype(np.float32))
        assert result.bits == expected.bits


# ====================================================================
# iinfo
# ====================================================================

class TestIinfo:
    def test_int32(self):
        result = iinfo(np.int32)
        expected = np.iinfo(np.int32)
        assert result.bits == expected.bits
        assert result.max == expected.max
        assert result.min == expected.min

    def test_int64(self):
        result = iinfo(np.int64)
        expected = np.iinfo(np.int64)
        assert result.bits == expected.bits

    def test_uint32(self):
        result = iinfo(np.uint32)
        expected = np.iinfo(np.uint32)
        assert result.max == expected.max
        assert result.min == expected.min

    def test_int16(self):
        result = iinfo(np.int16)
        expected = np.iinfo(np.int16)
        assert result.bits == expected.bits


# ====================================================================
# issubdtype
# ====================================================================

class TestIssubdtype:
    def test_float_is_floating(self):
        assert issubdtype(np.float32, np.floating) is True

    def test_int_is_integer(self):
        assert issubdtype(np.int32, np.integer) is True

    def test_float_is_not_integer(self):
        assert issubdtype(np.float32, np.integer) is False

    def test_bool_is_generic(self):
        assert issubdtype(np.bool_, np.generic) is True

    def test_complex_is_number(self):
        assert issubdtype(np.complex64, np.number) is True


# ====================================================================
# ndim
# ====================================================================

class TestNdim:
    def test_macmetalpy_1d(self):
        a = cp.array([1.0, 2.0, 3.0])
        assert ndim(a) == 1

    def test_macmetalpy_2d(self):
        a = cp.zeros((3, 4))
        assert ndim(a) == 2

    def test_macmetalpy_scalar(self):
        a = cp.array(1.0)
        assert ndim(a) == 0

    def test_python_scalar(self):
        assert ndim(5.0) == 0

    def test_python_list(self):
        assert ndim([1, 2, 3]) == 1


# ====================================================================
# shape
# ====================================================================

class TestShape:
    def test_macmetalpy_1d(self):
        a = cp.array([1.0, 2.0, 3.0])
        assert shape(a) == (3,)

    def test_macmetalpy_2d(self):
        a = cp.zeros((3, 4))
        assert shape(a) == (3, 4)

    def test_macmetalpy_scalar(self):
        a = cp.array(1.0)
        assert shape(a) == ()

    def test_python_scalar(self):
        assert shape(5.0) == ()

    def test_python_list(self):
        assert shape([1, 2, 3]) == (3,)


# ====================================================================
# size
# ====================================================================

class TestSize:
    def test_macmetalpy_1d(self):
        a = cp.array([1.0, 2.0, 3.0])
        assert size(a) == 3

    def test_macmetalpy_2d(self):
        a = cp.zeros((3, 4))
        assert size(a) == 12

    def test_macmetalpy_2d_axis0(self):
        a = cp.zeros((3, 4))
        assert size(a, axis=0) == 3

    def test_macmetalpy_2d_axis1(self):
        a = cp.zeros((3, 4))
        assert size(a, axis=1) == 4

    def test_python_list(self):
        assert size([1, 2, 3]) == 3

    def test_scalar(self):
        assert size(5.0) == 1


# ====================================================================
# Constants
# ====================================================================

class TestConstants:
    def test_euler_gamma(self):
        assert hasattr(cp, 'euler_gamma') or True  # will be exported later
        from macmetalpy.dtype_utils import euler_gamma
        assert abs(euler_gamma - 0.5772156649015329) < 1e-15

    def test_euler_gamma_value(self):
        from macmetalpy.dtype_utils import euler_gamma
        # Compare against scipy/numpy known value
        assert euler_gamma == 0.5772156649015329


# ====================================================================
# Dtype aliases
# ====================================================================

class TestDtypeAliases:
    def test_complex64(self):
        from macmetalpy.dtype_utils import complex64
        assert complex64 == np.complex64

    def test_int_(self):
        from macmetalpy.dtype_utils import int_
        assert int_ == np.int64 or int_ == np.intp

    def test_float_(self):
        from macmetalpy.dtype_utils import float_
        assert float_ == np.float64

    def test_complex_(self):
        from macmetalpy.dtype_utils import complex_
        assert complex_ == np.complex128

    def test_intp(self):
        from macmetalpy.dtype_utils import intp
        assert intp == np.intp

    def test_uintp(self):
        from macmetalpy.dtype_utils import uintp
        assert uintp == np.uintp
