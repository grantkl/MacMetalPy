"""Tests for missing dtype aliases, format functions, and utility classes.

Covers:
- dtype_utils.py: dtype aliases (int8, uint8, etc.), abstract types, dtype, broadcast,
  flatiter, nditer re-exports
- format_ops.py: array2string, array_repr, array_str, base_repr, binary_repr,
  format_float_positional, format_float_scientific, typename, mintypecode,
  issctype, obj2sctype, sctype2char
"""

import numpy as np
import pytest

import macmetalpy as cp
from macmetalpy.dtype_utils import (
    # New dtype aliases - integer types
    int8,
    uint8,
    byte,
    ubyte,
    short,
    ushort,
    intc,
    uintc,
    uint,
    longlong,
    ulonglong,
    # New dtype aliases - float types
    single,
    double,
    half,
    longdouble,
    # New dtype aliases - complex types
    csingle,
    cdouble,
    clongdouble,
    complex128,
    # Abstract types
    complexfloating,
    floating,
    integer,
    signedinteger,
    unsignedinteger,
    number,
    generic,
    inexact,
    # Utility re-exports
    dtype,
    broadcast,
    flatiter,
    nditer,
)
from macmetalpy.format_ops import (
    array2string,
    array_repr,
    array_str,
    base_repr,
    binary_repr,
    format_float_positional,
    format_float_scientific,
    typename,
    mintypecode,
)


# ====================================================================
# Dtype Aliases - Integer Types
# ====================================================================

class TestIntegerDtypeAliases:
    def test_int8_is_numpy_int8(self):
        assert int8 is np.int8

    def test_uint8_is_numpy_uint8(self):
        assert uint8 is np.uint8

    def test_byte_is_numpy_byte(self):
        assert byte is np.byte

    def test_ubyte_is_numpy_ubyte(self):
        assert ubyte is np.ubyte

    def test_short_is_numpy_short(self):
        assert short is np.short

    def test_ushort_is_numpy_ushort(self):
        assert ushort is np.ushort

    def test_intc_is_numpy_intc(self):
        assert intc is np.intc

    def test_uintc_is_numpy_uintc(self):
        assert uintc is np.uintc

    def test_uint_is_numpy_uint(self):
        assert uint is np.uint

    def test_longlong_is_numpy_longlong(self):
        assert longlong is np.longlong

    def test_ulonglong_is_numpy_ulonglong(self):
        assert ulonglong is np.ulonglong


# ====================================================================
# Dtype Aliases - Float Types
# ====================================================================

class TestFloatDtypeAliases:
    def test_single_is_numpy_single(self):
        assert single is np.single

    def test_double_is_numpy_double(self):
        assert double is np.double

    def test_half_is_numpy_half(self):
        assert half is np.half

    def test_longdouble_is_numpy_longdouble(self):
        assert longdouble is np.longdouble


# ====================================================================
# Dtype Aliases - Complex Types
# ====================================================================

class TestComplexDtypeAliases:
    def test_csingle_is_numpy_csingle(self):
        assert csingle is np.csingle

    def test_cdouble_is_numpy_cdouble(self):
        assert cdouble is np.cdouble

    def test_clongdouble_is_numpy_clongdouble(self):
        assert clongdouble is np.clongdouble

    def test_complex128_is_numpy_complex128(self):
        assert complex128 is np.complex128


# ====================================================================
# Abstract Types
# ====================================================================

class TestAbstractTypes:
    def test_complexfloating(self):
        assert complexfloating is np.complexfloating

    def test_floating(self):
        assert floating is np.floating

    def test_integer(self):
        assert integer is np.integer

    def test_signedinteger(self):
        assert signedinteger is np.signedinteger

    def test_unsignedinteger(self):
        assert unsignedinteger is np.unsignedinteger

    def test_number(self):
        assert number is np.number

    def test_generic(self):
        assert generic is np.generic

    def test_inexact(self):
        assert inexact is np.inexact


# ====================================================================
# Utility Re-exports: dtype, broadcast, flatiter, nditer
# ====================================================================

class TestUtilityReexports:
    def test_dtype_is_numpy_dtype(self):
        assert dtype is np.dtype

    def test_dtype_can_create(self):
        d = dtype('float32')
        assert d == np.dtype('float32')

    def test_broadcast_is_numpy_broadcast(self):
        assert broadcast is np.broadcast

    def test_broadcast_usage(self):
        x = np.array([1, 2, 3])
        y = np.array([[1], [2]])
        b = broadcast(x, y)
        assert b.shape == (2, 3)

    def test_flatiter_is_numpy_flatiter(self):
        assert flatiter is np.flatiter

    def test_nditer_is_numpy_nditer(self):
        assert nditer is np.nditer

    def test_nditer_usage(self):
        a = np.array([1, 2, 3])
        result = []
        for x in nditer(a):
            result.append(int(x))
        assert result == [1, 2, 3]


# ====================================================================
# format_ops: array2string
# ====================================================================

class TestArray2String:
    def test_basic(self):
        a = cp.array([1, 2, 3])
        result = array2string(a)
        assert isinstance(result, str)
        assert '1' in result and '2' in result and '3' in result

    def test_separator(self):
        a = cp.array([1, 2, 3])
        result = array2string(a, separator=', ')
        assert ', ' in result

    def test_precision(self):
        a = cp.array([1.123456789], dtype=np.float32)
        result = array2string(a, precision=2)
        # Should have reduced precision
        assert isinstance(result, str)

    def test_2d(self):
        a = cp.array([[1, 2], [3, 4]])
        result = array2string(a)
        assert isinstance(result, str)
        assert '1' in result and '4' in result

    def test_matches_numpy(self):
        np_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cp_a = cp.array([1.0, 2.0, 3.0], dtype=np.float32)
        np_result = np.array2string(np_a)
        cp_result = array2string(cp_a)
        assert cp_result == np_result


# ====================================================================
# format_ops: array_repr
# ====================================================================

class TestArrayRepr:
    def test_basic(self):
        a = cp.array([1, 2, 3])
        result = array_repr(a)
        assert isinstance(result, str)
        assert 'array' in result

    def test_precision(self):
        a = cp.array([1.1234567], dtype=np.float32)
        result = array_repr(a, precision=2)
        assert isinstance(result, str)

    def test_matches_numpy(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        cp_a = cp.array([1, 2, 3], dtype=np.int32)
        np_result = np.array_repr(np_a)
        cp_result = array_repr(cp_a)
        assert cp_result == np_result


# ====================================================================
# format_ops: array_str
# ====================================================================

class TestArrayStr:
    def test_basic(self):
        a = cp.array([1, 2, 3])
        result = array_str(a)
        assert isinstance(result, str)
        assert '1' in result and '3' in result

    def test_matches_numpy(self):
        np_a = np.array([1, 2, 3], dtype=np.int32)
        cp_a = cp.array([1, 2, 3], dtype=np.int32)
        np_result = np.array_str(np_a)
        cp_result = array_str(cp_a)
        assert cp_result == np_result


# ====================================================================
# format_ops: base_repr
# ====================================================================

class TestBaseRepr:
    def test_binary(self):
        result = base_repr(10, base=2)
        assert result == np.base_repr(10, base=2)

    def test_hex(self):
        result = base_repr(255, base=16)
        assert result == 'FF'

    def test_octal(self):
        result = base_repr(8, base=8)
        assert result == np.base_repr(8, base=8)

    def test_with_padding(self):
        result = base_repr(5, base=2, padding=8)
        assert result == np.base_repr(5, base=2, padding=8)


# ====================================================================
# format_ops: binary_repr
# ====================================================================

class TestBinaryRepr:
    def test_basic(self):
        result = binary_repr(5)
        assert result == '101'

    def test_with_width(self):
        result = binary_repr(5, width=8)
        assert result == '00000101'

    def test_negative(self):
        result = binary_repr(-1, width=8)
        assert result == np.binary_repr(-1, width=8)

    def test_zero(self):
        result = binary_repr(0)
        assert result == '0'


# ====================================================================
# format_ops: format_float_positional
# ====================================================================

class TestFormatFloatPositional:
    def test_basic(self):
        result = format_float_positional(1.5)
        assert isinstance(result, str)
        assert '1.5' in result

    def test_precision(self):
        result = format_float_positional(3.14159, precision=2)
        np_result = np.format_float_positional(3.14159, precision=2)
        assert result == np_result

    def test_trim(self):
        result = format_float_positional(1.0, trim='0')
        np_result = np.format_float_positional(1.0, trim='0')
        assert result == np_result

    def test_sign(self):
        result = format_float_positional(1.5, sign=True)
        assert '+' in result or result.startswith('+')


# ====================================================================
# format_ops: format_float_scientific
# ====================================================================

class TestFormatFloatScientific:
    def test_basic(self):
        result = format_float_scientific(1500.0)
        assert isinstance(result, str)
        assert 'e' in result.lower()

    def test_precision(self):
        result = format_float_scientific(1500.0, precision=3)
        np_result = np.format_float_scientific(1500.0, precision=3)
        assert result == np_result

    def test_exp_digits(self):
        result = format_float_scientific(1500.0, exp_digits=3)
        np_result = np.format_float_scientific(1500.0, exp_digits=3)
        assert result == np_result


# ====================================================================
# format_ops: typename
# ====================================================================

class TestTypename:
    def test_float(self):
        assert typename('f') == np.typename('f')

    def test_double(self):
        assert typename('d') == np.typename('d')

    def test_int(self):
        assert typename('i') == np.typename('i')

    def test_long(self):
        assert typename('l') == np.typename('l')


# ====================================================================
# format_ops: mintypecode
# ====================================================================

class TestMintypecode:
    def test_basic(self):
        result = mintypecode(['f', 'd'])
        assert result == np.mintypecode(['f', 'd'])

    def test_with_default(self):
        result = mintypecode([], default='f')
        assert result == np.mintypecode([], default='f')

    def test_complex(self):
        result = mintypecode(['f', 'F'])
        assert result == np.mintypecode(['f', 'F'])


