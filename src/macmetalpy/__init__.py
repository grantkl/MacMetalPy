"""macmetalpy — CuPy-compatible GPU array library for Apple Silicon."""

import sys as _sys

if _sys.platform != "darwin":
    raise ImportError("macmetalpy requires macOS with Metal GPU support")

from .ndarray import ndarray
from .raw_kernel import RawKernel
from .creation import (
    array,
    asarray,
    empty,
    zeros,
    ones,
    full,
    arange,
    zeros_like,
    ones_like,
    empty_like,
    full_like,
    linspace,
    eye,
    diag, identity, tri, triu, tril,
    logspace, meshgrid, indices,
    # Extended creation
    fromfunction, diagflat, vander, asanyarray,
    geomspace, frombuffer,
    # Final gap-filling creation
    asarray_chkfinite, fromiter, fromstring,
)
from .math_ops import (
    sqrt, exp, log, abs, power,
    dot, where, clip, fastCopyAndTranspose,
    concatenate, stack, vstack, hstack,
    sign, floor, ceil,
    sin, cos, tan,
    arcsin, arccos, arctan,
    sinh, cosh, tanh,
    log2, log10,
    square, negative,
    around, mod, remainder,
    isnan, isinf, isfinite, nan_to_num,
    isclose, allclose, array_equal, count_nonzero,
    copy, ascontiguousarray, trace, diagonal,
)
from .math_ops import round
from .reductions import (
    sum, mean, max, min, any, all, argmax, argmin,
    std, var, prod, median, percentile,
    cumsum, cumprod, diff, divmod,
    # Extended reductions
    ptp, quantile, average,
    # NumPy 2 cumulative aliases
    cumulative_sum, cumulative_prod,
)
from .ufunc import maximum, minimum
from ._config import get_config, set_config

# Sorting & searching
from .sorting import (
    sort, argsort, unique, searchsorted,
    lexsort, partition, argpartition, sort_complex,
    # NumPy 2 unique variants
    unique_all, unique_counts, unique_inverse, unique_values,
)

# Manipulation
from .manipulation import (
    tile, repeat, flip, roll, split, array_split,
    squeeze, ravel, moveaxis, swapaxes, broadcast_to,
    # Extended manipulation
    reshape, transpose, rollaxis,
    atleast_1d, atleast_2d, atleast_3d,
    dstack, column_stack, concat,
    dsplit, hsplit, vsplit,
    delete, append, resize, trim_zeros,
    fliplr, flipud, rot90,
    broadcast_arrays, copyto, pad,
    # Gap-filling manipulation
    block, insert, broadcast_shapes, asfortranarray,
    # NumPy 2 array manipulation
    astype, matrix_transpose, permute_dims, unstack,
)

# Linear algebra
from . import linalg
from .linalg_top import (
    vdot, inner, outer, tensordot, einsum, einsum_path, kron, matmul, cross,
    # NumPy 2 linalg shortcuts
    matvec, vecmat, vecdot,
)

# Math ufuncs
from .ufunc_ops import (
    add, subtract, multiply, divide, true_divide, floor_divide,
    float_power, fmod,
    exp2, expm1, log1p, cbrt, reciprocal, rint, trunc,
    absolute, fabs, positive,
    arctan2, hypot, logaddexp, logaddexp2,
    heaviside, copysign, nextafter,
    arcsinh, arccosh, arctanh,
    degrees, radians, deg2rad, rad2deg,
    signbit, modf, ldexp, frexp,
    amax, amin, fmax, fmin,
)

# NaN functions & statistics
from .nan_ops import (
    nansum, nanprod, nancumsum, nancumprod,
    nanmax, nanmin, nanmean, nanmedian,
    nanstd, nanvar, nanargmax, nanargmin,
    nanpercentile, nanquantile,
    histogram, histogram2d, histogramdd, histogram_bin_edges,
    bincount, digitize,
    ediff1d, gradient,
    corrcoef, correlate, cov,
)

# Logic & comparison
from .logic_ops import (
    logical_and, logical_or, logical_not, logical_xor,
    greater, greater_equal, less, less_equal, equal, not_equal,
    isneginf, isposinf, iscomplex, isreal, isscalar, array_equiv,
    iscomplexobj, isrealobj, isfortran,
)

# Bitwise operations
from .bitwise_ops import (
    bitwise_and, bitwise_or, bitwise_xor, invert, bitwise_not,
    left_shift, right_shift, packbits, unpackbits,
    gcd, lcm,
    # NumPy 2 bitwise
    bitwise_count,
)

# Indexing
from .indexing import (
    take, take_along_axis, put, put_along_axis, putmask, place,
    choose, compress, select, extract,
    diag_indices, diag_indices_from,
    tril_indices, tril_indices_from,
    triu_indices, triu_indices_from,
    ravel_multi_index, unravel_index,
    fill_diagonal, nonzero, flatnonzero, argwhere, ix_,
    mask_indices,
)

# Set operations
from .set_ops import (
    union1d, intersect1d, setdiff1d, setxor1d, isin,
)

# FFT & Random modules
from . import fft
from . import random

# Window functions
from .window import bartlett, blackman, hamming, hanning, kaiser

# Math extensions
from .math_ext import sinc, i0, convolve, interp, fix, unwrap, trapezoid, trapz, piecewise, spacing, isnat

# Complex number operations
from .complex_ops import angle, real, imag, conj, conjugate, real_if_close

# Dtype utilities
from .dtype_utils import (
    can_cast, promote_types, result_type, common_type, min_scalar_type,
    finfo, iinfo, issubdtype,
    ndim, shape, size,
    euler_gamma,
    int_, float_, complex_, intp, uintp,
    # Dtype aliases
    int8, uint8, byte, ubyte, short, ushort, intc, uintc, uint,
    longlong, ulonglong,
    single, double, half, longdouble, longfloat,
    csingle, cdouble, clongdouble, clongfloat, singlecomplex, longcomplex,
    complex128, cfloat,
    complexfloating, floating, integer, signedinteger, unsignedinteger,
    number, generic, inexact,
    dtype, broadcast, flatiter, nditer,
)

# Functional programming
from .functional import vectorize, apply_along_axis, apply_over_axes

# Index expression objects
from .index_tricks import c_, r_, s_, mgrid, ogrid

# Formatting & string representation
from .format_ops import (
    array2string, array_repr, array_str,
    base_repr, binary_repr,
    format_float_positional, format_float_scientific,
    typename, mintypecode,
)

# I/O
from . import io
from .io import save, load, savez, savez_compressed
from .io import loadtxt, savetxt, fromfile, genfromtxt, fromregex, from_dlpack

# Polynomial functions
from .poly_ops import (
    poly, polyval, polyfit, polyadd, polysub, polymul, polydiv,
    polyder, polyint, roots, poly1d,
)

# Print/buffer/error configuration
from .config_ops import (
    get_printoptions, set_printoptions, printoptions,
    getbufsize, setbufsize,
    geterr, seterr, geterrcall, seterrcall,
    geterrobj, seterrobj, set_numeric_ops,
    get_include,
    show_config, show_runtime,
)

# Utility functions
from .utils import (
    frompyfunc, require, iterable,
    may_share_memory, shares_memory,
    isdtype, info,
)


def expand_dims(a, axis):
    """Insert a new axis at the given position."""
    return a.expand_dims(axis)


def synchronize():
    """Block until all pending GPU work completes."""
    from ._metal_backend import MetalBackend
    MetalBackend().synchronize()


import numpy as _np
import math as _math

float16 = _np.float16
float32 = _np.float32
float64 = _np.float64
int16 = _np.int16
int32 = _np.int32
int64 = _np.int64
uint16 = _np.uint16
uint32 = _np.uint32
uint64 = _np.uint64
bool_ = _np.bool_
complex64 = _np.complex64
newaxis = _np.newaxis

# Bitwise alias
bitwise_invert = invert
bitwise_left_shift = left_shift
bitwise_right_shift = right_shift

# NumPy 2 math aliases (C99/IEEE-754 names)
acos = arccos
acosh = arccosh
asin = arcsin
asinh = arcsinh
atan = arctan
atan2 = arctan2
atanh = arctanh
pow = power

# NumPy 2 manipulation alias
row_stack = vstack

nan = float('nan')
inf = float('inf')
pi = _math.pi
e = _math.e

# Classes (re-export from numpy or thin wrappers)
errstate = _np.errstate
ndindex = _np.ndindex

# Constants
False_ = _np.False_
True_ = _np.True_
ScalarType = _np.ScalarType
index_exp = _np.index_exp
little_endian = _np.little_endian
typecodes = _np.typecodes


class ndenumerate:
    """Multidimensional index iterator (works on macmetalpy arrays via .get())."""

    def __init__(self, arr):
        from .ndarray import ndarray as _ndarray
        if isinstance(arr, _ndarray):
            arr = arr.get()
        self._impl = _np.ndenumerate(arr)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._impl)


# --- Type re-exports ---
bytes_ = _np.bytes_
character = _np.character
datetime64 = _np.datetime64
flexible = _np.flexible
object_ = _np.object_
str_ = _np.str_
timedelta64 = _np.timedelta64
void = _np.void

# --- Legacy class re-exports ---
matrix = _np.matrix
memmap = _np.memmap
recarray = _np.recarray
record = _np.record
sctypeDict = _np.sctypeDict
asmatrix = _np.asmatrix
bmat = _np.bmat

# --- Datetime functions ---
busday_count = _np.busday_count
busday_offset = _np.busday_offset
busdaycalendar = _np.busdaycalendar
is_busday = _np.is_busday
datetime_as_string = _np.datetime_as_string
datetime_data = _np.datetime_data

# --- Module re-exports ---
from numpy import char, dtypes, emath, exceptions, polynomial, rec, strings


__version__ = "0.1.0"
