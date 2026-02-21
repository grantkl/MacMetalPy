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
)
from .math_ops import (
    sqrt, exp, log, abs, power,
    dot, where, clip,
    concatenate, stack, vstack, hstack,
    sign, floor, ceil,
    sin, cos, tan,
    arcsin, arccos, arctan,
    sinh, cosh, tanh,
    log2, log10,
    square, negative,
    around, round_, mod, remainder,
    isnan, isinf, isfinite, nan_to_num,
    isclose, allclose, array_equal, count_nonzero,
    copy, ascontiguousarray, trace, diagonal,
)
from .reductions import (
    sum, mean, max, min, any, all, argmax, argmin,
    std, var, prod, median, percentile,
    cumsum, cumprod, diff,
    # Extended reductions
    ptp, quantile, average,
)
from .ufunc import maximum, minimum
from ._config import get_config, set_config

# Sorting & searching
from .sorting import (
    sort, argsort, unique, searchsorted,
    lexsort, partition, argpartition, msort, sort_complex,
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
)

# Linear algebra
from . import linalg
from .linalg_top import (
    vdot, inner, outer, tensordot, einsum, kron, matmul, cross,
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
    histogram, histogram2d, histogramdd,
    bincount, digitize,
    ediff1d, gradient,
    corrcoef, correlate, cov,
)

# Logic & comparison
from .logic_ops import (
    logical_and, logical_or, logical_not, logical_xor,
    greater, greater_equal, less, less_equal, equal, not_equal,
    isneginf, isposinf, iscomplex, isreal, isscalar, array_equiv,
)

# Bitwise operations
from .bitwise_ops import (
    bitwise_and, bitwise_or, bitwise_xor, invert,
    left_shift, right_shift, packbits, unpackbits,
    gcd, lcm,
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
)

# Set operations
from .set_ops import (
    union1d, intersect1d, setdiff1d, setxor1d, in1d, isin,
)

# FFT & Random modules
from . import fft
from . import random

# Window functions
from .window import bartlett, blackman, hamming, hanning, kaiser

# Math extensions
from .math_ext import sinc, i0, convolve, interp, fix, unwrap, trapezoid


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
int16 = _np.int16
int32 = _np.int32
uint16 = _np.uint16
uint32 = _np.uint32
int64 = _np.int64
uint64 = _np.uint64
bool_ = _np.bool_
float64 = _np.float64
newaxis = _np.newaxis

nan = float('nan')
inf = float('inf')
pi = _math.pi
e = _math.e

__version__ = "0.1.0"
