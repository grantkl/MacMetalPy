"""Array formatting and type-code utility functions.

Thin wrappers around NumPy formatting functions that also accept macmetalpy ndarrays.
"""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray


def _to_numpy(a):
    """Convert macmetalpy ndarray to numpy array for delegation."""
    if isinstance(a, ndarray):
        return a.get()
    return a


def array2string(a, max_line_width=None, precision=None, suppress_small=None,
                 separator=' ', prefix='', formatter=None,
                 threshold=None, edgeitems=None, sign=None, floatmode=None,
                 suffix='', *, legacy=None, **extra_kw):
    """Return a string representation of an array."""
    kwargs = dict(
        max_line_width=max_line_width, precision=precision,
        suppress_small=suppress_small, separator=separator,
        prefix=prefix, formatter=formatter,
        threshold=threshold, edgeitems=edgeitems, sign=sign,
        floatmode=floatmode, suffix=suffix, legacy=legacy,
    )
    return np.array2string(_to_numpy(a), **kwargs)


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """Return the string representation of an array."""
    return np.array_repr(_to_numpy(arr), max_line_width=max_line_width,
                         precision=precision, suppress_small=suppress_small)


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """Return a string representation of the data in an array."""
    return np.array_str(_to_numpy(a), max_line_width=max_line_width,
                        precision=precision, suppress_small=suppress_small)


def base_repr(number, base=2, padding=0):
    """Return a string representation of a number in the given base system."""
    return np.base_repr(number, base=base, padding=padding)


def binary_repr(num, width=None):
    """Return the binary representation of the input number as a string."""
    return np.binary_repr(num, width=width)


def format_float_positional(x, precision=None, unique=True, fractional=True,
                            trim='k', sign=False, pad_left=None,
                            pad_right=None, min_digits=None):
    """Format a floating-point scalar as a decimal string in positional notation."""
    return np.format_float_positional(
        x, precision=precision, unique=unique, fractional=fractional,
        trim=trim, sign=sign, pad_left=pad_left, pad_right=pad_right,
        min_digits=min_digits,
    )


def format_float_scientific(x, precision=None, unique=True, trim='k',
                            sign=False, pad_left=None, exp_digits=None,
                            min_digits=None):
    """Format a floating-point scalar as a decimal string in scientific notation."""
    return np.format_float_scientific(
        x, precision=precision, unique=unique, trim=trim,
        sign=sign, pad_left=pad_left, exp_digits=exp_digits,
        min_digits=min_digits,
    )


def typename(char):
    """Return a description for the given data type character code."""
    return np.typename(char)


def mintypecode(typechars, typeset='GDFgdf', default='d'):
    """Return the character for the minimum-size type to which given types can be safely cast."""
    return np.mintypecode(typechars, typeset=typeset, default=default)


def issctype(rep):
    """Determine if the given object represents a scalar data-type."""
    _issctype = getattr(np, 'issctype', None)
    if _issctype is not None:
        return _issctype(rep)
    return isinstance(rep, type) and issubclass(rep, np.generic)


def obj2sctype(rep, default=None):
    """Return the scalar dtype or NumPy equivalent of Python type of an object."""
    _obj2sctype = getattr(np, 'obj2sctype', None)
    if _obj2sctype is not None:
        return _obj2sctype(rep, default=default)
    try:
        return np.dtype(rep).type
    except (TypeError, KeyError):
        return default


def sctype2char(sctype):
    """Return the string representation of a scalar dtype."""
    _sctype2char = getattr(np, 'sctype2char', None)
    if _sctype2char is not None:
        return _sctype2char(sctype)
    return np.dtype(sctype).char
