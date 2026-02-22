"""Polynomial functions (NumPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def _to_np(x):
    """Convert macmetalpy array to numpy, or pass through."""
    if isinstance(x, ndarray):
        return x.get()
    return x


def poly(seq_of_zeros):
    """Find the coefficients of a polynomial with the given sequence of roots."""
    result = np.poly(_to_np(seq_of_zeros))
    return creation.array(result)


def polyval(p, x):
    """Evaluate a polynomial at specific values."""
    result = np.polyval(_to_np(p), _to_np(x))
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def polyfit(x, y, deg, **kwargs):
    """Least squares polynomial fit."""
    result = np.polyfit(_to_np(x), _to_np(y), deg, **kwargs)
    return creation.array(result)


def polyadd(a1, a2):
    """Find the sum of two polynomials."""
    result = np.polyadd(_to_np(a1), _to_np(a2))
    return creation.array(result)


def polysub(a1, a2):
    """Find the difference of two polynomials."""
    result = np.polysub(_to_np(a1), _to_np(a2))
    return creation.array(result)


def polymul(a1, a2):
    """Find the product of two polynomials."""
    result = np.polymul(_to_np(a1), _to_np(a2))
    return creation.array(result)


def polydiv(u, v):
    """Returns the quotient and remainder of polynomial division."""
    q, r = np.polydiv(_to_np(u), _to_np(v))
    return creation.array(q), creation.array(r)


def polyder(p, m=1):
    """Return the derivative of the specified order of a polynomial."""
    result = np.polyder(_to_np(p), m=m)
    return creation.array(result)


def polyint(p, m=1, k=0):
    """Return an antiderivative (indefinite integral) of a polynomial."""
    result = np.polyint(_to_np(p), m=m, k=k)
    return creation.array(result)


def roots(p):
    """Return the roots of a polynomial with coefficients given in p."""
    result = np.roots(_to_np(p))
    return creation.array(result)


# Re-export numpy's poly1d for API compatibility
poly1d = np.poly1d
