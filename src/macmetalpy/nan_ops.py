"""NaN-aware and extended statistics functions (CuPy-compatible)."""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


# ------------------------------------------------------------------ helpers

def _ensure_ndarray(a):
    if not isinstance(a, ndarray):
        return creation.asarray(a)
    return a


def _wrap_result(result):
    """Wrap a numpy scalar or array as an ndarray."""
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


# ================================================================== NaN reductions

def nansum(a, axis=None):
    """Sum of array elements, treating NaNs as zero."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nansum(a.get(), axis=axis))


def nanprod(a, axis=None):
    """Product of array elements, treating NaNs as one."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nanprod(a.get(), axis=axis))


def nancumsum(a, axis=None):
    """Cumulative sum, treating NaNs as zero."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nancumsum(a.get(), axis=axis))


def nancumprod(a, axis=None):
    """Cumulative product, treating NaNs as one."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nancumprod(a.get(), axis=axis))


def nanmax(a, axis=None):
    """Maximum of array elements, ignoring NaNs."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nanmax(a.get(), axis=axis))


def nanmin(a, axis=None):
    """Minimum of array elements, ignoring NaNs."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nanmin(a.get(), axis=axis))


def nanmean(a, axis=None):
    """Arithmetic mean, ignoring NaNs."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nanmean(a.get(), axis=axis))


def nanmedian(a, axis=None):
    """Median, ignoring NaNs."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nanmedian(a.get(), axis=axis))


def nanstd(a, axis=None):
    """Standard deviation, ignoring NaNs."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nanstd(a.get(), axis=axis))


def nanvar(a, axis=None):
    """Variance, ignoring NaNs."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.nanvar(a.get(), axis=axis))


def nanargmax(a, axis=None):
    """Indices of the maximum values, ignoring NaNs."""
    a = _ensure_ndarray(a)
    result = np.nanargmax(a.get(), axis=axis)
    if axis is None:
        return int(result)
    return _wrap_result(result)


def nanargmin(a, axis=None):
    """Indices of the minimum values, ignoring NaNs."""
    a = _ensure_ndarray(a)
    result = np.nanargmin(a.get(), axis=axis)
    if axis is None:
        return int(result)
    return _wrap_result(result)


# ================================================================== Extended stats

def ptp(a, axis=None):
    """Range of values (maximum - minimum) along an axis."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.ptp(a.get(), axis=axis))


def quantile(a, q, axis=None):
    """Compute the q-th quantile."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.quantile(a.get(), q, axis=axis))


def average(a, axis=None, weights=None):
    """Compute weighted average."""
    a = _ensure_ndarray(a)
    w = weights.get() if isinstance(weights, ndarray) else weights
    return _wrap_result(np.average(a.get(), axis=axis, weights=w))


def corrcoef(x, y=None):
    """Pearson correlation coefficient matrix."""
    x = _ensure_ndarray(x)
    y_np = y.get() if y is not None else None
    return _wrap_result(np.corrcoef(x.get(), y_np))


def correlate(a, v, mode="valid"):
    """Cross-correlation of two 1-D sequences."""
    a = _ensure_ndarray(a)
    v = _ensure_ndarray(v)
    return _wrap_result(np.correlate(a.get(), v.get(), mode=mode))


def cov(m, y=None):
    """Covariance matrix."""
    m = _ensure_ndarray(m)
    y_np = y.get() if y is not None else None
    return _wrap_result(np.cov(m.get(), y_np))


# ================================================================== Histograms

def histogram(a, bins=10):
    """Compute histogram."""
    a = _ensure_ndarray(a)
    hist, edges = np.histogram(a.get(), bins=bins)
    return creation.array(hist), creation.array(edges)


def histogram2d(x, y, bins=10):
    """Compute 2-D histogram."""
    x = _ensure_ndarray(x)
    y = _ensure_ndarray(y)
    H, xedges, yedges = np.histogram2d(x.get(), y.get(), bins=bins)
    return creation.array(H), creation.array(xedges), creation.array(yedges)


def histogramdd(sample, bins=10):
    """Compute multidimensional histogram."""
    sample = _ensure_ndarray(sample)
    H, edges = np.histogramdd(sample.get(), bins=bins)
    return creation.array(H), [creation.array(e) for e in edges]


def bincount(x, weights=None, minlength=0):
    """Count number of occurrences of each value in array of non-negative ints."""
    x = _ensure_ndarray(x)
    w = weights.get() if isinstance(weights, ndarray) else weights
    result = np.bincount(x.get().astype(int), weights=w, minlength=minlength)
    return creation.array(result)


def digitize(x, bins, right=False):
    """Return the indices of the bins to which each value belongs."""
    x = _ensure_ndarray(x)
    bins_np = bins.get() if isinstance(bins, ndarray) else bins
    result = np.digitize(x.get(), bins_np, right=right)
    return creation.array(result)


# ================================================================== Diff extensions

def ediff1d(ary, to_end=None, to_begin=None):
    """Differences between consecutive elements of an array."""
    ary = _ensure_ndarray(ary)
    te = to_end.get() if isinstance(to_end, ndarray) else to_end
    tb = to_begin.get() if isinstance(to_begin, ndarray) else to_begin
    result = np.ediff1d(ary.get(), to_end=te, to_begin=tb)
    return creation.array(result)


def gradient(f, *varargs):
    """Return the gradient of an N-dimensional array."""
    f = _ensure_ndarray(f)
    result = np.gradient(f.get(), *varargs)
    if isinstance(result, list):
        return [creation.array(r) for r in result]
    return creation.array(result)
