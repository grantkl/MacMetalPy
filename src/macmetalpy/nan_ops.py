"""NaN-aware and extended statistics functions (CuPy-compatible).

GPU-accelerated: NaN values are replaced on the GPU before reduction kernels run.
"""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray, _c_contiguous_strides
from . import creation
from ._kernel_cache import KernelCache
from ._metal_backend import MetalBackend


def _wrap_np(np_data):
    """Fast inline ndarray construction for known-good numpy arrays."""
    arr = ndarray.__new__(ndarray)
    arr._buffer = None
    arr._np_data = np_data
    arr._shape = np_data.shape
    arr._dtype = np_data.dtype
    arr._strides = _c_contiguous_strides(np_data.shape)
    arr._offset = 0
    arr._base = None
    return arr


# ------------------------------------------------------------------ helpers

def _ensure_ndarray(a):
    if not isinstance(a, ndarray):
        return creation.asarray(a)
    return a


def _cpu_view(a):
    """Get a zero-copy numpy view of a's data (syncs if GPU-resident)."""
    if a._np_data is None:
        MetalBackend().synchronize()
    return a._get_view()


def _get_np(a):
    """Get numpy data, preferring _np_data for CPU-resident arrays."""
    np_data = a._np_data
    if np_data is not None:
        return np_data
    return _cpu_view(a)


def _wrap_result(result):
    """Wrap a numpy scalar or array as an ndarray."""
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return ndarray._from_np_direct(result)


def _copy_to_out(result, out):
    """Copy result data into out array and return out."""
    tmp = result.astype(out.dtype)._ensure_contiguous()
    out._adopt_buffer(tmp._buffer)
    return out


def _get_where(where):
    """Extract numpy array from a where mask (may be ndarray or numpy)."""
    if where is None:
        return None
    if isinstance(where, ndarray):
        return where.get()
    return where


def _nan_replace(a_contig, kernel_name):
    """Replace NaN values using a GPU kernel, returning a clean ndarray."""
    a_contig._ensure_gpu()
    backend = MetalBackend()
    cache = KernelCache()
    shader = cache.get_shader("nan_elementwise", a_contig._dtype)
    clean_buf = backend.create_buffer(a_contig.size, a_contig._dtype)
    backend.execute_kernel(shader, kernel_name, a_contig.size, [a_contig._buffer, clean_buf])
    return ndarray._from_buffer(clean_buf, a_contig.shape, a_contig._dtype)


def _ensure_float(a):
    """Ensure array is a float type (NaN only exists for floats)."""
    if not np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32)
    return a


# ================================================================== NaN reductions

def nansum(a, axis=None, dtype=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """Sum of array elements, treating NaNs as zero."""
    a = _ensure_ndarray(a)

    # where parameter -> CPU fallback
    if where is not np._NoValue:
        kwargs = dict(axis=axis, dtype=dtype, keepdims=keepdims)
        if initial is not np._NoValue:
            kwargs['initial'] = initial
        kwargs['where'] = _get_where(where)
        result = _wrap_result(np.nansum(a.get(), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    # Non-float dtypes: NaN is impossible
    if not np.issubdtype(a.dtype, np.floating):
        from . import reductions
        return reductions.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)

    # CPU fallback for small/medium arrays (GPU kernel overhead dominates)
    if a.size < 4194304:
        kwargs = dict(axis=axis, keepdims=keepdims)
        if dtype is not None:
            kwargs['dtype'] = dtype
        if initial is not np._NoValue:
            kwargs['initial'] = initial
        result = _wrap_result(np.nansum(_get_np(a), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    # GPU path: replace NaN with 0, then reduce
    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    clean = _nan_replace(a_contig, "nan_replace_zero")

    if dtype is not None:
        clean = clean.astype(dtype)
    result = clean.sum(axis=axis if a_contig.shape == a.shape else None, keepdims=keepdims)

    if initial is not np._NoValue:
        initial_arr = creation.full(result.shape, initial, dtype=result.dtype)
        result = result._binary_op(initial_arr, "add_op")
    if out is not None:
        return _copy_to_out(result, out)
    return result


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """Product of array elements, treating NaNs as one."""
    a = _ensure_ndarray(a)

    if where is not np._NoValue:
        kwargs = dict(axis=axis, dtype=dtype, keepdims=keepdims)
        if initial is not np._NoValue:
            kwargs['initial'] = initial
        kwargs['where'] = _get_where(where)
        result = _wrap_result(np.nanprod(a.get(), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    if not np.issubdtype(a.dtype, np.floating):
        from . import reductions
        return reductions.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)

    if a.size < 4194304:
        kwargs = dict(axis=axis, keepdims=keepdims)
        if dtype is not None:
            kwargs['dtype'] = dtype
        if initial is not np._NoValue:
            kwargs['initial'] = initial
        result = _wrap_result(np.nanprod(_get_np(a), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    clean = _nan_replace(a_contig, "nan_replace_one")

    if dtype is not None:
        clean = clean.astype(dtype)
    result = clean.prod(axis=axis if a_contig.shape == a.shape else None, keepdims=keepdims)

    if initial is not np._NoValue:
        initial_arr = creation.full(result.shape, initial, dtype=result.dtype)
        result = result._binary_op(initial_arr, "mul_op")
    if out is not None:
        return _copy_to_out(result, out)
    return result


def nancumsum(a, axis=None, dtype=None, out=None):
    """Cumulative sum, treating NaNs as zero."""
    a = _ensure_ndarray(a)

    if not np.issubdtype(a.dtype, np.floating):
        from . import reductions
        return reductions.cumsum(a, axis=axis, dtype=dtype, out=out)

    if a.size < 4194304:
        kwargs = dict(axis=axis)
        if dtype is not None:
            kwargs['dtype'] = dtype
        result = _wrap_result(np.nancumsum(_get_np(a), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    clean = _nan_replace(a_contig, "nan_replace_zero")

    if dtype is not None:
        clean = clean.astype(dtype)
    result = clean.cumsum(axis=axis if a_contig.shape == a.shape else None)

    if out is not None:
        return _copy_to_out(result, out)
    return result


def nancumprod(a, axis=None, dtype=None, out=None):
    """Cumulative product, treating NaNs as one."""
    a = _ensure_ndarray(a)

    if not np.issubdtype(a.dtype, np.floating):
        from . import reductions
        return reductions.cumprod(a, axis=axis, dtype=dtype, out=out)

    if a.size < 4194304:
        kwargs = dict(axis=axis)
        if dtype is not None:
            kwargs['dtype'] = dtype
        result = _wrap_result(np.nancumprod(_get_np(a), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    clean = _nan_replace(a_contig, "nan_replace_one")

    if dtype is not None:
        clean = clean.astype(dtype)
    result = clean.cumprod(axis=axis if a_contig.shape == a.shape else None)

    if out is not None:
        return _copy_to_out(result, out)
    return result


def nanmax(a, axis=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """Maximum of array elements, ignoring NaNs."""
    a = _ensure_ndarray(a)

    if where is not np._NoValue:
        kwargs = dict(axis=axis, keepdims=keepdims)
        if initial is not np._NoValue:
            kwargs['initial'] = initial
        kwargs['where'] = _get_where(where)
        result = _wrap_result(np.nanmax(a.get(), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    if not np.issubdtype(a.dtype, np.floating):
        from . import reductions
        return reductions.max(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    if a.size < 4194304:
        np_data = _get_np(a)
        if initial is not np._NoValue:
            r = np.nanmax(np_data, axis=axis, keepdims=keepdims, initial=initial)
        else:
            r = np.nanmax(np_data, axis=axis, keepdims=keepdims)
        result = _wrap_result(r)
        if out is not None:
            return _copy_to_out(result, out)
        return result

    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    clean = _nan_replace(a_contig, "nan_replace_neg_inf")

    result = clean.max(axis=axis if a_contig.shape == a.shape else None, keepdims=keepdims)

    if initial is not np._NoValue:
        initial_arr = creation.full(result.shape, initial, dtype=result.dtype)
        result = result._binary_op(initial_arr, "max_op")
    if out is not None:
        return _copy_to_out(result, out)
    return result


def nanmin(a, axis=None, out=None, keepdims=False, initial=np._NoValue, where=np._NoValue):
    """Minimum of array elements, ignoring NaNs."""
    a = _ensure_ndarray(a)

    if where is not np._NoValue:
        kwargs = dict(axis=axis, keepdims=keepdims)
        if initial is not np._NoValue:
            kwargs['initial'] = initial
        kwargs['where'] = _get_where(where)
        result = _wrap_result(np.nanmin(a.get(), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    if not np.issubdtype(a.dtype, np.floating):
        from . import reductions
        return reductions.min(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    if a.size < 4194304:
        np_data = _get_np(a)
        if initial is not np._NoValue:
            r = np.nanmin(np_data, axis=axis, keepdims=keepdims, initial=initial)
        else:
            r = np.nanmin(np_data, axis=axis, keepdims=keepdims)
        result = _wrap_result(r)
        if out is not None:
            return _copy_to_out(result, out)
        return result

    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    clean = _nan_replace(a_contig, "nan_replace_pos_inf")

    result = clean.min(axis=axis if a_contig.shape == a.shape else None, keepdims=keepdims)

    if initial is not np._NoValue:
        initial_arr = creation.full(result.shape, initial, dtype=result.dtype)
        result = result._binary_op(initial_arr, "min_op")
    if out is not None:
        return _copy_to_out(result, out)
    return result


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, where=np._NoValue):
    """Arithmetic mean, ignoring NaNs."""
    a = _ensure_ndarray(a)

    if where is not np._NoValue:
        kwargs = dict(axis=axis, dtype=dtype, keepdims=keepdims)
        kwargs['where'] = _get_where(where)
        result = _wrap_result(np.nanmean(a.get(), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    if not np.issubdtype(a.dtype, np.floating):
        # No NaNs possible for integer types
        result = a.mean(axis=axis, keepdims=keepdims)
        if out is not None:
            return _copy_to_out(result, out)
        return result

    if a.size < 4194304:
        kwargs = dict(axis=axis, keepdims=keepdims)
        if dtype is not None:
            kwargs['dtype'] = dtype
        result = _wrap_result(np.nanmean(_get_np(a), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    # GPU path: nan_replace_zero for sum, nan_count for count
    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    clean_sum = _nan_replace(a_contig, "nan_replace_zero")
    clean_count = _nan_replace(a_contig, "nan_count")

    # Cast to requested dtype before summing for precision
    if dtype is not None:
        clean_sum = clean_sum.astype(dtype)
        clean_count = clean_count.astype(dtype)

    reduce_axis = axis if a_contig.shape == a.shape else None
    total = clean_sum.sum(axis=reduce_axis, keepdims=keepdims)
    count = clean_count.sum(axis=reduce_axis, keepdims=keepdims)

    # GPU division with NaN where count==0
    from . import math_ops
    out_dtype = dtype if dtype is not None else a.dtype
    total = total.astype(out_dtype)
    count = count.astype(out_dtype)
    quotient = total._binary_op(count, "div_op")
    count_positive = count._comparison_op(creation.full(count.shape, 0, dtype=count.dtype), "gt_op")
    nan_fill = creation.full(quotient.shape, np.nan, dtype=out_dtype)
    result = math_ops.where(count_positive, quotient, nan_fill)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Median, ignoring NaNs."""
    a = _ensure_ndarray(a)
    result = _wrap_result(np.nanmedian(_get_np(a), axis=axis, overwrite_input=overwrite_input, keepdims=keepdims))
    if out is not None:
        return _copy_to_out(result, out)
    return result


def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=np._NoValue, *, correction=None, mean=None):
    """Standard deviation, ignoring NaNs."""
    if correction is not None:
        ddof = correction
    a = _ensure_ndarray(a)

    if where is not np._NoValue:
        kwargs = dict(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        kwargs['where'] = _get_where(where)
        result = _wrap_result(np.nanstd(a.get(), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    # dtype casting requires CPU for precision matching
    if dtype is not None and a.dtype != np.dtype(dtype):
        kwargs = dict(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        result = _wrap_result(np.nanstd(a.get(), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    if a.size < 4194304:
        kwargs = dict(axis=axis, ddof=ddof, keepdims=keepdims)
        if dtype is not None:
            kwargs['dtype'] = dtype
        result = _wrap_result(np.nanstd(_get_np(a), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    # Compute variance then sqrt on GPU
    var_result = nanvar(a, axis=axis, ddof=ddof, keepdims=keepdims)
    out_dtype = dtype if dtype is not None else a.dtype
    result = var_result.astype(out_dtype)._unary_op("sqrt_op")
    if out is not None:
        return _copy_to_out(result, out)
    return result


def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=np._NoValue, *, correction=None, mean=None):
    """Variance, ignoring NaNs."""
    if correction is not None:
        ddof = correction
    a = _ensure_ndarray(a)

    if where is not np._NoValue:
        kwargs = dict(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        kwargs['where'] = _get_where(where)
        result = _wrap_result(np.nanvar(a.get(), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    if not np.issubdtype(a.dtype, np.floating):
        # No NaNs for integers -- use regular var
        from . import reductions
        return reductions.var(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

    if a.size < 4194304:
        kwargs = dict(axis=axis, ddof=ddof, keepdims=keepdims)
        if dtype is not None:
            kwargs['dtype'] = dtype
        result = _wrap_result(np.nanvar(_get_np(a), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    # dtype casting requires CPU for precision matching
    if dtype is not None and a.dtype != np.dtype(dtype):
        kwargs = dict(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        result = _wrap_result(np.nanvar(a.get(), **kwargs))
        if out is not None:
            return _copy_to_out(result, out)
        return result

    # Two-pass GPU: compute mean, then mean of squared deviations
    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    reduce_axis = axis if a_contig.shape == a.shape else None

    # Pass 1: compute nanmean on GPU
    from . import math_ops
    clean_sum = _nan_replace(a_contig, "nan_replace_zero")
    clean_count = _nan_replace(a_contig, "nan_count")
    total = clean_sum.sum(axis=reduce_axis, keepdims=True)
    count_kd = clean_count.sum(axis=reduce_axis, keepdims=True)

    # mean = total / count (0 where count==0 to avoid inf in diff)
    zero_arr = creation.full(count_kd.shape, 0.0, dtype=a.dtype)
    count_positive = count_kd._comparison_op(zero_arr, "gt_op")
    raw_mean = total._binary_op(count_kd, "div_op")
    mean_gpu = math_ops.where(count_positive, raw_mean, zero_arr)

    # Pass 2: compute (x - mean)^2, ignoring NaN positions, on GPU
    diff = a_contig._binary_op(mean_gpu, "sub_op")
    diff_sq = diff._binary_op(diff, "mul_op")
    # Zero out NaN positions using nan_replace_zero on diff_sq
    diff_sq_clean = _nan_replace(diff_sq, "nan_replace_zero")
    sum_sq = diff_sq_clean.sum(axis=reduce_axis, keepdims=keepdims)

    # Get count with correct keepdims
    count = clean_count.sum(axis=reduce_axis, keepdims=keepdims)

    # var = sum_sq / (count - ddof), NaN where count <= ddof
    out_dtype = dtype if dtype is not None else a.dtype
    ddof_arr = creation.full(count.shape, float(ddof), dtype=out_dtype)
    count_out = count.astype(out_dtype)
    denom = count_out._binary_op(ddof_arr, "sub_op")
    sum_sq_out = sum_sq.astype(out_dtype)
    raw_var = sum_sq_out._binary_op(denom, "div_op")
    denom_positive = denom._comparison_op(creation.full(denom.shape, 0.0, dtype=out_dtype), "gt_op")
    nan_fill = creation.full(raw_var.shape, np.nan, dtype=out_dtype)
    result = math_ops.where(denom_positive, raw_var, nan_fill)
    if out is not None:
        return _copy_to_out(result, out)
    return result


def nanargmax(a, axis=None, out=None, keepdims=False):
    """Indices of the maximum values, ignoring NaNs."""
    a = _ensure_ndarray(a)

    if not np.issubdtype(a.dtype, np.floating):
        from . import reductions
        return reductions.argmax(a, axis=axis, out=out, keepdims=keepdims)

    if a.size < 4194304:
        r = np.nanargmax(_get_np(a), axis=axis, keepdims=keepdims)
        if axis is None and not keepdims:
            return int(r)
        result = _wrap_result(r)
        if out is not None:
            return _copy_to_out(result, out)
        return result

    # GPU path: replace NaN with -inf, then argmax
    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    clean = _nan_replace(a_contig, "nan_replace_neg_inf")

    from . import reductions
    return reductions.argmax(clean, axis=axis if a_contig.shape == a.shape else None, out=out, keepdims=keepdims)


def nanargmin(a, axis=None, out=None, keepdims=False):
    """Indices of the minimum values, ignoring NaNs."""
    a = _ensure_ndarray(a)

    if not np.issubdtype(a.dtype, np.floating):
        from . import reductions
        return reductions.argmin(a, axis=axis, out=out, keepdims=keepdims)

    if a.size < 4194304:
        r = np.nanargmin(_get_np(a), axis=axis, keepdims=keepdims)
        if axis is None and not keepdims:
            return int(r)
        result = _wrap_result(r)
        if out is not None:
            return _copy_to_out(result, out)
        return result

    # GPU path: replace NaN with +inf, then argmin
    a_contig = a.ravel()._ensure_contiguous() if axis is None else a._ensure_contiguous()
    clean = _nan_replace(a_contig, "nan_replace_pos_inf")

    from . import reductions
    return reductions.argmin(clean, axis=axis if a_contig.shape == a.shape else None, out=out, keepdims=keepdims)


# ================================================================== Extended stats

def ptp(a, axis=None):
    """Range of values (maximum - minimum) along an axis."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.ptp(_get_np(a), axis=axis))


def quantile(a, q, axis=None):
    """Compute the q-th quantile."""
    a = _ensure_ndarray(a)
    return _wrap_result(np.quantile(_get_np(a), q, axis=axis))


def average(a, axis=None, weights=None):
    """Compute weighted average."""
    a = _ensure_ndarray(a)
    w = weights.get() if isinstance(weights, ndarray) else weights
    return _wrap_result(np.average(_get_np(a), axis=axis, weights=w))


def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue, dtype=None):
    """Pearson correlation coefficient matrix."""
    x = _ensure_ndarray(x)
    y_np = y.get() if isinstance(y, ndarray) else y
    kwargs = dict(rowvar=rowvar)
    if bias is not np._NoValue:
        kwargs['bias'] = bias
    if ddof is not np._NoValue:
        kwargs['ddof'] = ddof
    if dtype is not None:
        kwargs['dtype'] = dtype
    return _wrap_result(np.corrcoef(_get_np(x), y_np, **kwargs))


def correlate(a, v, mode="valid"):
    """Cross-correlation of two 1-D sequences."""
    a = _ensure_ndarray(a)
    v = _ensure_ndarray(v)
    return _wrap_result(np.correlate(_get_np(a), _get_np(v), mode=mode))


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, dtype=None):
    """Covariance matrix."""
    m = _ensure_ndarray(m)
    y_np = y.get() if isinstance(y, ndarray) else y
    fw = fweights.get() if isinstance(fweights, ndarray) else fweights
    aw = aweights.get() if isinstance(aweights, ndarray) else aweights
    kwargs = dict(rowvar=rowvar, bias=bias)
    if ddof is not None:
        kwargs['ddof'] = ddof
    if fw is not None:
        kwargs['fweights'] = fw
    if aw is not None:
        kwargs['aweights'] = aw
    if dtype is not None:
        kwargs['dtype'] = dtype
    return _wrap_result(np.cov(_get_np(m), y_np, **kwargs))


# ================================================================== Histograms

def histogram(a, bins=10, range=None, density=None, weights=None):
    """Compute histogram."""
    a = _ensure_ndarray(a)
    w = weights.get() if isinstance(weights, ndarray) else weights
    kwargs = dict(bins=bins)
    if range is not None:
        kwargs['range'] = range
    if density is not None:
        kwargs['density'] = density
    if w is not None:
        kwargs['weights'] = w
    hist, edges = np.histogram(_get_np(a), **kwargs)
    return ndarray._from_np_direct(hist), ndarray._from_np_direct(edges)


def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    """Compute 2-D histogram."""
    x = _ensure_ndarray(x)
    y = _ensure_ndarray(y)
    w = weights.get() if isinstance(weights, ndarray) else weights
    kwargs = dict(bins=bins)
    if range is not None:
        kwargs['range'] = range
    if density is not None:
        kwargs['density'] = density
    if w is not None:
        kwargs['weights'] = w
    H, xedges, yedges = np.histogram2d(_get_np(x), _get_np(y), **kwargs)
    return ndarray._from_np_direct(H), ndarray._from_np_direct(xedges), ndarray._from_np_direct(yedges)


def histogramdd(sample, bins=10, range=None, density=None, weights=None):
    """Compute multidimensional histogram."""
    sample = _ensure_ndarray(sample)
    w = weights.get() if isinstance(weights, ndarray) else weights
    kwargs = dict(bins=bins)
    if range is not None:
        kwargs['range'] = range
    if density is not None:
        kwargs['density'] = density
    if w is not None:
        kwargs['weights'] = w
    H, edges = np.histogramdd(_get_np(sample), **kwargs)
    return ndarray._from_np_direct(H), [ndarray._from_np_direct(e) for e in edges]


def bincount(x, weights=None, minlength=0):
    """Count number of occurrences of each value in array of non-negative ints."""
    x = _ensure_ndarray(x)
    w = weights.get() if isinstance(weights, ndarray) else weights
    result = np.bincount(_get_np(x).astype(int), weights=w, minlength=minlength)
    return ndarray._from_np_direct(result)


def digitize(x, bins, right=False):
    """Return the indices of the bins to which each value belongs."""
    x = _ensure_ndarray(x)
    bins_np = bins.get() if isinstance(bins, ndarray) else bins
    result = np.digitize(_get_np(x), bins_np, right=right)
    return ndarray._from_np_direct(result)


# ================================================================== Diff extensions

def ediff1d(ary, to_end=None, to_begin=None):
    """Differences between consecutive elements of an array."""
    ary = _ensure_ndarray(ary)
    np_data = _get_np(ary)
    te = to_end.get() if isinstance(to_end, ndarray) else to_end
    tb = to_begin.get() if isinstance(to_begin, ndarray) else to_begin
    result = np.ediff1d(np_data, to_end=te, to_begin=tb)
    return _wrap_np(result)


def gradient(f, *varargs, axis=None, edge_order=1):
    """Return the gradient of an N-dimensional array."""
    f = _ensure_ndarray(f)
    varargs = tuple(_get_np(v) if isinstance(v, ndarray) else v for v in varargs)
    result = np.gradient(_get_np(f), *varargs, axis=axis, edge_order=edge_order)
    if isinstance(result, list):
        return [_wrap_np(np.asarray(r)) for r in result]
    return _wrap_np(np.asarray(result))


# ================================================================== NaN percentile/quantile

def nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
                  method='linear', keepdims=False):
    """Compute the q-th percentile of data along the specified axis, ignoring NaN values.

    Parameters
    ----------
    a : array_like
        Input array.
    q : float or array_like of float
        Percentile(s) to compute, in range [0, 100].
    axis : int or None, optional
        Axis along which to compute.
    keepdims : bool, optional
        If True, keep reduced axes as size-1 dimensions.
    """
    a = _ensure_ndarray(a)
    result = np.nanpercentile(_get_np(a), q, axis=axis, keepdims=keepdims)
    return _wrap_result(result)


def nanquantile(a, q, axis=None, out=None, overwrite_input=False,
                method='linear', keepdims=False):
    """Compute the q-th quantile of data along the specified axis, ignoring NaN values.

    Parameters
    ----------
    a : array_like
        Input array.
    q : float or array_like of float
        Quantile(s) to compute, in range [0, 1].
    axis : int or None, optional
        Axis along which to compute.
    keepdims : bool, optional
        If True, keep reduced axes as size-1 dimensions.
    """
    a = _ensure_ndarray(a)
    result = np.nanquantile(_get_np(a), q, axis=axis, keepdims=keepdims)
    return _wrap_result(result)


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """Compute only the bin edges for a histogram (no counts).

    Parameters
    ----------
    a : array_like
        Input data.
    bins : int or str, optional
        Number of bins or binning strategy.
    range : (float, float) or None, optional
        Lower and upper range of the bins.
    weights : array_like or None, optional
        Weights (unused for edge computation but accepted for API compat).
    """
    a = _ensure_ndarray(a)
    w = weights.get() if isinstance(weights, ndarray) else weights
    result = np.histogram_bin_edges(_get_np(a), bins=bins, range=range, weights=w)
    return _wrap_result(result)
