"""Indexing routines (CuPy-compatible)."""
from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


# ------------------------------------------------------------------ Take / Put


def take(a, indices, axis=None):
    """Take elements from an array along an axis."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    ind_np = indices.get() if isinstance(indices, ndarray) else indices
    result = np.take(a.get(), ind_np, axis=axis)
    return creation.array(result)


def take_along_axis(arr, indices, axis):
    """Take values from the input array by matching 1d index and data slices along the given axis."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    result = np.take_along_axis(arr.get(), indices.get(), axis)
    return creation.array(result)


def put(a, ind, v):
    """Replaces specified elements of an array with given values. Mutates *a* in place."""
    ind_np = ind.get() if isinstance(ind, ndarray) else ind
    v_np = v.get() if isinstance(v, ndarray) else v
    tmp = a.get()
    np.put(tmp, ind_np, v_np)
    a.set(tmp)
    return None


def put_along_axis(arr, indices, values, axis):
    """Put values into the destination array by matching 1d index and data slices along the given axis."""
    indices_np = indices.get() if isinstance(indices, ndarray) else indices
    values_np = values.get() if isinstance(values, ndarray) else values
    tmp = arr.get()
    np.put_along_axis(tmp, indices_np, values_np, axis)
    arr.set(tmp)
    return None


def putmask(a, mask, values):
    """Changes elements of an array based on conditional and input values. Mutates *a*."""
    mask_np = mask.get() if isinstance(mask, ndarray) else mask
    val_np = values.get() if isinstance(values, ndarray) else values
    tmp = a.get()
    np.putmask(tmp, mask_np, val_np)
    a.set(tmp)
    return None


def place(arr, mask, vals):
    """Change elements of an array based on conditional and input values. Mutates *arr*."""
    mask_np = mask.get() if isinstance(mask, ndarray) else mask
    val_np = vals.get() if isinstance(vals, ndarray) else vals
    tmp = arr.get()
    np.place(tmp, mask_np, val_np)
    arr.set(tmp)
    return None


# ------------------------------------------------------------------ Selection


def choose(a, choices):
    """Construct an array from an index array and a set of arrays to choose from."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    np_choices = [c.get() if isinstance(c, ndarray) else c for c in choices]
    result = np.choose(a.get(), np_choices)
    return creation.array(result)


def compress(condition, a, axis=None):
    """Return selected slices of an array along given axis."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    cond_np = condition.get() if isinstance(condition, ndarray) else condition
    result = np.compress(cond_np, a.get(), axis=axis)
    return creation.array(result)


def select(condlist, choicelist, default=0):
    """Return an array drawn from elements in choicelist, depending on conditions."""
    np_conds = [c.get() if isinstance(c, ndarray) else c for c in condlist]
    np_choices = [ch.get() if isinstance(ch, ndarray) else ch for ch in choicelist]
    result = np.select(np_conds, np_choices, default=default)
    return creation.array(result)


def extract(condition, arr):
    """Return the elements of an array that satisfy some condition."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    cond_np = condition.get() if isinstance(condition, ndarray) else condition
    result = np.extract(cond_np, arr.get())
    return creation.array(result)


# ------------------------------------------------------------------ Index arrays


def diag_indices(n, ndim=2):
    """Return the indices to access the main diagonal of an array."""
    return tuple(creation.array(x) for x in np.diag_indices(n, ndim))


def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional array."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    return tuple(creation.array(x) for x in np.diag_indices_from(arr.get()))


def tril_indices(n, k=0, m=None):
    """Return the indices for the lower-triangle of an (n, m) array."""
    return tuple(creation.array(x) for x in np.tril_indices(n, k, m))


def tril_indices_from(arr, k=0):
    """Return the indices for the lower-triangle of arr."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    return tuple(creation.array(x) for x in np.tril_indices_from(arr.get(), k))


def triu_indices(n, k=0, m=None):
    """Return the indices for the upper-triangle of an (n, m) array."""
    return tuple(creation.array(x) for x in np.triu_indices(n, k, m))


def triu_indices_from(arr, k=0):
    """Return the indices for the upper-triangle of arr."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    return tuple(creation.array(x) for x in np.triu_indices_from(arr.get(), k))


# ------------------------------------------------------------------ Conversion


def ravel_multi_index(multi_index, dims):
    """Converts a tuple of index arrays into an array of flat indices."""
    np_mi = [x.get() if isinstance(x, ndarray) else x for x in multi_index]
    result = np.ravel_multi_index(np_mi, dims)
    return creation.array(np.asarray(result))


def unravel_index(indices, shape):
    """Converts a flat index or array of flat indices into a tuple of coordinate arrays."""
    ind_np = indices.get() if isinstance(indices, ndarray) else indices
    return tuple(creation.array(np.asarray(x)) for x in np.unravel_index(ind_np, shape))


# ------------------------------------------------------------------ Fill


def fill_diagonal(a, val):
    """Fill the main diagonal of the given array. Mutates *a* in place."""
    tmp = a.get()
    np.fill_diagonal(tmp, val)
    a.set(tmp)
    return None


# ------------------------------------------------------------------ Search


def nonzero(a):
    """Return the indices of the elements that are non-zero."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return tuple(creation.array(x) for x in np.nonzero(a.get()))


def flatnonzero(a):
    """Return indices that are non-zero in the flattened version of a."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return creation.array(np.flatnonzero(a.get()))


def argwhere(a):
    """Find the indices of array elements that are non-zero, grouped by element."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return creation.array(np.argwhere(a.get()))


# ------------------------------------------------------------------ Advanced


def ix_(*args):
    """Construct an open mesh from multiple sequences."""
    np_args = [a.get() if isinstance(a, ndarray) else a for a in args]
    return tuple(creation.array(x) for x in np.ix_(*np_args))
