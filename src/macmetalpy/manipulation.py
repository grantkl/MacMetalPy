"""Array manipulation functions (CuPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def tile(a, reps):
    """Construct an array by repeating a the number of times given by reps."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.tile(a.get(), reps)
    return creation.array(result)


def repeat(a, repeats, axis=None):
    """Repeat elements of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.repeat(a.get(), repeats, axis=axis)
    return creation.array(result)


def flip(a, axis=None):
    """Reverse the order of elements along the given axis."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.flip(a.get(), axis=axis)
    return creation.array(np.ascontiguousarray(result))


def roll(a, shift, axis=None):
    """Roll array elements along a given axis."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.roll(a.get(), shift, axis=axis)
    return creation.array(result)


def split(a, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    results = np.split(a.get(), indices_or_sections, axis=axis)
    return [creation.array(r) for r in results]


def array_split(a, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays (allows unequal division)."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    results = np.array_split(a.get(), indices_or_sections, axis=axis)
    return [creation.array(r) for r in results]


def squeeze(a, axis=None):
    """Remove single-dimensional entries from the shape of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.squeeze(axis=axis)


def ravel(a):
    """Return a contiguous flattened array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.ravel()


def moveaxis(a, source, destination):
    """Move axes of an array to new positions."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.moveaxis(a.get(), source, destination)
    return creation.array(np.ascontiguousarray(result))


def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.swapaxes(a.get(), axis1, axis2)
    return creation.array(np.ascontiguousarray(result))


def broadcast_to(a, shape):
    """Broadcast an array to a new shape."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.broadcast_to(a.get(), shape)
    return creation.array(np.ascontiguousarray(result))


# ------------------------------------------------------------------ extended manipulation

def reshape(a, newshape):
    """Give a new shape to an array without changing its data."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.reshape(newshape)


def transpose(a, axes=None):
    """Permute the dimensions of an array."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    return a.transpose(axes)


def rollaxis(a, axis, start=0):
    """Roll the specified axis backwards."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.rollaxis(a.get(), axis, start)
    return creation.array(np.ascontiguousarray(result))


def atleast_1d(*arys):
    """Convert inputs to arrays with at least one dimension."""
    results = []
    for a in arys:
        if isinstance(a, ndarray):
            a_np = a.get()
        else:
            a_np = np.asarray(a)
        results.append(creation.array(np.atleast_1d(a_np)))
    return results[0] if len(results) == 1 else results


def atleast_2d(*arys):
    """Convert inputs to arrays with at least two dimensions."""
    results = []
    for a in arys:
        if isinstance(a, ndarray):
            a_np = a.get()
        else:
            a_np = np.asarray(a)
        results.append(creation.array(np.atleast_2d(a_np)))
    return results[0] if len(results) == 1 else results


def atleast_3d(*arys):
    """Convert inputs to arrays with at least three dimensions."""
    results = []
    for a in arys:
        if isinstance(a, ndarray):
            a_np = a.get()
        else:
            a_np = np.asarray(a)
        results.append(creation.array(np.atleast_3d(a_np)))
    return results[0] if len(results) == 1 else results


def dstack(tup):
    """Stack arrays in sequence depth wise (along third axis)."""
    np_arrays = [a.get() if isinstance(a, ndarray) else np.asarray(a) for a in tup]
    return creation.array(np.dstack(np_arrays))


def column_stack(tup):
    """Stack 1-D arrays as columns into a 2-D array."""
    np_arrays = [a.get() if isinstance(a, ndarray) else np.asarray(a) for a in tup]
    return creation.array(np.column_stack(np_arrays))


def concat(arrays, axis=0):
    """Alias for concatenate."""
    np_arrays = [a.get() if isinstance(a, ndarray) else np.asarray(a) for a in arrays]
    return creation.array(np.concatenate(np_arrays, axis=axis))


def dsplit(ary, indices_or_sections):
    """Split array along third axis."""
    if not isinstance(ary, ndarray):
        ary = creation.asarray(ary)
    results = np.dsplit(ary.get(), indices_or_sections)
    return [creation.array(r) for r in results]


def hsplit(ary, indices_or_sections):
    """Split array along second axis."""
    if not isinstance(ary, ndarray):
        ary = creation.asarray(ary)
    results = np.hsplit(ary.get(), indices_or_sections)
    return [creation.array(r) for r in results]


def vsplit(ary, indices_or_sections):
    """Split array along first axis."""
    if not isinstance(ary, ndarray):
        ary = creation.asarray(ary)
    results = np.vsplit(ary.get(), indices_or_sections)
    return [creation.array(r) for r in results]


def delete(arr, obj, axis=None):
    """Return a new array with sub-arrays along an axis deleted."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    result = np.delete(arr.get(), obj, axis=axis)
    return creation.array(result)


def append(arr, values, axis=None):
    """Append values to the end of an array."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    val_np = values.get() if isinstance(values, ndarray) else np.asarray(values)
    result = np.append(arr.get(), val_np, axis=axis)
    return creation.array(result)


def resize(a, new_shape):
    """Return a new array with the given shape."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.resize(a.get(), new_shape)
    return creation.array(result)


def trim_zeros(filt, trim='fb'):
    """Trim the leading and/or trailing zeros from a 1-D array."""
    if not isinstance(filt, ndarray):
        filt = creation.asarray(filt)
    result = np.trim_zeros(filt.get(), trim)
    return creation.array(result)


def fliplr(m):
    """Reverse the order of elements along axis 1 (left/right)."""
    if not isinstance(m, ndarray):
        m = creation.asarray(m)
    return creation.array(np.ascontiguousarray(np.fliplr(m.get())))


def flipud(m):
    """Reverse the order of elements along axis 0 (up/down)."""
    if not isinstance(m, ndarray):
        m = creation.asarray(m)
    return creation.array(np.ascontiguousarray(np.flipud(m.get())))


def rot90(m, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the plane specified by axes."""
    if not isinstance(m, ndarray):
        m = creation.asarray(m)
    result = np.rot90(m.get(), k=k, axes=axes)
    return creation.array(np.ascontiguousarray(result))


def broadcast_arrays(*args):
    """Broadcast any number of arrays against each other."""
    np_arrays = [a.get() if isinstance(a, ndarray) else np.asarray(a) for a in args]
    results = np.broadcast_arrays(*np_arrays)
    return [creation.array(np.ascontiguousarray(r)) for r in results]


def copyto(dst, src):
    """Copy values from one array to another."""
    if not isinstance(src, ndarray):
        src = creation.asarray(src)
    dst.set(src.get())


def pad(array, pad_width, mode='constant', **kwargs):
    """Pad an array."""
    if not isinstance(array, ndarray):
        array = creation.asarray(array)
    result = np.pad(array.get(), pad_width, mode=mode, **kwargs)
    return creation.array(result)


def block(arrays):
    """Assemble an ndarray from nested lists of blocks."""
    def _to_numpy(obj):
        if isinstance(obj, ndarray):
            return obj.get()
        if isinstance(obj, list):
            return [_to_numpy(item) for item in obj]
        return np.asarray(obj)

    np_arrays = _to_numpy(arrays)
    result = np.block(np_arrays)
    return creation.array(result)


def insert(arr, obj, values, axis=None):
    """Insert values along the given axis before the given indices."""
    if not isinstance(arr, ndarray):
        arr = creation.asarray(arr)
    val_np = values.get() if isinstance(values, ndarray) else np.asarray(values)
    result = np.insert(arr.get(), obj, val_np, axis=axis)
    return creation.array(result)


def broadcast_shapes(*args):
    """Broadcast shapes and return the resulting shape."""
    return np.broadcast_shapes(*args)


def asfortranarray(a, dtype=None):
    """Return an array laid out in Fortran order in memory.

    Since Metal is row-major only, this returns a contiguous (C-order) copy
    with the correct values and dtype, matching the NumPy interface.
    """
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if dtype is not None:
        return creation.array(a.get(), dtype=dtype)
    return creation.array(a.get())
