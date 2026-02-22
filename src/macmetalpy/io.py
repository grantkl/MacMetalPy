"""I/O functions for saving and loading arrays (CuPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def save(file, arr, allow_pickle=True):
    """Save an array to a binary file in NumPy .npy format."""
    if isinstance(arr, ndarray):
        arr = arr.get()
    np.save(file, arr, allow_pickle=allow_pickle)


def load(file, mmap_mode=None, allow_pickle=False, encoding='ASCII', fix_imports=True):
    """Load arrays from .npy or .npz files, returning GPU arrays."""
    result = np.load(file, mmap_mode=mmap_mode, allow_pickle=allow_pickle,
                     encoding=encoding, fix_imports=fix_imports)
    if isinstance(result, np.lib.npyio.NpzFile):
        return {key: creation.array(result[key]) for key in result.files}
    return ndarray._from_np_direct(result)


def savez(file, *args, **kwds):
    """Save several arrays into a single .npz file (uncompressed)."""
    cpu_args = [a.get() if isinstance(a, ndarray) else np.asarray(a) for a in args]
    cpu_kwds = {k: v.get() if isinstance(v, ndarray) else np.asarray(v) for k, v in kwds.items()}
    np.savez(file, *cpu_args, **cpu_kwds)


def savez_compressed(file, *args, **kwds):
    """Save several arrays into a single compressed .npz file."""
    cpu_args = [a.get() if isinstance(a, ndarray) else np.asarray(a) for a in args]
    cpu_kwds = {k: v.get() if isinstance(v, ndarray) else np.asarray(v) for k, v in kwds.items()}
    np.savez_compressed(file, *cpu_args, **cpu_kwds)


def _to_np(x):
    """Convert macmetalpy array to numpy, or pass through."""
    if isinstance(x, ndarray):
        return x.get()
    return x


def loadtxt(fname, **kwargs):
    """Load data from a text file."""
    result = np.loadtxt(fname, **kwargs)
    return ndarray._from_np_direct(result)


def savetxt(fname, X, **kwargs):
    """Save an array to a text file."""
    np.savetxt(fname, _to_np(X), **kwargs)


def fromfile(file, dtype=float, count=-1, sep='', **kwargs):
    """Construct an array from data in a binary or text file."""
    result = np.fromfile(file, dtype=dtype, count=count, sep=sep, **kwargs)
    return ndarray._from_np_direct(result)


def genfromtxt(fname, **kwargs):
    """Load data from a text file, with missing values handled."""
    result = np.genfromtxt(fname, **kwargs)
    return ndarray._from_np_direct(result)


def fromregex(file, regexp, dtype, **kwargs):
    """Construct an array from a text file using regular expression parsing."""
    result = np.fromregex(file, regexp, dtype, **kwargs)
    # Structured/void dtypes (named fields) can't be GPU-accelerated;
    # return the numpy structured array directly in that case.
    if result.dtype.names is not None:
        return result
    return ndarray._from_np_direct(result)


def from_dlpack(x):
    """Create an array from a DLPack capsule."""
    if isinstance(x, ndarray):
        return creation.array(x.get())
    result = np.from_dlpack(x)
    return ndarray._from_np_direct(result)
