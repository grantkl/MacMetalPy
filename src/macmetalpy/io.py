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


def load(file, mmap_mode=None, allow_pickle=False):
    """Load arrays from .npy or .npz files, returning GPU arrays."""
    result = np.load(file, mmap_mode=mmap_mode, allow_pickle=allow_pickle)
    if isinstance(result, np.lib.npyio.NpzFile):
        return {key: creation.array(result[key]) for key in result.files}
    return creation.array(result)


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
