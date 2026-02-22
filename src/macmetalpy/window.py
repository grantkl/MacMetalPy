"""Window functions (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray


def bartlett(M):
    """Return the Bartlett window."""
    return ndarray._from_np_direct(np.bartlett(M).astype(np.float32))


def blackman(M):
    """Return the Blackman window."""
    return ndarray._from_np_direct(np.blackman(M).astype(np.float32))


def hamming(M):
    """Return the Hamming window."""
    return ndarray._from_np_direct(np.hamming(M).astype(np.float32))


def hanning(M):
    """Return the Hanning window."""
    return ndarray._from_np_direct(np.hanning(M).astype(np.float32))


def kaiser(M, beta):
    """Return the Kaiser window."""
    return ndarray._from_np_direct(np.kaiser(M, beta).astype(np.float32))
