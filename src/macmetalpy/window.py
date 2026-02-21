"""Window functions (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from . import creation


def bartlett(M):
    """Return the Bartlett window."""
    return creation.array(np.bartlett(M).astype(np.float32))


def blackman(M):
    """Return the Blackman window."""
    return creation.array(np.blackman(M).astype(np.float32))


def hamming(M):
    """Return the Hamming window."""
    return creation.array(np.hamming(M).astype(np.float32))


def hanning(M):
    """Return the Hanning window."""
    return creation.array(np.hanning(M).astype(np.float32))


def kaiser(M, beta):
    """Return the Kaiser window."""
    return creation.array(np.kaiser(M, beta).astype(np.float32))
