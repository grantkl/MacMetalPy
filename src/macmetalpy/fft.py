"""FFT functions (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray
from . import creation


def _ensure(x):
    return x if isinstance(x, ndarray) else creation.asarray(x)


def fft(a, n=None, axis=-1):
    """Compute the one-dimensional discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.fft(a.get(), n=n, axis=axis)
    return creation.array(result)


def ifft(a, n=None, axis=-1):
    """Compute the one-dimensional inverse discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.ifft(a.get(), n=n, axis=axis)
    return creation.array(result)


def fft2(a, s=None, axes=(-2, -1)):
    """Compute the 2-dimensional discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.fft2(a.get(), s=s, axes=axes)
    return creation.array(result)


def ifft2(a, s=None, axes=(-2, -1)):
    """Compute the 2-dimensional inverse discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.ifft2(a.get(), s=s, axes=axes)
    return creation.array(result)


def fftn(a, s=None, axes=None):
    """Compute the N-dimensional discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.fftn(a.get(), s=s, axes=axes)
    return creation.array(result)


def ifftn(a, s=None, axes=None):
    """Compute the N-dimensional inverse discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.ifftn(a.get(), s=s, axes=axes)
    return creation.array(result)


def rfft(a, n=None, axis=-1):
    """Compute the one-dimensional DFT for real input."""
    a = _ensure(a)
    result = np.fft.rfft(a.get(), n=n, axis=axis)
    return creation.array(result)


def irfft(a, n=None, axis=-1):
    """Compute the inverse of rfft."""
    a = _ensure(a)
    result = np.fft.irfft(a.get(), n=n, axis=axis)
    return creation.array(result)


def rfft2(a, s=None, axes=(-2, -1)):
    """Compute the 2-dimensional FFT of a real array."""
    a = _ensure(a)
    result = np.fft.rfft2(a.get(), s=s, axes=axes)
    return creation.array(result)


def irfft2(a, s=None, axes=(-2, -1)):
    """Compute the inverse of rfft2."""
    a = _ensure(a)
    result = np.fft.irfft2(a.get(), s=s, axes=axes)
    return creation.array(result)


def rfftn(a, s=None, axes=None):
    """Compute the N-dimensional FFT of real input."""
    a = _ensure(a)
    result = np.fft.rfftn(a.get(), s=s, axes=axes)
    return creation.array(result)


def irfftn(a, s=None, axes=None):
    """Compute the inverse of rfftn."""
    a = _ensure(a)
    result = np.fft.irfftn(a.get(), s=s, axes=axes)
    return creation.array(result)


def hfft(a, n=None, axis=-1):
    """Compute the FFT of a signal with Hermitian symmetry."""
    a = _ensure(a)
    result = np.fft.hfft(a.get(), n=n, axis=axis)
    return creation.array(result)


def ihfft(a, n=None, axis=-1):
    """Compute the inverse FFT of a signal with Hermitian symmetry."""
    a = _ensure(a)
    result = np.fft.ihfft(a.get(), n=n, axis=axis)
    return creation.array(result)


def fftfreq(n, d=1.0):
    """Return the DFT sample frequencies."""
    result = np.fft.fftfreq(n, d=d)
    return creation.array(result)


def rfftfreq(n, d=1.0):
    """Return the DFT sample frequencies for rfft."""
    result = np.fft.rfftfreq(n, d=d)
    return creation.array(result)


def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum."""
    x = _ensure(x)
    result = np.fft.fftshift(x.get(), axes=axes)
    return creation.array(result)


def ifftshift(x, axes=None):
    """Inverse of fftshift."""
    x = _ensure(x)
    result = np.fft.ifftshift(x.get(), axes=axes)
    return creation.array(result)
