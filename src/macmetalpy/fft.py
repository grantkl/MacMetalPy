"""FFT functions (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray as _ndarray
from . import creation as _creation


def _ensure(x):
    return x if isinstance(x, _ndarray) else _creation.asarray(x)


def _get_np(a):
    """Get zero-copy numpy view (syncs if GPU-resident)."""
    if a._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
    return a._get_view()


def _get_np(a):
    """Get numpy data from ndarray, preferring _np_data."""
    np_data = a._np_data
    if np_data is not None:
        return np_data
    return _get_np(a)


def fft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.fft(_get_np(a), n=n, axis=axis, norm=norm)
    return _ndarray._from_np_direct(result)


def ifft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional inverse discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.ifft(_get_np(a), n=n, axis=axis, norm=norm)
    return _ndarray._from_np_direct(result)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-dimensional discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.fft2(_get_np(a), s=s, axes=axes, norm=norm)
    return _ndarray._from_np_direct(result)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-dimensional inverse discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.ifft2(_get_np(a), s=s, axes=axes, norm=norm)
    return _ndarray._from_np_direct(result)


def fftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.fftn(_get_np(a), s=s, axes=axes, norm=norm)
    return _ndarray._from_np_direct(result)


def ifftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional inverse discrete Fourier Transform."""
    a = _ensure(a)
    result = np.fft.ifftn(_get_np(a), s=s, axes=axes, norm=norm)
    return _ndarray._from_np_direct(result)


def rfft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional DFT for real input."""
    a = _ensure(a)
    result = np.fft.rfft(_get_np(a), n=n, axis=axis, norm=norm)
    return _ndarray._from_np_direct(result)


def irfft(a, n=None, axis=-1, norm=None):
    """Compute the inverse of rfft."""
    a = _ensure(a)
    result = np.fft.irfft(_get_np(a), n=n, axis=axis, norm=norm)
    return _ndarray._from_np_direct(result)


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-dimensional FFT of a real array."""
    a = _ensure(a)
    result = np.fft.rfft2(_get_np(a), s=s, axes=axes, norm=norm)
    return _ndarray._from_np_direct(result)


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the inverse of rfft2."""
    a = _ensure(a)
    result = np.fft.irfft2(_get_np(a), s=s, axes=axes, norm=norm)
    return _ndarray._from_np_direct(result)


def rfftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional FFT of real input."""
    a = _ensure(a)
    result = np.fft.rfftn(_get_np(a), s=s, axes=axes, norm=norm)
    return _ndarray._from_np_direct(result)


def irfftn(a, s=None, axes=None, norm=None):
    """Compute the inverse of rfftn."""
    a = _ensure(a)
    result = np.fft.irfftn(_get_np(a), s=s, axes=axes, norm=norm)
    return _ndarray._from_np_direct(result)


def hfft(a, n=None, axis=-1, norm=None):
    """Compute the FFT of a signal with Hermitian symmetry."""
    a = _ensure(a)
    result = np.fft.hfft(_get_np(a), n=n, axis=axis, norm=norm)
    return _ndarray._from_np_direct(result)


def ihfft(a, n=None, axis=-1, norm=None):
    """Compute the inverse FFT of a signal with Hermitian symmetry."""
    a = _ensure(a)
    result = np.fft.ihfft(_get_np(a), n=n, axis=axis, norm=norm)
    return _ndarray._from_np_direct(result)


def fftfreq(n, d=1.0):
    """Return the DFT sample frequencies."""
    return _ndarray._from_np_direct(np.fft.fftfreq(n, d=d).astype(np.float32, copy=False))


def rfftfreq(n, d=1.0):
    """Return the DFT sample frequencies for rfft."""
    return _ndarray._from_np_direct(np.fft.rfftfreq(n, d=d).astype(np.float32, copy=False))


def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum."""
    x = _ensure(x)
    return _ndarray._from_np_direct(np.fft.fftshift(_get_np(x), axes=axes))


def ifftshift(x, axes=None):
    """Inverse of fftshift."""
    x = _ensure(x)
    return _ndarray._from_np_direct(np.fft.ifftshift(_get_np(x), axes=axes))
