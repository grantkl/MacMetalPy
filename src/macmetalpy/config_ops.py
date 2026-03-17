"""Print/buffer/error configuration and info utilities.

Thin wrappers around NumPy configuration functions.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Print options
# ---------------------------------------------------------------------------

def get_printoptions():
    """Return the current print options."""
    return np.get_printoptions()


def set_printoptions(**kwargs):
    """Set printing options."""
    np.set_printoptions(**kwargs)


def printoptions(**kwargs):
    """Context manager for setting print options."""
    return np.printoptions(**kwargs)


# ---------------------------------------------------------------------------
# Buffer size
# ---------------------------------------------------------------------------

def getbufsize():
    """Return the size of the buffer used in ufuncs."""
    return np.getbufsize()


def setbufsize(size):
    """Set the size of the buffer used in ufuncs."""
    return np.setbufsize(size)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def geterr():
    """Get the current way of handling floating-point errors."""
    return np.geterr()


def seterr(**kwargs):
    """Set how floating-point errors are handled."""
    return np.seterr(**kwargs)


def geterrcall():
    """Return the current callback function used on floating-point errors."""
    return np.geterrcall()


def seterrcall(func):
    """Set the floating-point error callback function or log object."""
    return np.seterrcall(func)


def geterrobj():
    """Return the current object that defines floating-point error handling.

    Returns a list of [buffer_size, error_mask, callback] matching the legacy
    NumPy geterrobj() contract.  Removed in NumPy 2.0; we reconstruct it from
    the still-available geterr/getbufsize/geterrcall helpers.
    """
    err = np.geterr()
    _err_map = {'ignore': 0, 'warn': 1, 'raise': 2, 'call': 3, 'print': 4, 'log': 5}
    mask = (_err_map.get(err.get('divide', 'warn'), 1)
            | (_err_map.get(err.get('over', 'warn'), 1) << 3)
            | (_err_map.get(err.get('under', 'ignore'), 0) << 6)
            | (_err_map.get(err.get('invalid', 'warn'), 1) << 9))
    return [np.getbufsize(), mask, np.geterrcall()]


def seterrobj(errobj):
    """Set the object that defines floating-point error handling.

    Accepts a list of [buffer_size, error_mask, callback] matching the legacy
    NumPy seterrobj() contract.  Removed in NumPy 2.0; we translate to the
    still-available seterr/setbufsize/seterrcall helpers.
    """
    bufsize, mask, callback = errobj
    np.setbufsize(bufsize)
    _modes = {0: 'ignore', 1: 'warn', 2: 'raise', 3: 'call', 4: 'print', 5: 'log'}
    np.seterr(
        divide=_modes.get(mask & 7, 'warn'),
        over=_modes.get((mask >> 3) & 7, 'warn'),
        under=_modes.get((mask >> 6) & 7, 'ignore'),
        invalid=_modes.get((mask >> 9) & 7, 'warn'),
    )
    np.seterrcall(callback)


def set_numeric_ops(**ops):
    """Define element-wise operations.

    Deprecated and removed in NumPy 2.0.  Provided as a no-op stub for API
    compatibility; returns an empty dict.
    """
    return {}


# ---------------------------------------------------------------------------
# Misc config / info
# ---------------------------------------------------------------------------

def get_include():
    """Return the directory that contains the NumPy C header files."""
    return np.get_include()


def show_config():
    """Print macmetalpy and NumPy build configuration information."""
    from . import __version__
    print(f"macmetalpy version: {__version__}")
    print(f"NumPy version: {np.__version__}")
    print("Backend: Apple Metal (GPU)")


def show_runtime():
    """Print runtime information for macmetalpy."""
    import sys
    import platform
    from . import __version__
    print(f"macmetalpy version: {__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print("Backend: Apple Metal (GPU)")
