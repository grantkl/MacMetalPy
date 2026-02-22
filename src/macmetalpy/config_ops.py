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
