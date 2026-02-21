"""Tests for config integration with array operations.

Consolidates test_config.py behavioral integration tests.
Target: ~10 cases.
"""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

import macmetalpy as cp
from macmetalpy._config import get_config, set_config


# ======================================================================
# Float64 downcast behavior
# ======================================================================

class TestFloat64Downcast:
    def test_array_downcasts_float64(self):
        """array() with float64 data should downcast to float32."""
        cfg = get_config()
        cfg.float64_behavior = "downcast"
        a = cp.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert a.dtype == np.float32
        npt.assert_allclose(a.get(), [1.0, 2.0, 3.0], rtol=1e-5)

    def test_array_float64_without_explicit_dtype(self):
        """Python floats default to default_float_dtype, not float64."""
        cfg = get_config()
        cfg.default_float_dtype = "float32"
        a = cp.array([1.0, 2.0, 3.0])
        assert a.dtype == np.float32


# ======================================================================
# warn_on_downcast
# ======================================================================

class TestWarnOnDowncast:
    def test_warn_on_downcast_true_emits_warning(self):
        """When warn_on_downcast=True and float64 input, a warning is emitted."""
        cfg = get_config()
        cfg.float64_behavior = "downcast"
        cfg.warn_on_downcast = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = cp.array(np.array([1.0], dtype=np.float64))
            downcast_warnings = [x for x in w if "downcast" in str(x.message).lower()
                                 or "float64" in str(x.message).lower()]
            assert len(downcast_warnings) >= 1

    def test_warn_on_downcast_false_silent(self):
        """When warn_on_downcast=False and float64 input, no warning."""
        cfg = get_config()
        cfg.float64_behavior = "downcast"
        cfg.warn_on_downcast = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = cp.array(np.array([1.0], dtype=np.float64))
            downcast_warnings = [x for x in w if "downcast" in str(x.message).lower()
                                 or "float64" in str(x.message).lower()]
            assert len(downcast_warnings) == 0

    def test_no_warning_for_float32(self):
        """float32 input should never produce a downcast warning."""
        cfg = get_config()
        cfg.warn_on_downcast = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = cp.array([1.0, 2.0], dtype=np.float32)
            downcast_warnings = [x for x in w if "downcast" in str(x.message).lower()
                                 or "float64" in str(x.message).lower()]
            assert len(downcast_warnings) == 0


# ======================================================================
# default_float_dtype
# ======================================================================

class TestDefaultFloatDtype:
    def test_default_float32(self):
        """When default_float_dtype='float32', untyped array() uses float32."""
        cfg = get_config()
        cfg.default_float_dtype = "float32"
        a = cp.array([1.0, 2.0, 3.0])
        assert a.dtype == np.float32

    def test_default_float16(self):
        """When default_float_dtype='float16', untyped array() uses float16."""
        cfg = get_config()
        cfg.default_float_dtype = "float16"
        a = cp.array([1.0, 2.0, 3.0])
        assert a.dtype == np.float16


# ======================================================================
# Config change mid-session
# ======================================================================

class TestConfigMidSession:
    def test_change_default_dtype_mid_session(self):
        """Changing config mid-session affects subsequent operations."""
        cfg = get_config()
        cfg.default_float_dtype = "float32"
        a = cp.array([1.0, 2.0])
        assert a.dtype == np.float32

        cfg.default_float_dtype = "float16"
        b = cp.array([1.0, 2.0])
        assert b.dtype == np.float16

    def test_change_warn_setting_mid_session(self):
        """Changing warn_on_downcast mid-session takes effect."""
        cfg = get_config()
        cfg.float64_behavior = "downcast"

        cfg.warn_on_downcast = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.array(np.array([1.0], dtype=np.float64))
            w1 = [x for x in w if "downcast" in str(x.message).lower()
                   or "float64" in str(x.message).lower()]

        cfg.warn_on_downcast = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cp.array(np.array([1.0], dtype=np.float64))
            w2 = [x for x in w if "downcast" in str(x.message).lower()
                   or "float64" in str(x.message).lower()]

        assert len(w1) == 0
        assert len(w2) >= 1
