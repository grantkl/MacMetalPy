"""Tests for macmetalpy._config — singleton config, get/set, validation."""

import pytest

from macmetalpy._config import MacMetalConfig, get_config, set_config


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset the config singleton to defaults before each test."""
    cfg = get_config()
    cfg.float64_behavior = "downcast"
    cfg.warn_on_downcast = True
    cfg.default_float_dtype = "float32"
    yield


class TestMacMetalConfigSingleton:
    def test_singleton_identity(self):
        a = MacMetalConfig()
        b = MacMetalConfig()
        assert a is b

    def test_get_config_returns_singleton(self):
        cfg = get_config()
        assert cfg is MacMetalConfig()


class TestGetConfig:
    def test_default_float64_behavior(self):
        cfg = get_config()
        assert cfg.float64_behavior == "downcast"

    def test_default_warn_on_downcast(self):
        cfg = get_config()
        assert cfg.warn_on_downcast is True

    def test_default_float_dtype(self):
        cfg = get_config()
        assert cfg.default_float_dtype == "float32"


class TestSetConfig:
    def test_set_float64_behavior_downcast(self):
        set_config(float64_behavior="downcast")
        assert get_config().float64_behavior == "downcast"

    def test_set_float64_behavior_cpu_fallback(self):
        set_config(float64_behavior="cpu_fallback")
        assert get_config().float64_behavior == "cpu_fallback"

    def test_set_warn_on_downcast_false(self):
        set_config(warn_on_downcast=False)
        assert get_config().warn_on_downcast is False

    def test_set_warn_on_downcast_true(self):
        set_config(warn_on_downcast=False)
        set_config(warn_on_downcast=True)
        assert get_config().warn_on_downcast is True

    def test_set_default_float_dtype(self):
        set_config(default_float_dtype="float16")
        assert get_config().default_float_dtype == "float16"

    def test_set_multiple_values(self):
        set_config(float64_behavior="cpu_fallback", warn_on_downcast=False)
        cfg = get_config()
        assert cfg.float64_behavior == "cpu_fallback"
        assert cfg.warn_on_downcast is False


class TestSetConfigValidation:
    def test_invalid_float64_behavior_raises(self):
        with pytest.raises(ValueError, match="float64_behavior"):
            set_config(float64_behavior="invalid")

    def test_invalid_float64_behavior_none_string_raises(self):
        with pytest.raises(ValueError, match="float64_behavior"):
            set_config(float64_behavior="error")

    def test_none_values_do_not_change_config(self):
        set_config(float64_behavior="cpu_fallback")
        set_config()  # no arguments
        assert get_config().float64_behavior == "cpu_fallback"
