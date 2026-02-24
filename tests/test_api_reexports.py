"""Tests for top-level re-exports that must match numpy equivalents."""
import numpy as np
import macmetalpy as mp
import pytest

# Type re-exports
@pytest.mark.parametrize("name", [
    "bytes_", "character", "datetime64", "flexible",
    "object_", "str_", "timedelta64", "void",
])
def test_type_reexports(name):
    assert hasattr(mp, name), f"mp.{name} missing"
    assert getattr(mp, name) is getattr(np, name)

# Legacy class re-exports
@pytest.mark.parametrize("name", [
    "matrix", "memmap", "recarray", "record",
    "sctypeDict", "asmatrix", "bmat",
])
def test_legacy_reexports(name):
    assert hasattr(mp, name), f"mp.{name} missing"
    assert getattr(mp, name) is getattr(np, name)

# Datetime functions
@pytest.mark.parametrize("name", [
    "busday_count", "busday_offset", "busdaycalendar",
    "is_busday", "datetime_as_string", "datetime_data",
])
def test_datetime_reexports(name):
    assert hasattr(mp, name), f"mp.{name} missing"
    assert getattr(mp, name) is getattr(np, name)

# Module re-exports
@pytest.mark.parametrize("name", [
    "char", "dtypes", "emath", "exceptions", "polynomial", "rec", "strings",
])
def test_module_reexports(name):
    assert hasattr(mp, name), f"mp.{name} missing"
