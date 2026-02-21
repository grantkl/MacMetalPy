"""Tests for any, all, argmax, argmin operations."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import macmetalpy as cp


class TestAny:
    def test_any_true(self):
        a = cp.array([False, True, False], dtype=cp.bool_)
        assert a.any() == True

    def test_any_false(self):
        a = cp.array([False, False, False], dtype=cp.bool_)
        assert a.any() == False

    def test_any_all_true(self):
        a = cp.array([True, True, True], dtype=cp.bool_)
        assert a.any() == True

    def test_any_module_level(self):
        a = cp.array([False, True, False], dtype=cp.bool_)
        assert cp.any(a) == True

    def test_any_numeric(self):
        """Non-zero values are truthy."""
        a = cp.array([0.0, 0.0, 1.0], dtype=cp.float32)
        assert a.any() == True

    def test_any_numeric_false(self):
        a = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)
        assert a.any() == False


class TestAll:
    def test_all_true(self):
        a = cp.array([True, True, True], dtype=cp.bool_)
        assert a.all() == True

    def test_all_false(self):
        a = cp.array([True, False, True], dtype=cp.bool_)
        assert a.all() == False

    def test_all_module_level(self):
        a = cp.array([True, True, True], dtype=cp.bool_)
        assert cp.all(a) == True


class TestArgmax:
    def test_argmax_basic(self):
        a = cp.array([1.0, 3.0, 2.0], dtype=cp.float32)
        assert cp.argmax(a) == 1

    def test_argmax_first_occurrence(self):
        """argmax returns the FIRST index of the max value."""
        a = cp.array([1.0, 3.0, 3.0, 2.0], dtype=cp.float32)
        assert cp.argmax(a) == 1

    def test_argmax_bool_first_true(self):
        """Trading bot pattern: argmax on bool array finds first True."""
        a = cp.array([False, False, True, True, False], dtype=cp.bool_)
        assert cp.argmax(a) == 2

    def test_argmax_int(self):
        a = cp.array([10, 5, 20, 3], dtype=cp.int32)
        assert cp.argmax(a) == 2


class TestArgmin:
    def test_argmin_basic(self):
        a = cp.array([3.0, 1.0, 2.0], dtype=cp.float32)
        assert cp.argmin(a) == 1

    def test_argmin_first_occurrence(self):
        a = cp.array([3.0, 1.0, 1.0, 2.0], dtype=cp.float32)
        assert cp.argmin(a) == 1

    def test_argmin_int(self):
        a = cp.array([10, 5, 20, 3], dtype=cp.int32)
        assert cp.argmin(a) == 3
