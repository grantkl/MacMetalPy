"""Tests for random module completion items."""
import macmetalpy as mp
import numpy as np
import pytest

def test_bit_generator_module():
    assert hasattr(mp.random, 'bit_generator')

def test_mtrand_module():
    assert hasattr(mp.random, 'mtrand')

def test_get_bit_generator():
    assert hasattr(mp.random, 'get_bit_generator')
    assert callable(mp.random.get_bit_generator)

def test_set_bit_generator():
    assert hasattr(mp.random, 'set_bit_generator')
    assert callable(mp.random.set_bit_generator)

def test_random_integers():
    assert hasattr(mp.random, 'random_integers')
    result = mp.random.random_integers(1, 10, size=5)
    assert len(result) == 5
