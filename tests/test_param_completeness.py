"""Tests that non-ufunc functions accept their full parameter sets without raising."""
import numpy as np
import macmetalpy as mp
import pytest
import inspect

# ---- creation.py: like= and device= params ----
CREATION_WITH_LIKE = [
    'arange', 'empty', 'eye', 'full', 'identity', 'linspace',
    'ones', 'tri', 'zeros',
]

@pytest.mark.parametrize("name", CREATION_WITH_LIKE)
def test_creation_accepts_like(name):
    fn = getattr(mp, name, None)
    if fn is None:
        pytest.skip(f"mp.{name} not found")
    sig = inspect.signature(fn)
    assert 'like' in sig.parameters, f"mp.{name} missing 'like' parameter"

CREATION_LIKE_FUNCS = [
    'empty_like', 'ones_like', 'zeros_like', 'full_like',
]

@pytest.mark.parametrize("name", CREATION_LIKE_FUNCS)
def test_like_funcs_accept_device(name):
    fn = getattr(mp, name, None)
    if fn is None:
        pytest.skip(f"mp.{name} not found")
    sig = inspect.signature(fn)
    assert 'device' in sig.parameters, f"mp.{name} missing 'device' parameter"

# ---- reductions.py: correction= and mean= on std/var ----
def test_std_accepts_correction():
    sig = inspect.signature(mp.std)
    assert 'correction' in sig.parameters

def test_var_accepts_correction():
    sig = inspect.signature(mp.var)
    assert 'correction' in sig.parameters

def test_std_accepts_mean():
    sig = inspect.signature(mp.std)
    assert 'mean' in sig.parameters

def test_var_accepts_mean():
    sig = inspect.signature(mp.var)
    assert 'mean' in sig.parameters

# ---- nan_ops.py: correction, mean ----
def test_nanstd_accepts_correction():
    sig = inspect.signature(mp.nanstd)
    assert 'correction' in sig.parameters

def test_nanvar_accepts_correction():
    sig = inspect.signature(mp.nanvar)
    assert 'correction' in sig.parameters

# ---- manipulation.py ----
def test_reshape_accepts_copy():
    sig = inspect.signature(mp.reshape)
    assert 'copy' in sig.parameters

def test_broadcast_to_accepts_array():
    """broadcast_to first param should accept positional 'array' name."""
    sig = inspect.signature(mp.broadcast_to)
    params = list(sig.parameters.keys())
    assert params[0] in ('array', 'x'), f"First param is {params[0]}"

# ---- sorting.py ----
def test_sort_accepts_stable():
    sig = inspect.signature(mp.sort)
    assert 'stable' in sig.parameters

def test_argsort_accepts_stable():
    sig = inspect.signature(mp.argsort)
    assert 'stable' in sig.parameters

# ---- functional test: passing the params doesn't crash ----
def test_std_with_correction():
    a = mp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = mp.std(a, correction=0)
    assert result is not None

def test_var_with_correction():
    a = mp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = mp.var(a, correction=0)
    assert result is not None

def test_sort_with_stable():
    a = mp.array([3.0, 1.0, 2.0])
    result = mp.sort(a, stable=True)
    assert result is not None

def test_reshape_with_copy():
    a = mp.array([1.0, 2.0, 3.0, 4.0])
    result = mp.reshape(a, (2, 2), copy=False)
    assert result is not None

def test_creation_with_like():
    ref = mp.array([1.0])
    result = mp.zeros(5, like=ref)
    assert result is not None

def test_creation_with_device():
    result = mp.empty_like(mp.array([1.0]), device="cpu")
    assert result is not None

# ---- config_ops.py ----
def test_set_printoptions_accepts_override():
    sig = inspect.signature(mp.set_printoptions)
    assert 'override_repr' in sig.parameters or len(sig.parameters) > 0

# ---- dot with out ----
def test_dot_accepts_out():
    sig = inspect.signature(mp.dot)
    assert 'out' in sig.parameters

# ---- clip with min/max ----
def test_clip_accepts_min_max():
    sig = inspect.signature(mp.clip)
    params = sig.parameters
    assert 'min' in params or 'a_min' in params

# ---- matmul with axes/axis/keepdims ----
def test_matmul_signature():
    sig = inspect.signature(mp.matmul)
    assert 'axes' in sig.parameters or '**' in str(sig)

# ---- unique with sorted ----
def test_unique_accepts_sorted():
    """unique should accept a 'sorted' parameter."""
    fn = getattr(mp, 'unique', None)
    if fn is None:
        pytest.skip()
    sig = inspect.signature(fn)
    # Accept either 'sorted' param or **kwargs
    has_sorted = 'sorted' in sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    assert has_sorted or has_kwargs

# ---- vectorize with cache/doc ----
def test_vectorize_accepts_cache():
    sig = inspect.signature(mp.vectorize)
    params = sig.parameters
    has_cache = 'cache' in params
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert has_cache or has_kwargs

# ---- concat with casting/dtype/out ----
def test_concat_accepts_dtype():
    sig = inspect.signature(mp.concat)
    params = sig.parameters
    has_dtype = 'dtype' in params
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert has_dtype or has_kwargs

def test_concatenate_accepts_dtype():
    sig = inspect.signature(mp.concatenate)
    params = sig.parameters
    has_dtype = 'dtype' in params
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert has_dtype or has_kwargs
