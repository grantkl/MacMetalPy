# Test Coverage Plan: CuPy-Parity Testing Suite

> **Current:** 890 tests, 25 files, 6754 LOC
> **Target:** ~15,000+ parametrized test cases mirroring CuPy's testing infrastructure
> **Public API surface:** 401 functions/methods across 27 modules
>
> **Reference:**
> - CuPy testing module: https://docs.cupy.dev/en/stable/reference/testing.html
> - CuPy tests: `tests/cupy_tests/core_tests/`, `tests/cupy_tests/math_tests/`
> - NumPy tests: `numpy/core/tests/`, `numpy/linalg/tests/`

---

## 1. Testing infrastructure — mirror CuPy's `cupy.testing`

CuPy's power comes from two things:
1. **Dtype group decorators** that parametrize tests across all supported types
2. **Auto-comparison decorators** that run the same code on both NumPy and CuPy

We build the same, adapted for pytest + Metal's type constraints.

### 1.1 Dtype groups — mapped from CuPy

CuPy has 16 types. We have 10 (Metal doesn't support float64, int8, uint8,
complex128). Our groups mirror CuPy's exactly, minus unsupported types.

```python
# tests/conftest.py
import numpy as np
import pytest

# ── CuPy-equivalent dtype groups ──────────────────────────────
# Ref: cupy.testing.for_all_dtypes, for_float_dtypes, etc.

FLOAT_DTYPES = [np.float16, np.float32]
#  CuPy: [float16, float32, float64]
#  We drop float64 (no Metal support)

SIGNED_DTYPES = [np.int16, np.int32, np.int64]
#  CuPy: [int8, int16, int32, int64]
#  We drop int8 (no Metal support)

UNSIGNED_DTYPES = [np.uint16, np.uint32, np.uint64]
#  CuPy: [uint8, uint16, uint32, uint64]
#  We drop uint8 (no Metal support)

INT_DTYPES = SIGNED_DTYPES + UNSIGNED_DTYPES
#  CuPy: signed + unsigned + optional bool

COMPLEX_DTYPES = [np.complex64]
#  CuPy: [complex64, complex128]
#  We drop complex128 (downcast to complex64)

ALL_DTYPES = [np.bool_] + FLOAT_DTYPES + SIGNED_DTYPES + UNSIGNED_DTYPES + COMPLEX_DTYPES
#  10 types total (CuPy: 16)

# Subsets matching CuPy's no_* flags
ALL_DTYPES_NO_BOOL = [d for d in ALL_DTYPES if d != np.bool_]         # 9
ALL_DTYPES_NO_FLOAT16 = [d for d in ALL_DTYPES if d != np.float16]    # 9
ALL_DTYPES_NO_COMPLEX = [d for d in ALL_DTYPES if d != np.complex64]  # 9
NUMERIC_DTYPES = FLOAT_DTYPES + INT_DTYPES                            # 8
```

### 1.2 Shape groups — from CuPy's arithmetic tests

CuPy tests these shapes (from `test_arithmetic.py`, `test_ndarray_elementwise_op.py`):

```python
# ── Shape groups ──────────────────────────────────────────────
SHAPES_SCALAR = [()]                            # 0-d
SHAPES_1D = [(1,), (5,), (1024,)]               # 1-D
SHAPES_2D = [(2, 3), (3, 2), (1, 1)]           # 2-D
SHAPES_3D = [(2, 3, 4)]                         # 3-D
SHAPES_ZERO = [(0,), (3, 0, 2)]                 # zero-size (CuPy tests these)
SHAPES_ALL = SHAPES_SCALAR + SHAPES_1D + SHAPES_2D + SHAPES_3D + SHAPES_ZERO  # 10

# Broadcast pairs (from CuPy's test_ndarray_elementwise_op.py)
BROADCAST_PAIRS = [
    ((2, 3), (3,)),             # trailing dim
    ((2, 3), (2, 1)),           # size-1 dim
    ((2, 1, 3), (3, 1)),       # doubly-broadcast
    ((1,), (2, 3)),            # scalar-like broadcast
]
```

### 1.3 Tolerance groups — from CuPy's `numpy_cupy_allclose`

CuPy uses `rtol=1e-7, atol=0` by default, with per-category overrides.
We need wider tolerance for float16 and match CuPy's category-specific values.

```python
# ── Tolerances (match CuPy's numpy_cupy_allclose patterns) ────
# Ref: CuPy defaults rtol=1e-7, atol=0
# CuPy arithmetic tests use atol=1e-4
# CuPy unary math uses atol=1e-5
# CuPy power uses atol=1.0, rtol=1e-6

def tol_for(dtype, category="default"):
    """Return dict(rtol=, atol=) matching CuPy's tolerance patterns."""
    dtype = np.dtype(dtype)

    if category == "arithmetic":
        # Ref: CuPy ArithmeticBinaryBase.check_binary atol=1e-4
        if dtype == np.float16:
            return dict(rtol=1e-2, atol=1e-2)
        return dict(rtol=1e-6, atol=1e-4)

    if category == "unary_math":
        # Ref: CuPy TestArithmeticUnary.test_unary atol=1e-5
        if dtype == np.float16:
            return dict(rtol=1e-2, atol=1e-2)
        return dict(rtol=1e-5, atol=1e-5)

    if category == "power":
        # Ref: CuPy power tests atol=1.0, rtol=1e-6
        return dict(rtol=1e-6, atol=1.0)

    # Default: match CuPy's numpy_cupy_allclose defaults
    if dtype == np.float16:
        return dict(rtol=1e-2, atol=1e-3)
    if dtype in (np.float32, np.complex64):
        return dict(rtol=1e-5, atol=1e-6)
    # Exact for int/bool
    return dict(rtol=0, atol=0)
```

### 1.4 Comparison helper — equivalent to CuPy's `@numpy_cupy_allclose`

CuPy runs the same test body with NumPy and CuPy, then diffs. We do the same
with a helper function and optionally a decorator.

```python
import macmetalpy as cp
import numpy.testing as npt

def assert_op(op_func, *np_args, dtype=np.float32, category="default",
              accept_error=None):
    """Run op_func with NumPy arrays and macmetalpy arrays, compare results.

    Mirrors CuPy's @numpy_cupy_allclose pattern.

    Parameters
    ----------
    op_func : callable
        Takes (xp, *args) where xp is numpy or macmetalpy module.
    np_args : array-like
        Input data as numpy arrays.
    accept_error : type or tuple, optional
        Exception types that are acceptable from both libs.
    """
    try:
        np_result = op_func(np, *[np.array(a, dtype=dtype) for a in np_args])
    except Exception as e:
        if accept_error and isinstance(e, accept_error):
            # NumPy raised — verify macmetalpy also raises
            with pytest.raises(type(e)):
                op_func(cp, *[cp.array(a, dtype=dtype) for a in np_args])
            return
        raise

    gpu_result = op_func(cp, *[cp.array(a, dtype=dtype) for a in np_args])
    result = gpu_result.get() if hasattr(gpu_result, 'get') else gpu_result
    npt.assert_allclose(result, np_result, **tol_for(dtype, category))


def assert_eq(gpu_arr, np_ref, dtype=None, category="default"):
    """Direct comparison of GPU array against NumPy reference."""
    result = gpu_arr.get() if hasattr(gpu_arr, 'get') else gpu_arr
    d = dtype or (np_ref.dtype if hasattr(np_ref, 'dtype') else np.float32)
    npt.assert_allclose(result, np_ref, **tol_for(d, category))
```

### 1.5 Dtype combination testing — from CuPy's `for_all_dtypes_combination`

CuPy tests binary ops with **all N×N dtype pairs**. With our 10 types,
that's 100 combinations per binary op (CuPy: 256).

```python
import itertools

DTYPE_PAIRS = list(itertools.product(ALL_DTYPES_NO_BOOL, repeat=2))  # 81 pairs
DTYPE_PAIRS_WITH_BOOL = list(itertools.product(ALL_DTYPES, repeat=2))  # 100 pairs
```

### 1.6 conftest.py fixtures

```python
@pytest.fixture(params=ALL_DTYPES, ids=lambda d: np.dtype(d).name)
def dtype(request): return request.param

@pytest.fixture(params=ALL_DTYPES_NO_BOOL, ids=lambda d: np.dtype(d).name)
def dtype_no_bool(request): return request.param

@pytest.fixture(params=FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
def float_dtype(request): return request.param

@pytest.fixture(params=INT_DTYPES, ids=lambda d: np.dtype(d).name)
def int_dtype(request): return request.param

@pytest.fixture(params=SIGNED_DTYPES, ids=lambda d: np.dtype(d).name)
def signed_dtype(request): return request.param

@pytest.fixture(params=UNSIGNED_DTYPES, ids=lambda d: np.dtype(d).name)
def unsigned_dtype(request): return request.param

@pytest.fixture(params=NUMERIC_DTYPES, ids=lambda d: np.dtype(d).name)
def numeric_dtype(request): return request.param

@pytest.fixture(params=SHAPES_ALL, ids=str)
def shape(request): return request.param

@pytest.fixture(params=SHAPES_1D + SHAPES_2D + SHAPES_3D, ids=str)
def nonzero_shape(request): return request.param

@pytest.fixture(params=SHAPES_2D, ids=str)
def shape_2d(request): return request.param

@pytest.fixture(params=BROADCAST_PAIRS, ids=lambda p: f"{p[0]}+{p[1]}")
def broadcast_pair(request): return request.param
```

---

## 2. Module-by-module test matrix

Each table shows: the function, how many test scenarios, which dtype group
is parametrized, which shape group, and the resulting case count.

**Dtype group sizes for reference:**
- `ALL` = 10, `ALL_NO_BOOL` = 9, `FLOAT` = 2, `INT` = 6
- `NUMERIC` = 8, `SIGNED` = 3, `UNSIGNED` = 3, `COMPLEX` = 1
- `PAIRS` = 81 (no-bool × no-bool), `PAIRS_ALL` = 100

**Shape group sizes:**
- `ALL` = 10, `1D` = 3, `2D` = 3, `3D` = 1, `NONZERO` = 7, `BCAST` = 4

---

### 2.1 `test_creation.py` — 25 functions

Ref: `cupy_tests/creation_tests/test_basic.py`

| Function | Scenarios | × dtypes | × shapes | Cases |
|----------|-----------|----------|----------|-------|
| `zeros` | values correct | ALL(10) | ALL(10) | 100 |
| `ones` | values correct | ALL(10) | ALL(10) | 100 |
| `empty` | shape+dtype only | ALL(10) | ALL(10) | 100 |
| `full` | scalar, 0, nan, inf, negative | ALL(10) | NONZERO(7) | 350 |
| `arange` | int step, float step, neg step, large, start=stop | NUMERIC(8) | — | 40 |
| `array` | list, nested, tuple, ndarray, scalar, empty, dtype override | ALL(10) | 3 | 210 |
| `asarray` | passthrough, conversion | ALL(10) | 3 | 60 |
| `zeros_like` | same dtype, override dtype | ALL(10) | NONZERO(7) | 140 |
| `ones_like` | same | ALL(10) | NONZERO(7) | 140 |
| `empty_like` | same | ALL(10) | NONZERO(7) | 140 |
| `full_like` | same dtype, override | ALL(10) | NONZERO(7) | 140 |
| `linspace` | basic, endpoint=False, num=1, num=0 | FLOAT(2) | — | 8 |
| `eye` | square, non-square, k=+1, k=-1, k=0 | NUMERIC(8) | — | 40 |
| `diag` | 1-D→2-D, 2-D→1-D, k offset | NUMERIC(8) | — | 24 |
| `identity` | N=1, N=4 | NUMERIC(8) | — | 16 |
| `tri` | square, non-square, k offset | NUMERIC(8) | — | 24 |
| `triu` / `tril` | k=0, k=+1, k=-1 | NUMERIC(8) | 2D(3) | 144 |
| `logspace` | basic, num, dtype | FLOAT(2) | — | 4 |
| `meshgrid` | 2 args, 3 args, indexing='ij'/'xy' | FLOAT(2) | — | 8 |
| `indices` | 2-D, 3-D, dtype | 4 | — | 8 |
| `fromfunction` | 1-D, 2-D | 2 | — | 4 |
| `diagflat` | basic, k offset | NUMERIC(8) | — | 16 |
| `vander` | basic, N, increasing | FLOAT(2) | — | 6 |
| `asanyarray` | passthrough, list | 2 | — | 4 |

**Subtotal: ~1,866**

---

### 2.2 `test_ndarray.py` — properties + methods + operators

Ref: `cupy_tests/core_tests/test_ndarray.py`, `test_ndarray_elementwise_op.py`

**Properties:**

| Property | × dtypes | × shapes | Cases |
|----------|----------|----------|-------|
| shape, ndim, size, nbytes, itemsize (5 props) | ALL(10) | ALL(10) | 500 |
| dtype (verify type) | ALL(10) | 3 | 30 |
| strides (match NumPy) | ALL(10) | NONZERO(7) | 70 |
| T (transpose result) | ALL(10) | 2D(3) | 30 |

**Data transfer:**

| Method | Scenarios | × dtypes | × shapes | Cases |
|--------|-----------|----------|----------|-------|
| `get()` roundtrip | basic | ALL(10) | ALL(10) | 100 |
| `get()` non-contiguous | transposed, sliced | ALL(10) | 2D(3) | 60 |
| `set()` roundtrip | basic | ALL(10) | ALL(10) | 100 |

**Shape ops:**

| Method | Scenarios | × dtypes | × shapes | Cases |
|--------|-----------|----------|----------|-------|
| `reshape` | compatible, -1 infer, error (2 -1s), error (bad size) | ALL(10) | 3 | 120 |
| `transpose` | default (reverse), explicit axes | ALL(10) | 2 (2D,3D) | 40 |
| `flatten` | contiguous, non-contiguous | ALL(10) | 3 | 60 |
| `ravel` | contiguous→view, non-contiguous→copy | ALL(10) | 3 | 60 |
| `squeeze` | size-1, all, specific axis, error | ALL(10) | 2 | 80 |
| `expand_dims` | positive, negative axis | ALL(10) | 3 | 60 |

**Type casting:**

| Method | Scenarios | × dtypes | Cases |
|--------|-----------|----------|-------|
| `astype` | same (noop), all dtype pairs | PAIRS(81) | 81 |
| `copy` | verify independence | ALL(10) | 3 shapes × 10 = 30 |

**Arithmetic operators (CuPy tests these with `for_all_dtypes_combination`):**

| Operator group | Ops | Scenarios | × dtype pairs | Cases |
|---------------|------|-----------|---------------|-------|
| Binary (+, -, *, /, **) | 5 | array-array, array-scalar, scalar-array | PAIRS(81) × 3 | 1,215 |
| Reverse (radd, rsub, rtruediv, rpow) | 4 | scalar-array | PAIRS(81) | 324 |
| Unary (neg, abs) | 2 | basic | ALL_NO_BOOL(9) × ALL(10) shapes | 180 |
| Matmul (@) | 1 | basic, shape mismatch error | FLOAT(2) × 3 | 6 |
| Broadcast binary | 5 ops | BCAST_PAIRS(4) | NUMERIC(8) | 160 |

**Comparison operators:**

| Op group | Ops | Scenarios | × dtype pairs | Cases |
|----------|------|-----------|---------------|-------|
| <, <=, >, >=, ==, != | 6 | array-array, array-scalar, nan, inf | PAIRS(81) × 4 | 1,944 |

**Boolean operators:**

| Op | Scenarios | × shapes | Cases |
|----|-----------|----------|-------|
| &, \|, ~, \|= | 4 | bool arrays, int arrays | 3 × 4 × 2 = 24 |

**Reduction methods:**

| Method | Scenarios | × dtypes | × shapes | Cases |
|--------|-----------|----------|----------|-------|
| sum, max, min, mean (4) | axis=None, 0, -1, keepdims | NUMERIC(8) | 3 | 384 |
| std, var (2) | axis=None, 0, keepdims | FLOAT(2) | 3 | 36 |
| prod (1) | axis=None, 0, keepdims | NUMERIC(8) | 3 | 72 |
| cumsum, cumprod (2) | axis=None, 0, -1 | NUMERIC(8) | 3 | 144 |
| any, all (2) | axis=None, 0, all-true, all-false | 3 | 3 | 36 |

**Indexing:**

| Op | Scenarios | × dtypes | × shapes | Cases |
|----|-----------|----------|----------|-------|
| `__getitem__` | int, neg, slice, step, ellipsis, newaxis, bool mask, fancy | ALL(10) | 3 | 240 |
| `__setitem__` | int, slice, bool mask, scalar, broadcast | ALL(10) | 3 | 150 |

**Dunder methods:**

| Method | Scenarios | × dtypes | Cases |
|--------|-----------|----------|-------|
| `__float__`, `__int__` | size-1 ok, size-N error | ALL(10) | 40 |
| `__len__` | 1-D, 2-D, 0-D error | 3 | 9 |
| `__repr__`, `__str__` | basic | ALL(10) | 20 |

**Subtotal: ~6,400**

---

### 2.3 `test_math_ops.py` — 44 functions

Ref: `cupy_tests/math_tests/test_arithmetic.py`, `test_trigonometric.py`

CuPy pattern: `@testing.for_all_dtypes(no_complex=True)` + `@testing.numpy_cupy_allclose(atol=1e-5)`

| Group | Functions | Scenarios/func | × dtypes | × shapes | Cases |
|-------|-----------|----------------|----------|----------|-------|
| Unary float (20): sqrt, exp, log, sign, floor, ceil, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, log2, log10, square, negative, abs | basic, 0, neg, nan, inf, boundary | FLOAT(2) | NONZERO(7) | 1,680 |
| Binary (5): power, dot, mod, remainder, around | basic, scalar, edge | NUMERIC(8) | 3 | 360 |
| Array ops (4): where, clip, concatenate, stack/vstack/hstack | basic, scalar, axis, >2 arrays | NUMERIC(8) | 3 | 288 |
| Predicates (5): isnan, isinf, isfinite, isclose, allclose | all special values (nan, inf, -inf, 0, normal) | FLOAT(2) | 3 | 150 |
| Utility (6): nan_to_num, array_equal, count_nonzero, copy, ascontiguousarray, trace, diagonal | basic, edge | NUMERIC(8) | 3 | 144 |
| math_ext (7): sinc, i0, convolve, interp, fix, unwrap, trapezoid | basic, edge, params | FLOAT(2) | 3 | 126 |

**Subtotal: ~2,748**

---

### 2.4 `test_ufunc_ops.py` — 42 functions

Ref: `cupy_tests/math_tests/test_arithmetic.py` (CuPy uses `for_all_dtypes_combination` for binary ufuncs)

| Group | Functions (count) | Scenarios/func | × dtypes | × shapes | Cases |
|-------|-------------------|----------------|----------|----------|-------|
| Binary arith (8): add, subtract, multiply, divide, true_divide, floor_divide, float_power, fmod | array-array, array-scalar, 0-div, nan, inf, overflow | PAIRS(81) × 6 / 8 ops | 3 | 1,458 |
| Unary math (13): exp2, expm1, log1p, cbrt, reciprocal, rint, trunc, absolute, fabs, positive, arcsinh, arccosh, arctanh | basic, 0, neg, nan, inf, boundary | FLOAT(2) | NONZERO(7) | 1,092 |
| Binary math (8): arctan2, hypot, logaddexp, logaddexp2, heaviside, copysign, nextafter | basic, 0, signs, nan | FLOAT(2) | 3 | 336 |
| Angle (4): degrees, radians, deg2rad, rad2deg | 0, 90, 180, 360, neg | FLOAT(2) | 3 | 120 |
| Special (4): signbit, modf, ldexp, frexp | basic, special vals | FLOAT(2) | 3 | 72 |
| Reductive (4): amax, amin, fmax, fmin | basic, nan, axis | NUMERIC(8) | 3 | 288 |
| Ufunc objects (2): maximum, minimum + .accumulate | basic, nan, 2-D, axis | NUMERIC(8) | 3 | 144 |

**Subtotal: ~3,510**

---

### 2.5 `test_reductions.py` — 19 functions

Ref: `cupy_tests/math_tests/test_sumprod.py`, `test_misc.py`

| Function | Scenarios | × dtypes | × shapes | Cases |
|----------|-----------|----------|----------|-------|
| sum, mean (2) | axis=None, 0, 1, -1, -2, keepdims, multi-axis (if supported) | NUMERIC(8) | 3 (1D,2D,3D) | 336 |
| max, min (2) | axis=None, 0, 1, -1, keepdims, all-nan | NUMERIC(8) | 3 | 288 |
| std, var (2) | axis=None, 0, keepdims | FLOAT(2) | 3 | 36 |
| prod (1) | axis=None, 0, keepdims, overflow | NUMERIC(8) | 3 | 96 |
| argmax, argmin (2) | axis=None, 0, ties, all-equal | NUMERIC(8) | 3 | 144 |
| any, all (2) | axis=None, 0, all-T, all-F, mixed, empty | 4 | 3 | 72 |
| cumsum, cumprod (2) | axis=None, 0, -1 | NUMERIC(8) | 3 | 144 |
| diff (1) | n=1, n=2, axis=0, axis=-1 | NUMERIC(8) | 3 | 96 |
| median (1) | 1-D, 2-D, even/odd count, axis | FLOAT(2) | 3 | 18 |
| percentile, quantile (2) | q=0, 25, 50, 75, 100, axis | FLOAT(2) | 3 | 60 |
| ptp (1) | axis=None, 0 | NUMERIC(8) | 3 | 48 |
| average (1) | basic, axis, weights | FLOAT(2) | 3 | 18 |

**Subtotal: ~1,356**

---

### 2.6 `test_manipulation.py` — 33 functions

Ref: `cupy_tests/manipulation_tests/`

| Group | Functions (count) | Scenarios/func | × dtypes | × shapes | Cases |
|-------|-------------------|----------------|----------|----------|-------|
| Shape (5): reshape, transpose, ravel, flatten, squeeze | basic, -1, axes, error | ALL(10) | 3 | 450 |
| Tile/repeat (2) | 1-D reps, N-D reps, axis | NUMERIC(8) | 3 | 144 |
| Flip (4): flip, fliplr, flipud, rot90 | basic, multi-axis, k values | NUMERIC(8) | 2D(3) | 288 |
| Roll (1) | int shift, tuple shift+axis | NUMERIC(8) | 3 | 72 |
| Split (5): split, array_split, hsplit, vsplit, dsplit | equal, unequal, error | NUMERIC(8) | 2 | 240 |
| Join (7): concatenate, stack, vstack, hstack, dstack, column_stack, concat | axis=0, axis=1, >2 arrays, dtype mix | NUMERIC(8) | 2 | 336 |
| Axis (3): moveaxis, swapaxes, rollaxis | various axes | NUMERIC(8) | 1 (3D) | 72 |
| Broadcast (2): broadcast_to, broadcast_arrays | compatible, error | NUMERIC(8) | 3 | 48 |
| Atleast (3): atleast_1d, atleast_2d, atleast_3d | scalar, 1-D, 2-D, 3-D | ALL(10) | 4 inputs | 120 |
| Misc (6): delete, append, resize, trim_zeros, copyto, pad | basic, axis, modes | NUMERIC(8) | 2 | 288 |

**Subtotal: ~2,058**

---

### 2.7 `test_indexing.py` — 23 functions

Ref: `cupy_tests/indexing_tests/`

| Function | Scenarios | × dtypes | × shapes | Cases |
|----------|-----------|----------|----------|-------|
| take (1) | basic, axis, negative idx | NUMERIC(8) | 3 | 72 |
| take_along_axis (1) | basic, axis=0, 1 | NUMERIC(8) | 2 | 48 |
| put, put_along_axis (2) | basic, mask, mode | NUMERIC(8) | 2 | 96 |
| putmask, place (2) | basic, mask | NUMERIC(8) | 2 | 96 |
| choose, compress, select, extract (4) | basic, multi | NUMERIC(8) | 2 | 192 |
| diag_indices, tril_indices, triu_indices + _from (6) | basic, k | 2 | 2 | 72 |
| ravel_multi_index, unravel_index (2) | basic, 2-D, 3-D | 2 | 2 | 24 |
| fill_diagonal (1) | square, non-square | NUMERIC(8) | 2 | 16 |
| nonzero, flatnonzero, argwhere (3) | basic, all-zero, all-nonzero | NUMERIC(8) | 3 | 72 |
| ix_ (1) | 2-D, 3-D | 4 | 1 | 8 |

**Subtotal: ~696**

---

### 2.8 `test_linalg.py` — 25 functions

Ref: `cupy_tests/linalg_tests/`

Linalg is float-only. CuPy tests with `for_float_dtypes`.

| Function | Scenarios | × dtypes | Cases |
|----------|-----------|----------|-------|
| norm (1) | ord=None/1/2/inf/'fro', vec vs mat, axis | FLOAT(2) | 28 |
| inv (1) | basic, 1×1, large, singular error, non-square error | FLOAT(2) | 10 |
| det (1) | basic, identity=1, singular=0, 1×1 | FLOAT(2) | 8 |
| solve (1) | basic, overdetermined, singular error | FLOAT(2) | 6 |
| eigh (1) | basic, identity, diagonal | FLOAT(2) | 6 |
| svd (1) | basic, full_matrices T/F, rank-deficient | FLOAT(2) | 6 |
| cholesky (1) | basic, non-PD error | FLOAT(2) | 4 |
| matrix_power (1) | n=0, 1, 2, -1, large | FLOAT(2) | 10 |
| qr (1) | reduced, complete, tall, wide | FLOAT(2) | 8 |
| eig, eigvals, eigvalsh (3) | basic, diagonal, symmetric | FLOAT(2) | 18 |
| cond, matrix_rank, slogdet (3) | basic, singular, well-conditioned | FLOAT(2) | 18 |
| lstsq, pinv (2) | overdetermined, underdetermined, full-rank | FLOAT(2) | 12 |
| vdot, inner, outer (3) | basic, large | FLOAT(2) | 18 |
| tensordot, einsum, kron (3) | basic, axes/subscripts | FLOAT(2) | 18 |
| matmul, cross (2) | basic, shape variants, error | FLOAT(2) | 12 |

**Subtotal: ~182**

---

### 2.9 `test_nan_ops.py` — 22 functions

Ref: `cupy_tests/math_tests/test_misc.py`, NumPy `test_nanfunctions.py`

| Function | Scenarios | × dtypes | × shapes | Cases |
|----------|-----------|----------|----------|-------|
| nansum, nanprod, nanmax, nanmin, nanmean (5) | basic, all-nan, no-nan, axis | FLOAT(2) | 3 | 120 |
| nanmedian, nanstd, nanvar (3) | basic, all-nan, axis | FLOAT(2) | 3 | 54 |
| nanargmax, nanargmin (2) | basic, all-nan error | FLOAT(2) | 3 | 36 |
| nancumsum, nancumprod (2) | basic, axis | FLOAT(2) | 3 | 36 |
| histogram, histogram2d, histogramdd (3) | basic, bins param, weights | FLOAT(2) | 1 | 18 |
| bincount, digitize (2) | basic, weights, empty | 2 | 1 | 12 |
| corrcoef, cov, correlate (3) | basic, x+y, mode | FLOAT(2) | 1 | 18 |
| ediff1d, gradient (2) | basic, to_begin/end, varargs | FLOAT(2) | 1 | 12 |

**Subtotal: ~306**

---

### 2.10 `test_logic_bitwise.py` — 26 functions

Ref: `cupy_tests/logic_tests/`, `cupy_tests/binary_tests/`

| Group | Functions (count) | Scenarios/func | × dtypes | × shapes | Cases |
|-------|-------------------|----------------|----------|----------|-------|
| Logical (4): and, or, not, xor | basic, mixed truthy | ALL(10) | 3 | 120 |
| Comparison funcs (6): greater, ge, less, le, equal, ne | basic, nan, inf, cross-dtype pairs | PAIRS(81) | 3 | 1,458 |
| Predicate (6): isneginf, isposinf, iscomplex, isreal, isscalar, array_equiv | special values | FLOAT(2) | 3 | 108 |
| Bitwise (4): and, or, xor, invert | basic, edge bits | INT(6) | 3 | 216 |
| Shift (2): left_shift, right_shift | basic, shift amounts | INT(6) | 3 | 108 |
| Pack (2): packbits, unpackbits | basic | 1 | 1 | 6 |
| GCD/LCM (2): gcd, lcm | basic, edge (0, negative) | INT(6) | 1 | 36 |

**Subtotal: ~2,052**

---

### 2.11 `test_sorting.py` — 9 functions

Ref: `cupy_tests/sorting_tests/`

| Function | Scenarios | × dtypes | × shapes | Cases |
|----------|-----------|----------|----------|-------|
| sort (1) | axis=-1, 0, already-sorted, reverse, stable | NUMERIC(8) | 3 | 120 |
| argsort (1) | axis=-1, 0, stable | NUMERIC(8) | 3 | 72 |
| unique (1) | basic, all-same, all-unique, empty | NUMERIC(8) | 1 | 32 |
| searchsorted (1) | left, right, scalar, sorted edge | NUMERIC(8) | 1 | 32 |
| lexsort (1) | 2 keys, 3 keys | NUMERIC(8) | 1 | 16 |
| partition, argpartition (2) | basic, kth variants | NUMERIC(8) | 1 | 32 |
| msort (1) | basic | NUMERIC(8) | 1 | 8 |
| sort_complex (1) | basic | FLOAT(2) | 1 | 2 |

**Subtotal: ~314**

---

### 2.12 `test_set_ops.py` — 6 functions

| Function | Scenarios | × dtypes | Cases |
|----------|-----------|----------|-------|
| union1d, intersect1d, setdiff1d, setxor1d (4) | basic, disjoint, identical, empty, return_indices | NUMERIC(8) | 160 |
| in1d, isin (2) | basic, invert, assume_unique | NUMERIC(8) | 48 |

**Subtotal: ~208**

---

### 2.13 `test_fft.py` — 18 functions

Ref: `cupy_tests/fft_tests/`

| Function | Scenarios | × dtypes | Cases |
|----------|-----------|----------|-------|
| fft/ifft roundtrip (2) | power-of-2, prime, odd, n param, single-element | FLOAT(2) + COMPLEX(1) | 30 |
| fft2/ifft2 (2) | basic, s param | 3 | 12 |
| fftn/ifftn (2) | basic, s/axes | 3 | 12 |
| rfft/irfft (2) | basic, n param | FLOAT(2) | 8 |
| rfft2/irfft2, rfftn/irfftn (4) | basic | FLOAT(2) | 16 |
| hfft/ihfft (2) | basic | FLOAT(2) | 4 |
| fftfreq, rfftfreq (2) | basic, d param, value check | FLOAT(2) | 12 |
| fftshift, ifftshift (2) | 1-D, 2-D, axes | FLOAT(2) | 12 |

**Subtotal: ~106**

---

### 2.14 `test_random.py` — 40 functions

Ref: `cupy_tests/random_tests/`

CuPy tests: shape, dtype, range, seed reproducibility, invalid params.
We add moment tests (mean/var of 100k samples ≈ theoretical ± tolerance).

| Group | Functions | Tests/func | Cases |
|-------|-----------|------------|-------|
| Basic (8): seed, rand, randn, randint, random, shuffle, permutation, choice | shape, dtype, range, repro, error | 40 |
| Continuous (19): normal, uniform, beta, exponential, gamma, etc. | shape, dtype, range, moments, repro, invalid → error | 6 each = 114 |
| Discrete (8): binomial, poisson, geometric, etc. | shape, dtype, range, moments, repro | 6 each = 48 |
| Multivariate (5): multinomial, multivariate_normal, dirichlet, etc. | shape, dtype, range, repro | 6 each = 30 |

**Subtotal: ~232**

---

### 2.15 `test_window.py` — 5 functions

| Function | Scenarios | Cases |
|----------|-----------|-------|
| bartlett, blackman, hamming, hanning (4) | M=0, 1, 5, 64, symmetry, values match NumPy | 20 |
| kaiser (1) | beta variants, M=0, 1, 64 | 6 |

**Subtotal: ~26**

---

## 3. Cross-cutting test files (NEW)

### 3.1 `test_dtype_system.py`

| Area | Cases |
|------|-------|
| `resolve_dtype` for all 10 supported dtypes | 10 |
| `resolve_dtype(float64)` → downcast + warning | 3 |
| `resolve_dtype(complex128)` → complex64 | 1 |
| `resolve_dtype(unsupported)` → TypeError | 3 |
| `result_dtype` all pairs (10×10) vs `np.result_type` | 100 |
| `numpy_to_metal` / `metal_to_numpy` roundtrip | 10 |
| Array create + `.get()` roundtrip per dtype | 10 × ALL(10) shapes = 100 |
| Binary op cross-dtype: 81 no-bool pairs, verify result dtype | 81 |

**Subtotal: ~308**

---

### 3.2 `test_strides_views.py`

| Area | × dtypes | × shapes | Cases |
|------|----------|----------|-------|
| `reshape` returns view (mutate → check base) | NUMERIC(8) | 3 | 24 |
| `transpose` returns view | NUMERIC(8) | 2D(3) | 24 |
| `ravel` contiguous→view, noncontiguous→copy | NUMERIC(8) | 2 | 32 |
| Slice `a[::2]` as kernel input | NUMERIC(8) | 1D(3) | 24 |
| Slice `a[1:]` offset correctness | NUMERIC(8) | 1D(3) | 24 |
| Transposed array as kernel input | NUMERIC(8) | 2D(3) | 24 |
| `_is_c_contiguous` true/false | 1 | 4 | 4 |
| `.strides` matches NumPy (C-contiguous) | ALL(10) | NONZERO(7) | 70 |
| `.strides` after transpose | ALL(10) | 2D(3) | 30 |
| `ascontiguousarray` no-copy on contiguous | NUMERIC(8) | 3 | 24 |

**Subtotal: ~280**

---

### 3.3 `test_error_handling.py`

Mirrors CuPy's `accept_error` pattern — verify both libs raise same errors.

| Area | Cases |
|------|-------|
| Creation: bad shape, bad dtype, step=0 | 10 |
| Reshape: incompatible size, two -1s | 4 |
| Squeeze: non-size-1 axis | 2 |
| Matmul: shape mismatch | 3 |
| Linalg: non-square, singular, non-PD | 10 |
| Broadcast: incompatible shapes | 3 |
| Concat/stack: shape mismatch | 4 |
| Index: out of bounds (positive, negative) | 4 |
| Scalar: `float()`/`int()`/`len()` wrong size | 6 |
| Split: bad sections | 2 |
| Invalid axis | 5 |
| RawKernel: bad grid tuple | 2 |
| Unsupported dtype combos → TypeError (like CuPy's `accept_error`) | 10 |

**Subtotal: ~65**

---

### 3.4 `test_edge_shapes.py`

CuPy explicitly tests `(3, 0, 2)` and `()` shapes in their arithmetic tests.

| Area | × dtypes | Cases |
|------|----------|-------|
| Zero-size 1-D: creation + get | ALL(10) | 10 |
| Zero-size 2-D: `(0,3)`, `(3,0)` | ALL(10) | 20 |
| Zero-size reductions (sum, max, min) | FLOAT(2) | 6 |
| Zero-size concat, stack | FLOAT(2) | 4 |
| Zero-size elementwise ops | NUMERIC(8) | 8 |
| 0-d scalar: creation + properties | ALL(10) | 10 |
| 0-d arithmetic: scalar + scalar | NUMERIC(8) | 8 |
| 0-d reduction | 4 | 4 |
| Single-element reductions | NUMERIC(8) | 24 |
| Shape (1,) vs (1,1) vs () | 4 | 12 |
| Large array 1M: creation + sum + op | FLOAT(2) | 6 |

**Subtotal: ~112**

---

### 3.5 `test_numeric_edges.py`

CuPy tests division by zero, nan propagation, overflow in their arithmetic tests.

| Area | Cases |
|------|-------|
| Float32 specials (nan, inf, -inf, 0, -0) through 20 unary ops | 100 |
| Float32 specials through 8 binary ops (5 combos each) | 200 |
| Float16 near-epsilon arithmetic | 10 |
| Float16 overflow (>65504) → inf | 5 |
| Float16 underflow (<6e-8) → 0 | 5 |
| Int32 overflow (MAX+1, MIN-1) | 5 |
| Int16/uint16 overflow | 5 |
| `isnan/isinf/isfinite` on all specials | 15 |
| `signbit` on -0, +0, -inf, +inf, nan | 5 |
| `copysign`/`nextafter` specials | 10 |
| `nan_to_num` custom values | 5 |
| NaN propagation: nan+1, nan*0, nan==nan, nan!=nan | 8 |
| Inf arithmetic: inf+1, inf-inf, inf*0, 1/0 | 8 |
| `-0.0 == 0.0` → True | 1 |

**Subtotal: ~382**

---

### 3.6 `test_inplace_ops.py`

Ref: `cupy_tests/core_tests/test_ndarray_elementwise_op.py` (CuPy tests `__iadd__` etc.)

| Area | × dtypes | × shapes | Cases |
|------|----------|----------|-------|
| `+=, -=, *=, /=, **=` array-array | NUMERIC(8) | 3 | 120 |
| In-place with scalar RHS | NUMERIC(8) | 3 | 120 |
| In-place with broadcast RHS | NUMERIC(8) | BCAST(4) | 160 |
| In-place preserves dtype (no promotion) | NUMERIC(8) | 1 | 40 |
| `\|=, &=` on bool arrays | 1 | 3 | 6 |
| In-place on view modifies base | NUMERIC(8) | 2 | 16 |

**Subtotal: ~462**

---

### 3.7 `test_interop.py`

| Area | × dtypes | Cases |
|------|----------|-------|
| `len()` on 1-D, 2-D, 0-D error | ALL(10) | 30 |
| `float()` size-1 ok + size-N error | ALL(10) | 20 |
| `int()` size-1 ok + size-N error | ALL(10) | 20 |
| `repr()`/`str()` includes correct values | ALL(10) | 20 |

**Subtotal: ~90**

---

### 3.8 `test_rawkernel.py`

Ref: `cupy_tests/core_tests/test_raw.py`

| Area | Cases |
|------|-------|
| Basic elementwise (float32, int32, float16, uint32) | 4 |
| 2-D grid via MetalSize | 1 |
| 3-D grid via tuple | 1 |
| 1-tuple, 2-tuple, 3-tuple grid | 3 |
| >3-tuple grid → ValueError | 1 |
| Auto-append _sync | 1 |
| Source already has _sync | 1 |
| 8+ buffers | 1 |
| Reuse same RawKernel | 1 |
| Two different RawKernels | 1 |
| Large grid (100k threads) | 1 |

**Subtotal: ~16**

---

### 3.9 `test_synchronize.py`

| Area | Cases |
|------|-------|
| No pending → no-op | 1 |
| After single op | 1 |
| After chained ops | 1 |
| `.get()` implicit sync | 1 |
| `.set()` implicit sync | 1 |
| Multiple arrays, one sync | 1 |
| Back-to-back sync (second no-op) | 1 |
| `_has_pending` transitions | 3 |

**Subtotal: ~10**

---

### 3.10 `test_config_integration.py`

| Area | Cases |
|------|-------|
| Downcast float64 in `array()` | 2 |
| `warn_on_downcast=True` emits warning | 2 |
| `warn_on_downcast=False` silent | 2 |
| `default_float_dtype` affects untyped `array()` | 2 |
| Config change mid-session | 2 |

**Subtotal: ~10**

---

## 4. Grand total

| File | Cases |
|------|-------|
| `test_creation.py` | 1,866 |
| `test_ndarray.py` | 6,400 |
| `test_math_ops.py` | 2,748 |
| `test_ufunc_ops.py` | 3,510 |
| `test_reductions.py` | 1,356 |
| `test_manipulation.py` | 2,058 |
| `test_indexing.py` | 696 |
| `test_linalg.py` | 182 |
| `test_nan_ops.py` | 306 |
| `test_logic_bitwise.py` | 2,052 |
| `test_sorting.py` | 314 |
| `test_set_ops.py` | 208 |
| `test_fft.py` | 106 |
| `test_random.py` | 232 |
| `test_window.py` | 26 |
| `test_dtype_system.py` | 308 |
| `test_strides_views.py` | 280 |
| `test_error_handling.py` | 65 |
| `test_edge_shapes.py` | 112 |
| `test_numeric_edges.py` | 382 |
| `test_inplace_ops.py` | 462 |
| `test_interop.py` | 90 |
| `test_rawkernel.py` | 16 |
| `test_synchronize.py` | 10 |
| `test_config_integration.py` | 10 |
| **Grand total** | **~17,835** |

---

## 5. Execution order

### Wave 1 — Infrastructure (blocks everything)
1. Rewrite `conftest.py` — dtype groups, shape groups, tolerance helpers, `assert_op`/`assert_eq`
2. `test_dtype_system.py` — validate the foundation works

### Wave 2 — Core correctness
3. `test_strides_views.py` — memory model
4. `test_error_handling.py` — error paths
5. `test_edge_shapes.py` — zero-size + 0-d arrays
6. `test_numeric_edges.py` — special values

### Wave 3 — ndarray (biggest payoff)
7. `test_ndarray.py` rewrite — 6,400 cases
8. `test_inplace_ops.py` — in-place operators
9. `test_interop.py` — Python protocols

### Wave 4 — Math (largest function count)
10. `test_math_ops.py` — consolidate elementwise + math_ext
11. `test_ufunc_ops.py` — full ufunc coverage
12. `test_creation.py` — creation functions

### Wave 5 — Collections
13. `test_reductions.py`
14. `test_manipulation.py` — consolidate manip_ext + sort_manip manip parts
15. `test_indexing.py`
16. `test_logic_bitwise.py`

### Wave 6 — Specialized
17. `test_linalg.py` — consolidate both linalg test files
18. `test_nan_ops.py`
19. `test_sorting.py` + `test_set_ops.py`
20. `test_fft.py`
21. `test_random.py` — add moments + reproducibility
22. `test_window.py`

### Wave 7 — New features
23. `test_rawkernel.py`
24. `test_synchronize.py`
25. `test_config_integration.py`

---

## 6. Conventions (matching CuPy patterns)

### Test pattern
```python
# Mirror CuPy's @numpy_cupy_allclose + @for_all_dtypes pattern
@pytest.mark.parametrize("dtype", ALL_DTYPES_NO_COMPLEX)
@pytest.mark.parametrize("shape", [(2, 3), (), (3, 0, 2)])
def test_add_basic(dtype, shape):
    np_a = np.random.randn(*shape).astype(dtype) if shape else np.array(1.0, dtype=dtype)
    np_b = np.random.randn(*shape).astype(dtype) if shape else np.array(2.0, dtype=dtype)
    np_result = np.add(np_a, np_b)
    gpu_result = cp.add(cp.array(np_a), cp.array(np_b))
    assert_eq(gpu_result, np_result, dtype=dtype, category="arithmetic")
```

### Dtype combination pattern (binary ops)
```python
# Mirror CuPy's @for_all_dtypes_combination
@pytest.mark.parametrize("x_type,y_type", DTYPE_PAIRS)
def test_add_dtype_combination(x_type, y_type):
    np_a = np.array([1, 2, 3], dtype=x_type)
    np_b = np.array([4, 5, 6], dtype=y_type)
    np_result = np.add(np_a, np_b)
    gpu_result = cp.add(cp.array(np_a), cp.array(np_b))
    assert_eq(gpu_result, np_result, dtype=np_result.dtype)
```

### Error pattern
```python
# Mirror CuPy's accept_error pattern
def test_reshape_incompatible():
    a = cp.zeros((2, 3))
    with pytest.raises(ValueError, match="cannot reshape"):
        a.reshape((4, 4))
```

### Naming
- Files: `test_<module>.py` (one per source module)
- Functions: `test_<function>_<scenario>` (e.g. `test_sum_axis_negative`)
- Classes: `TestAdd`, `TestSubtract` (group by function, like CuPy)

### Tolerances
- Always use `tol_for(dtype, category)` — never hardcode rtol/atol
- float16: rtol=1e-2, atol=1e-2 (CuPy also relaxes for float16)
- float32: rtol=1e-5, atol=1e-6 (default), 1e-4 (arithmetic), 1e-5 (unary math)
- int/bool: exact (rtol=0, atol=0)

### References
- Every test group links to the CuPy/NumPy test it mirrors:
  ```python
  # Ref: cupy_tests/math_tests/test_arithmetic.py::TestArithmeticBinary
  ```
