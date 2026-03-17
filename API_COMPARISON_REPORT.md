# NumPy vs MacMetalPy API Comparison Report

NumPy version analyzed: 1.26.4

---

## 1. Top-Level Function/Class/Constant Coverage

- **NumPy top-level items**: 553
- **MacMetalPy top-level items**: 426
- **In common**: 396
- **Missing from MacMetalPy**: 157
- **Extra in MacMetalPy (not in NumPy)**: 30

### 1.1 Missing: RELEVANT (a GPU library should have these)

**6 items:**

| Name | Type | Notes |
|------|------|-------|
| `fastCopyAndTranspose` | function |  |
| `from_dlpack` | function |  |
| `frompyfunc` | function |  |
| `geterrobj` | function |  |
| `set_numeric_ops` | function |  |
| `seterrobj` | function |  |

### 1.2 Missing: NOT APPLICABLE (strings, datetime, I/O, matrix, etc.)

**115 items** (collapsed for brevity):

<details>
<summary>Click to expand</summary>

| Name | Type |
|------|------|
| `CLIP` | constant |
| `DataSource` | class |
| `ERR_CALL` | constant |
| `ERR_DEFAULT` | constant |
| `ERR_IGNORE` | constant |
| `ERR_LOG` | constant |
| `ERR_PRINT` | constant |
| `ERR_RAISE` | constant |
| `ERR_WARN` | constant |
| `FLOATING_POINT_SUPPORT` | constant |
| `FPE_DIVIDEBYZERO` | constant |
| `FPE_INVALID` | constant |
| `FPE_OVERFLOW` | constant |
| `FPE_UNDERFLOW` | constant |
| `MAXDIMS` | constant |
| `MAY_SHARE_BOUNDS` | constant |
| `MAY_SHARE_EXACT` | constant |
| `RAISE` | constant |
| `RankWarning` | class |
| `SHIFT_DIVIDEBYZERO` | constant |
| `SHIFT_INVALID` | constant |
| `SHIFT_OVERFLOW` | constant |
| `SHIFT_UNDERFLOW` | constant |
| `UFUNC_BUFSIZE_DEFAULT` | constant |
| `UFUNC_PYVALS_NAME` | constant |
| `WRAP` | constant |
| `add_docstring` | function |
| `add_newdoc` | function |
| `add_newdoc_ufunc` | function |
| `busday_count` | function |
| `busday_offset` | function |
| `busdaycalendar` | class |
| `byte_bounds` | function |
| `bytes_` | class |
| `char` | module |
| `character` | class |
| `chararray` | class |
| `compat` | module |
| `ctypeslib` | module |
| `datetime64` | class |
| `datetime_as_string` | function |
| `datetime_data` | function |
| `deprecate` | function |
| `deprecate_with_doc` | function |
| `disp` | function |
| `dtypes` | module |
| `emath` | module |
| `errstate` | class |
| `exceptions` | module |
| `flexible` | class |
| `format_parser` | class |
| `fromfile` | function |
| `fromregex` | function |
| `genfromtxt` | function |
| `get_array_wrap` | function |
| `get_include` | function |
| `get_printoptions` | function |
| `getbufsize` | function |
| `geterr` | function |
| `geterrcall` | function |
| `index_exp` | constant |
| `info` | function |
| `is_busday` | function |
| `iterable` | function |
| `lib` | module |
| `little_endian` | constant |
| `loadtxt` | function |
| `lookfor` | function |
| `ma` | module |
| `may_share_memory` | function |
| `memmap` | class |
| `ndenumerate` | class |
| `ndindex` | class |
| `nested_iters` | function |
| `numarray` | constant |
| `oldnumeric` | constant |
| `poly` | function |
| `poly1d` | class |
| `polyadd` | function |
| `polyder` | function |
| `polydiv` | function |
| `polyfit` | function |
| `polyint` | function |
| `polymul` | function |
| `polynomial` | module |
| `polysub` | function |
| `polyval` | function |
| `printoptions` | function |
| `rec` | module |
| `recarray` | class |
| `recfromcsv` | function |
| `recfromtxt` | function |
| `record` | class |
| `require` | function |
| `roots` | function |
| `safe_eval` | function |
| `savetxt` | function |
| `set_printoptions` | function |
| `set_string_function` | function |
| `setbufsize` | function |
| `seterr` | function |
| `seterrcall` | function |
| `shares_memory` | function |
| `show_config` | function |
| `show_runtime` | function |
| `source` | function |
| `str_` | class |
| `test` | function |
| `testing` | module |
| `timedelta64` | class |
| `tracemalloc_domain` | constant |
| `typecodes` | constant |
| `version` | module |
| `void` | class |
| `who` | function |

</details>

### 1.3 Missing: LEGACY / DEPRECATED

**36 items** (collapsed for brevity):

<details>
<summary>Click to expand</summary>

| Name | Type |
|------|------|
| `ALLOW_THREADS` | constant |
| `BUFSIZE` | constant |
| `False_` | constant |
| `Inf` | constant |
| `Infinity` | constant |
| `NAN` | constant |
| `NINF` | constant |
| `NZERO` | constant |
| `NaN` | constant |
| `PINF` | constant |
| `PZERO` | constant |
| `ScalarType` | constant |
| `True_` | constant |
| `alltrue` | function |
| `asmatrix` | function |
| `bmat` | function |
| `cast` | constant |
| `compare_chararrays` | function |
| `cumproduct` | function |
| `find_common_type` | function |
| `infty` | constant |
| `issubclass_` | function |
| `issubsctype` | function |
| `mat` | function |
| `matrix` | class |
| `maximum_sctype` | function |
| `nbytes` | constant |
| `object_` | class |
| `product` | function |
| `row_stack` | function |
| `sctypeDict` | constant |
| `sctypes` | constant |
| `sometrue` | function |
| `string_` | class |
| `trapz` | function |
| `unicode_` | class |

</details>

### 1.4 Extra in MacMetalPy (not in NumPy)

**30 items:**

| Name | Type |
|------|------|
| `RawKernel` | class |
| `bitwise_invert` | function |
| `bitwise_left_shift` | function |
| `bitwise_ops` | module |
| `bitwise_right_shift` | function |
| `complex_ops` | module |
| `concat` | function |
| `creation` | module |
| `dtype_utils` | module |
| `format_ops` | module |
| `functional` | module |
| `get_config` | function |
| `index_tricks` | module |
| `indexing` | module |
| `io` | module |
| `linalg_top` | module |
| `logic_ops` | module |
| `manipulation` | module |
| `math_ext` | module |
| `math_ops` | module |
| `nan_ops` | module |
| `raw_kernel` | module |
| `reductions` | module |
| `set_config` | function |
| `set_ops` | module |
| `sorting` | module |
| `synchronize` | function |
| `trapezoid` | function |
| `ufunc_ops` | module |
| `window` | module |

---

## 2. Submodule Coverage

### 2.1 numpy.linalg vs macmetalpy.linalg

- **NumPy items**: 23
- **MacMetalPy items**: 25
- **In common**: 21
- **Missing from MacMetalPy**: 2

| Function | Type |
|----------|------|
| `linalg` | module |
| `test` | function |

#### Extra in MacMetalPy (not in NumPy):

| Function | Type |
|----------|------|
| `annotations` | constant |
| `creation` | module |
| `ndarray` | class |
| `np` | module |

#### Parameter gaps in common linalg functions:

| Function | Missing Parameters |
|----------|-------------------|
| `matrix_rank` | `A` |


### 2.2 numpy.fft vs macmetalpy.fft

- **NumPy items**: 20
- **MacMetalPy items**: 22
- **In common**: 18
- **Missing from MacMetalPy**: 2

| Function | Type |
|----------|------|
| `helper` | module |
| `test` | function |

#### Extra in MacMetalPy (not in NumPy):

| Function | Type |
|----------|------|
| `annotations` | constant |
| `creation` | module |
| `ndarray` | class |
| `np` | module |


### 2.3 numpy.random vs macmetalpy.random

- **NumPy items**: 65
- **MacMetalPy items**: 52
- **In common**: 48
- **Missing from MacMetalPy**: 17

| Function | Type |
|----------|------|
| `BitGenerator` | class |
| `MT19937` | class |
| `PCG64` | class |
| `PCG64DXSM` | class |
| `Philox` | class |
| `RandomState` | class |
| `SFC64` | class |
| `SeedSequence` | class |
| `bit_generator` | module |
| `bytes` | function |
| `get_bit_generator` | function |
| `get_state` | function |
| `mtrand` | module |
| `random_integers` | function |
| `set_bit_generator` | function |
| `set_state` | function |
| `test` | function |

#### Extra in MacMetalPy (not in NumPy):

| Function | Type |
|----------|------|
| `annotations` | constant |
| `creation` | module |
| `np` | module |
| `permuted` | function |


---

## 3. ndarray Method and Property Coverage

### 3.1 ndarray Methods

- **NumPy ndarray methods**: 146
- **MacMetalPy ndarray methods**: 138
- **In common**: 135
- **Missing from MacMetalPy**: 11

| Method | Type | Relevance |
|--------|------|-----------|
| `__array__` | operator | internal |
| `__array_finalize__` | operator | internal |
| `__array_function__` | operator | internal |
| `__array_prepare__` | operator | internal |
| `__array_ufunc__` | operator | internal |
| `__array_wrap__` | operator | internal |
| `__class_getitem__` | operator | internal |
| `__delitem__` | operator | internal |
| `__dlpack__` | operator | internal |
| `__dlpack_device__` | operator | internal |
| `__setstate__` | operator | internal |

#### Extra ndarray methods in MacMetalPy:

| Method | Type |
|--------|------|
| `expand_dims` | method |
| `get` | method |
| `set` | method |

### 3.2 ndarray Properties

- **NumPy ndarray properties**: 0
- **MacMetalPy ndarray properties**: 12
- **Missing from MacMetalPy**: 0

### 3.3 Parameter Gaps in Common ndarray Methods

| Method | Missing Parameters |
|--------|-------------------|
| `__init__` | `args`, `kwargs` |

---

## 4. Parameter-Level Comparison (Top-Level Functions)

This is the most important section. For every function that exists in BOTH
NumPy and MacMetalPy, we compare parameter lists and report missing parameters.

- **Functions compared**: 396
- **Functions with parameter gaps**: 121
- **Total missing parameters**: 218
  - HIGH severity: 0
  - MEDIUM severity: 0
  - LOW severity: 218
- **NumPy functions with uninspectable signatures**: 29
- **MacMetalPy functions with uninspectable signatures**: 0

### 4.1 HIGH Severity: Missing Commonly-Used Parameters

These are parameters that real-world code frequently uses: `out=`, `dtype=`
on reductions, `axis=`, `keepdims=`, `ddof=`, `return_index/inverse/counts`, etc.

**No HIGH severity parameter gaps remain!**

### 4.2 MEDIUM Severity: Missing Moderately-Used Parameters

Parameters like `order=`, `casting=`, `where=`, `initial=`, `kind=`, `mode=`, etc.

**No MEDIUM severity parameter gaps remain!**

### 4.3 LOW Severity: Missing Rarely-Used Parameters

Parameters like `subok=`, `like=`, `signature=`, `extobj=`, etc.

<details>
<summary>Click to expand (121 functions affected)</summary>

| Function | Missing Parameter(s) |
|----------|---------------------|
| `abs` | `args`, `kwargs` |
| `absolute` | `args`, `kwargs` |
| `add` | `args`, `kwargs` |
| `arccos` | `args`, `kwargs` |
| `arccosh` | `args`, `kwargs` |
| `arcsin` | `args`, `kwargs` |
| `arcsinh` | `args`, `kwargs` |
| `arctan` | `args`, `kwargs` |
| `arctan2` | `args`, `kwargs` |
| `arctanh` | `args`, `kwargs` |
| `around` | `a` |
| `array_equal` | `a1`, `a2` |
| `array_split` | `ary` |
| `bitwise_and` | `args`, `kwargs` |
| `bitwise_not` | `args`, `kwargs` |
| `bitwise_or` | `args`, `kwargs` |
| `bitwise_xor` | `args`, `kwargs` |
| `broadcast_arrays` | `subok` |
| `broadcast_to` | `array`, `subok` |
| `cbrt` | `args`, `kwargs` |
| `ceil` | `args`, `kwargs` |
| `clip` | `kwargs` |
| `conj` | `args`, `kwargs` |
| `conjugate` | `args`, `kwargs` |
| `copy` | `subok` |
| `copysign` | `args`, `kwargs` |
| `cos` | `args`, `kwargs` |
| `cosh` | `args`, `kwargs` |
| `deg2rad` | `args`, `kwargs` |
| `degrees` | `args`, `kwargs` |
| `divide` | `args`, `kwargs` |
| `divmod` | `args`, `kwargs` |
| `einsum_path` | `einsum_call` |
| `equal` | `args`, `kwargs` |
| `exp` | `args`, `kwargs` |
| `exp2` | `args`, `kwargs` |
| `expm1` | `args`, `kwargs` |
| `eye` | `like` |
| `fabs` | `args`, `kwargs` |
| `flip` | `m` |
| `float_power` | `args`, `kwargs` |
| `floor` | `args`, `kwargs` |
| `floor_divide` | `args`, `kwargs` |
| `fmax` | `args`, `kwargs` |
| `fmin` | `args`, `kwargs` |
| `fmod` | `args`, `kwargs` |
| `frexp` | `args`, `kwargs` |
| `fromfunction` | `kwargs`, `like` |
| `full` | `like` |
| `full_like` | `shape`, `subok` |
| `gcd` | `args`, `kwargs` |
| `greater` | `args`, `kwargs` |
| `greater_equal` | `args`, `kwargs` |
| `heaviside` | `args`, `kwargs` |
| `hstack` | `tup` |
| `hypot` | `args`, `kwargs` |
| `identity` | `like` |
| `invert` | `args`, `kwargs` |
| `isfinite` | `args`, `kwargs` |
| `isinf` | `args`, `kwargs` |
| `isnan` | `args`, `kwargs` |
| `isnat` | `args`, `kwargs` |
| `lcm` | `args`, `kwargs` |
| `ldexp` | `args`, `kwargs` |
| `left_shift` | `args`, `kwargs` |
| `less` | `args`, `kwargs` |
| `less_equal` | `args`, `kwargs` |
| `load` | `max_header_size` |
| `log` | `args`, `kwargs` |
| `log10` | `args`, `kwargs` |
| `log1p` | `args`, `kwargs` |
| `log2` | `args`, `kwargs` |
| `logaddexp` | `args`, `kwargs` |
| `logaddexp2` | `args`, `kwargs` |
| `logical_and` | `args`, `kwargs` |
| `logical_not` | `args`, `kwargs` |
| `logical_or` | `args`, `kwargs` |
| `logical_xor` | `args`, `kwargs` |
| `matmul` | `args`, `kwargs` |
| `maximum` | `args`, `kwargs` |
| `minimum` | `args`, `kwargs` |
| `mod` | `args`, `kwargs` |
| `modf` | `args`, `kwargs` |
| `multiply` | `args`, `kwargs` |
| `nanpercentile` | `interpolation` |
| `nanquantile` | `interpolation` |
| `negative` | `args`, `kwargs` |
| `nextafter` | `args`, `kwargs` |
| `not_equal` | `args`, `kwargs` |
| `ones` | `like` |
| `ones_like` | `shape`, `subok` |
| `percentile` | `interpolation` |
| `positive` | `args`, `kwargs` |
| `power` | `args`, `kwargs` |
| `quantile` | `interpolation` |
| `rad2deg` | `args`, `kwargs` |
| `radians` | `args`, `kwargs` |
| `reciprocal` | `args`, `kwargs` |
| `remainder` | `args`, `kwargs` |
| `right_shift` | `args`, `kwargs` |
| `rint` | `args`, `kwargs` |
| `round` | `a` |
| `round_` | `a` |
| `save` | `fix_imports` |
| `sign` | `args`, `kwargs` |
| `signbit` | `args`, `kwargs` |
| `sin` | `args`, `kwargs` |
| `sinh` | `args`, `kwargs` |
| `spacing` | `args`, `kwargs` |
| `split` | `ary` |
| `sqrt` | `args`, `kwargs` |
| `square` | `args`, `kwargs` |
| `subtract` | `args`, `kwargs` |
| `tan` | `args`, `kwargs` |
| `tanh` | `args`, `kwargs` |
| `tile` | `A` |
| `tri` | `like` |
| `true_divide` | `args`, `kwargs` |
| `trunc` | `args`, `kwargs` |
| `vstack` | `tup` |
| `zeros_like` | `shape`, `subok` |

</details>

---

## 5. Summary

- **Function coverage**: 325/399 (81.5%)
- **Parameter coverage**: 693/911 (76.1%)
- **HIGH severity gaps remaining**: 0
- **MEDIUM severity gaps remaining**: 0
- **LOW severity gaps remaining**: 218
