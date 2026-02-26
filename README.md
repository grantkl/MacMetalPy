```
  __  __            __  __      _        _ ____
 |  \/  | __ _  ___|  \/  | ___| |_ __ _| |  _ \ _   _
 | |\/| |/ _` |/ __| |\/| |/ _ \ __/ _` | | |_) | | | |
 | |  | | (_| | (__| |  | |  __/ || (_| | |  __/| |_| |
 |_|  |_|\__,_|\___|_|  |_|\___|\__\__,_|_|_|    \__, |
                                                  |___/
```

# MacMetalPy

### Shred data on Apple Silicon. No CUDA required.

A **CuPy-compatible** GPU array library that rips through computation on Apple Silicon using the **Metal** backend. Drop it into your existing CuPy code, swap the import, and let your M-series chip absolutely shred.

> **Heads up:** Metal GPUs operate in **float32** — there is no hardware float64. MacMetalPy auto-downcasts float64 → float32 by default (with warnings), or can fall back to CPU. See [Float Precision](#float-precision--the-float64-question) for details.

```python
import macmetalpy as cp

a = cp.random.randn(4096, 4096)
b = cp.random.randn(4096, 4096)
c = a @ b  # 🔥 Metal GPU goes brrr
```

---

## The Setlist

- **Drop-in CuPy replacement** — `import macmetalpy as cp` and your existing code just works
- **200+ NumPy-compatible functions** — creation, math, linalg, FFT, random, indexing, sorting, reductions, and more
- **Async Metal dispatch** — operations fire off to the GPU and don't wait around
- **RawKernel** — write your own Metal Shading Language kernels when the built-in riffs aren't enough
- **17,000+ passing tests** — battle-tested across 10 dtypes and every edge case we could throw at it
- **Zero CUDA dependency** — pure Apple Silicon, pure Metal

---

## Plug In & Play

```bash
pip install macmetalpy
```

**Requirements:**
- macOS (Apple Silicon — M1/M2/M3/M4)
- Python >= 3.10
- numpy >= 2.0
- metalgpu >= 0.0.5

---

## Soundcheck

**Create arrays on the GPU:**

```python
import macmetalpy as cp

a = cp.zeros((1000, 1000))
b = cp.ones((1000, 1000))
c = cp.arange(0, 100, dtype=cp.int32)    # explicit int dtype
d = cp.linspace(0, 1, 256, dtype=cp.float16)  # half precision
```

**Rip through math:**

```python
import macmetalpy as cp

x = cp.random.randn(10000)

# Elementwise operations — all on the GPU
y = cp.sqrt(cp.abs(x)) + cp.exp(-x ** 2)

# Reductions
total = cp.sum(y)
avg = cp.mean(y)
```

**Linear algebra:**

```python
import macmetalpy as cp

A = cp.random.randn(512, 512)
b = cp.random.randn(512)

x = cp.linalg.solve(A, b)          # Solve Ax = b
U, S, Vt = cp.linalg.svd(A)        # SVD
eigenvalues = cp.linalg.eigvalsh(A @ A.T)  # Eigenvalues
```

**Pull results back to CPU:**

```python
gpu_result = cp.sum(cp.random.randn(1000000))
numpy_array = gpu_result.get()  # Transfer to NumPy
```

---

## Benchmarks — When Does the GPU Shred?

MacMetalPy vs NumPy on an **M4 Mac Mini**, float32. Small arrays use optimized CPU paths (NumPy SIMD is hard to beat below 100K elements), while the GPU shines on large compute-heavy workloads and specialized operations.

### The Scaling Story

| Operation | 1K | 100K | 1M |
|---|---|---|---|
| `a + b` | 0.37x | 0.90x | 0.91x |
| `sin(a)` | 0.68x | 0.96x | **1.92x** |
| `exp(a)` | 0.70x | 0.95x | **2.07x** |
| `cumsum(a)` | 0.58x | **1.38x** | **1.37x** |
| `floor_divide` | 0.77x | **3.01x** | **11.66x** |
| `mod(a, b)` | 0.68x | **1.51x** | **5.73x** |
| `randn(a)` | **2.62x** | **3.33x** | **3.23x** |
| `normal(a)` | **1.90x** | **2.26x** | **2.24x** |
| `sort(a)` | 0.78x | 0.87x | **1.46x** |
| `searchsorted` | 0.01x | **3.24x** | **18.84x** |

> Values are speedup vs NumPy (higher = faster). **Bold** = MacMetalPy wins.

### Where MacMetalPy Shreds

| Category | Avg Speedup | Highlights |
|---|---|---|
| **Creation (f64)** | **10.59x** | `array()` 111x at 1M — skips float64 intermediates |
| **Creation** | **2.57x** | `array()` 51x at 1M |
| **Sorting** | **2.26x** | `searchsorted` 18.8x at 1M, `sort` 1.5x at 1M |
| **Ufuncs** | **1.58x** | `fabs` 10x, `logaddexp` 8.5x at 1M |
| **Trig** | **1.73x** | `sin` 1.9x, `exp` 2.1x at 1M — GPU shines at scale |
| **Random** | **1.70x** | `randn` 3.2x, `normal` 2.2x — native float32 generation |
| **Math** | **1.63x** | `floor_divide` 12x, `mod` 8.1x at 1M |

### By Category at 100K / 1M Elements

| Category | 100K | 1M | Notes |
|---|---|---|---|
| **Random** | **1.85x** | **1.81x** | Native float32 via Generator API |
| **Creation** | **1.32x** | **5.94x** | Dtype conversion bypass at scale |
| **Ufuncs** | **1.32x** | **2.85x** | GPU dominates at scale |
| **Sorting** | **1.33x** | **4.85x** | `searchsorted` 18.8x at 1M |
| **Math** | **1.16x** | **3.17x** | `floor_divide` 12x at 1M |
| **Trig** | 0.98x | **3.49x** | GPU wins decisively at 1M |
| **Reductions** | 0.93x | 0.96x | `cumsum` 1.37x |
| **Comparisons** | 0.90x | 0.89x | Near-parity |

### The Rule of Thumb

| Array Size | Who Wins | Why |
|---|---|---|
| **< 10K** | NumPy | Python dispatch overhead dominates |
| **10K – 100K** | Roughly even | CPU SIMD paths match NumPy |
| **100K – 1M** | **GPU wins many** | Trig, math, sorting, ufuncs all >1x; random/creation dominate |
| **1M+** | **GPU shreds** | Metal dispatch amortized, massive parallelism wins |

> Run the benchmarks yourself: `python benchmarks/bench_vs_numpy.py --numpy-cache`

---

## The Lineup

| Module | Functions | What it shreds |
|---|---|---|
| **Creation** | 25 | `zeros`, `ones`, `arange`, `linspace`, `eye`, `meshgrid`, ... |
| **Math** | 94 | `sqrt`, `exp`, `log`, `sin`, `cos`, `dot`, `where`, `clip`, ... |
| **Reductions** | 21 | `sum`, `mean`, `std`, `var`, `argmax`, `cumsum`, `median`, ... |
| **Linalg** | 25 | `solve`, `inv`, `svd`, `eigh`, `qr`, `det`, `norm`, `einsum`, ... |
| **Manipulation** | 33 | `reshape`, `transpose`, `concatenate`, `stack`, `pad`, `tile`, ... |
| **Indexing** | 23 | `take`, `put`, `nonzero`, `argwhere`, `fill_diagonal`, ... |
| **Sorting** | 9 | `sort`, `argsort`, `unique`, `searchsorted`, `partition`, ... |
| **FFT** | 19 | `fft`, `ifft`, `rfft`, `fft2`, `fftn`, `fftfreq`, ... |
| **Random** | 40+ | `randn`, `uniform`, `normal`, `poisson`, `choice`, `shuffle`, ... |
| **Logic & Bitwise** | 30 | `logical_and`, `greater`, `bitwise_xor`, `gcd`, `lcm`, ... |
| **NaN Ops** | 27 | `nansum`, `nanmean`, `histogram`, `corrcoef`, `gradient`, ... |
| **Set Ops** | 7 | `union1d`, `intersect1d`, `setdiff1d`, `isin`, ... |

---

## Custom Riffs

When the built-in operations don't cut it, write your own Metal Shading Language kernels with `RawKernel`:

```python
from macmetalpy import RawKernel
import macmetalpy as cp

# Write a custom Metal kernel
kernel_source = """
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device float *x [[buffer(0)]],
                  device float *y [[buffer(1)]],
                  device float *out [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    float alpha = 2.5f;
    out[id] = alpha * x[id] + y[id];
}
"""

saxpy = RawKernel(kernel_source, 'saxpy')

N = 1_000_000
x = cp.random.randn(N)
y = cp.random.randn(N)
out = cp.empty(N)

saxpy(N, (x, y, out))  # Launch N GPU threads

result = out.get()
```

Grid sizes can be 1D, 2D, or 3D:

```python
kernel(N, args)              # 1D — N threads
kernel((W, H), args)         # 2D grid
kernel((W, H, D), args)      # 3D grid
```

---

## Float Precision & The float64 Question

**This is the biggest difference between MacMetalPy and NumPy/CuPy.**

Apple's Metal GPU has **no native float64 (double) support**. All GPU computation runs in **float32** (single precision) or **float16** (half precision). This is a hardware limitation — not a software one.

### What this means in practice

| Scenario | What happens |
|---|---|
| `cp.array([1.0, 2.0])` | Created as **float32** (NumPy would default to float64) |
| `cp.zeros(10, dtype=np.float64)` | **Downcast to float32** with a warning (by default) |
| `cp.linalg.solve(A, b)` | Runs in float32 — ~7 decimal digits of precision |
| `cp.sum(x, dtype=np.float64)` | Accumulates in float32 |
| `complex128` input | **Downcast to complex64** (two float32 values) |

### When float32 is fine (most cases)

- Machine learning / deep learning (models train in float16/float32 anyway)
- Image and signal processing
- General scientific computing where ~7 digits of precision is sufficient
- Data analysis and statistics on reasonably-scaled data
- FFT, random number generation, sorting, indexing

### When you might need float64

- Numerical methods sensitive to rounding (e.g., ill-conditioned linear systems)
- Financial calculations requiring exact decimal precision
- Accumulating very large sums (billions of elements) where error compounds
- Algorithms that rely on the full 15-16 digits of float64 precision

### Configuring float64 behavior

```python
from macmetalpy import set_config

# DEFAULT: Downcast float64 → float32, emit a warning
set_config(float64_behavior="downcast", warn_on_downcast=True)

# Silence the warnings if you know what you're doing
set_config(float64_behavior="downcast", warn_on_downcast=False)

# Fall back to CPU (NumPy) for any float64 operation
set_config(float64_behavior="cpu_fallback")

# Set the default float dtype for creation functions
set_config(default_float_dtype="float32")
```

### Comparison with NumPy and CuPy

| | NumPy (CPU) | CuPy (CUDA) | MacMetalPy (Metal) |
|---|---|---|---|
| Default float | float64 | float64 | **float32** |
| float64 support | Native | Native | Downcast or CPU fallback |
| float16 support | Software | Native | Native |
| complex128 | Native | Native | Downcast to complex64 |
| int8 / uint8 | Native | Native | **Not supported** |
| Precision digits | ~15-16 | ~15-16 | **~7** (float32) |

---

## Supported Amps

| Dtype | Metal Type | Notes |
|---|---|---|
| `float32` | `float` | Default float — full GPU support |
| `float16` | `half` | Half precision — fastest for large arrays |
| `int32` | `int` | Default int type |
| `int64` | `long` | 64-bit integer |
| `int16` | `short` | 16-bit integer |
| `uint32` | `uint` | Unsigned 32-bit |
| `uint64` | `uint64_t` | Unsigned 64-bit |
| `uint16` | `uint16_t` | Unsigned 16-bit |
| `bool` | `bool` | Boolean |
| `complex64` | float32 pairs | Stored as real/imag float32 |

**Not supported by Metal:** `float64`, `complex128`, `int8`, `uint8`, `longdouble`, `str_`, `bytes_`, `object_`

---

## Acknowledgments

MacMetalPy stands on the shoulders of giants:

- **[NumPy](https://numpy.org/)** — The foundation. MacMetalPy's API is modeled after NumPy's, because they got it right the first time.
- **[CuPy](https://cupy.dev/)** — The blueprint for GPU array libraries. CuPy proved that a drop-in NumPy replacement on the GPU is both possible and practical.
- **[metalgpu](https://github.com/Al0den/metalgpu)** — The engine under the hood. Without metalgpu's Python-to-Metal bridge, MacMetalPy wouldn't exist.

---

## The Crew

**License:** MIT

**Contributing:** Issues and PRs welcome. If you find a bug or want to add a new function, open an issue or submit a pull request.

**Built by** [@grantkl](https://github.com/grantkl)
