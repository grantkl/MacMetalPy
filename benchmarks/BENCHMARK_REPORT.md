# macmetalpy vs NumPy Benchmark Report

**Platform**: arm  
**Python**: 3.11.14  
**Date**: 2026-02-22 16:06  
**Warmup**: 2 runs  

## Category Summary

| Category | small speedup | medium speedup | large speedup | Avg Speedup |
|---|---|---|---|---|
| arithmetic_f64 | 0.44x | 0.95x | 0.99x | **0.80x** |
| bitwise | 0.36x | 0.89x | 0.96x | **0.74x** |
| casting_f64 | 0.28x | 0.92x | 0.98x | **0.73x** |
| comparison | 0.33x | 0.93x | 0.99x | **0.75x** |
| comparison_f64 | 0.36x | 0.94x | 0.99x | **0.76x** |
| complex_ops | 0.45x | 0.65x | 0.67x | **0.59x** |
| creation | 0.53x | 1.93x | 12.23x | **4.89x** |
| creation_extra | 0.67x | 0.80x | 0.82x | **0.76x** |
| creation_f64 | 0.40x | 3.08x | 32.16x | **11.88x** |
| fft | 0.91x | 1.00x | 1.00x | **0.97x** |
| fft_extra | 0.78x | 0.91x | 0.91x | **0.86x** |
| indexing | 0.46x | 0.99x | 1.04x | **0.83x** |
| indexing_extra | 0.72x | 1.00x | 1.62x | **1.10x** |
| io | 0.98x | 0.99x | - | **0.99x** |
| linalg | 0.80x | 0.97x | - | **0.88x** |
| linalg_extra | 0.77x | 0.89x | 0.77x | **0.82x** |
| logic | 0.45x | 0.88x | 0.97x | **0.77x** |
| manipulation | 0.41x | 0.58x | 0.63x | **0.54x** |
| manipulation_extra | 0.53x | 0.91x | 4.21x | **1.88x** |
| math | 0.63x | 1.21x | 2.68x | **1.51x** |
| math_ext | 0.71x | 0.95x | 0.99x | **0.88x** |
| misc | 0.65x | 0.92x | 0.98x | **0.84x** |
| nan_extra | 0.83x | 0.93x | 0.98x | **0.91x** |
| nan_ops | 0.68x | 0.82x | 0.97x | **0.83x** |
| poly | 0.95x | 1.00x | 1.00x | **0.98x** |
| poly_extra | 0.68x | - | - | **0.68x** |
| random | 0.86x | 0.97x | 0.96x | **0.93x** |
| reduction | 0.51x | 0.90x | 0.98x | **0.80x** |
| reduction_extra | 0.66x | 1.04x | 1.49x | **1.06x** |
| reduction_f64 | 0.45x | 0.84x | 0.98x | **0.75x** |
| set_extra | 0.93x | 1.00x | 0.99x | **0.97x** |
| set_ops | 0.94x | 1.00x | 1.00x | **0.98x** |
| sorting | 0.67x | 1.65x | 5.56x | **2.63x** |
| sorting_extra | 0.78x | 0.97x | 0.99x | **0.91x** |
| stats | 0.85x | 0.97x | 0.98x | **0.93x** |
| trig | 0.79x | 1.05x | 3.95x | **1.93x** |
| trig_f64 | 0.70x | 0.98x | 0.99x | **0.89x** |
| ufuncs_extra | 0.65x | 1.51x | 3.06x | **1.74x** |
| window | 0.88x | 0.96x | 0.97x | **0.94x** |

## Top 10: macmetalpy Fastest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | array_f64 | creation_f64 | large | **125.68x** |
| 2 | array | creation | large | **113.65x** |
| 3 | copyto | manipulation_extra | large | **96.64x** |
| 4 | searchsorted | sorting | large | **21.56x** |
| 5 | put_along_axis | indexing_extra | large | **13.95x** |
| 6 | arcsin | trig | large | **10.64x** |
| 7 | arccos | trig | large | **10.38x** |
| 8 | fabs | ufuncs_extra | large | **10.38x** |
| 9 | array | creation | medium | **10.10x** |
| 10 | heaviside | ufuncs_extra | large | **9.91x** |

## Top 10: macmetalpy Slowest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | frombuffer | creation_extra | large | **0.00x** |
| 2 | searchsorted | sorting | small | **0.01x** |
| 3 | frombuffer | creation_extra | medium | **0.03x** |
| 4 | asfortranarray | manipulation_extra | medium | **0.08x** |
| 5 | flip | manipulation | medium | **0.13x** |
| 6 | flip | manipulation | large | **0.14x** |
| 7 | flip | manipulation | small | **0.14x** |
| 8 | asanyarray | creation_extra | medium | **0.14x** |
| 9 | asanyarray | creation_extra | small | **0.14x** |
| 10 | asanyarray | creation_extra | large | **0.14x** |

## Category Aggregates

| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |
|----------|------------|----------|-------------|----------------|-----|-----|
| arithmetic_f64 | 18 | 0 | 0.80x | 0.95x | 0.22x | 1.00x |
| bitwise | 27 | 2 | 0.74x | 0.90x | 0.24x | 1.02x |
| casting_f64 | 6 | 0 | 0.73x | 0.92x | 0.26x | 0.99x |
| comparison | 12 | 0 | 0.75x | 0.93x | 0.32x | 0.99x |
| comparison_f64 | 12 | 1 | 0.76x | 0.94x | 0.35x | 1.00x |
| complex_ops | 15 | 0 | 0.59x | 0.71x | 0.17x | 1.00x |
| creation | 30 | 6 | 4.89x | 0.88x | 0.21x | 113.65x |
| creation_extra | 51 | 2 | 0.76x | 0.90x | 0.00x | 1.02x |
| creation_f64 | 12 | 2 | 11.88x | 0.96x | 0.28x | 125.68x |
| fft | 18 | 7 | 0.97x | 1.00x | 0.87x | 1.02x |
| fft_extra | 18 | 2 | 0.86x | 0.89x | 0.57x | 1.00x |
| indexing | 15 | 2 | 0.83x | 0.98x | 0.20x | 1.25x |
| indexing_extra | 56 | 9 | 1.10x | 0.94x | 0.21x | 13.95x |
| io | 4 | 1 | 0.99x | 0.99x | 0.98x | 1.00x |
| linalg | 20 | 2 | 0.88x | 0.98x | 0.48x | 1.00x |
| linalg_extra | 27 | 5 | 0.82x | 0.97x | 0.31x | 1.02x |
| logic | 48 | 1 | 0.77x | 0.92x | 0.22x | 1.00x |
| manipulation | 33 | 1 | 0.54x | 0.53x | 0.13x | 1.01x |
| manipulation_extra | 81 | 2 | 1.88x | 0.66x | 0.08x | 96.64x |
| math | 42 | 12 | 1.51x | 0.95x | 0.16x | 8.49x |
| math_ext | 32 | 1 | 0.88x | 0.97x | 0.30x | 1.00x |
| misc | 31 | 2 | 0.84x | 0.93x | 0.36x | 1.00x |
| nan_extra | 27 | 0 | 0.91x | 0.95x | 0.68x | 1.00x |
| nan_ops | 15 | 0 | 0.83x | 0.83x | 0.55x | 0.98x |
| poly | 7 | 1 | 0.98x | 1.00x | 0.86x | 1.00x |
| poly_extra | 7 | 0 | 0.68x | 0.59x | 0.35x | 0.99x |
| random | 15 | 0 | 0.93x | 0.96x | 0.72x | 0.99x |
| reduction | 39 | 1 | 0.80x | 0.94x | 0.34x | 1.00x |
| reduction_extra | 27 | 4 | 1.06x | 0.93x | 0.41x | 5.53x |
| reduction_f64 | 24 | 2 | 0.75x | 0.83x | 0.31x | 1.06x |
| set_extra | 3 | 0 | 0.97x | 0.99x | 0.93x | 1.00x |
| set_ops | 15 | 2 | 0.98x | 0.99x | 0.90x | 1.02x |
| sorting | 15 | 5 | 2.63x | 0.99x | 0.01x | 21.56x |
| sorting_extra | 9 | 2 | 0.91x | 0.97x | 0.75x | 1.00x |
| stats | 24 | 3 | 0.93x | 0.94x | 0.63x | 1.03x |
| trig | 45 | 13 | 1.93x | 0.99x | 0.74x | 10.64x |
| trig_f64 | 18 | 0 | 0.89x | 0.98x | 0.46x | 1.00x |
| ufuncs_extra | 102 | 27 | 1.74x | 0.98x | 0.18x | 10.38x |
| window | 15 | 0 | 0.94x | 0.96x | 0.82x | 1.00x |

## Failed Benchmarks

| API | Category | Size | Error |
|-----|----------|------|-------|
| mask_indices | indexing_extra | small | TypeError: no implementation found for 'numpy.nonzero' on types that implement __array_function__: [<class 'macmetalpy.ndarray.ndarray'>] |
| mask_indices | indexing_extra | medium | TypeError: no implementation found for 'numpy.nonzero' on types that implement __array_function__: [<class 'macmetalpy.ndarray.ndarray'>] |
| mask_indices | indexing_extra | large | TypeError: no implementation found for 'numpy.nonzero' on types that implement __array_function__: [<class 'macmetalpy.ndarray.ndarray'>] |

## Optimization Guidance

### Where macmetalpy is slower than NumPy

- **manipulation_extra**: 79/81 benchmarks slower (avg 1.88x)
- **ufuncs_extra**: 75/102 benchmarks slower (avg 1.74x)
- **creation_extra**: 49/51 benchmarks slower (avg 0.76x)
- **indexing_extra**: 47/56 benchmarks slower (avg 1.10x)
- **logic**: 47/48 benchmarks slower (avg 0.77x)
- **reduction**: 38/39 benchmarks slower (avg 0.80x)
- **manipulation**: 32/33 benchmarks slower (avg 0.54x)
- **trig**: 32/45 benchmarks slower (avg 1.93x)
- **math_ext**: 31/32 benchmarks slower (avg 0.88x)
- **math**: 30/42 benchmarks slower (avg 1.51x)
- **misc**: 29/31 benchmarks slower (avg 0.84x)
- **nan_extra**: 27/27 benchmarks slower (avg 0.91x)
- **bitwise**: 25/27 benchmarks slower (avg 0.74x)
- **creation**: 24/30 benchmarks slower (avg 4.89x)
- **reduction_extra**: 23/27 benchmarks slower (avg 1.06x)
- **linalg_extra**: 22/27 benchmarks slower (avg 0.82x)
- **reduction_f64**: 22/24 benchmarks slower (avg 0.75x)
- **stats**: 21/24 benchmarks slower (avg 0.93x)
- **arithmetic_f64**: 18/18 benchmarks slower (avg 0.80x)
- **linalg**: 18/20 benchmarks slower (avg 0.88x)
- **trig_f64**: 18/18 benchmarks slower (avg 0.89x)
- **fft_extra**: 16/18 benchmarks slower (avg 0.86x)
- **complex_ops**: 15/15 benchmarks slower (avg 0.59x)
- **nan_ops**: 15/15 benchmarks slower (avg 0.83x)
- **random**: 15/15 benchmarks slower (avg 0.93x)
- **window**: 15/15 benchmarks slower (avg 0.94x)
- **indexing**: 13/15 benchmarks slower (avg 0.83x)
- **set_ops**: 13/15 benchmarks slower (avg 0.98x)
- **comparison**: 12/12 benchmarks slower (avg 0.75x)
- **comparison_f64**: 11/12 benchmarks slower (avg 0.76x)
- **fft**: 11/18 benchmarks slower (avg 0.97x)
- **creation_f64**: 10/12 benchmarks slower (avg 11.88x)
- **sorting**: 10/15 benchmarks slower (avg 2.63x)
- **poly_extra**: 7/7 benchmarks slower (avg 0.68x)
- **sorting_extra**: 7/9 benchmarks slower (avg 0.91x)
- **casting_f64**: 6/6 benchmarks slower (avg 0.73x)
- **poly**: 6/7 benchmarks slower (avg 0.98x)
- **io**: 3/4 benchmarks slower (avg 0.99x)
- **set_extra**: 3/3 benchmarks slower (avg 0.97x)

### General Observations

- **small** (1,000 elements): GPU wins 3/354, avg speedup 0.64x
- **medium** (100,000 elements): GPU wins 40/345, avg speedup 1.05x
- **large** (1,000,000 elements): GPU wins 77/316, avg speedup 2.54x

### Recommendations

1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.
2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).
3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.
4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.

