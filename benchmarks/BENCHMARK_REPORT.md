# macmetalpy vs NumPy Benchmark Report

**Platform**: arm  
**Python**: 3.11.14  
**Date**: 2026-02-22 18:31  
**Warmup**: 2 runs  

## Category Summary

| Category | small speedup | medium speedup | large speedup | Avg Speedup |
|---|---|---|---|---|
| arithmetic_f64 | 0.44x | 0.95x | 1.00x | **0.80x** |
| bitwise | 0.36x | 0.89x | 0.96x | **0.74x** |
| casting_f64 | 0.31x | 0.92x | 0.97x | **0.73x** |
| comparison | 0.34x | 0.93x | 0.98x | **0.75x** |
| comparison_f64 | 0.35x | 0.94x | 0.98x | **0.76x** |
| complex_ops | 0.44x | 0.68x | 0.66x | **0.59x** |
| creation | 0.53x | 2.00x | 11.92x | **4.82x** |
| creation_extra | 0.66x | 0.80x | 0.82x | **0.76x** |
| creation_f64 | 0.40x | 3.08x | 33.09x | **12.19x** |
| fft | 0.90x | 1.00x | 1.00x | **0.97x** |
| fft_extra | 0.78x | 0.91x | 0.92x | **0.87x** |
| indexing | 0.47x | 0.98x | 1.04x | **0.83x** |
| indexing_extra | 0.72x | 1.01x | 1.52x | **1.07x** |
| io | 0.98x | 1.00x | - | **0.99x** |
| linalg | 0.80x | 0.97x | - | **0.88x** |
| linalg_extra | 0.78x | 0.90x | 0.78x | **0.83x** |
| logic | 0.45x | 0.89x | 0.98x | **0.77x** |
| manipulation | 0.41x | 0.58x | 0.64x | **0.54x** |
| manipulation_extra | 0.54x | 0.94x | 4.43x | **1.97x** |
| math | 0.63x | 1.20x | 2.92x | **1.58x** |
| math_ext | 0.71x | 0.95x | 0.99x | **0.88x** |
| misc | 0.66x | 0.93x | 0.98x | **0.85x** |
| nan_extra | 0.84x | 0.94x | 0.99x | **0.92x** |
| nan_ops | 0.69x | 0.85x | 0.96x | **0.83x** |
| poly | 0.94x | 1.00x | 1.00x | **0.98x** |
| poly_extra | 0.67x | - | - | **0.67x** |
| random | 0.85x | 0.98x | 0.97x | **0.93x** |
| reduction | 0.52x | 0.90x | 0.99x | **0.80x** |
| reduction_extra | 0.67x | 1.03x | 1.58x | **1.10x** |
| reduction_f64 | 0.45x | 0.84x | 0.95x | **0.75x** |
| set_extra | 0.97x | 0.99x | 0.99x | **0.99x** |
| set_ops | 0.92x | 1.00x | 1.00x | **0.97x** |
| sorting | 0.68x | 1.64x | 4.81x | **2.37x** |
| sorting_extra | 0.78x | 0.98x | 1.00x | **0.92x** |
| stats | 0.85x | 0.97x | 0.97x | **0.92x** |
| trig | 0.80x | 1.04x | 3.14x | **1.66x** |
| trig_f64 | 0.71x | 0.98x | 0.99x | **0.89x** |
| ufuncs_extra | 0.64x | 1.51x | 2.93x | **1.70x** |
| window | 0.88x | 0.96x | 0.97x | **0.94x** |

## Top 10: macmetalpy Fastest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | array_f64 | creation_f64 | large | **129.38x** |
| 2 | array | creation | large | **111.00x** |
| 3 | copyto | manipulation_extra | large | **102.56x** |
| 4 | searchsorted | sorting | large | **17.70x** |
| 5 | put_along_axis | indexing_extra | large | **12.13x** |
| 6 | array | creation | medium | **10.86x** |
| 7 | fabs | ufuncs_extra | large | **10.51x** |
| 8 | floor_divide | math | large | **10.42x** |
| 9 | heaviside | ufuncs_extra | large | **10.02x** |
| 10 | fabs | ufuncs_extra | medium | **9.52x** |

## Top 10: macmetalpy Slowest vs NumPy

| Rank | API | Category | Size | Speedup |
|------|-----|----------|------|---------|
| 1 | frombuffer | creation_extra | large | **0.00x** |
| 2 | searchsorted | sorting | small | **0.01x** |
| 3 | frombuffer | creation_extra | medium | **0.03x** |
| 4 | asanyarray | creation_extra | small | **0.07x** |
| 5 | asfortranarray | manipulation_extra | small | **0.09x** |
| 6 | flip | manipulation | medium | **0.14x** |
| 7 | flip | manipulation | large | **0.14x** |
| 8 | asanyarray | creation_extra | large | **0.14x** |
| 9 | asanyarray | creation_extra | medium | **0.14x** |
| 10 | flip | manipulation | small | **0.15x** |

## Category Aggregates

| Category | Benchmarks | GPU Wins | Avg Speedup | Median Speedup | Min | Max |
|----------|------------|----------|-------------|----------------|-----|-----|
| arithmetic_f64 | 18 | 2 | 0.80x | 0.95x | 0.21x | 1.01x |
| bitwise | 27 | 1 | 0.74x | 0.90x | 0.21x | 1.00x |
| casting_f64 | 6 | 0 | 0.73x | 0.92x | 0.27x | 0.99x |
| comparison | 12 | 0 | 0.75x | 0.93x | 0.33x | 0.99x |
| comparison_f64 | 12 | 0 | 0.76x | 0.94x | 0.35x | 0.99x |
| complex_ops | 15 | 2 | 0.59x | 0.72x | 0.16x | 1.01x |
| creation | 30 | 5 | 4.82x | 0.89x | 0.20x | 111.00x |
| creation_extra | 51 | 2 | 0.76x | 0.91x | 0.00x | 1.01x |
| creation_f64 | 12 | 3 | 12.19x | 0.97x | 0.30x | 129.38x |
| fft | 18 | 5 | 0.97x | 1.00x | 0.86x | 1.01x |
| fft_extra | 18 | 2 | 0.87x | 0.90x | 0.57x | 1.00x |
| indexing | 15 | 3 | 0.83x | 0.98x | 0.21x | 1.23x |
| indexing_extra | 56 | 11 | 1.07x | 0.93x | 0.19x | 12.13x |
| io | 4 | 1 | 0.99x | 0.99x | 0.96x | 1.01x |
| linalg | 20 | 3 | 0.88x | 0.98x | 0.45x | 1.00x |
| linalg_extra | 27 | 3 | 0.83x | 0.96x | 0.32x | 1.00x |
| logic | 48 | 4 | 0.77x | 0.93x | 0.23x | 1.02x |
| manipulation | 33 | 0 | 0.54x | 0.53x | 0.14x | 0.99x |
| manipulation_extra | 81 | 4 | 1.97x | 0.68x | 0.09x | 102.56x |
| math | 42 | 13 | 1.58x | 0.96x | 0.16x | 10.42x |
| math_ext | 32 | 4 | 0.88x | 0.96x | 0.31x | 1.01x |
| misc | 31 | 3 | 0.85x | 0.94x | 0.32x | 1.10x |
| nan_extra | 27 | 2 | 0.92x | 0.94x | 0.68x | 1.01x |
| nan_ops | 15 | 0 | 0.83x | 0.87x | 0.56x | 0.97x |
| poly | 7 | 1 | 0.98x | 1.00x | 0.86x | 1.00x |
| poly_extra | 7 | 0 | 0.67x | 0.63x | 0.34x | 0.99x |
| random | 15 | 2 | 0.93x | 0.95x | 0.72x | 1.01x |
| reduction | 39 | 2 | 0.80x | 0.95x | 0.36x | 1.01x |
| reduction_extra | 27 | 3 | 1.10x | 0.96x | 0.43x | 6.39x |
| reduction_f64 | 24 | 1 | 0.75x | 0.83x | 0.30x | 1.04x |
| set_extra | 3 | 0 | 0.99x | 0.99x | 0.97x | 0.99x |
| set_ops | 15 | 5 | 0.97x | 0.99x | 0.90x | 1.03x |
| sorting | 15 | 5 | 2.37x | 0.99x | 0.01x | 17.70x |
| sorting_extra | 9 | 2 | 0.92x | 0.96x | 0.76x | 1.04x |
| stats | 24 | 0 | 0.92x | 0.94x | 0.63x | 1.00x |
| trig | 45 | 14 | 1.66x | 0.99x | 0.75x | 6.64x |
| trig_f64 | 18 | 1 | 0.89x | 0.98x | 0.48x | 1.02x |
| ufuncs_extra | 102 | 27 | 1.70x | 0.98x | 0.19x | 10.51x |
| window | 15 | 1 | 0.94x | 0.95x | 0.82x | 1.00x |

## Failed Benchmarks

| API | Category | Size | Error |
|-----|----------|------|-------|
| mask_indices | indexing_extra | small | TypeError: no implementation found for 'numpy.nonzero' on types that implement __array_function__: [<class 'macmetalpy.ndarray.ndarray'>] |
| mask_indices | indexing_extra | medium | TypeError: no implementation found for 'numpy.nonzero' on types that implement __array_function__: [<class 'macmetalpy.ndarray.ndarray'>] |
| mask_indices | indexing_extra | large | TypeError: no implementation found for 'numpy.nonzero' on types that implement __array_function__: [<class 'macmetalpy.ndarray.ndarray'>] |

## Optimization Guidance

### Where macmetalpy is slower than NumPy

- **manipulation_extra**: 77/81 benchmarks slower (avg 1.97x)
- **ufuncs_extra**: 75/102 benchmarks slower (avg 1.70x)
- **creation_extra**: 49/51 benchmarks slower (avg 0.76x)
- **indexing_extra**: 45/56 benchmarks slower (avg 1.07x)
- **logic**: 44/48 benchmarks slower (avg 0.77x)
- **reduction**: 37/39 benchmarks slower (avg 0.80x)
- **manipulation**: 33/33 benchmarks slower (avg 0.54x)
- **trig**: 31/45 benchmarks slower (avg 1.66x)
- **math**: 29/42 benchmarks slower (avg 1.58x)
- **math_ext**: 28/32 benchmarks slower (avg 0.88x)
- **misc**: 28/31 benchmarks slower (avg 0.85x)
- **bitwise**: 26/27 benchmarks slower (avg 0.74x)
- **creation**: 25/30 benchmarks slower (avg 4.82x)
- **nan_extra**: 25/27 benchmarks slower (avg 0.92x)
- **linalg_extra**: 24/27 benchmarks slower (avg 0.83x)
- **reduction_extra**: 24/27 benchmarks slower (avg 1.10x)
- **stats**: 24/24 benchmarks slower (avg 0.92x)
- **reduction_f64**: 23/24 benchmarks slower (avg 0.75x)
- **linalg**: 17/20 benchmarks slower (avg 0.88x)
- **trig_f64**: 17/18 benchmarks slower (avg 0.89x)
- **arithmetic_f64**: 16/18 benchmarks slower (avg 0.80x)
- **fft_extra**: 16/18 benchmarks slower (avg 0.87x)
- **nan_ops**: 15/15 benchmarks slower (avg 0.83x)
- **window**: 14/15 benchmarks slower (avg 0.94x)
- **complex_ops**: 13/15 benchmarks slower (avg 0.59x)
- **fft**: 13/18 benchmarks slower (avg 0.97x)
- **random**: 13/15 benchmarks slower (avg 0.93x)
- **comparison**: 12/12 benchmarks slower (avg 0.75x)
- **comparison_f64**: 12/12 benchmarks slower (avg 0.76x)
- **indexing**: 12/15 benchmarks slower (avg 0.83x)
- **set_ops**: 10/15 benchmarks slower (avg 0.97x)
- **sorting**: 10/15 benchmarks slower (avg 2.37x)
- **creation_f64**: 9/12 benchmarks slower (avg 12.19x)
- **poly_extra**: 7/7 benchmarks slower (avg 0.67x)
- **sorting_extra**: 7/9 benchmarks slower (avg 0.92x)
- **casting_f64**: 6/6 benchmarks slower (avg 0.73x)
- **poly**: 6/7 benchmarks slower (avg 0.98x)
- **io**: 3/4 benchmarks slower (avg 0.99x)
- **set_extra**: 3/3 benchmarks slower (avg 0.99x)

### General Observations

- **small** (1,000 elements): GPU wins 3/354, avg speedup 0.64x
- **medium** (100,000 elements): GPU wins 44/345, avg speedup 1.06x
- **large** (1,000,000 elements): GPU wins 90/316, avg speedup 2.51x

### Recommendations

1. **Small arrays (<1K)**: GPU overhead often dominates; keep data on CPU for tiny operations.
2. **Medium arrays (~100K)**: GPU starts to show benefits for compute-heavy ops (math, FFT).
3. **Large arrays (1M+)**: GPU excels; batch operations where possible to amortize transfer cost.
4. **Transfer minimization**: Chain GPU operations to avoid repeated CPU<->GPU transfers.

