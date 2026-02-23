# NumPy vs MacMetalPy API Comparison Report

NumPy version analyzed: 2.4.2
Report generated: 2026-02-22

---

## 1. Top-Level Function/Class/Constant Coverage

- **NumPy top-level items (excluding skipped)**: 483
- **MacMetalPy top-level items**: 483
- **Coverage**: **100.0%** (483/483)
- **Missing from MacMetalPy**: 0

Skipped items (not applicable): `core`, `ctypeslib`, `f2py`, `lib`, `ma`, `testing`, `typing`, `test`, `nested_iters`, `bool`, `long`, `ulong`, `overrides`, `AxisError`

---

## 2. Submodule Coverage

### 2.1 numpy.linalg

- **Coverage**: 21/23 (91.3%)
- **Missing**: `linalg` (self-ref), `test`

### 2.2 numpy.fft

- **Coverage**: 18/20 (90.0%)
- **Missing**: `helper`, `test`

### 2.3 numpy.random

- **Coverage**: 64/65 (98.5%)
- **Missing**: `test`

---

## 3. ndarray Method and Property Coverage

### 3.1 ndarray Methods & Properties

- **Coverage**: 70/70 (100.0%)
- **Missing methods/properties**: 0

### 3.2 ndarray Dunders

- **Coverage**: 95/95 (100.0%)
- **Missing dunders**: 0

---

## 4. Benchmark Summary

Total benchmarks: 1015 | GPU wins: 120 | Failed: 3

### By Size Tier

| Size | Avg Speedup | Median | GPU Wins |
|------|-------------|--------|----------|
| small | 0.64x | 0.69x | 3/354 |
| medium | 1.05x | 0.96x | 40/345 |
| large | 2.54x | 0.99x | 77/316 |

### Top GPU-Winning Categories

| Category | Avg Speedup | GPU Wins |
|----------|-------------|----------|
| creation_f64 | 11.88x | 2/12 |
| creation | 4.89x | 6/30 |
| sorting | 2.63x | 5/15 |
| trig | 1.93x | 13/45 |
| manipulation_extra | 1.88x | 2/81 |
| ufuncs_extra | 1.74x | 27/102 |
| math | 1.51x | 12/42 |
| indexing_extra | 1.10x | 9/56 |
| reduction_extra | 1.06x | 4/27 |

---

## 5. Summary

- **Top-level API coverage**: 483/483 (100.0%)
- **ndarray coverage**: 100.0% (methods, properties, and dunders)
- **Random module coverage**: 64/65 (98.5%)
- **Linalg/FFT coverage**: 91-90%
- **All ufunc functions accept standard ufunc kwargs** (out, where, casting, subok, dtype, order, signature)
- **All creation functions accept `like=` and `device=` parameters**
- **All reduction functions accept `correction=` and `mean=` where applicable**
- **17,471 tests passing, 0 failures**
