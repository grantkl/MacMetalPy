"""Top-level linear algebra functions (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray
from . import creation

def _cpu_view(a):
    """Get zero-copy numpy view (syncs if GPU-resident)."""
    if a._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
    return a._get_view()


def vdot(a, b):
    """Return the dot product of two vectors."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    # CPU fast path — dot product is memory-bound
    if a.size < 4194304:
        a_np = a._np_data if a._np_data is not None else _cpu_view(a)
        b_np = b._np_data if b._np_data is not None else _cpu_view(b)
        return ndarray._from_np_direct(np.asarray(np.vdot(a_np, b_np)))
    a_flat = a.ravel()
    b_flat = b.ravel()
    return a_flat._reduce_dot(b_flat)

def inner(a, b):
    """Inner product of two arrays."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    # CPU fast path
    if a.size < 4194304 and b.size < 4194304:
        a_np = a._np_data if a._np_data is not None else _cpu_view(a)
        b_np = b._np_data if b._np_data is not None else _cpu_view(b)
        result = np.inner(a_np, b_np)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        return ndarray._from_np_direct(result)
    if a.ndim == 1 and b.ndim == 1:
        return a._reduce_dot(b)
    result = np.inner(a.get(), b.get())
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return ndarray._from_np_direct(result)

def outer(a, b, out=None):
    """Compute the outer product of two vectors."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    a_np = a._np_data if a._np_data is not None else _cpu_view(a)
    b_np = b._np_data if b._np_data is not None else _cpu_view(b)
    result = ndarray._from_np_direct(np.outer(a_np, b_np))
    if out is not None:
        out.set(result.get())
        return out
    return result

def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    a_np = a._np_data if a._np_data is not None else _cpu_view(a)
    b_np = b._np_data if b._np_data is not None else _cpu_view(b)
    result = np.tensordot(a_np, b_np, axes=axes)
    return ndarray._from_np_direct(np.asarray(result))

def einsum(*operands, out=None, optimize=False, **kwargs):
    """Evaluates the Einstein summation convention."""
    new_operands = []
    for op in operands:
        if isinstance(op, ndarray):
            new_operands.append(op.get())
        else:
            new_operands.append(op)
    result = np.einsum(*new_operands, optimize=optimize, **kwargs)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    gpu_result = ndarray._from_np_direct(result)
    if out is not None:
        out._np_data = gpu_result.get()
        out._buffer = None
        return out
    return gpu_result

def kron(a, b):
    """Kronecker product of two arrays."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    a_np = a._np_data if a._np_data is not None else _cpu_view(a)
    b_np = b._np_data if b._np_data is not None else _cpu_view(b)
    return ndarray._from_np_direct(np.kron(a_np, b_np))

def matmul(x1, x2, out=None, *, axes=None, axis=None, keepdims=False):
    """Matrix product of two arrays."""
    if not isinstance(x1, ndarray): x1 = creation.asarray(x1)
    if not isinstance(x2, ndarray): x2 = creation.asarray(x2)
    # CPU fast path for small/medium arrays
    if x1.size < 4194304 and x2.size < 4194304:
        x1_np = x1._np_data if x1._np_data is not None else _cpu_view(x1)
        x2_np = x2._np_data if x2._np_data is not None else _cpu_view(x2)
        result = ndarray._from_np_direct(np.matmul(x1_np, x2_np))
    elif x1.ndim == 2 and x2.ndim == 2:
        result = x1.__matmul__(x2)
    else:
        result = ndarray._from_np_direct(np.matmul(x1.get(), x2.get()))
    if out is not None:
        out._np_data = result.astype(out.dtype).get()
        out._buffer = None
        return out
    return result

def einsum_path(*operands, optimize='greedy', **kwargs):
    """Evaluates the lowest cost contraction order for an einsum expression."""
    new_operands = []
    for op in operands:
        if isinstance(op, ndarray):
            new_operands.append(op.get())
        else:
            new_operands.append(op)
    return np.einsum_path(*new_operands, optimize=optimize, **kwargs)

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Return the cross product of two vectors."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    result = np.cross(a.get(), b.get(), axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)
    return ndarray._from_np_direct(result)


# ── NumPy 2 linear algebra shortcuts ─────────────────────────────────

def matvec(m, v):
    """Matrix-vector product: m @ v."""
    if not isinstance(m, ndarray): m = creation.asarray(m)
    if not isinstance(v, ndarray): v = creation.asarray(v)
    return matmul(m, v)


def vecmat(v, m):
    """Vector-matrix product: v @ m."""
    if not isinstance(v, ndarray): v = creation.asarray(v)
    if not isinstance(m, ndarray): m = creation.asarray(m)
    v_np = v._np_data if v._np_data is not None else _cpu_view(v)
    m_np = m._np_data if m._np_data is not None else _cpu_view(m)
    result = np.matmul(v_np, m_np)
    return ndarray._from_np_direct(result)


def vecdot(a, b):
    """Vector dot product: sum(a * b) along the last axis."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    a_np = a._np_data if a._np_data is not None else _cpu_view(a)
    b_np = b._np_data if b._np_data is not None else _cpu_view(b)
    result = np.vecdot(a_np, b_np)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return ndarray._from_np_direct(result)
