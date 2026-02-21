"""Top-level linear algebra functions (CuPy-compatible)."""
from __future__ import annotations
import numpy as np
from .ndarray import ndarray
from . import creation

def vdot(a, b):
    """Return the dot product of two vectors."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    a_flat = a.ravel()
    b_flat = b.ravel()
    product = a_flat._binary_op(b_flat, "mul_op")
    return product._reduce("reduce_sum")

def inner(a, b):
    """Inner product of two arrays."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        product = a._binary_op(b, "mul_op")
        return product._reduce("reduce_sum")
    result = np.inner(a.get(), b.get())
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)

def outer(a, b):
    """Compute the outer product of two vectors."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    return creation.array(np.outer(a.get(), b.get()))

def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    result = np.tensordot(a.get(), b.get(), axes=axes)
    return creation.array(result)

def einsum(*operands):
    """Evaluates the Einstein summation convention."""
    new_operands = []
    for op in operands:
        if isinstance(op, ndarray):
            new_operands.append(op.get())
        else:
            new_operands.append(op)
    result = np.einsum(*new_operands)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)

def kron(a, b):
    """Kronecker product of two arrays."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    return creation.array(np.kron(a.get(), b.get()))

def matmul(x1, x2):
    """Matrix product of two arrays."""
    if not isinstance(x1, ndarray): x1 = creation.asarray(x1)
    if not isinstance(x2, ndarray): x2 = creation.asarray(x2)
    if x1.ndim == 2 and x2.ndim == 2:
        return x1.__matmul__(x2)
    return creation.array(np.matmul(x1.get(), x2.get()))

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Return the cross product of two vectors."""
    if not isinstance(a, ndarray): a = creation.asarray(a)
    if not isinstance(b, ndarray): b = creation.asarray(b)
    result = np.cross(a.get(), b.get(), axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)
    return creation.array(result)
