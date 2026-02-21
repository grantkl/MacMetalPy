"""Linear algebra functions (CuPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray
from . import creation


def norm(x, ord=None, axis=None):
    """Matrix or vector norm."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    result = np.linalg.norm(x.get(), ord=ord, axis=axis)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def inv(a):
    """Compute the inverse of a matrix."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.linalg.inv(a.get())
    return creation.array(result)


def det(a):
    """Compute the determinant of a matrix."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.linalg.det(a.get())
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def solve(a, b):
    """Solve a linear matrix equation."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    result = np.linalg.solve(a.get(), b.get())
    return creation.array(result)


def eigh(a):
    """Eigenvalues and eigenvectors of a symmetric matrix."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    w, v = np.linalg.eigh(a.get())
    return creation.array(w), creation.array(v)


def svd(a, full_matrices=True):
    """Singular Value Decomposition."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    u, s, vh = np.linalg.svd(a.get(), full_matrices=full_matrices)
    return creation.array(u), creation.array(s), creation.array(vh)


def cholesky(a):
    """Cholesky decomposition."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.linalg.cholesky(a.get())
    return creation.array(result)


def matrix_power(a, n):
    """Raise a square matrix to the (integer) power n."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.linalg.matrix_power(a.get(), n)
    return creation.array(result)


def qr(a, mode='reduced'):
    """Compute the qr factorization."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.linalg.qr(a.get(), mode=mode)
    if mode == 'r':
        return creation.array(result)
    return creation.array(result[0]), creation.array(result[1])


def eig(a):
    """Compute eigenvalues and right eigenvectors."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    w, v = np.linalg.eig(a.get())
    return creation.array(w), creation.array(v)


def eigvals(a):
    """Compute eigenvalues of a general matrix."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.linalg.eigvals(a.get())
    return creation.array(result)


def eigvalsh(a):
    """Compute eigenvalues of a symmetric matrix."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.linalg.eigvalsh(a.get())
    return creation.array(result)


def cond(x, p=None):
    """Compute the condition number of a matrix."""
    if not isinstance(x, ndarray):
        x = creation.asarray(x)
    result = np.linalg.cond(x.get(), p=p)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def matrix_rank(M, tol=None):
    """Return matrix rank using SVD method."""
    if not isinstance(M, ndarray):
        M = creation.asarray(M)
    result = np.linalg.matrix_rank(M.get(), tol=tol)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return creation.array(result)


def slogdet(a):
    """Compute sign and natural log of the determinant."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    sign, logdet = np.linalg.slogdet(a.get())
    if not isinstance(sign, np.ndarray):
        sign = np.array(sign)
    if not isinstance(logdet, np.ndarray):
        logdet = np.array(logdet)
    return creation.array(sign), creation.array(logdet)


def lstsq(a, b, rcond=None):
    """Return the least-squares solution."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    if not isinstance(b, ndarray):
        b = creation.asarray(b)
    x, residuals, rank, sv = np.linalg.lstsq(a.get(), b.get(), rcond=rcond)
    return (creation.array(x),
            creation.array(np.asarray(residuals)) if len(residuals) > 0 else creation.array(np.array([])),
            int(rank),
            creation.array(sv))


def pinv(a, rcond=1e-15):
    """Compute the pseudo-inverse of a matrix."""
    if not isinstance(a, ndarray):
        a = creation.asarray(a)
    result = np.linalg.pinv(a.get(), rcond=rcond)
    return creation.array(result)
