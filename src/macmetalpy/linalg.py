"""Linear algebra functions (CuPy-compatible)."""

from __future__ import annotations

import numpy as np

from .ndarray import ndarray as _ndarray
from . import creation as _creation


def _get_np(a):
    """Get zero-copy numpy view (syncs if GPU-resident)."""
    if a._np_data is None:
        from ._metal_backend import MetalBackend
        MetalBackend().synchronize()
    return a._get_view()


def _get_np(a):
    """Get numpy data, preferring _np_data for CPU-resident arrays."""
    np_data = a._np_data
    if np_data is not None:
        return np_data
    return _get_np(a)


def norm(x, ord=None, axis=None, keepdims=False):
    """Matrix or vector norm."""
    if not isinstance(x, _ndarray):
        x = _creation.asarray(x)
    result = np.linalg.norm(_get_np(x), ord=ord, axis=axis, keepdims=keepdims)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return _ndarray._from_np_direct(result)


def inv(a):
    """Compute the inverse of a matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.inv(_get_np(a)))


def det(a):
    """Compute the determinant of a matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    result = np.linalg.det(_get_np(a))
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return _ndarray._from_np_direct(result)


def solve(a, b):
    """Solve a linear matrix equation."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    if not isinstance(b, _ndarray):
        b = _creation.asarray(b)
    return _ndarray._from_np_direct(np.linalg.solve(_get_np(a), _get_np(b)))


def eigh(a, UPLO='L'):
    """Eigenvalues and eigenvectors of a symmetric matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    w, v = np.linalg.eigh(_get_np(a), UPLO=UPLO)
    return _ndarray._from_np_direct(w), _ndarray._from_np_direct(v)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """Singular Value Decomposition."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    result = np.linalg.svd(_get_np(a), full_matrices=full_matrices,
                           compute_uv=compute_uv, hermitian=hermitian)
    if not compute_uv:
        return _ndarray._from_np_direct(result)
    u, s, vh = result
    return _ndarray._from_np_direct(u), _ndarray._from_np_direct(s), _ndarray._from_np_direct(vh)


def cholesky(a):
    """Cholesky decomposition."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.cholesky(_get_np(a)))


def matrix_power(a, n):
    """Raise a square matrix to the (integer) power n."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.matrix_power(_get_np(a), n))


def qr(a, mode='reduced'):
    """Compute the qr factorization."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    result = np.linalg.qr(_get_np(a), mode=mode)
    if mode == 'r':
        return _ndarray._from_np_direct(result)
    return _ndarray._from_np_direct(result[0]), _ndarray._from_np_direct(result[1])


def eig(a):
    """Compute eigenvalues and right eigenvectors."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    w, v = np.linalg.eig(_get_np(a))
    return _ndarray._from_np_direct(w), _ndarray._from_np_direct(v)


def eigvals(a):
    """Compute eigenvalues of a general matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.eigvals(_get_np(a)))


def eigvalsh(a, UPLO='L'):
    """Compute eigenvalues of a symmetric matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.eigvalsh(_get_np(a), UPLO=UPLO))


def cond(x, p=None):
    """Compute the condition number of a matrix."""
    if not isinstance(x, _ndarray):
        x = _creation.asarray(x)
    result = np.linalg.cond(_get_np(x), p=p)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return _ndarray._from_np_direct(result)


def matrix_rank(M, tol=None, hermitian=False):
    """Return matrix rank using SVD method."""
    if not isinstance(M, _ndarray):
        M = _creation.asarray(M)
    result = np.linalg.matrix_rank(_get_np(M), tol=tol, hermitian=hermitian)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return _ndarray._from_np_direct(result)


def slogdet(a):
    """Compute sign and natural log of the determinant."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    sign, logdet = np.linalg.slogdet(_get_np(a))
    if not isinstance(sign, np.ndarray):
        sign = np.array(sign)
    if not isinstance(logdet, np.ndarray):
        logdet = np.array(logdet)
    return _ndarray._from_np_direct(sign), _ndarray._from_np_direct(logdet)


def lstsq(a, b, rcond=None):
    """Return the least-squares solution."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    if not isinstance(b, _ndarray):
        b = _creation.asarray(b)
    x, residuals, rank, sv = np.linalg.lstsq(_get_np(a), _get_np(b), rcond=rcond)
    return (_ndarray._from_np_direct(x),
            _ndarray._from_np_direct(np.asarray(residuals)) if len(residuals) > 0 else _ndarray._from_np_direct(np.array([])),
            int(rank),
            _ndarray._from_np_direct(sv))


def pinv(a, rcond=1e-15, hermitian=False):
    """Compute the pseudo-inverse of a matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.pinv(_get_np(a), rcond=rcond, hermitian=hermitian))


# Re-export numpy's LinAlgError for API compatibility
LinAlgError = np.linalg.LinAlgError


def multi_dot(arrays, *, out=None):
    """Compute the dot product of two or more arrays in a single call."""
    np_arrays = []
    for a in arrays:
        if isinstance(a, _ndarray):
            np_arrays.append(_get_np(a))
        else:
            np_arrays.append(np.asarray(a))
    result = np.linalg.multi_dot(np_arrays)
    gpu_result = _ndarray._from_np_direct(result)
    if out is not None:
        out._np_data = gpu_result.get()
        out._buffer = None
        return out
    return gpu_result


def tensorsolve(a, b, axes=None):
    """Solve the tensor equation a x = b for x."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    if not isinstance(b, _ndarray):
        b = _creation.asarray(b)
    return _ndarray._from_np_direct(np.linalg.tensorsolve(_get_np(a), _get_np(b), axes=axes))


def tensorinv(a, ind=2):
    """Compute the tensor inverse of an array."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.tensorinv(_get_np(a), ind=ind))


def cross(a, b):
    """Compute the cross product of two vectors."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    if not isinstance(b, _ndarray):
        b = _creation.asarray(b)
    return _ndarray._from_np_direct(np.linalg.cross(_get_np(a), _get_np(b)))


def diagonal(a):
    """Return the diagonal of a matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.diagonal(_get_np(a)))


def matmul(a, b):
    """Matrix product of two arrays."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    if not isinstance(b, _ndarray):
        b = _creation.asarray(b)
    return _ndarray._from_np_direct(np.linalg.matmul(_get_np(a), _get_np(b)))


def matrix_norm(a, ord=None):
    """Compute the norm of a matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    result = np.linalg.matrix_norm(_get_np(a), ord=ord)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return _ndarray._from_np_direct(result)


def matrix_transpose(a):
    """Transpose the last two dimensions of a matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.matrix_transpose(_get_np(a)))


def outer(a, b):
    """Compute the outer product of two vectors."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    if not isinstance(b, _ndarray):
        b = _creation.asarray(b)
    return _ndarray._from_np_direct(np.linalg.outer(_get_np(a), _get_np(b)))


def svdvals(a):
    """Compute the singular values of a matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    return _ndarray._from_np_direct(np.linalg.svdvals(_get_np(a)))


def tensordot(a, b, axes=2):
    """Compute the tensor dot product of two arrays."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    if not isinstance(b, _ndarray):
        b = _creation.asarray(b)
    return _ndarray._from_np_direct(np.linalg.tensordot(_get_np(a), _get_np(b), axes=axes))


def trace(a):
    """Compute the trace of a matrix."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    result = np.linalg.trace(_get_np(a))
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return _ndarray._from_np_direct(result)


def vecdot(a, b):
    """Compute the vector dot product."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    if not isinstance(b, _ndarray):
        b = _creation.asarray(b)
    a_np = a._np_data if a._np_data is not None else _get_np(a)
    b_np = b._np_data if b._np_data is not None else _get_np(b)
    result = np.linalg.vecdot(a_np, b_np)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return _ndarray._from_np_direct(result)


def vector_norm(a, ord=2):
    """Compute the norm of a vector."""
    if not isinstance(a, _ndarray):
        a = _creation.asarray(a)
    result = np.linalg.vector_norm(_get_np(a), ord=ord)
    if not isinstance(result, np.ndarray):
        result = np.array(result)
    return _ndarray._from_np_direct(result)
