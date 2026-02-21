"""GPU-resident N-dimensional array backed by Metal buffers."""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple, Union

import numpy as np

from ._broadcasting import broadcast_shapes, needs_broadcast
from ._dtypes import (
    METAL_TYPE_NAMES,
    is_float_dtype,
    numpy_to_metal,
    resolve_dtype,
    result_dtype,
)
from ._kernel_cache import KernelCache
from ._metal_backend import MetalBackend

__all__ = ["ndarray"]


class _FlatIterator:
    """Simple flat iterator for ndarray.flat property."""

    def __init__(self, arr):
        self._data = arr.get().ravel()
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        val = self._data[self._index]
        self._index += 1
        return val

    def __len__(self):
        return len(self._data)


def _c_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return C-contiguous *element* strides for *shape*."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class ndarray:
    """CuPy-compatible GPU array for Apple Silicon via MetalGPU.

    Parameters are internal; prefer :mod:`macmetalpy.creation` functions.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        buffer,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        strides: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
        base: Optional[ndarray] = None,
    ) -> None:
        self._buffer = buffer
        self._shape = tuple(shape)
        self._dtype = np.dtype(dtype)
        self._strides = strides if strides is not None else _c_contiguous_strides(self._shape)
        self._offset = offset
        self._base = base

    # ------------------------------------------------------------------ classmethods
    @classmethod
    def _from_buffer(
        cls,
        buffer,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        strides: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
        base: Optional[ndarray] = None,
    ) -> ndarray:
        """Create an ndarray from an existing Metal buffer without copying."""
        return cls(buffer, shape, dtype, strides=strides, offset=offset, base=base)

    # ------------------------------------------------------------------ properties
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return math.prod(self._shape) if self._shape else 1

    @property
    def itemsize(self) -> int:
        return self._dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    @property
    def strides(self) -> Tuple[int, ...]:
        """Byte strides (element strides * itemsize) for CuPy compatibility."""
        return tuple(s * self.itemsize for s in self._strides)

    @property
    def T(self) -> ndarray:
        return self.transpose()

    @property
    def real(self) -> ndarray:
        """Real part of the array."""
        if self._dtype == np.complex64:
            from . import creation
            return creation.array(self.get().real)
        return self

    @property
    def imag(self) -> ndarray:
        """Imaginary part of the array."""
        if self._dtype == np.complex64:
            from . import creation
            return creation.array(self.get().imag)
        from . import creation
        return creation.array(np.zeros(self._shape, dtype=self._dtype))

    @property
    def flat(self):
        """Return a flat iterator over the array."""
        return _FlatIterator(self)

    @property
    def base(self):
        """Return the base object if memory is from some other object."""
        return self._base

    # ------------------------------------------------------------------ contiguity
    def _is_c_contiguous(self) -> bool:
        expected = _c_contiguous_strides(self._shape)
        return self._strides == expected and self._offset == 0

    def _ensure_contiguous(self) -> ndarray:
        """Return *self* if contiguous, otherwise a contiguous copy."""
        if self._is_c_contiguous():
            return self
        return self._contiguous_copy()

    def _contiguous_copy(self) -> ndarray:
        """Materialise a contiguous copy via CPU round-trip."""
        np_data = self.get()
        backend = MetalBackend()
        buf = backend.array_to_buffer(np.ascontiguousarray(np_data))
        return ndarray._from_buffer(buf, self._shape, self._dtype)

    # ------------------------------------------------------------------ data transfer
    def get(self) -> np.ndarray:
        """Transfer data to host and return as a NumPy array."""
        MetalBackend().synchronize()
        raw = np.array(self._buffer.contents, copy=False)
        if self._dtype == np.complex64:
            # Buffer stores float32 pairs; reconstruct complex64
            offset2 = self._offset * 2
            flat = raw[offset2 : offset2 + self.size * 2].copy()
            return flat.view(np.complex64).reshape(self._shape)
        if self._is_c_contiguous():
            result = raw[self._offset : self._offset + self.size].reshape(self._shape).copy()
            if self._dtype == np.float16:
                result = result.view(np.float16)
            return result
        # Non-contiguous view: use stride_tricks
        byte_strides = tuple(s * self.itemsize for s in self._strides)
        flat = raw[self._offset:]
        arr = np.lib.stride_tricks.as_strided(flat, shape=self._shape, strides=byte_strides)
        result = arr.copy()
        if self._dtype == np.float16:
            result = result.view(np.float16)
        return result

    def set(self, np_array: np.ndarray) -> None:
        """Write *np_array* into this array's buffer."""
        MetalBackend().synchronize()
        np_array = np.asarray(np_array, dtype=self._dtype)
        if self._dtype == np.complex64:
            float_view = np.ascontiguousarray(np_array).view(np.float32).ravel()
            offset2 = self._offset * 2
            self._buffer.contents[offset2 : offset2 + self.size * 2] = float_view
        else:
            flat = np_array.ravel()
            if self._dtype == np.float16:
                flat = flat.view(np.int16)
            self._buffer.contents[self._offset : self._offset + self.size] = flat

    # ------------------------------------------------------------------ shape ops
    def reshape(self, *new_shape) -> ndarray:
        """Return a reshaped view (or copy if not contiguous)."""
        # Accept reshape(2, 3) or reshape((2, 3))
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        else:
            new_shape = tuple(new_shape)

        # Handle -1 dimension
        if -1 in new_shape:
            neg_count = new_shape.count(-1)
            if neg_count > 1:
                raise ValueError("can only specify one unknown dimension")
            known = math.prod(s for s in new_shape if s != -1)
            if known == 0 or self.size % known != 0:
                raise ValueError(
                    f"cannot reshape array of size {self.size} into shape {new_shape}"
                )
            new_shape = tuple(self.size // known if s == -1 else s for s in new_shape)

        if math.prod(new_shape) != self.size:
            raise ValueError(
                f"cannot reshape array of size {self.size} into shape {new_shape}"
            )

        if self._is_c_contiguous():
            new_strides = _c_contiguous_strides(new_shape)
            return ndarray._from_buffer(
                self._buffer, new_shape, self._dtype,
                strides=new_strides, offset=self._offset, base=self._base or self,
            )
        arr = self._contiguous_copy()
        return arr.reshape(new_shape)

    def transpose(self, axes=None) -> ndarray:
        """Return a view with permuted axes."""
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))
        else:
            axes = tuple(axes)
        # Complex64 is stored as float32 pairs; stride-based views break
        # the pair structure, so fall back to CPU transpose.
        if self._dtype == np.complex64:
            from . import creation
            return creation.array(np.transpose(self.get(), axes))
        new_shape = tuple(self._shape[a] for a in axes)
        new_strides = tuple(self._strides[a] for a in axes)
        return ndarray._from_buffer(
            self._buffer, new_shape, self._dtype,
            strides=new_strides, offset=self._offset, base=self._base or self,
        )

    def flatten(self) -> ndarray:
        """Return a 1-D contiguous copy."""
        arr = self._ensure_contiguous()
        return ndarray._from_buffer(
            arr._buffer, (self.size,), self._dtype,
            strides=(1,), offset=arr._offset, base=arr._base or arr,
        )

    def ravel(self) -> ndarray:
        """Return a 1-D array; view if contiguous, copy otherwise."""
        if self._is_c_contiguous():
            return ndarray._from_buffer(
                self._buffer, (self.size,), self._dtype,
                strides=(1,), offset=self._offset, base=self._base or self,
            )
        return self.flatten()

    def squeeze(self, axis=None) -> ndarray:
        """Remove dimensions of size 1."""
        if axis is None:
            new_shape = tuple(s for s in self._shape if s != 1)
            new_strides = tuple(
                st for s, st in zip(self._shape, self._strides) if s != 1
            )
        else:
            if isinstance(axis, int):
                axis = (axis,)
            axis = tuple(a % self.ndim for a in axis)
            for a in axis:
                if self._shape[a] != 1:
                    raise ValueError(
                        f"cannot squeeze axis {a} with size {self._shape[a]}"
                    )
            new_shape = tuple(s for i, s in enumerate(self._shape) if i not in axis)
            new_strides = tuple(
                st for i, st in enumerate(self._strides) if i not in axis
            )
        if not new_shape:
            new_shape = ()
            new_strides = ()
        return ndarray._from_buffer(
            self._buffer, new_shape, self._dtype,
            strides=new_strides, offset=self._offset, base=self._base or self,
        )

    def expand_dims(self, axis: int) -> ndarray:
        """Insert a size-1 dimension at *axis*."""
        ndim_new = self.ndim + 1
        if axis < 0:
            axis = ndim_new + axis
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        new_shape.insert(axis, 1)
        # stride for a size-1 dim doesn't matter; use 0
        new_strides.insert(axis, 0)
        return ndarray._from_buffer(
            self._buffer, tuple(new_shape), self._dtype,
            strides=tuple(new_strides), offset=self._offset, base=self._base or self,
        )

    # ------------------------------------------------------------------ indexing
    def __getitem__(self, key):
        # MVP: CPU round-trip
        from . import creation

        np_data = self.get()
        result = np_data[key]
        if not isinstance(result, np.ndarray):
            # Wrap scalar as 0-d ndarray (CuPy behavior)
            result = np.array(result)
        return creation.array(result, dtype=self._dtype)

    def __setitem__(self, key, value):
        np_data = self.get()
        if isinstance(value, ndarray):
            value = value.get()
        np_data[key] = value
        self.set(np_data)

    # ------------------------------------------------------------------ operators
    def _binary_op(self, other, op_name: str) -> ndarray:
        """Execute a binary elementwise GPU kernel."""
        from . import creation

        backend = MetalBackend()
        cache = KernelCache()

        # Wrap scalar — use numpy-style promotion for Python float scalars
        if isinstance(other, (int, float)):
            if isinstance(other, float) and np.issubdtype(self._dtype, np.integer):
                # Float scalar with integer array: promote to float (match numpy)
                scalar_dtype = resolve_dtype(np.float64)
            else:
                scalar_dtype = self._dtype
            other = creation.full(self._shape, other, dtype=scalar_dtype)
        elif not isinstance(other, ndarray):
            other = creation.array(other, dtype=self._dtype)

        # Determine result dtype
        rdtype = result_dtype(self._dtype, other._dtype)

        # CPU fallback for complex64 (no Metal shader support)
        if rdtype == np.complex64:
            _NP_OPS = {"add_op": np.add, "sub_op": np.subtract, "mul_op": np.multiply,
                       "div_op": np.true_divide, "pow_op": np.power,
                       "mod_op": np.mod, "min_op": np.minimum, "max_op": np.maximum}
            a_np = self.get().astype(np.complex64) if self._dtype != np.complex64 else self.get()
            b_np = other.get().astype(np.complex64) if other._dtype != np.complex64 else other.get()
            # Handle broadcasting
            if a_np.shape != b_np.shape:
                out_shape = np.broadcast_shapes(a_np.shape, b_np.shape)
                a_np = np.broadcast_to(a_np, out_shape)
                b_np = np.broadcast_to(b_np, out_shape)
            op_func = _NP_OPS.get(op_name)
            if op_func is not None:
                result_np = op_func(a_np, b_np)
            else:
                result_np = a_np
            return creation.array(result_np)

        # Cast operands to result dtype if needed
        a = self.astype(rdtype) if self._dtype != rdtype else self
        b = other.astype(rdtype) if other._dtype != rdtype else other

        # Broadcasting (MVP: CPU expand)
        if needs_broadcast(a._shape, b._shape):
            out_shape = broadcast_shapes(a._shape, b._shape)
            a_np = np.broadcast_to(a.get(), out_shape)
            b_np = np.broadcast_to(b.get(), out_shape)
            a = creation.array(np.ascontiguousarray(a_np), dtype=rdtype)
            b = creation.array(np.ascontiguousarray(b_np), dtype=rdtype)

        a = a._ensure_contiguous()
        b = b._ensure_contiguous()

        shader = cache.get_shader("elementwise", rdtype)
        out_buf = backend.create_buffer(a.size, rdtype)
        backend.execute_kernel(shader, op_name, a.size, [a._buffer, b._buffer, out_buf])
        return ndarray._from_buffer(out_buf, a._shape, rdtype)

    def _unary_op(self, op_name: str) -> ndarray:
        """Execute a unary elementwise GPU kernel."""
        # CPU fallback for complex64 (no Metal shader support)
        if self._dtype == np.complex64:
            from . import creation
            _NP_OPS = {"neg_op": np.negative, "abs_op": np.abs,
                       "sqrt_op": np.sqrt, "exp_op": np.exp, "log_op": np.log,
                       "sin_op": np.sin, "cos_op": np.cos, "tan_op": np.tan,
                       "ceil_op": np.ceil, "floor_op": np.floor, "sign_op": np.sign,
                       "square_op": np.square}
            a_np = self.get()
            op_func = _NP_OPS.get(op_name)
            if op_func is not None:
                return creation.array(op_func(a_np))
            return creation.array(a_np)

        backend = MetalBackend()
        cache = KernelCache()
        a = self._ensure_contiguous()
        shader = cache.get_shader("elementwise", a._dtype)
        out_buf = backend.create_buffer(a.size, a._dtype)
        backend.execute_kernel(shader, op_name, a.size, [a._buffer, out_buf])
        return ndarray._from_buffer(out_buf, a._shape, a._dtype)

    def _unary_predicate_op(self, op_name: str) -> ndarray:
        """Execute a unary predicate GPU kernel (typed input → bool output)."""
        # CPU fallback for complex64 (no Metal shader support)
        if self._dtype == np.complex64:
            from . import creation
            _PRED_OPS = {"isnan_op": np.isnan, "isinf_op": np.isinf, "isfinite_op": np.isfinite}
            op_func = _PRED_OPS.get(op_name)
            if op_func is not None:
                return creation.array(op_func(self.get()))
            return creation.array(np.zeros(self._shape, dtype=bool))

        backend = MetalBackend()
        cache = KernelCache()
        a = self._ensure_contiguous()
        shader = cache.get_shader("predicate", a._dtype)
        out_buf = backend.create_buffer(a.size, np.int32)
        backend.execute_kernel(shader, op_name, a.size, [a._buffer, out_buf])
        int_arr = ndarray._from_buffer(out_buf, a._shape, np.int32)
        return int_arr.astype(np.bool_)

    # Arithmetic
    def __add__(self, other):
        return self._binary_op(other, "add_op")

    def __radd__(self, other):
        return self._binary_op(other, "add_op")

    def __sub__(self, other):
        return self._binary_op(other, "sub_op")

    def __rsub__(self, other):
        from . import creation
        if isinstance(other, (int, float)):
            if isinstance(other, float) and np.issubdtype(self._dtype, np.integer):
                scalar_dtype = resolve_dtype(np.float64)
            else:
                scalar_dtype = self._dtype
            other = creation.full(self._shape, other, dtype=scalar_dtype)
        return other._binary_op(self, "sub_op")

    def __mul__(self, other):
        return self._binary_op(other, "mul_op")

    def __rmul__(self, other):
        return self._binary_op(other, "mul_op")

    def __truediv__(self, other):
        # True division always returns float (match NumPy)
        if np.issubdtype(self._dtype, np.integer):
            from . import creation
            lhs = creation.array(self.get().astype(np.float32))
            if isinstance(other, ndarray) and np.issubdtype(other._dtype, np.integer):
                other = creation.array(other.get().astype(np.float32))
            return lhs._binary_op(other, "div_op")
        return self._binary_op(other, "div_op")

    def __rtruediv__(self, other):
        from . import creation
        if isinstance(other, (int, float)):
            other = creation.full(self._shape, other, dtype=np.float32 if np.issubdtype(self._dtype, np.integer) else self._dtype)
        lhs = other
        rhs = self
        if isinstance(rhs, ndarray) and np.issubdtype(rhs._dtype, np.integer):
            rhs = creation.array(rhs.get().astype(np.float32))
        if isinstance(lhs, ndarray) and np.issubdtype(lhs._dtype, np.integer):
            lhs = creation.array(lhs.get().astype(np.float32))
        return lhs._binary_op(rhs, "div_op")

    def __pow__(self, other):
        return self._binary_op(other, "pow_op")

    def __rpow__(self, other):
        from . import creation
        if isinstance(other, (int, float)):
            other = creation.full(self._shape, other, dtype=self._dtype)
        return other._binary_op(self, "pow_op")

    def __neg__(self):
        return self._unary_op("neg_op")

    def __abs__(self):
        return self._unary_op("abs_op")

    def __matmul__(self, other):
        if not isinstance(other, ndarray):
            from . import creation
            other = creation.array(other, dtype=self._dtype)

        rdtype = result_dtype(self._dtype, other._dtype)
        a = self.astype(rdtype) if self._dtype != rdtype else self
        b = other.astype(rdtype) if other._dtype != rdtype else other
        a = a._ensure_contiguous()
        b = b._ensure_contiguous()

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul requires 2-D arrays")
        M, K = a._shape
        K2, N = b._shape
        if K != K2:
            raise ValueError(
                f"matmul shape mismatch: ({M},{K}) @ ({K2},{N})"
            )

        backend = MetalBackend()
        cache = KernelCache()
        shader = cache.get_shader("matmul", rdtype)

        dims_np = np.array([M, N, K], dtype=np.uint32)
        dims_buf = backend.array_to_buffer(dims_np)
        out_buf = backend.create_buffer(M * N, rdtype)

        from ._metal_backend import MetalSize
        grid = MetalSize(N, M, 1)
        backend.execute_kernel(shader, "matmul_op", grid, [a._buffer, b._buffer, out_buf, dims_buf])
        return ndarray._from_buffer(out_buf, (M, N), rdtype)

    # ------------------------------------------------------------------ comparison
    def _comparison_op(self, other, op_name: str) -> ndarray:
        """Execute a comparison GPU kernel, returning a bool-typed result."""
        from . import creation

        backend = MetalBackend()
        cache = KernelCache()

        # Wrap scalar
        if isinstance(other, (int, float)):
            other = creation.full(self._shape, other, dtype=self._dtype)
        elif not isinstance(other, ndarray):
            other = creation.array(other, dtype=self._dtype)

        # Promote operands to common type for comparison
        rdtype = result_dtype(self._dtype, other._dtype)

        # CPU fallback for complex64 (no Metal shader support)
        if rdtype == np.complex64:
            _CMP_OPS = {"eq_op": np.equal, "ne_op": np.not_equal}
            op_func = _CMP_OPS.get(op_name)
            if op_func is None:
                raise TypeError("'>' not supported between instances of 'complex'")
            a_np = self.get().astype(np.complex64) if self._dtype != np.complex64 else self.get()
            b_np = other.get().astype(np.complex64) if other._dtype != np.complex64 else other.get()
            # Handle broadcasting
            if a_np.shape != b_np.shape:
                out_shape = np.broadcast_shapes(a_np.shape, b_np.shape)
                a_np = np.broadcast_to(a_np, out_shape)
                b_np = np.broadcast_to(b_np, out_shape)
            return creation.array(op_func(a_np, b_np))

        a = self.astype(rdtype) if self._dtype != rdtype else self
        b = other.astype(rdtype) if other._dtype != rdtype else other

        # Broadcasting
        if needs_broadcast(a._shape, b._shape):
            out_shape = broadcast_shapes(a._shape, b._shape)
            a_np = np.broadcast_to(a.get(), out_shape)
            b_np = np.broadcast_to(b.get(), out_shape)
            a = creation.array(np.ascontiguousarray(a_np), dtype=rdtype)
            b = creation.array(np.ascontiguousarray(b_np), dtype=rdtype)

        a = a._ensure_contiguous()
        b = b._ensure_contiguous()

        shader = cache.get_shader("comparison", rdtype)
        out_buf = backend.create_buffer(a.size, np.int32)
        backend.execute_kernel(shader, op_name, a.size, [a._buffer, b._buffer, out_buf])

        int_arr = ndarray._from_buffer(out_buf, a._shape, np.int32)
        return int_arr.astype(np.bool_)

    def _boolean_op(self, other, op_name: str) -> ndarray:
        """Execute a boolean logic GPU kernel on bool arrays."""
        from . import creation

        backend = MetalBackend()
        cache = KernelCache()

        if not isinstance(other, ndarray):
            other = creation.array(other, dtype=np.bool_)

        # Convert both to int32 for GPU
        a = self.astype(np.int32)._ensure_contiguous()
        b = other.astype(np.int32)._ensure_contiguous()

        shader = cache.get_shader("boolean", np.int32)
        out_buf = backend.create_buffer(a.size, np.int32)
        backend.execute_kernel(shader, op_name, a.size, [a._buffer, b._buffer, out_buf])

        int_arr = ndarray._from_buffer(out_buf, a._shape, np.int32)
        return int_arr.astype(np.bool_)

    # Comparison operators
    def __lt__(self, other):
        return self._comparison_op(other, "lt_op")

    def __le__(self, other):
        return self._comparison_op(other, "le_op")

    def __gt__(self, other):
        return self._comparison_op(other, "gt_op")

    def __ge__(self, other):
        return self._comparison_op(other, "ge_op")

    def __eq__(self, other):
        return self._comparison_op(other, "eq_op")

    def __ne__(self, other):
        return self._comparison_op(other, "ne_op")

    # Boolean logic operators
    def __and__(self, other):
        return self._boolean_op(other, "and_op")

    def __or__(self, other):
        return self._boolean_op(other, "or_op")

    def __invert__(self):
        backend = MetalBackend()
        cache = KernelCache()
        a = self.astype(np.int32)._ensure_contiguous()
        shader = cache.get_shader("boolean", np.int32)
        out_buf = backend.create_buffer(a.size, np.int32)
        backend.execute_kernel(shader, "not_op", a.size, [a._buffer, out_buf])
        int_arr = ndarray._from_buffer(out_buf, a._shape, np.int32)
        return int_arr.astype(np.bool_)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            from . import creation
            other = creation.full(self._shape, self._dtype.type(other), dtype=self._dtype)
        result = self._binary_op(other, "add_op")
        self.set(result.astype(self._dtype).get())
        return self

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            from . import creation
            other = creation.full(self._shape, self._dtype.type(other), dtype=self._dtype)
        result = self._binary_op(other, "sub_op")
        self.set(result.astype(self._dtype).get())
        return self

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            from . import creation
            other = creation.full(self._shape, self._dtype.type(other), dtype=self._dtype)
        result = self._binary_op(other, "mul_op")
        self.set(result.astype(self._dtype).get())
        return self

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            from . import creation
            other = creation.full(self._shape, self._dtype.type(other), dtype=self._dtype)
        result = self._binary_op(other, "div_op")
        self.set(result.astype(self._dtype).get())
        return self

    def __ipow__(self, other):
        if isinstance(other, (int, float)):
            from . import creation
            other = creation.full(self._shape, self._dtype.type(other), dtype=self._dtype)
        result = self._binary_op(other, "pow_op")
        self.set(result.astype(self._dtype).get())
        return self

    def __ior__(self, other):
        result = self._boolean_op(other, "or_op")
        self.set(result.get())
        self._dtype = np.dtype(np.bool_)
        return self

    # ------------------------------------------------------------------ reductions
    def sum(self, axis=None, keepdims=False):
        """Sum of array elements."""
        if axis is not None:
            return self._reduce_axis("reduce_sum_axis", axis, keepdims)
        result = self._reduce("reduce_sum")
        if keepdims:
            result = result.reshape((1,) * self.ndim)
        return result

    def max(self, axis=None, keepdims=False):
        """Maximum of array elements."""
        if axis is not None:
            return self._reduce_axis("reduce_max_axis", axis, keepdims)
        result = self._reduce("reduce_max")
        if keepdims:
            result = result.reshape((1,) * self.ndim)
        return result

    def min(self, axis=None, keepdims=False):
        """Minimum of array elements."""
        if axis is not None:
            return self._reduce_axis("reduce_min_axis", axis, keepdims)
        result = self._reduce("reduce_min")
        if keepdims:
            result = result.reshape((1,) * self.ndim)
        return result

    def mean(self, axis=None, keepdims=False):
        """Mean of array elements."""
        from . import creation
        # For integer dtypes, cast to float32 first (matches NumPy behavior)
        arr = self
        if not np.issubdtype(self._dtype, np.floating) and self._dtype != np.complex64:
            arr = creation.array(self.get().astype(np.float32))
        if axis is not None:
            s = arr._reduce_axis("reduce_sum_axis", axis, keepdims=False)
            count = float(arr._shape[axis % arr.ndim if axis < 0 else axis])
            count_arr = creation.full(s._shape, count, dtype=s._dtype)
            result = s._binary_op(count_arr, "div_op")
            if keepdims:
                shape = list(self._shape)
                shape[axis % self.ndim if axis < 0 else axis] = 1
                result = result.reshape(tuple(shape))
            return result
        s = arr._reduce("reduce_sum")
        count_arr = creation.full((), float(arr.size), dtype=arr._dtype)
        result = s._binary_op(count_arr, "div_op")
        if keepdims:
            result = result.reshape((1,) * self.ndim)
        return result

    def std(self, axis=None, keepdims=False):
        """Standard deviation of array elements."""
        m = self.mean(axis=axis, keepdims=True)
        diff = self._binary_op(m, "sub_op")
        sq = diff._unary_op("square_op")
        result = sq.mean(axis=axis, keepdims=keepdims)
        return result._unary_op("sqrt_op")

    def var(self, axis=None, keepdims=False):
        """Variance of array elements."""
        m = self.mean(axis=axis, keepdims=True)
        diff = self._binary_op(m, "sub_op")
        sq = diff._unary_op("square_op")
        return sq.mean(axis=axis, keepdims=keepdims)

    def prod(self, axis=None, keepdims=False):
        """Product of array elements."""
        if axis is not None:
            return self._reduce_axis("reduce_prod_axis", axis, keepdims)
        # For axis=None, flatten and use axis reduction with outer=1
        from . import creation
        flat = self.ravel()._ensure_contiguous()
        backend = MetalBackend()
        cache = KernelCache()
        shader = cache.get_shader("axis_reduction", flat._dtype)
        dims_buf = backend.array_to_buffer(np.array([1, flat.size], dtype=np.uint32))
        out_buf = backend.create_buffer(1, flat._dtype)
        backend.execute_kernel(shader, "reduce_prod_axis", 1, [flat._buffer, out_buf, dims_buf])
        result = ndarray._from_buffer(out_buf, (), flat._dtype)
        if keepdims:
            result = result.reshape((1,) * self.ndim)
        return result

    def cumsum(self, axis=None):
        """Cumulative sum of array elements."""
        return self._prefix_scan("prefix_sum", axis)

    def cumprod(self, axis=None):
        """Cumulative product of array elements."""
        return self._prefix_scan("prefix_prod", axis)

    def any(self, axis=None, keepdims=False):
        """Test whether any array element evaluates to True."""
        if axis is not None:
            np_data = self.get()
            from . import creation
            return creation.array(np_data.any(axis=axis, keepdims=keepdims))
        return bool(self.get().any())

    def all(self, axis=None, keepdims=False):
        """Test whether all array elements evaluate to True."""
        if axis is not None:
            np_data = self.get()
            from . import creation
            return creation.array(np_data.all(axis=axis, keepdims=keepdims))
        return bool(self.get().all())

    def _reduce(self, kernel_name: str):
        """Run a full-array GPU reduction and return a 0-d ndarray."""
        from . import creation

        backend = MetalBackend()
        cache = KernelCache()
        a = self._ensure_contiguous()
        shader = cache.get_shader("reduction", a._dtype)

        # Apple Silicon uses threadgroup size = maxTotalThreadsPerThreadgroup = 1024
        # for 1-D dispatch.  We must dispatch full threadgroups so the
        # parallel reduction sees the correct threads_per_threadgroup.
        TGROUP = 1024
        n = a.size
        num_groups = (n + TGROUP - 1) // TGROUP
        grid = num_groups * TGROUP  # round up → full threadgroups

        n_buf = backend.array_to_buffer(np.array([n], dtype=np.uint32))
        out_buf = backend.create_buffer(max(num_groups, 1), a._dtype)

        backend.execute_kernel(shader, kernel_name, grid, [a._buffer, out_buf, n_buf])

        # Iteratively reduce until we have a single value
        while num_groups > 1:
            in_buf = out_buf
            n = num_groups
            num_groups = (n + TGROUP - 1) // TGROUP
            grid = num_groups * TGROUP
            n_buf = backend.array_to_buffer(np.array([n], dtype=np.uint32))
            out_buf = backend.create_buffer(max(num_groups, 1), a._dtype)
            backend.execute_kernel(shader, kernel_name, n, [in_buf, out_buf, n_buf])

        return ndarray._from_buffer(out_buf, (), a._dtype)

    def _reduce_axis(self, kernel_name: str, axis: int, keepdims: bool = False, out_dtype=None):
        """GPU axis reduction using axis_reduction_shader."""
        from . import creation
        axis = axis % self.ndim if axis < 0 else axis

        # Move reduction axis to last position
        axes = list(range(self.ndim))
        axes.append(axes.pop(axis))
        transposed = self.transpose(axes)

        # Reshape to 2D (outer, inner) via contiguous copy
        inner = self._shape[axis]
        outer = self.size // inner
        flat_np = np.ascontiguousarray(transposed.get().reshape(outer, inner))
        flat = creation.array(flat_np, dtype=self._dtype)._ensure_contiguous()

        backend = MetalBackend()
        cache = KernelCache()
        shader = cache.get_shader("axis_reduction", flat._dtype)
        dims_buf = backend.array_to_buffer(np.array([outer, inner], dtype=np.uint32))

        _out_dtype = out_dtype if out_dtype is not None else flat._dtype
        out_buf = backend.create_buffer(outer, _out_dtype)
        backend.execute_kernel(shader, kernel_name, outer, [flat._buffer, out_buf, dims_buf])

        # Compute output shape
        out_shape = list(self._shape)
        if keepdims:
            out_shape[axis] = 1
        else:
            out_shape.pop(axis)
        out_shape = tuple(out_shape) if out_shape else ()

        return ndarray._from_buffer(out_buf, out_shape, _out_dtype)

    def _prefix_scan(self, kernel_name: str, axis=None):
        """GPU prefix scan (cumsum/cumprod) using axis_reduction_shader."""
        from . import creation
        if axis is None:
            flat = self.ravel()._ensure_contiguous()
            backend = MetalBackend()
            cache = KernelCache()
            shader = cache.get_shader("axis_reduction", flat._dtype)
            dims_buf = backend.array_to_buffer(np.array([1, flat.size], dtype=np.uint32))
            out_buf = backend.create_buffer(flat.size, flat._dtype)
            backend.execute_kernel(shader, kernel_name, 1, [flat._buffer, out_buf, dims_buf])
            return ndarray._from_buffer(out_buf, (flat.size,), flat._dtype)

        axis = axis % self.ndim if axis < 0 else axis
        axes = list(range(self.ndim))
        axes.append(axes.pop(axis))
        transposed = self.transpose(axes)
        inner = self._shape[axis]
        outer = self.size // inner
        flat_np = np.ascontiguousarray(transposed.get().reshape(outer, inner))
        flat = creation.array(flat_np, dtype=self._dtype)._ensure_contiguous()

        backend = MetalBackend()
        cache = KernelCache()
        shader = cache.get_shader("axis_reduction", flat._dtype)
        dims_buf = backend.array_to_buffer(np.array([outer, inner], dtype=np.uint32))
        out_buf = backend.create_buffer(flat.size, flat._dtype)
        backend.execute_kernel(shader, kernel_name, outer, [flat._buffer, out_buf, dims_buf])

        # Reshape back: (outer, inner) → transposed shape → inverse transpose
        result_2d = ndarray._from_buffer(out_buf, (outer, inner), flat._dtype)
        result_nd = creation.array(result_2d.get().reshape(transposed._shape))
        inv_axes = [0] * len(axes)
        for i, a in enumerate(axes):
            inv_axes[a] = i
        return result_nd.transpose(inv_axes)

    # ------------------------------------------------------------------ type casting
    def astype(self, dtype) -> ndarray:
        """Return array cast to *dtype* (CPU round-trip)."""
        dtype = np.dtype(dtype)
        if dtype == self._dtype:
            return self
        np_data = self.get().astype(dtype)
        backend = MetalBackend()
        buf = backend.array_to_buffer(np.ascontiguousarray(np_data))
        return ndarray._from_buffer(buf, self._shape, dtype)

    # ------------------------------------------------------------------ copy
    def copy(self) -> ndarray:
        """Return a deep copy."""
        np_data = self.get()
        backend = MetalBackend()
        buf = backend.array_to_buffer(np.ascontiguousarray(np_data))
        return ndarray._from_buffer(buf, self._shape, self._dtype)

    # ------------------------------------------------------------------ gap methods
    def item(self, *args):
        """Extract a Python scalar from the array."""
        return self.get().item(*args)

    def tolist(self):
        """Return the array as a nested list of Python scalars."""
        return self.get().tolist()

    def fill(self, value):
        """Fill the array with a scalar value (in-place)."""
        self.set(np.full(self._shape, value, dtype=self._dtype))

    def round(self, decimals=0):
        """Round to the given number of decimals."""
        from . import math_ops
        return math_ops.around(self, decimals=decimals)

    def clip(self, min=None, max=None):
        """Clip values to [min, max]."""
        from . import math_ops
        return math_ops.clip(self, min, max)

    def conj(self):
        """Return the complex conjugate."""
        if self._dtype == np.complex64:
            from . import creation
            return creation.array(np.conj(self.get()))
        return self.copy()

    def diagonal(self, offset=0):
        """Return specified diagonals."""
        from . import math_ops
        return math_ops.diagonal(self, offset=offset)

    def trace(self, offset=0):
        """Return the sum along diagonals."""
        from . import math_ops
        return math_ops.trace(self, offset=offset)

    def repeat(self, repeats, axis=None):
        """Repeat elements of the array."""
        from . import manipulation
        return manipulation.repeat(self, repeats, axis=axis)

    def take(self, indices, axis=None):
        """Take elements from the array."""
        from . import indexing
        return indexing.take(self, indices, axis=axis)

    def put(self, indices, values):
        """Set a.flat[n] = values[n] for all n in indices (in-place)."""
        from . import indexing
        indexing.put(self, indices, values)

    def choose(self, choices):
        """Construct an array from an index array and choices."""
        from . import indexing
        return indexing.choose(self, choices)

    def compress(self, condition, axis=None):
        """Return selected slices along given axis."""
        from . import indexing
        return indexing.compress(condition, self, axis=axis)

    def searchsorted(self, v, side='left'):
        """Find indices where elements should be inserted."""
        from . import sorting as sorting_mod
        return sorting_mod.searchsorted(self, v, side=side)

    def nonzero(self):
        """Return the indices of non-zero elements."""
        from . import indexing
        return indexing.nonzero(self)

    def sort(self, axis=-1):
        """Sort the array in-place."""
        from . import sorting as sorting_mod
        result = sorting_mod.sort(self, axis=axis)
        self.set(result.get())

    def argsort(self, axis=-1):
        """Return the indices that would sort the array."""
        from . import sorting as sorting_mod
        return sorting_mod.argsort(self, axis=axis)

    def argmax(self, axis=None):
        """Return indices of the maximum values."""
        from . import reductions as reductions_mod
        return reductions_mod.argmax(self, axis=axis)

    def argmin(self, axis=None):
        """Return indices of the minimum values."""
        from . import reductions as reductions_mod
        return reductions_mod.argmin(self, axis=axis)

    def ptp(self, axis=None):
        """Peak-to-peak (max - min) along an axis."""
        from . import reductions as reductions_mod
        return reductions_mod.ptp(self, axis=axis)

    def partition(self, kth, axis=-1):
        """Partially sort the array in-place."""
        from . import sorting as sorting_mod
        result = sorting_mod.partition(self, kth, axis=axis)
        self.set(result.get())

    def argpartition(self, kth, axis=-1):
        """Return indices that would partition the array."""
        from . import sorting as sorting_mod
        return sorting_mod.argpartition(self, kth, axis=axis)

    def tobytes(self):
        """Return the array data as bytes."""
        return self.get().tobytes()

    def view(self, dtype):
        """View the array with a different dtype (reinterpret buffer)."""
        from . import creation
        dtype = np.dtype(dtype)
        np_data = self.get()
        viewed = np_data.view(dtype)
        return creation.array(viewed)

    # ------------------------------------------------------------------ repr
    def __repr__(self) -> str:
        np_data = self.get()
        data_str = repr(np_data)
        return f"macmetalpy.ndarray({data_str}, dtype={self._dtype})"

    def __str__(self) -> str:
        return str(self.get())

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self._shape[0]

    def __float__(self) -> float:
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return float(self.get().ravel()[0])

    def __int__(self) -> int:
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return int(self.get().ravel()[0])

    def __bool__(self) -> bool:
        if self.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous."
            )
        return bool(self.get().ravel()[0])

    def __pos__(self):
        return self.copy()

    def __complex__(self) -> complex:
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return complex(self.get().ravel()[0])

    def __index__(self) -> int:
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        if not np.issubdtype(self._dtype, np.integer):
            raise TypeError("only integer arrays can be used as indices")
        return int(self.get().ravel()[0])

    def __contains__(self, item) -> bool:
        return bool(item in self.get())

    # Floor division //
    def __floordiv__(self, other):
        from . import ufunc_ops
        from . import creation
        if not isinstance(other, ndarray):
            other = creation.array(np.asarray(other), dtype=self._dtype)
        return ufunc_ops.floor_divide(self, other)

    def __rfloordiv__(self, other):
        from . import ufunc_ops
        from . import creation
        if not isinstance(other, ndarray):
            other = creation.array(np.asarray(other), dtype=self._dtype)
        return ufunc_ops.floor_divide(other, self)

    def __ifloordiv__(self, other):
        from . import ufunc_ops
        from . import creation
        if not isinstance(other, ndarray):
            other = creation.array(np.asarray(other), dtype=self._dtype)
        result = ufunc_ops.floor_divide(self, other)
        self.set(result.astype(self._dtype).get())
        return self

    # Modulo %
    def __mod__(self, other):
        from . import math_ops
        from . import creation
        if not isinstance(other, ndarray):
            other = creation.array(np.asarray(other), dtype=self._dtype)
        return math_ops.mod(self, other)

    def __rmod__(self, other):
        from . import math_ops
        from . import creation
        if not isinstance(other, ndarray):
            other = creation.array(np.asarray(other), dtype=self._dtype)
        return math_ops.mod(other, self)

    def __imod__(self, other):
        from . import math_ops
        from . import creation
        if not isinstance(other, ndarray):
            other = creation.array(np.asarray(other), dtype=self._dtype)
        result = math_ops.mod(self, other)
        self.set(result.astype(self._dtype).get())
        return self

    # Left shift <<
    def __lshift__(self, other):
        from . import bitwise_ops
        return bitwise_ops.left_shift(self, other)

    def __rlshift__(self, other):
        from . import bitwise_ops
        return bitwise_ops.left_shift(other, self)

    # Right shift >>
    def __rshift__(self, other):
        from . import bitwise_ops
        return bitwise_ops.right_shift(self, other)

    def __rrshift__(self, other):
        from . import bitwise_ops
        return bitwise_ops.right_shift(other, self)

    # In-place &=
    def __iand__(self, other):
        result = self._boolean_op(other, "and_op")
        self.set(result.get())
        self._dtype = np.dtype(np.bool_)
        return self

    # XOR ^
    def __xor__(self, other):
        return self._boolean_op(other, "xor_op")

    # In-place ^=
    def __ixor__(self, other):
        result = self._boolean_op(other, "xor_op")
        self.set(result.get())
        self._dtype = np.dtype(np.bool_)
        return self

    # Reverse matmul
    def __rmatmul__(self, other):
        if not isinstance(other, ndarray):
            from . import creation
            other = creation.array(other, dtype=self._dtype)
        return other.__matmul__(self)

    # In-place matmul
    def __imatmul__(self, other):
        result = self.__matmul__(other)
        self.set(result.get())
        return self

    # divmod
    def __divmod__(self, other):
        return (self // other, self % other)
