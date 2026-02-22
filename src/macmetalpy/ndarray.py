"""GPU-resident N-dimensional array backed by Metal buffers."""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple, Union

import numpy as np

from ._broadcasting import broadcast_shapes, needs_broadcast
from ._dtypes import (
    METAL_TYPE_NAMES,
    _PASSTHROUGH_DTYPES,
    is_float_dtype,
    numpy_to_metal,
    resolve_dtype,
    result_dtype,
)
from ._kernel_cache import (
    KernelCache,
    get_astype_shader,
    get_broadcast_shader,
    get_copy_shader,
    get_offset_copy_shader,
    get_scalar_copy_shader,
    get_scalar_write_shader,
    get_strided_copy_shader,
    get_strided_write_shader,
)
from ._metal_backend import MetalBackend

try:
    from . import _accelerator
except ImportError:
    _accelerator = None

__all__ = ["ndarray"]

# Opt 2: arrays below this element count run on CPU via NumPy
_GPU_THRESHOLD = 8192  # default for heavy ops (transcendentals, power, mod)
_GPU_THRESHOLD_LIGHT = 262144  # 256K for light ops (simple arithmetic, sqrt, etc.)
_GPU_THRESHOLD_MEMORY = 4194304  # 4M for pure memory ops — GPU can rarely beat CPU SIMD

# Ops that are pure memory operations (copy/negate/compare) — CPU SIMD is almost
# always faster because GPU dispatch overhead dominates.
_MEMORY_UNARY_OPS = frozenset({
    "neg_op", "abs_op", "ceil_op", "floor_op", "sign_op", "square_op",
    "rint_op", "trunc_op", "reciprocal_op", "degrees_op", "radians_op",
    "negative_op", "positive_op", "sqrt_op",
})

# Ops where GPU kernel does minimal per-element work — NumPy SIMD is faster
# until arrays are large enough to amortise dispatch overhead.
_LIGHT_UNARY_OPS = frozenset({
    "sin_op", "cos_op", "tan_op", "exp_op", "log_op",
    "asin_op", "acos_op", "atan_op",
    "sinh_op", "cosh_op", "tanh_op",
    "asinh_op", "acosh_op", "atanh_op",
    "exp2_op", "expm1_op", "log2_op", "log10_op", "log1p_op",
    "cbrt_op", "erf_op",
}) | _MEMORY_UNARY_OPS
_LIGHT_BINARY_OPS = frozenset({
    "add_op", "sub_op", "mul_op", "div_op", "min_op", "max_op",
    "copysign_op", "fmax_op", "fmin_op", "hypot_op",
})
_MEMORY_BINARY_OPS = frozenset({
    "add_op", "sub_op", "mul_op", "div_op", "min_op", "max_op",
    "fmax_op", "fmin_op", "copysign_op",
})

# Map op_name -> numpy function for CPU fallback paths
_NP_UNARY = {
    "neg_op": np.negative, "abs_op": np.abs, "sqrt_op": np.sqrt,
    "exp_op": np.exp, "log_op": np.log, "sin_op": np.sin, "cos_op": np.cos,
    "tan_op": np.tan, "ceil_op": np.ceil, "floor_op": np.floor,
    "sign_op": np.sign, "square_op": np.square, "exp2_op": np.exp2,
    "log2_op": np.log2, "log10_op": np.log10, "tanh_op": np.tanh,
    "sinh_op": np.sinh, "cosh_op": np.cosh, "asin_op": np.arcsin,
    "acos_op": np.arccos, "atan_op": np.arctan, "asinh_op": np.arcsinh,
    "acosh_op": np.arccosh, "atanh_op": np.arctanh, "rint_op": np.rint,
    "trunc_op": np.trunc, "expm1_op": np.expm1, "log1p_op": np.log1p,
    "cbrt_op": np.cbrt, "reciprocal_op": np.reciprocal,
    "degrees_op": np.degrees, "radians_op": np.radians,
    "negative_op": np.negative, "positive_op": np.positive,
    "floor_op": np.floor, "ceil_op": np.ceil,
}
_NP_BINARY = {
    "add_op": np.add, "sub_op": np.subtract, "mul_op": np.multiply,
    "div_op": np.true_divide, "pow_op": np.power,
    "mod_op": np.mod, "min_op": np.minimum, "max_op": np.maximum,
    "atan2_op": np.arctan2, "hypot_op": np.hypot,
    "fmod_op": np.fmod, "copysign_op": np.copysign,
    "logaddexp_op": np.logaddexp, "logaddexp2_op": np.logaddexp2,
    "fmax_op": np.fmax, "fmin_op": np.fmin,
    "floor_divide_op": np.floor_divide, "heaviside_op": np.heaviside,
    "nextafter_op": np.nextafter,
}
_NP_CMP = {
    "lt_op": np.less, "le_op": np.less_equal, "gt_op": np.greater,
    "ge_op": np.greater_equal, "eq_op": np.equal, "ne_op": np.not_equal,
}
_NP_PRED = {
    "isnan_op": np.isnan, "isinf_op": np.isinf,
    "isfinite_op": np.isfinite, "signbit_op": np.signbit,
}
_NP_REDUCE = {
    "reduce_sum": np.sum, "reduce_max": np.max,
    "reduce_min": np.min, "reduce_prod": np.prod,
}

# Pre-computed thresholds: op_name -> element count threshold for CPU fallback
_UNARY_THRESHOLD: dict[str, int] = {}
for _op in _NP_UNARY:
    if _op in _MEMORY_UNARY_OPS:
        _UNARY_THRESHOLD[_op] = _GPU_THRESHOLD_MEMORY
    elif _op in _LIGHT_UNARY_OPS:
        _UNARY_THRESHOLD[_op] = _GPU_THRESHOLD_LIGHT
    else:
        _UNARY_THRESHOLD[_op] = _GPU_THRESHOLD

_BOOL_DTYPE = np.dtype(np.bool_)

_BINARY_THRESHOLD: dict[str, int] = {}
for _op in _NP_BINARY:
    if _op in _MEMORY_BINARY_OPS:
        _BINARY_THRESHOLD[_op] = _GPU_THRESHOLD_MEMORY
    elif _op in _LIGHT_BINARY_OPS:
        _BINARY_THRESHOLD[_op] = _GPU_THRESHOLD_LIGHT
    else:
        _BINARY_THRESHOLD[_op] = _GPU_THRESHOLD

# Op IDs for C dispatch tables (must match order passed to init_dispatch)
# Binary ops
_OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV = 0, 1, 2, 3
_OP_POW, _OP_MOD, _OP_FLOOR_DIV = 4, 5, 6
# Unary ops
_UOP_NEG, _UOP_ABS = 0, 1
# Comparison ops
_COP_LT, _COP_LE, _COP_GT, _COP_GE, _COP_EQ, _COP_NE = 0, 1, 2, 3, 4, 5

_F64_DTYPE = np.dtype(np.float64)
_C128_DTYPE = np.dtype(np.complex128)


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


_STRIDE_CACHE: dict[Tuple[int, ...], Tuple[int, ...]] = {}


def _c_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return C-contiguous *element* strides for *shape* (cached)."""
    s = _STRIDE_CACHE.get(shape)
    if s is not None:
        return s
    if not shape:
        _STRIDE_CACHE[shape] = ()
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    result = tuple(strides)
    if len(_STRIDE_CACHE) < 256:
        _STRIDE_CACHE[shape] = result
    return result


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
        self._np_data: Optional[np.ndarray] = None  # CPU-resident backing store
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

    @classmethod
    def _from_np_direct(cls, np_array: np.ndarray) -> ndarray:
        """Create a CPU-resident ndarray (no Metal buffer allocation).

        The data stays in a plain numpy array until a GPU kernel needs it,
        at which point ``_ensure_gpu()`` uploads it lazily.
        """
        if type(np_array) is not np.ndarray:
            np_array = np.asarray(np_array)
        orig_shape = np_array.shape
        cur_dtype = np_array.dtype
        # Fast path: skip resolve_dtype for common supported types
        if cur_dtype in _PASSTHROUGH_DTYPES:
            target = cur_dtype
        else:
            target = resolve_dtype(cur_dtype)
            if cur_dtype != target:
                np_array = np.ascontiguousarray(np_array, dtype=target)
        # C extension fast path for contiguous arrays
        if _accelerator is not None and np_array.flags['C_CONTIGUOUS']:
            arr = _accelerator.wrap_result(cls, np_array)
            if arr is not None:
                arr._dtype = target
                # Fix shape/strides — np.ascontiguousarray can change 0-d to 1-d
                if np_array.shape != orig_shape:
                    arr._shape = orig_shape
                    arr._strides = _c_contiguous_strides(orig_shape)
                return arr
        # Allow non-contiguous views (flip, transpose, etc.) — lazily
        # made contiguous only when GPU needs it via _ensure_gpu().
        arr = cls.__new__(cls)
        arr._buffer = None
        arr._np_data = np_array
        arr._shape = orig_shape
        arr._dtype = target
        if np_array.flags['C_CONTIGUOUS']:
            arr._strides = _c_contiguous_strides(orig_shape)
        else:
            itemsize = np_array.itemsize
            arr._strides = tuple(s // itemsize for s in np_array.strides)
        arr._offset = 0
        arr._base = None
        return arr

    @classmethod
    def _from_numpy(cls, np_array: np.ndarray) -> ndarray:
        """Create a GPU ndarray from a numpy array (convenience for CPU fallback)."""
        orig_shape = np_array.shape
        np_array = np.ascontiguousarray(np_array)
        dtype = resolve_dtype(np_array.dtype)
        np_array = np_array.astype(dtype, copy=False)
        backend = MetalBackend()
        buf = backend.array_to_buffer(np_array)
        return cls._from_buffer(buf, orig_shape, dtype)

    def _ensure_gpu(self) -> ndarray:
        """Ensure data is in a Metal buffer (lazy upload from CPU)."""
        if self._dtype == _F64_DTYPE or self._dtype == _C128_DTYPE:
            raise TypeError(
                "float64/complex128 arrays cannot be uploaded to Metal GPU. "
                "Use .astype(np.float32) to convert, or keep operations on CPU."
            )
        if self._np_data is not None:
            backend = MetalBackend()
            data = np.ascontiguousarray(self._np_data)
            self._buffer = backend.array_to_buffer(data)
            self._np_data = None
            # Data is now contiguous after upload
            self._strides = _c_contiguous_strides(self._shape)
            self._offset = 0
        return self

    def _adopt_buffer(self, buf) -> None:
        """Replace backing store with *buf*, clearing CPU-resident data."""
        self._buffer = buf
        self._np_data = None

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
            return ndarray._from_np_direct(self.get().real)
        return self

    @property
    def imag(self) -> ndarray:
        """Imaginary part of the array."""
        if self._dtype == np.complex64:
            return ndarray._from_np_direct(self.get().imag)
        return ndarray._from_np_direct(np.zeros(self._shape, dtype=self._dtype))

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
        if self._np_data is not None:
            # CPU-resident: make contiguous if needed, then upload
            if not self._np_data.flags['C_CONTIGUOUS']:
                self._np_data = np.ascontiguousarray(self._np_data)
                self._strides = _c_contiguous_strides(self._shape)
            self._ensure_gpu()
            return self
        if self._is_c_contiguous():
            return self
        return self._contiguous_copy()

    def _contiguous_copy(self) -> ndarray:
        """Materialise a contiguous copy via GPU strided copy."""
        if self._dtype in METAL_TYPE_NAMES and self.size > 0:
            metal_type = METAL_TYPE_NAMES[self._dtype]
            ndim = len(self._shape)
            backend = MetalBackend()
            if ndim == 0:
                shader_src = get_scalar_copy_shader(metal_type)
                params = np.array([self._offset], dtype=np.uint32)
                params_buf = backend.array_to_buffer(params)
                out_buf = backend.create_buffer(1, self._dtype)
                backend.execute_kernel(shader_src, "scalar_copy", 1,
                                      [self._buffer, out_buf, params_buf])
                return ndarray._from_buffer(out_buf, self._shape, self._dtype)

            shader_src = get_strided_copy_shader(metal_type)
            shape_buf = backend.array_to_buffer(np.array(self._shape, dtype=np.uint32))
            stride_buf = backend.array_to_buffer(np.array(self._strides, dtype=np.uint32))
            params = np.array([ndim, self._offset], dtype=np.uint32)
            params_buf = backend.array_to_buffer(params)
            out_buf = backend.create_buffer(self.size, self._dtype)
            backend.execute_kernel(shader_src, "strided_copy", self.size,
                                  [self._buffer, out_buf, shape_buf, stride_buf, params_buf])
            return ndarray._from_buffer(out_buf, self._shape, self._dtype)

        # CPU fallback for unsupported dtypes
        np_data = self.get()
        backend = MetalBackend()
        buf = backend.array_to_buffer(np.ascontiguousarray(np_data))
        return ndarray._from_buffer(buf, self._shape, self._dtype)

    @staticmethod
    def _gpu_broadcast_to(arr, shape):
        """GPU N-dimensional broadcast of arr to target shape."""
        if arr._shape == tuple(shape):
            return arr
        if arr._dtype not in METAL_TYPE_NAMES or arr.size == 0:
            # CPU fallback
            from . import creation
            np_result = np.broadcast_to(arr.get(), shape)
            return creation.array(np.ascontiguousarray(np_result), dtype=arr._dtype)

        a_c = arr._ensure_contiguous()
        out_size = 1
        for s in shape:
            out_size *= s
        if out_size == 0:
            from . import creation
            return creation.array(np.empty(shape, dtype=arr._dtype))

        ndim = len(shape)
        src_shape = (1,) * (ndim - arr.ndim) + arr._shape
        metal_type = METAL_TYPE_NAMES[arr._dtype]
        shader_src = get_broadcast_shader(metal_type)

        backend = MetalBackend()
        src_shape_buf = backend.array_to_buffer(np.array(src_shape, dtype=np.uint32))
        out_shape_buf = backend.array_to_buffer(np.array(shape, dtype=np.uint32))
        ndim_buf = backend.array_to_buffer(np.array([ndim], dtype=np.uint32))
        out_buf = backend.create_buffer(out_size, arr._dtype)
        backend.execute_kernel(shader_src, "broadcast_nd", out_size,
                              [a_c._buffer, out_buf, src_shape_buf, out_shape_buf, ndim_buf])
        return ndarray._from_buffer(out_buf, tuple(shape), arr._dtype)

    # ------------------------------------------------------------------ data transfer
    def _get_view(self) -> np.ndarray:
        """Return a *read-only* numpy view of the backing data (zero-copy).

        For CPU-resident arrays, returns the numpy array directly.
        For Metal-backed arrays on Apple Silicon, returns a view of unified memory.
        Callers MUST synchronize() beforehand for Metal-backed arrays.
        """
        if self._np_data is not None:
            d = self._np_data
            if d.shape != self._shape:
                d = d.ravel()[0:1].reshape(()) if not self._shape else d.reshape(self._shape)
            return d
        raw = np.array(self._buffer.contents, copy=False)
        if self._dtype == np.float16:
            # contents stored as int16
            if self._is_c_contiguous():
                view = raw[self._offset : self._offset + self.size].reshape(self._shape)
                return view.view(np.float16)
            byte_strides = tuple(s * self.itemsize for s in self._strides)
            flat = raw[self._offset:]
            return np.lib.stride_tricks.as_strided(flat, shape=self._shape, strides=byte_strides).view(np.float16)
        if self._is_c_contiguous():
            return raw[self._offset : self._offset + self.size].reshape(self._shape)
        byte_strides = tuple(s * self.itemsize for s in self._strides)
        flat = raw[self._offset:]
        return np.lib.stride_tricks.as_strided(flat, shape=self._shape, strides=byte_strides)

    def get(self, *, _force_copy: bool = False) -> np.ndarray:
        """Transfer data to host and return as a NumPy array."""
        if self._np_data is not None:
            data = self._np_data
            if data.shape != self._shape:
                data = data.reshape(self._shape) if self._shape else data.ravel()[0:1].reshape(())
            return data.copy() if _force_copy else data
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
        self._ensure_gpu()
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

        # CPU-resident: reshape the numpy data directly (preserving view/base)
        if self._np_data is not None:
            d = self._np_data
            if d.shape != self._shape:
                d = d.ravel()[0:1].reshape(()) if not self._shape else d.reshape(self._shape)
            arr = ndarray.__new__(ndarray)
            arr._buffer = None
            arr._np_data = d.reshape(new_shape)
            arr._shape = new_shape
            arr._dtype = self._dtype
            arr._strides = _c_contiguous_strides(new_shape)
            arr._offset = 0
            arr._base = self._base if self._base is not None else self
            return arr

        if self._is_c_contiguous():
            new_strides = _c_contiguous_strides(new_shape)
            return ndarray._from_buffer(
                self._buffer, new_shape, self._dtype,
                strides=new_strides, offset=self._offset, base=(self._base if self._base is not None else self),
            )
        arr = self._contiguous_copy()
        return arr.reshape(new_shape)

    def transpose(self, axes=None) -> ndarray:
        """Return a view with permuted axes."""
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))
        else:
            axes = tuple(axes)
        # CPU-resident: transpose via numpy (preserve non-contiguous view)
        if self._np_data is not None:
            d = self._np_data
            if d.shape != self._shape:
                d = d.ravel()[0:1].reshape(()) if not self._shape else d.reshape(self._shape)
            new_shape = tuple(self._shape[a] for a in axes)
            new_strides = tuple(self._strides[a] for a in axes)
            arr = ndarray.__new__(ndarray)
            arr._buffer = None
            arr._np_data = np.transpose(d, axes)
            arr._shape = new_shape
            arr._dtype = self._dtype
            arr._strides = new_strides
            arr._offset = 0
            arr._base = self._base if self._base is not None else self
            return arr
        # Complex64 is stored as float32 pairs; stride-based views break
        # the pair structure, so fall back to CPU transpose.
        if self._dtype == np.complex64:
            from . import creation
            return creation.array(np.transpose(self.get(), axes))
        new_shape = tuple(self._shape[a] for a in axes)
        new_strides = tuple(self._strides[a] for a in axes)
        return ndarray._from_buffer(
            self._buffer, new_shape, self._dtype,
            strides=new_strides, offset=self._offset, base=(self._base if self._base is not None else self),
        )

    def flatten(self) -> ndarray:
        """Return a 1-D contiguous copy."""
        if self._np_data is not None:
            return ndarray._from_np_direct(self._np_data.ravel().copy())
        arr = self._ensure_contiguous()
        return ndarray._from_buffer(
            arr._buffer, (self.size,), self._dtype,
            strides=(1,), offset=arr._offset, base=(arr._base if arr._base is not None else arr),
        )

    def ravel(self) -> ndarray:
        """Return a 1-D array; view if contiguous, copy otherwise."""
        if self._np_data is not None:
            return ndarray._from_np_direct(self._np_data.ravel())
        if self._is_c_contiguous():
            return ndarray._from_buffer(
                self._buffer, (self.size,), self._dtype,
                strides=(1,), offset=self._offset, base=(self._base if self._base is not None else self),
            )
        return self.flatten()

    def squeeze(self, axis=None) -> ndarray:
        """Remove dimensions of size 1."""
        if self._np_data is not None:
            d = self._np_data
            if d.shape != self._shape:
                d = d.ravel()[0:1].reshape(()) if not self._shape else d.reshape(self._shape)
            return ndarray._from_np_direct(d.squeeze(axis=axis))
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
            strides=new_strides, offset=self._offset, base=(self._base if self._base is not None else self),
        )

    def expand_dims(self, axis: int) -> ndarray:
        """Insert a size-1 dimension at *axis*."""
        if self._np_data is not None:
            d = self._np_data
            if d.shape != self._shape:
                d = d.ravel()[0:1].reshape(()) if not self._shape else d.reshape(self._shape)
            return ndarray._from_np_direct(np.expand_dims(d, axis))
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
            strides=tuple(new_strides), offset=self._offset, base=(self._base if self._base is not None else self),
        )

    # ------------------------------------------------------------------ indexing
    def _try_basic_getitem(self, key):
        """Return a zero-copy view for basic indexing, or None for advanced.

        Basic keys: int, slice, None (newaxis), Ellipsis.
        Advanced keys (bool/int arrays, lists) return None → CPU fallback.
        """
        # CPU-resident: fall through to numpy indexing in __getitem__
        if self._np_data is not None:
            return None
        # Complex64 stores float32 pairs; stride-based views break pair structure
        if self._dtype == np.complex64:
            return None

        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Reject non-basic elements (check bool before int since bool is subclass of int)
        for k in key:
            if isinstance(k, (bool, np.bool_)):
                return None
            if isinstance(k, (ndarray, np.ndarray, list)):
                return None
            if not isinstance(k, (int, slice, type(None), type(Ellipsis))):
                return None

        # At most one Ellipsis
        n_ellipsis = sum(1 for k in key if k is Ellipsis)
        if n_ellipsis > 1:
            return None

        # Expand Ellipsis → fill with slice(None) to match remaining dims
        n_newaxis = sum(1 for k in key if k is None)
        n_index = len(key) - n_newaxis - n_ellipsis
        if n_ellipsis == 1:
            n_fill = self.ndim - n_index
            expanded = []
            for k in key:
                if k is Ellipsis:
                    expanded.extend([slice(None)] * n_fill)
                else:
                    expanded.append(k)
            key = tuple(expanded)

        # Walk key elements, building new shape/strides/offset
        new_shape = []
        new_strides = []
        offset = self._offset
        dim = 0

        for k in key:
            if k is None:
                # New axis: size-1, stride-0 (no dimension consumed)
                new_shape.append(1)
                new_strides.append(0)
            elif isinstance(k, int):
                if dim >= self.ndim:
                    return None
                size = self._shape[dim]
                idx = k
                if idx < 0:
                    idx += size
                if idx < 0 or idx >= size:
                    return None  # out of bounds → let NumPy raise
                offset += idx * self._strides[dim]
                dim += 1
            elif isinstance(k, slice):
                if dim >= self.ndim:
                    return None
                size = self._shape[dim]
                start, stop, step = k.indices(size)
                if step > 0:
                    length = max(0, (stop - start + step - 1) // step)
                else:
                    length = max(0, (stop - start + step + 1) // step)
                offset += start * self._strides[dim]
                new_shape.append(length)
                new_strides.append(self._strides[dim] * step)
                dim += 1

        # Append remaining unconsumed dimensions
        while dim < self.ndim:
            new_shape.append(self._shape[dim])
            new_strides.append(self._strides[dim])
            dim += 1

        new_shape = tuple(new_shape)
        new_strides = tuple(new_strides)
        base = self._base if self._base is not None else self

        return ndarray._from_buffer(
            self._buffer, new_shape, self._dtype,
            strides=new_strides, offset=offset, base=base,
        )

    def __getitem__(self, key):
        result = self._try_basic_getitem(key)
        if result is not None:
            return result
        # CPU fallback for advanced indexing (boolean/integer arrays)
        if self._np_data is not None:
            np_data = self._np_data
        else:
            np_data = self.get()
        # Convert ndarray keys to numpy before delegating
        if isinstance(key, ndarray):
            key = key.get()
        elif isinstance(key, tuple):
            key = tuple(k.get() if isinstance(k, ndarray) else k for k in key)
        result = np_data[key]
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        return ndarray._from_np_direct(result)

    @staticmethod
    def _gpu_setitem_view(target, value):
        """Write *value* into *target* view's buffer positions on GPU."""
        from . import creation

        if target.size == 0:
            return

        # Convert value to GPU ndarray with matching dtype
        if isinstance(value, (int, float)):
            value = creation.full(target._shape if target._shape else (), value, dtype=target._dtype)
        elif isinstance(value, np.ndarray):
            value = creation.array(value, dtype=target._dtype)
        elif isinstance(value, ndarray):
            if value._dtype != target._dtype:
                value = value.astype(target._dtype)
        else:
            value = creation.array(np.asarray(value), dtype=target._dtype)

        # Broadcast value to target shape if needed
        if value._shape != target._shape and target._shape != ():
            value = ndarray._gpu_broadcast_to(value, target._shape)

        value = value._ensure_contiguous()

        metal_type = METAL_TYPE_NAMES[target._dtype]
        backend = MetalBackend()

        if target._shape == ():
            shader_src = get_scalar_write_shader(metal_type)
            params = np.array([target._offset], dtype=np.uint32)
            params_buf = backend.array_to_buffer(params)
            backend.execute_kernel(shader_src, "scalar_write", 1,
                                  [target._buffer, value._buffer, params_buf])
        elif target._strides == _c_contiguous_strides(target._shape):
            shader_src = get_offset_copy_shader(metal_type)
            params = np.array([target._offset], dtype=np.uint32)
            params_buf = backend.array_to_buffer(params)
            backend.execute_kernel(shader_src, "offset_copy", target.size,
                                  [target._buffer, value._buffer, params_buf])
        else:
            ndim = len(target._shape)
            shader_src = get_strided_write_shader(metal_type)
            shape_buf = backend.array_to_buffer(np.array(target._shape, dtype=np.uint32))
            stride_buf = backend.array_to_buffer(np.array(target._strides, dtype=np.uint32))
            params = np.array([ndim, target._offset], dtype=np.uint32)
            params_buf = backend.array_to_buffer(params)
            backend.execute_kernel(shader_src, "strided_write", target.size,
                                  [target._buffer, value._buffer, shape_buf, stride_buf, params_buf])

    def __setitem__(self, key, value):
        target = self._try_basic_getitem(key)
        if target is not None and self._dtype in METAL_TYPE_NAMES:
            self._gpu_setitem_view(target, value)
            return
        # CPU fallback for advanced indexing
        np_data = self.get()
        if isinstance(value, ndarray):
            value = value.get()
        if isinstance(key, ndarray):
            key = key.get()
        elif isinstance(key, tuple):
            key = tuple(k.get() if isinstance(k, ndarray) else k for k in key)
        np_data[key] = value
        self.set(np_data)

    # ------------------------------------------------------------------ operators
    def _binary_op(self, other, op_name: str) -> ndarray:
        """Execute a binary elementwise GPU kernel."""
        from . import creation

        # ── Ultra-fast path: two ndarray, same dtype, CPU-resident ──
        if type(other) is ndarray:
            np_a = self._np_data
            np_b = other._np_data
            if np_a is not None and np_b is not None:
                sdtype = self._dtype
                threshold = _BINARY_THRESHOLD.get(op_name)
                is_f64 = sdtype == _F64_DTYPE or other._dtype == _F64_DTYPE
                if threshold is not None and (is_f64 or np_a.size < threshold):
                    np_func = _NP_BINARY.get(op_name)
                    if np_func is not None:
                        # Same dtype: skip result_dtype entirely (~350ns saved)
                        if sdtype is other._dtype or sdtype == other._dtype:
                            result = np_func(np_a, np_b)
                        else:
                            rdtype = result_dtype(sdtype, other._dtype)
                            result = np_func(
                                np_a if np_a.dtype == rdtype else np_a.astype(rdtype, copy=False),
                                np_b if np_b.dtype == rdtype else np_b.astype(rdtype, copy=False),
                            )
                        arr = ndarray.__new__(ndarray)
                        arr._buffer = None
                        arr._np_data = result
                        arr._shape = result.shape
                        arr._dtype = result.dtype
                        arr._strides = _c_contiguous_strides(result.shape)
                        arr._offset = 0
                        arr._base = None
                        return arr

        # Wrap scalar — use numpy-style promotion for Python float scalars
        if isinstance(other, (int, float)):
            if isinstance(other, float) and np.issubdtype(self._dtype, np.integer):
                scalar_dtype = resolve_dtype(np.float64)
            else:
                scalar_dtype = self._dtype
            # CPU fallback for small arrays (single dict lookup for threshold)
            threshold = _BINARY_THRESHOLD.get(op_name)
            if threshold is not None and self.size < threshold:
                np_func = _NP_BINARY[op_name]
                # Inline fast path: skip result_dtype + _from_np_direct overhead
                np_data = self._np_data
                if np_data is None:
                    MetalBackend().synchronize()
                    np_data = self._get_view()
                # Common case: same dtype (float32 + float scalar → float32)
                if scalar_dtype == self._dtype:
                    rdtype = self._dtype
                    src = np_data
                else:
                    rdtype = result_dtype(self._dtype, scalar_dtype)
                    src = np_data.astype(rdtype, copy=False)
                result = np_func(src, rdtype.type(other))
                arr = ndarray.__new__(ndarray)
                arr._buffer = None
                arr._np_data = result
                arr._shape = result.shape
                arr._dtype = rdtype
                arr._strides = _c_contiguous_strides(result.shape)
                arr._offset = 0
                arr._base = None
                return arr
            other = creation.full(self._shape, other, dtype=scalar_dtype)
        elif not isinstance(other, ndarray):
            other = creation.array(other, dtype=self._dtype)

        # Determine result dtype
        rdtype = result_dtype(self._dtype, other._dtype)

        # CPU fallback for complex64 (no Metal shader support)
        if rdtype == np.complex64:
            a_np = self.get().astype(np.complex64) if self._dtype != np.complex64 else self.get()
            b_np = other.get().astype(np.complex64) if other._dtype != np.complex64 else other.get()
            if a_np.shape != b_np.shape:
                out_shape = np.broadcast_shapes(a_np.shape, b_np.shape)
                a_np = np.broadcast_to(a_np, out_shape)
                b_np = np.broadcast_to(b_np, out_shape)
            op_func = _NP_BINARY.get(op_name)
            if op_func is not None:
                return creation.array(op_func(a_np, b_np))
            return creation.array(a_np)

        # Float64/complex128: always run on CPU (Metal doesn't support these)
        if rdtype == _F64_DTYPE or rdtype == _C128_DTYPE:
            np_func = _NP_BINARY.get(op_name)
            if np_func is not None:
                return self._cpu_binary(other, np_func, rdtype)

        # CPU fallback for small arrays (single dict lookup for threshold)
        threshold = _BINARY_THRESHOLD.get(op_name)
        if threshold is not None and self.size < threshold and other.size < threshold:
            return self._cpu_binary(other, _NP_BINARY[op_name], rdtype)

        # Cast operands to result dtype if needed
        a = self.astype(rdtype) if self._dtype != rdtype else self
        b = other.astype(rdtype) if other._dtype != rdtype else other

        # Broadcasting (GPU-accelerated)
        if needs_broadcast(a._shape, b._shape):
            out_shape = broadcast_shapes(a._shape, b._shape)
            a = self._gpu_broadcast_to(a, out_shape)
            b = self._gpu_broadcast_to(b, out_shape)

        a = a._ensure_contiguous()
        b = b._ensure_contiguous()

        backend = MetalBackend()
        cache = KernelCache()
        shader = cache.get_shader("elementwise", rdtype)
        out_buf = backend.create_buffer(a.size, rdtype)
        backend.execute_kernel(shader, op_name, a.size, [a._buffer, b._buffer, out_buf])
        return ndarray._from_buffer(out_buf, a._shape, rdtype)

    def _cpu_unary(self, np_func) -> ndarray:
        """Run a unary op on CPU — result stays CPU-resident (no Metal buffer)."""
        # Inline hot path: skip _get_view + _from_np_direct overhead
        np_data = self._np_data
        if np_data is not None:
            result = np_func(np_data)
        else:
            MetalBackend().synchronize()
            result = np_func(self._get_view())
        if not isinstance(result, np.ndarray):
            result = np.asarray(result)
        arr = ndarray.__new__(ndarray)
        arr._buffer = None
        arr._np_data = result
        arr._shape = result.shape
        arr._dtype = result.dtype if result.dtype in _PASSTHROUGH_DTYPES else resolve_dtype(result.dtype)
        arr._strides = _c_contiguous_strides(result.shape)
        arr._offset = 0
        arr._base = None
        return arr

    def _cpu_binary(self, other, np_func, rdtype) -> ndarray:
        """Run a binary op on CPU — result stays CPU-resident (no Metal buffer)."""
        # Inline hot path: skip _get_view + _from_np_direct overhead
        np_a = self._np_data
        np_b = other._np_data if isinstance(other, ndarray) else None
        if np_a is None or (isinstance(other, ndarray) and np_b is None):
            MetalBackend().synchronize()
            np_a = self._get_view() if np_a is None else np_a
            np_b = other._get_view() if isinstance(other, ndarray) and np_b is None else np_b
        src_a = np_a.astype(rdtype, copy=False) if np_a.dtype != rdtype else np_a
        if isinstance(other, ndarray):
            src_b = np_b.astype(rdtype, copy=False) if np_b.dtype != rdtype else np_b
        else:
            src_b = other
        result = np_func(src_a, src_b)
        arr = ndarray.__new__(ndarray)
        arr._buffer = None
        arr._np_data = result
        arr._shape = result.shape
        arr._dtype = rdtype
        arr._strides = _c_contiguous_strides(result.shape)
        arr._offset = 0
        arr._base = None
        return arr

    def _unary_op(self, op_name: str) -> ndarray:
        """Execute a unary elementwise GPU kernel."""
        np_data = self._np_data

        # ── Ultra-fast path: CPU-resident, non-complex ──
        if np_data is not None and self._dtype != np.complex64:
            threshold = _UNARY_THRESHOLD.get(op_name)
            is_f64 = self._dtype == _F64_DTYPE
            if threshold is not None and (is_f64 or np_data.size < threshold):
                np_func = _NP_UNARY.get(op_name)
                if np_func is not None:
                    result = np_func(np_data)
                    if type(result) is not np.ndarray:
                        result = np.asarray(result)
                    arr = ndarray.__new__(ndarray)
                    arr._buffer = None
                    arr._np_data = result
                    arr._shape = result.shape
                    arr._dtype = result.dtype
                    arr._strides = _c_contiguous_strides(result.shape)
                    arr._offset = 0
                    arr._base = None
                    return arr

        # CPU fallback for complex64 (no Metal shader support)
        if self._dtype == np.complex64:
            from . import creation
            op_func = _NP_UNARY.get(op_name)
            if op_func is not None:
                return creation.array(op_func(self.get()))
            return creation.array(self.get())

        # Float64: always run on CPU (Metal doesn't support float64)
        if self._dtype == _F64_DTYPE:
            np_func = _NP_UNARY.get(op_name)
            if np_func is not None:
                return self._cpu_unary(np_func)

        # CPU fallback for GPU-resident small arrays
        threshold = _UNARY_THRESHOLD.get(op_name)
        if threshold is not None and self.size < threshold:
            return self._cpu_unary(_NP_UNARY[op_name])

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
            op_func = _NP_PRED.get(op_name)
            if op_func is not None:
                return creation.array(op_func(self.get()))
            return creation.array(np.zeros(self._shape, dtype=bool))

        # CPU fallback — predicates are pure memory ops, CPU SIMD wins
        if self.size < _GPU_THRESHOLD_MEMORY:
            op_func = _NP_PRED.get(op_name)
            if op_func is not None:
                np_data = self._np_data
                if np_data is not None:
                    result = op_func(np_data)
                else:
                    MetalBackend().synchronize()
                    result = op_func(self._get_view())
                arr = ndarray.__new__(ndarray)
                arr._buffer = None
                arr._np_data = result
                arr._shape = result.shape
                arr._dtype = result.dtype
                arr._strides = _c_contiguous_strides(result.shape)
                arr._offset = 0
                arr._base = None
                return arr

        # Float64: always run on CPU (Metal doesn't support float64)
        if self._dtype == _F64_DTYPE:
            op_func = _NP_PRED.get(op_name)
            if op_func is not None:
                np_data = self._np_data if self._np_data is not None else self.get()
                result = op_func(np_data)
                arr = ndarray.__new__(ndarray)
                arr._buffer = None
                arr._np_data = result
                arr._shape = result.shape
                arr._dtype = result.dtype
                arr._strides = _c_contiguous_strides(result.shape)
                arr._offset = 0
                arr._base = None
                return arr

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
        if _fast_binary is not None:
            r = _fast_binary(self, other, _OP_ADD)
            if r is not None:
                return r
        return self._binary_op(other, "add_op")

    def __radd__(self, other):
        if _fast_binary is not None:
            r = _fast_binary(self, other, _OP_ADD)
            if r is not None:
                return r
        return self._binary_op(other, "add_op")

    def __sub__(self, other):
        if _fast_binary is not None:
            r = _fast_binary(self, other, _OP_SUB)
            if r is not None:
                return r
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
        if _fast_binary is not None:
            r = _fast_binary(self, other, _OP_MUL)
            if r is not None:
                return r
        return self._binary_op(other, "mul_op")

    def __rmul__(self, other):
        if _fast_binary is not None:
            r = _fast_binary(self, other, _OP_MUL)
            if r is not None:
                return r
        return self._binary_op(other, "mul_op")

    def __truediv__(self, other):
        if _fast_binary is not None:
            r = _fast_binary(self, other, _OP_DIV)
            if r is not None:
                return r
        # True division always returns float (match NumPy)
        if np.issubdtype(self._dtype, np.integer):
            lhs = self.astype(np.float32)
            if isinstance(other, ndarray) and np.issubdtype(other._dtype, np.integer):
                other = other.astype(np.float32)
            return lhs._binary_op(other, "div_op")
        return self._binary_op(other, "div_op")

    def __rtruediv__(self, other):
        from . import creation
        if isinstance(other, (int, float)):
            other = creation.full(self._shape, other, dtype=np.float32 if np.issubdtype(self._dtype, np.integer) else self._dtype)
        lhs = other
        rhs = self
        if isinstance(rhs, ndarray) and np.issubdtype(rhs._dtype, np.integer):
            rhs = rhs.astype(np.float32)
        if isinstance(lhs, ndarray) and np.issubdtype(lhs._dtype, np.integer):
            lhs = lhs.astype(np.float32)
        return lhs._binary_op(rhs, "div_op")

    def __pow__(self, other):
        if _fast_binary is not None:
            r = _fast_binary(self, other, _OP_POW)
            if r is not None:
                return r
        return self._binary_op(other, "pow_op")

    def __rpow__(self, other):
        from . import creation
        if isinstance(other, (int, float)):
            other = creation.full(self._shape, other, dtype=self._dtype)
        return other._binary_op(self, "pow_op")

    def __neg__(self):
        if _fast_unary is not None:
            r = _fast_unary(self, _UOP_NEG)
            if r is not None:
                return r
        return self._unary_op("neg_op")

    def __abs__(self):
        if _fast_unary is not None:
            r = _fast_unary(self, _UOP_ABS)
            if r is not None:
                return r
        return self._unary_op("abs_op")

    def __matmul__(self, other):
        if not isinstance(other, ndarray):
            from . import creation
            other = creation.array(other, dtype=self._dtype)

        # Handle 1-D arrays: compute inner product like NumPy (GPU)
        if self.ndim == 1 and other.ndim == 1:
            return (self * other).sum()

        rdtype = result_dtype(self._dtype, other._dtype)
        a = self.astype(rdtype) if self._dtype != rdtype else self
        b = other.astype(rdtype) if other._dtype != rdtype else other

        # Handle 1-D @ 2-D and 2-D @ 1-D by promoting
        a_1d = a.ndim == 1
        b_1d = b.ndim == 1
        if a_1d:
            a = a.reshape(1, -1)
        if b_1d:
            b = b.reshape(-1, 1)

        a = a._ensure_contiguous()
        b = b._ensure_contiguous()

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul requires 1-D or 2-D arrays")
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
        result = ndarray._from_buffer(out_buf, (M, N), rdtype)
        # Squeeze dimensions that were promoted from 1-D
        if a_1d and b_1d:
            return result.squeeze()
        elif a_1d:
            return result.squeeze(axis=0)
        elif b_1d:
            return result.squeeze(axis=1)
        return result

    # ------------------------------------------------------------------ comparison
    def _comparison_op(self, other, op_name: str) -> ndarray:
        """Execute a comparison GPU kernel, returning a bool-typed result."""
        from . import creation

        # ── Ultra-fast path: two ndarray, CPU-resident ──
        if type(other) is ndarray:
            np_a = self._np_data
            np_b = other._np_data
            if np_a is not None and np_b is not None and self._dtype != np.complex64:
                np_func = _NP_CMP.get(op_name)
                is_f64 = self._dtype == _F64_DTYPE or other._dtype == _F64_DTYPE
                if np_func is not None and (is_f64 or np_a.size < _GPU_THRESHOLD_MEMORY):
                    sdtype = self._dtype
                    if sdtype is other._dtype or sdtype == other._dtype:
                        result = np_func(np_a, np_b)
                    else:
                        rdtype = result_dtype(sdtype, other._dtype)
                        result = np_func(
                            np_a if np_a.dtype == rdtype else np_a.astype(rdtype, copy=False),
                            np_b if np_b.dtype == rdtype else np_b.astype(rdtype, copy=False),
                        )
                    arr = ndarray.__new__(ndarray)
                    arr._buffer = None
                    arr._np_data = result
                    arr._shape = result.shape
                    arr._dtype = _BOOL_DTYPE
                    arr._strides = _c_contiguous_strides(result.shape)
                    arr._offset = 0
                    arr._base = None
                    return arr

        # Wrap scalar
        if isinstance(other, (int, float)):
            # CPU fallback — comparisons are memory-bound, CPU SIMD wins
            if self.size < _GPU_THRESHOLD_MEMORY and self._dtype != np.complex64:
                np_func = _NP_CMP.get(op_name)
                if np_func is not None:
                    np_data = self._np_data
                    if np_data is not None:
                        result = np_func(np_data, other)
                    else:
                        MetalBackend().synchronize()
                        result = np_func(self._get_view(), other)
                    arr = ndarray.__new__(ndarray)
                    arr._buffer = None
                    arr._np_data = result
                    arr._shape = result.shape
                    arr._dtype = _BOOL_DTYPE
                    arr._strides = _c_contiguous_strides(result.shape)
                    arr._offset = 0
                    arr._base = None
                    return arr
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
            if a_np.shape != b_np.shape:
                out_shape = np.broadcast_shapes(a_np.shape, b_np.shape)
                a_np = np.broadcast_to(a_np, out_shape)
                b_np = np.broadcast_to(b_np, out_shape)
            return creation.array(op_func(a_np, b_np))

        # Float64/complex128: always run on CPU (Metal doesn't support these)
        if rdtype == _F64_DTYPE or rdtype == _C128_DTYPE:
            np_func = _NP_CMP.get(op_name)
            if np_func is not None:
                a_np = self._np_data if self._np_data is not None else self.get()
                b_np = other._np_data if other._np_data is not None else other.get()
                a_np = a_np.astype(rdtype, copy=False)
                b_np = b_np.astype(rdtype, copy=False)
                if a_np.shape != b_np.shape:
                    out_shape = np.broadcast_shapes(a_np.shape, b_np.shape)
                    a_np = np.broadcast_to(a_np, out_shape)
                    b_np = np.broadcast_to(b_np, out_shape)
                result = np_func(a_np, b_np)
                return ndarray._from_np_direct(result)

        # CPU fallback — comparisons are memory-bound, CPU SIMD wins
        if self.size < _GPU_THRESHOLD_MEMORY and other.size < _GPU_THRESHOLD_MEMORY:
            np_func = _NP_CMP.get(op_name)
            if np_func is not None:
                np_a = self._np_data
                np_b = other._np_data
                if np_a is None or np_b is None:
                    MetalBackend().synchronize()
                    np_a = self._get_view() if np_a is None else np_a
                    np_b = other._get_view() if np_b is None else np_b
                a_np = np_a.astype(rdtype, copy=False) if np_a.dtype != rdtype else np_a
                b_np = np_b.astype(rdtype, copy=False) if np_b.dtype != rdtype else np_b
                result = np_func(a_np, b_np)
                arr = ndarray.__new__(ndarray)
                arr._buffer = None
                arr._np_data = result
                arr._shape = result.shape
                arr._dtype = np.dtype(np.bool_)
                arr._strides = _c_contiguous_strides(result.shape)
                arr._offset = 0
                arr._base = None
                return arr

        a = self.astype(rdtype) if self._dtype != rdtype else self
        b = other.astype(rdtype) if other._dtype != rdtype else other

        # Broadcasting (GPU-accelerated)
        if needs_broadcast(a._shape, b._shape):
            out_shape = broadcast_shapes(a._shape, b._shape)
            a = self._gpu_broadcast_to(a, out_shape)
            b = self._gpu_broadcast_to(b, out_shape)

        a = a._ensure_contiguous()
        b = b._ensure_contiguous()

        # Opt 3: use comparison_bool shader that outputs uchar directly
        backend = MetalBackend()
        cache = KernelCache()
        shader = cache.get_shader("comparison_bool", rdtype)
        out_buf = backend.create_buffer(a.size, np.bool_)
        backend.execute_kernel(shader, op_name, a.size, [a._buffer, b._buffer, out_buf])
        return ndarray._from_buffer(out_buf, a._shape, np.bool_)

    def _boolean_op(self, other, op_name: str) -> ndarray:
        """Execute a boolean logic GPU kernel on bool arrays."""
        from . import creation

        if not isinstance(other, ndarray):
            other = creation.array(other, dtype=np.bool_)

        # CPU fallback — boolean ops are memory-bound, CPU SIMD wins
        if self.size < _GPU_THRESHOLD_MEMORY and other.size < _GPU_THRESHOLD_MEMORY:
            _BOOL_NP = {"and_op": np.logical_and, "or_op": np.logical_or,
                        "xor_op": np.logical_xor, "bool_and_op": np.logical_and,
                        "bool_or_op": np.logical_or, "bool_xor_op": np.logical_xor}
            np_func = _BOOL_NP.get(op_name)
            if np_func is not None:
                if self._np_data is None or other._np_data is None:
                    MetalBackend().synchronize()
                return ndarray._from_np_direct(np_func(self._get_view(), other._get_view()))

        backend = MetalBackend()
        cache = KernelCache()

        # Fast path: if both inputs are already bool, use native bool shader
        if self._dtype == np.bool_ and other.dtype == np.bool_:
            a = self._ensure_contiguous()
            b = other._ensure_contiguous()
            shader = cache.get_shader("bool_logic", np.bool_)
            out_buf = backend.create_buffer(a.size, np.bool_)
            backend.execute_kernel(shader, "bool_" + op_name, a.size,
                                   [a._buffer, b._buffer, out_buf])
            return ndarray._from_buffer(out_buf, a._shape, np.bool_)

        # Fallback: convert to int32 for GPU
        a = self.astype(np.int32)._ensure_contiguous()
        b = other.astype(np.int32)._ensure_contiguous()

        shader = cache.get_shader("boolean", np.int32)
        out_buf = backend.create_buffer(a.size, np.int32)
        backend.execute_kernel(shader, op_name, a.size, [a._buffer, b._buffer, out_buf])

        int_arr = ndarray._from_buffer(out_buf, a._shape, np.int32)
        return int_arr.astype(np.bool_)

    # Comparison operators
    def __lt__(self, other):
        if _fast_cmp is not None:
            r = _fast_cmp(self, other, _COP_LT)
            if r is not None:
                return r
        return self._comparison_op(other, "lt_op")

    def __le__(self, other):
        if _fast_cmp is not None:
            r = _fast_cmp(self, other, _COP_LE)
            if r is not None:
                return r
        return self._comparison_op(other, "le_op")

    def __gt__(self, other):
        if _fast_cmp is not None:
            r = _fast_cmp(self, other, _COP_GT)
            if r is not None:
                return r
        return self._comparison_op(other, "gt_op")

    def __ge__(self, other):
        if _fast_cmp is not None:
            r = _fast_cmp(self, other, _COP_GE)
            if r is not None:
                return r
        return self._comparison_op(other, "ge_op")

    def __eq__(self, other):
        if _fast_cmp is not None:
            r = _fast_cmp(self, other, _COP_EQ)
            if r is not None:
                return r
        return self._comparison_op(other, "eq_op")

    def __ne__(self, other):
        if _fast_cmp is not None:
            r = _fast_cmp(self, other, _COP_NE)
            if r is not None:
                return r
        return self._comparison_op(other, "ne_op")

    # Boolean logic operators
    def __and__(self, other):
        if np.issubdtype(self._dtype, np.integer):
            from . import bitwise_ops
            return bitwise_ops.bitwise_and(self, other)
        return self._boolean_op(other, "and_op")

    def __or__(self, other):
        if np.issubdtype(self._dtype, np.integer):
            from . import bitwise_ops
            return bitwise_ops.bitwise_or(self, other)
        return self._boolean_op(other, "or_op")

    def __invert__(self):
        if np.issubdtype(self._dtype, np.integer):
            from . import bitwise_ops
            return bitwise_ops.invert(self)
        # Fast path for bool: use native bool shader
        if self._dtype == np.bool_:
            backend = MetalBackend()
            cache = KernelCache()
            a = self._ensure_contiguous()
            shader = cache.get_shader("bool_logic", np.bool_)
            out_buf = backend.create_buffer(a.size, np.bool_)
            backend.execute_kernel(shader, "bool_not_op", a.size,
                                   [a._buffer, out_buf])
            return ndarray._from_buffer(out_buf, a._shape, np.bool_)
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
        result = self._binary_op(other, "add_op").astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
        return self

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            from . import creation
            other = creation.full(self._shape, self._dtype.type(other), dtype=self._dtype)
        result = self._binary_op(other, "sub_op").astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
        return self

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            from . import creation
            other = creation.full(self._shape, self._dtype.type(other), dtype=self._dtype)
        result = self._binary_op(other, "mul_op").astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
        return self

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            from . import creation
            other = creation.full(self._shape, self._dtype.type(other), dtype=self._dtype)
        result = self._binary_op(other, "div_op").astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
        return self

    def __ipow__(self, other):
        if isinstance(other, (int, float)):
            from . import creation
            other = creation.full(self._shape, self._dtype.type(other), dtype=self._dtype)
        result = self._binary_op(other, "pow_op").astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
        return self

    def __ior__(self, other):
        result = self._boolean_op(other, "or_op")
        self._adopt_buffer(result._ensure_contiguous()._buffer)
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
        # CPU fallback — reductions are memory-bound, CPU wins up to ~4M
        if self.size < _GPU_THRESHOLD_MEMORY:
            if self._np_data is None:
                MetalBackend().synchronize()
            r = np.mean(self._get_view(), axis=axis, keepdims=keepdims)
            return ndarray._from_np_direct(np.asarray(r))
        # For integer dtypes, cast to float32 first (matches NumPy behavior)
        arr = self
        if not np.issubdtype(self._dtype, np.floating) and self._dtype != np.complex64:
            arr = self.astype(np.float32)
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
        # CPU fallback — reductions are memory-bound
        if self.size < _GPU_THRESHOLD_MEMORY:
            if self._np_data is None:
                MetalBackend().synchronize()
            r = np.std(self._get_view(), axis=axis, keepdims=keepdims)
            return ndarray._from_np_direct(np.asarray(r))
        m = self.mean(axis=axis, keepdims=True)
        diff = self._binary_op(m, "sub_op")
        sq = diff._unary_op("square_op")
        result = sq.mean(axis=axis, keepdims=keepdims)
        return result._unary_op("sqrt_op")

    def var(self, axis=None, keepdims=False):
        """Variance of array elements."""
        # CPU fallback — reductions are memory-bound
        if self.size < _GPU_THRESHOLD_MEMORY:
            if self._np_data is None:
                MetalBackend().synchronize()
            r = np.var(self._get_view(), axis=axis, keepdims=keepdims)
            return ndarray._from_np_direct(np.asarray(r))
        m = self.mean(axis=axis, keepdims=True)
        diff = self._binary_op(m, "sub_op")
        sq = diff._unary_op("square_op")
        return sq.mean(axis=axis, keepdims=keepdims)

    def prod(self, axis=None, keepdims=False):
        """Product of array elements."""
        if axis is not None:
            return self._reduce_axis("reduce_prod_axis", axis, keepdims)
        result = self._reduce("reduce_prod")
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
        """Test whether any array element evaluates to True (GPU-accelerated)."""
        from . import creation
        # CPU fallback — reductions are memory-bound
        if self.size < _GPU_THRESHOLD_MEMORY:
            if self._np_data is None:
                MetalBackend().synchronize()
            r = np.any(self._get_view(), axis=axis, keepdims=keepdims)
            if axis is not None:
                return ndarray._from_np_direct(np.asarray(r))
            return bool(r)
        # GPU: compare != 0, then reduce_max (any true → max is 1)
        zero = creation.zeros(self._shape, dtype=self._dtype)
        mask = self._comparison_op(zero, "ne_op")  # bool as int32
        mask_f = mask.astype(np.float32)
        if axis is not None:
            result = mask_f.max(axis=axis, keepdims=keepdims)
            return result.astype(np.bool_)
        result = mask_f._reduce("reduce_max")
        return bool(result.get())

    def all(self, axis=None, keepdims=False):
        """Test whether all array elements evaluate to True (GPU-accelerated)."""
        from . import creation
        # CPU fallback — reductions are memory-bound
        if self.size < _GPU_THRESHOLD_MEMORY:
            if self._np_data is None:
                MetalBackend().synchronize()
            r = np.all(self._get_view(), axis=axis, keepdims=keepdims)
            if axis is not None:
                return ndarray._from_np_direct(np.asarray(r))
            return bool(r)
        # GPU: compare != 0, then reduce_min (all true → min is 1)
        zero = creation.zeros(self._shape, dtype=self._dtype)
        mask = self._comparison_op(zero, "ne_op")  # bool as int32
        mask_f = mask.astype(np.float32)
        if axis is not None:
            result = mask_f.min(axis=axis, keepdims=keepdims)
            return result.astype(np.bool_)
        result = mask_f._reduce("reduce_min")
        return bool(result.get())

    def _reduce(self, kernel_name: str):
        """Run a full-array GPU reduction and return a 0-d ndarray."""
        # CPU fallback — reductions are memory-bound, CPU wins up to ~4M
        if self.size < _GPU_THRESHOLD_MEMORY:
            np_func = _NP_REDUCE.get(kernel_name)
            if np_func is not None:
                np_data = self._np_data
                if np_data is not None:
                    scalar = np_func(np_data)
                else:
                    MetalBackend().synchronize()
                    scalar = np_func(self._get_view())
                result = np.asarray(scalar)
                arr = ndarray.__new__(ndarray)
                arr._buffer = None
                arr._np_data = result
                arr._shape = result.shape
                arr._dtype = result.dtype if result.dtype in _PASSTHROUGH_DTYPES else resolve_dtype(result.dtype)
                arr._strides = ()
                arr._offset = 0
                arr._base = None
                return arr

        backend = MetalBackend()
        cache = KernelCache()
        a = self._ensure_contiguous()
        shader = cache.get_shader("reduction", a._dtype)

        TGROUP = 1024
        n = a.size
        num_groups = (n + TGROUP - 1) // TGROUP
        grid = num_groups * TGROUP

        n_buf = backend.array_to_buffer(np.array([n], dtype=np.uint32))
        out_buf = backend.create_buffer(max(num_groups, 1), a._dtype)

        backend.execute_kernel(shader, kernel_name, grid, [a._buffer, out_buf, n_buf])

        while num_groups > 1:
            in_buf = out_buf
            n = num_groups
            num_groups = (n + TGROUP - 1) // TGROUP
            grid = num_groups * TGROUP
            n_buf = backend.array_to_buffer(np.array([n], dtype=np.uint32))
            out_buf = backend.create_buffer(max(num_groups, 1), a._dtype)
            backend.execute_kernel(shader, kernel_name, n, [in_buf, out_buf, n_buf])

        return ndarray._from_buffer(out_buf, (), a._dtype)

    def _reduce_dot(self, other):
        """Fused multiply-and-sum reduction for dot products (single kernel)."""
        backend = MetalBackend()
        cache = KernelCache()
        a = self._ensure_contiguous()
        b = other._ensure_contiguous()
        shader = cache.get_shader("reduction", a._dtype)

        TGROUP = 1024
        n = a.size
        num_groups = (n + TGROUP - 1) // TGROUP
        grid = num_groups * TGROUP

        n_buf = backend.array_to_buffer(np.array([n], dtype=np.uint32))
        out_buf = backend.create_buffer(max(num_groups, 1), a._dtype)

        backend.execute_kernel(shader, "reduce_dot", grid,
                               [a._buffer, b._buffer, out_buf, n_buf])

        while num_groups > 1:
            in_buf = out_buf
            n = num_groups
            num_groups = (n + TGROUP - 1) // TGROUP
            grid = num_groups * TGROUP
            n_buf = backend.array_to_buffer(np.array([n], dtype=np.uint32))
            out_buf = backend.create_buffer(max(num_groups, 1), a._dtype)
            backend.execute_kernel(shader, "reduce_sum", n,
                                   [in_buf, out_buf, n_buf])

        return ndarray._from_buffer(out_buf, (), a._dtype)

    def _reduce_axis(self, kernel_name: str, axis: int, keepdims: bool = False, out_dtype=None):
        """GPU axis reduction using axis_reduction_shader."""
        from . import creation
        axis = axis % self.ndim if axis < 0 else axis

        # CPU fallback — reductions are memory-bound
        if self.size < _GPU_THRESHOLD_MEMORY:
            _AXIS_NP = {
                "reduce_sum_axis": np.sum, "reduce_max_axis": np.max,
                "reduce_min_axis": np.min, "reduce_prod_axis": np.prod,
                "argmax_axis": np.argmax, "argmin_axis": np.argmin,
            }
            np_func = _AXIS_NP.get(kernel_name)
            if np_func is not None:
                if self._np_data is None:
                    MetalBackend().synchronize()
                r = np_func(self._get_view(), axis=axis, keepdims=keepdims)
                if out_dtype is not None:
                    r = np.asarray(r, dtype=out_dtype)
                return ndarray._from_np_direct(np.asarray(r))

        # Move reduction axis to last position
        axes = list(range(self.ndim))
        axes.append(axes.pop(axis))
        transposed = self.transpose(axes)

        # Reshape to 2D (outer, inner) via GPU contiguous copy
        inner = self._shape[axis]
        outer = self.size // inner
        flat = transposed._ensure_contiguous().reshape(outer, inner)._ensure_contiguous()

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
        """GPU prefix scan (cumsum/cumprod) using block-parallel scan."""
        from . import creation
        # CPU fallback — reductions are memory-bound
        if self.size < _GPU_THRESHOLD_MEMORY:
            if self._np_data is None:
                MetalBackend().synchronize()
            d = self._get_view()
            if kernel_name == "prefix_sum":
                r = np.cumsum(d, axis=axis)
            else:
                r = np.cumprod(d, axis=axis)
            return ndarray._from_np_direct(r)
        if axis is None:
            flat = self.ravel()._ensure_contiguous()
            backend = MetalBackend()
            cache = KernelCache()
            n = flat.size

            # Use block-parallel scan for large arrays
            block_size = 1024
            n_blocks = (n + block_size - 1) // block_size
            if n_blocks <= 1:
                # Small array: single-thread scan is fine
                shader = cache.get_shader("axis_reduction", flat._dtype)
                dims_buf = backend.array_to_buffer(np.array([1, n], dtype=np.uint32))
                out_buf = backend.create_buffer(n, flat._dtype)
                backend.execute_kernel(shader, kernel_name, 1, [flat._buffer, out_buf, dims_buf])
                return ndarray._from_buffer(out_buf, (n,), flat._dtype)

            is_sum = kernel_name == "prefix_sum"
            shader = cache.get_shader("parallel_scan", flat._dtype)
            params_buf = backend.array_to_buffer(np.array([n, block_size], dtype=np.uint32))
            out_buf = backend.create_buffer(n, flat._dtype)
            totals_buf = backend.create_buffer(n_blocks, flat._dtype)

            # Pass 1: block-local prefix scans
            block_kernel = "block_scan_sum" if is_sum else "block_scan_prod"
            backend.execute_kernel(
                shader, block_kernel, n_blocks,
                [flat._buffer, out_buf, totals_buf, params_buf],
            )

            # Pass 2: prefix scan on block totals (single thread, small array)
            totals_params = backend.array_to_buffer(np.array([n_blocks, 0], dtype=np.uint32))
            totals_kernel = "scan_block_totals_sum" if is_sum else "scan_block_totals_prod"
            backend.execute_kernel(
                shader, totals_kernel, 1,
                [totals_buf, totals_params],
            )

            # Pass 3: propagate block prefixes
            prop_kernel = "propagate_sum" if is_sum else "propagate_prod"
            backend.execute_kernel(
                shader, prop_kernel, n_blocks,
                [out_buf, totals_buf, params_buf],
            )

            return ndarray._from_buffer(out_buf, (n,), flat._dtype)

        axis = axis % self.ndim if axis < 0 else axis
        axes = list(range(self.ndim))
        axes.append(axes.pop(axis))
        transposed = self.transpose(axes)
        inner = self._shape[axis]
        outer = self.size // inner
        flat = transposed._ensure_contiguous().reshape(outer, inner)._ensure_contiguous()

        backend = MetalBackend()
        cache = KernelCache()
        shader = cache.get_shader("axis_reduction", flat._dtype)
        dims_buf = backend.array_to_buffer(np.array([outer, inner], dtype=np.uint32))
        out_buf = backend.create_buffer(flat.size, flat._dtype)
        backend.execute_kernel(shader, kernel_name, outer, [flat._buffer, out_buf, dims_buf])

        # Reshape back: (outer, inner) → transposed shape → inverse transpose
        result_2d = ndarray._from_buffer(out_buf, (outer, inner), flat._dtype)
        result_nd = result_2d.reshape(transposed._shape)
        inv_axes = [0] * len(axes)
        for i, a in enumerate(axes):
            inv_axes[a] = i
        return result_nd.transpose(inv_axes)

    # ------------------------------------------------------------------ type casting
    def astype(self, dtype) -> ndarray:
        """Return array cast to *dtype* (GPU-accelerated for Metal types)."""
        dtype = np.dtype(dtype)
        if dtype == self._dtype:
            return self

        # Float64/complex128: always handle on CPU (Metal doesn't support these)
        if dtype == _F64_DTYPE or dtype == _C128_DTYPE or self._dtype == _F64_DTYPE or self._dtype == _C128_DTYPE:
            np_data = self._np_data if self._np_data is not None else self.get()
            return ndarray._from_np_direct(np_data.astype(dtype, copy=False))

        # Opt 2: CPU fallback for small arrays (astype is light, CPU-resident)
        if self.size < _GPU_THRESHOLD_LIGHT and self._dtype != np.complex64:
            if self._np_data is None:
                MetalBackend().synchronize()
            return ndarray._from_np_direct(self._get_view().astype(dtype))

        _BOOL_METAL = "uchar"

        src_metal = METAL_TYPE_NAMES.get(self._dtype)
        dst_metal = METAL_TYPE_NAMES.get(dtype)
        src_is_bool = self._dtype == np.bool_
        dst_is_bool = dtype == np.bool_

        if src_is_bool:
            src_metal = _BOOL_METAL
        if dst_is_bool:
            dst_metal = _BOOL_METAL

        if src_metal and dst_metal and self.size > 0:
            a = self._ensure_contiguous()
            # Opt 5: use cached astype shader
            shader_src = get_astype_shader(src_metal, dst_metal, dst_is_bool)
            backend = MetalBackend()
            out_buf = backend.create_buffer(a.size, dtype)
            backend.execute_kernel(shader_src, "cast_op", a.size, [a._buffer, out_buf])
            return ndarray._from_buffer(out_buf, self._shape, dtype)

        # CPU fallback for complex64 and other unsupported types
        np_data = self.get().astype(dtype)
        backend = MetalBackend()
        buf = backend.array_to_buffer(np.ascontiguousarray(np_data))
        return ndarray._from_buffer(buf, self._shape, dtype)

    # ------------------------------------------------------------------ copy
    def copy(self) -> ndarray:
        """Return a deep copy."""
        # CPU fast path — inline to avoid _from_np_direct overhead
        if self._np_data is not None:
            result = self._np_data.copy()
            arr = ndarray.__new__(ndarray)
            arr._buffer = None
            arr._np_data = result
            arr._shape = result.shape
            arr._dtype = self._dtype
            arr._strides = _c_contiguous_strides(result.shape)
            arr._offset = 0
            arr._base = None
            return arr
        if self._dtype in METAL_TYPE_NAMES and self.size > 0:
            a = self._ensure_contiguous()
            metal_type = METAL_TYPE_NAMES[self._dtype]
            shader_src = get_copy_shader(metal_type)
            backend = MetalBackend()
            out_buf = backend.create_buffer(a.size, self._dtype)
            backend.execute_kernel(shader_src, "copy_buf", a.size, [a._buffer, out_buf])
            return ndarray._from_buffer(out_buf, self._shape, self._dtype)
        # CPU fallback
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
            return ndarray._from_np_direct(np.conj(self.get()))
        return self.copy()

    def conjugate(self):
        """Return the complex conjugate (alias for conj)."""
        return self.conj()

    def dot(self, b, out=None):
        """Dot product of two arrays."""
        if self._np_data is not None:
            a_np = self._np_data
        else:
            a_np = self.get()
        b_np = b.get() if hasattr(b, 'get') else b
        return ndarray._from_np_direct(np.dot(a_np, b_np))

    def swapaxes(self, axis1, axis2):
        """Interchange two axes of the array."""
        if self._np_data is not None:
            return ndarray._from_np_direct(np.swapaxes(self._np_data, axis1, axis2))
        return ndarray._from_np_direct(np.swapaxes(self.get(), axis1, axis2))

    def byteswap(self, inplace=False):
        """Swap bytes of the array elements."""
        return ndarray._from_np_direct(self.get().byteswap(inplace=False))

    def itemset(self, *args):
        """Insert scalar into the array.

        Usage: a.itemset(value) for 0-d arrays,
               a.itemset(index, value) for flat index,
               a.itemset((i, j, ...), value) for multi-index.
        """
        np_arr = self.get()
        if len(args) == 1:
            # 0-d array: itemset(value)
            np_arr.flat[0] = args[0]
        elif len(args) == 2:
            index, value = args
            if isinstance(index, tuple):
                # Multi-dimensional index: a.itemset((i, j), value)
                np_arr[index] = value
            else:
                # Flat index: a.itemset(flat_index, value)
                np_arr.flat[index] = value
        else:
            raise TypeError(
                f"itemset() takes 1 or 2 arguments, got {len(args)}"
            )
        self.set(np_arr)

    def dump(self, file):
        """Dump a pickle of the array to the specified file."""
        self.get().dump(file)

    def dumps(self):
        """Return the pickle of the array as a string."""
        return self.get().dumps()

    def tofile(self, fid, sep='', format='%s'):
        """Write array to a file as text or binary."""
        self.get().tofile(fid, sep=sep, format=format)

    def tostring(self, order='C'):
        """Return the array data as bytes (deprecated alias for tobytes)."""
        return self.tobytes(order=order)

    def newbyteorder(self, new_order='S'):
        """Return the array with the same data viewed with a different byte order."""
        from . import creation
        np_data = self.get()
        # Use view with dtype.newbyteorder() instead of removed ndarray.newbyteorder()
        new_dt = np_data.dtype.newbyteorder(new_order)
        np_result = np_data.view(new_dt)
        # Convert back to native byte order for GPU storage
        return creation.array(np_result.astype(np_result.dtype.newbyteorder('=')))

    def getfield(self, dtype, offset=0):
        """Return a field of the given array as a certain type."""
        from . import creation
        return creation.array(self.get().getfield(dtype, offset))

    def setfield(self, val, dtype, offset=0):
        """Put a value into a specified place in a field defined by a data-type."""
        np_arr = self.get()
        np_arr.setfield(val, dtype, offset)
        self.set(np_arr)

    def setflags(self, write=None, align=None, uic=None):
        """Set array flags (no-op for GPU arrays)."""
        pass

    def resize(self, new_shape, refcheck=True):
        """Change shape and size of array in-place."""
        np_arr = self.get()
        np_arr.resize(new_shape, refcheck=False)
        from . import creation
        new = creation.array(np_arr)
        self._buffer = new._buffer
        self._np_data = new._np_data
        self._shape = new._shape
        self._strides = new._strides
        self._offset = new._offset

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
        self._adopt_buffer(result._ensure_contiguous()._buffer)

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
        self._adopt_buffer(result._ensure_contiguous()._buffer)

    def argpartition(self, kth, axis=-1):
        """Return indices that would partition the array."""
        from . import sorting as sorting_mod
        return sorting_mod.argpartition(self, kth, axis=axis)

    def tobytes(self, order='C'):
        """Return the array data as bytes."""
        return self.get().tobytes(order=order)

    def view(self, dtype):
        """View the array with a different dtype (reinterpret buffer)."""
        dtype = np.dtype(dtype)
        np_data = self.get()
        viewed = np_data.view(dtype)
        return ndarray._from_np_direct(viewed)

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
        result = ufunc_ops.floor_divide(self, other).astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
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
        result = math_ops.mod(self, other).astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
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
        if np.issubdtype(self._dtype, np.integer):
            from . import bitwise_ops
            result = bitwise_ops.bitwise_and(self, other)
        else:
            result = self._boolean_op(other, "and_op")
        result = result.astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
        return self

    # XOR ^
    def __xor__(self, other):
        if np.issubdtype(self._dtype, np.integer):
            from . import bitwise_ops
            return bitwise_ops.bitwise_xor(self, other)
        return self._boolean_op(other, "xor_op")

    # In-place ^=
    def __ixor__(self, other):
        if np.issubdtype(self._dtype, np.integer):
            from . import bitwise_ops
            result = bitwise_ops.bitwise_xor(self, other)
        else:
            result = self._boolean_op(other, "xor_op")
        result = result.astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
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
        self._adopt_buffer(result._ensure_contiguous()._buffer)
        return self

    # In-place left shift <<=
    def __ilshift__(self, other):
        from . import bitwise_ops
        result = bitwise_ops.left_shift(self, other).astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
        return self

    # In-place right shift >>=
    def __irshift__(self, other):
        from . import bitwise_ops
        result = bitwise_ops.right_shift(self, other).astype(self._dtype)
        self._adopt_buffer(result._ensure_contiguous()._buffer)
        return self

    # Reflected bitwise AND
    def __rand__(self, other):
        if np.issubdtype(self._dtype, np.integer):
            from . import bitwise_ops
            return bitwise_ops.bitwise_and(other, self)
        return self._boolean_op(other, "and_op")

    # Reflected bitwise OR
    def __ror__(self, other):
        if np.issubdtype(self._dtype, np.integer):
            from . import bitwise_ops
            return bitwise_ops.bitwise_or(other, self)
        return self._boolean_op(other, "or_op")

    # Reflected bitwise XOR
    def __rxor__(self, other):
        if np.issubdtype(self._dtype, np.integer):
            from . import bitwise_ops
            return bitwise_ops.bitwise_xor(other, self)
        return self._boolean_op(other, "xor_op")

    # Reflected divmod
    def __rdivmod__(self, other):
        return (other // self, other % self)

    # divmod
    def __divmod__(self, other):
        return (self // other, self % other)

    # Iteration over first axis
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # copy module support
    def __copy__(self):
        from . import creation
        return creation.array(self.get().copy())

    def __deepcopy__(self, memo):
        from . import creation
        return creation.array(self.get().copy())


# ── C dispatch table initialization ──
_fast_binary = _fast_unary = _fast_cmp = None

if _accelerator is not None:
    try:
        _FLAG_FLOAT_ONLY = 1
        _accelerator.init_dispatch(
            ndarray, _BOOL_DTYPE,
            [  # binary_list - indexed by op_id
                (np.add, _BINARY_THRESHOLD["add_op"], 0),           # 0 = _OP_ADD
                (np.subtract, _BINARY_THRESHOLD["sub_op"], 0),      # 1 = _OP_SUB
                (np.multiply, _BINARY_THRESHOLD["mul_op"], 0),      # 2 = _OP_MUL
                (np.true_divide, _BINARY_THRESHOLD["div_op"], _FLAG_FLOAT_ONLY),  # 3 = _OP_DIV
                (np.power, _BINARY_THRESHOLD.get("pow_op", _GPU_THRESHOLD), 0),  # 4 = _OP_POW
                (np.mod, _BINARY_THRESHOLD.get("mod_op", _GPU_THRESHOLD), 0),    # 5 = _OP_MOD
                (np.floor_divide, _BINARY_THRESHOLD.get("floor_divide_op", _GPU_THRESHOLD), 0),  # 6 = _OP_FLOOR_DIV
            ],
            [  # unary_list - indexed by op_id
                (np.negative, _UNARY_THRESHOLD["neg_op"], 0),  # 0 = _UOP_NEG
                (np.abs, _UNARY_THRESHOLD["abs_op"], 0),       # 1 = _UOP_ABS
            ],
            [  # cmp_list - indexed by op_id
                (np.less, _GPU_THRESHOLD_MEMORY, 0),           # 0 = _COP_LT
                (np.less_equal, _GPU_THRESHOLD_MEMORY, 0),     # 1 = _COP_LE
                (np.greater, _GPU_THRESHOLD_MEMORY, 0),        # 2 = _COP_GT
                (np.greater_equal, _GPU_THRESHOLD_MEMORY, 0),  # 3 = _COP_GE
                (np.equal, _GPU_THRESHOLD_MEMORY, 0),          # 4 = _COP_EQ
                (np.not_equal, _GPU_THRESHOLD_MEMORY, 0),      # 5 = _COP_NE
            ],
        )
        _fast_binary = _accelerator.fast_binary
        _fast_unary = _accelerator.fast_unary
        _fast_cmp = _accelerator.fast_cmp
    except Exception:
        pass
