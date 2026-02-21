"""Thread-safe singleton wrapper around the MetalGPU native library.

We load the ``libmetalgpucpp-arm.dylib`` compiled by ``python -m metalgpu build``
directly via ctypes with correct argument types.  The PyPI metalgpu package is
kept as a dependency only for its build tooling (compiling the C++ shared lib).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import threading

import numpy as np
import numpy.ctypeslib

from ._dtypes import numpy_to_metal

__all__ = ["MetalBackend"]

# ── dtype → ctypes type ──────────────────────────────────────────────
_NP_TO_CTYPE: dict[np.dtype, type] = {
    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.float16): ctypes.c_int16,   # no c_half; use int16
    np.dtype(np.int32): ctypes.c_int,
    np.dtype(np.uint32): ctypes.c_uint,
    np.dtype(np.int64): ctypes.c_long,
    np.dtype(np.uint64): ctypes.c_ulong,
    np.dtype(np.int16): ctypes.c_int16,
    np.dtype(np.uint16): ctypes.c_uint16,
    np.dtype(np.bool_): ctypes.c_bool,
    np.dtype(np.complex64): ctypes.c_float,  # stored as float32 pairs
}

# ── Metal type string → ctypes type (mirrors metalgpu's utils.py) ───
_METAL_TO_CTYPE: dict[str, type] = {
    "float": ctypes.c_float,
    "half": ctypes.c_int16,
    "int": ctypes.c_int,
    "uint": ctypes.c_uint,
    "long": ctypes.c_long,
    "uint64_t": ctypes.c_ulong,
    "short": ctypes.c_int16,
    "uint16_t": ctypes.c_uint16,
    "bool": ctypes.c_bool,
}


def _find_dylib() -> str:
    """Locate the compiled MetalGPU shared library."""
    import metalgpu as _mg

    dylib = os.path.join(os.path.dirname(_mg.__file__), "lib", "libmetalgpucpp-arm.dylib")
    if not os.path.isfile(dylib):
        raise RuntimeError(
            "MetalGPU native library not found. "
            "Run `python -m metalgpu build` first."
        )
    return dylib


class _Buffer:
    """Lightweight wrapper around a single Metal buffer."""

    __slots__ = ("contents", "bufNum", "_interface_ptr", "_lib", "bufType")

    def __init__(
        self,
        pointer,
        size: int,
        dtype: np.dtype,
        buf_num: int,
        interface_ptr,
        lib,
    ) -> None:
        dtype = np.dtype(dtype)
        self.bufNum = buf_num
        self.bufType = dtype
        self._interface_ptr = interface_ptr
        self._lib = lib
        if size == 0:
            # Zero-size buffer: use an empty numpy array (no Metal memory)
            storage_dtype = np.float32 if dtype == np.complex64 else dtype
            self.contents: np.ndarray = np.empty((0,), dtype=storage_dtype)
        else:
            ctype = _NP_TO_CTYPE[dtype]
            lib.getBufferPointer.restype = ctypes.POINTER(ctype)
            raw_ptr = lib.getBufferPointer(interface_ptr, buf_num)
            self.contents = numpy.ctypeslib.as_array(raw_ptr, shape=(size,))

    def release(self) -> None:
        if self.bufNum >= 0:
            self._lib.releaseBuffer(self._interface_ptr, self.bufNum)


class MetalSize:
    """3-D grid size descriptor."""

    __slots__ = ("width", "height", "depth")

    def __init__(self, width: int, height: int, depth: int) -> None:
        self.width = width
        self.height = height
        self.depth = depth


class MetalBackend:
    """Singleton providing thread-safe access to the Metal GPU.

    All GPU interaction (buffer creation, kernel dispatch) goes through this
    class so that shader loading is deduplicated and access is serialised.
    """

    _instance: MetalBackend | None = None
    _init_lock = threading.Lock()

    def __new__(cls) -> MetalBackend:
        with cls._init_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._setup()
                cls._instance = inst
            return cls._instance

    def _setup(self) -> None:
        dylib_path = _find_dylib()
        self._lib = ctypes.cdll.LoadLibrary(dylib_path)
        self._declare_api()
        self._ptr = self._lib.init()  # Instance* pointer
        self._lock = threading.Lock()
        self._current_shader_hash: str | None = None
        self._totbuf = -1
        self._has_pending = False

        # Load a no-op shader so the library is in a valid state
        noop = b"#include <metal_stdlib>\nusing namespace metal;\nkernel void _noop() {}\nkernel void _sync() {}\n"
        self._lib.createLibraryFromString(self._ptr, noop)
        self._lib.setFunction(self._ptr, b"_noop")

    def _declare_api(self) -> None:
        """Declare argtypes / restypes for every C entry point."""
        L = self._lib
        vp = ctypes.c_void_p  # Instance*

        L.init.argtypes = []
        L.init.restype = vp

        L.deleteInstance.argtypes = [vp]
        L.deleteInstance.restype = None

        L.createBuffer.argtypes = [vp, ctypes.c_int]
        L.createBuffer.restype = ctypes.c_int

        L.releaseBuffer.argtypes = [vp, ctypes.c_int]
        L.releaseBuffer.restype = None

        L.getBufferPointer.argtypes = [vp, ctypes.c_int]
        L.getBufferPointer.restype = ctypes.POINTER(ctypes.c_int)  # overridden per-call

        L.createLibrary.argtypes = [vp, ctypes.c_char_p]
        L.createLibrary.restype = None

        L.createLibraryFromString.argtypes = [vp, ctypes.c_char_p]
        L.createLibraryFromString.restype = None

        L.setFunction.argtypes = [vp, ctypes.c_char_p]
        L.setFunction.restype = None

        L.runFunction.argtypes = [
            vp,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_bool,
        ]
        L.runFunction.restype = None

        L.maxThreadsPerGroup.argtypes = [vp]
        L.maxThreadsPerGroup.restype = ctypes.c_int

        L.threadExecutionWidth.argtypes = [vp]
        L.threadExecutionWidth.restype = ctypes.c_int

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def create_buffer(self, size: int, dtype: np.dtype) -> _Buffer:
        """Create an uninitialised Metal buffer for *size* elements of *dtype*."""
        dtype = np.dtype(dtype)
        # complex64 is stored as float32 pairs → double the element count
        buf_elems = size * 2 if dtype == np.complex64 else size
        if buf_elems == 0:
            return _Buffer(None, 0, dtype, -1, self._ptr, self._lib)
        ctype = _NP_TO_CTYPE[dtype]
        byte_size = ctypes.sizeof(ctype) * buf_elems
        buf_num = self._lib.createBuffer(self._ptr, byte_size)
        return _Buffer(None, buf_elems, dtype, buf_num, self._ptr, self._lib)

    def array_to_buffer(self, np_array: np.ndarray) -> _Buffer:
        """Create a Metal buffer and copy *np_array* into it."""
        if np_array.ndim == 0:
            np_array = np_array.reshape(1)
        np_array = np.ascontiguousarray(np_array)
        size = np_array.size
        dtype = np.dtype(np_array.dtype)
        if size == 0:
            return self.create_buffer(0, dtype)
        if dtype == np.complex64:
            # View complex64 as float32 pairs for Metal storage
            float_view = np_array.view(np.float32).ravel()
            buf = self.create_buffer(size, dtype)
            buf.contents[:] = float_view
        elif dtype == np.float16:
            buf = self.create_buffer(size, dtype)
            buf.contents[:] = np_array.ravel().view(np.int16)
        else:
            buf = self.create_buffer(size, dtype)
            buf.contents[:] = np_array.ravel()
        return buf

    # ------------------------------------------------------------------
    # Kernel execution
    # ------------------------------------------------------------------

    def execute_kernel(
        self,
        shader_src: str,
        func_name: str,
        grid_size: int | MetalSize,
        buffers: list[_Buffer],
    ) -> None:
        """Compile (if needed), select, and dispatch a Metal kernel."""
        src_hash = hashlib.sha256(shader_src.encode()).hexdigest()

        if isinstance(grid_size, int):
            grid_size = MetalSize(grid_size, 1, 1)

        buf_nums = np.array(
            [b.bufNum if b is not None else -1 for b in buffers],
            dtype=np.int32,
        )
        metal_size = np.array(
            [grid_size.width, grid_size.height, grid_size.depth],
            dtype=np.int32,
        )
        size_ptr = metal_size.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        buf_ptr = buf_nums.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        with self._lock:
            if src_hash != self._current_shader_hash:
                self._lib.createLibraryFromString(
                    self._ptr, shader_src.encode("utf-8")
                )
                self._current_shader_hash = src_hash
            self._lib.setFunction(self._ptr, func_name.encode("utf-8"))
            self._lib.runFunction(
                self._ptr, size_ptr, buf_ptr, len(buf_nums), False
            )
            self._has_pending = True

    def synchronize(self) -> None:
        """Block until all pending GPU work completes."""
        if not self._has_pending:
            return
        with self._lock:
            if not self._has_pending:
                return
            self._lib.setFunction(self._ptr, b"_sync")
            size = np.array([1, 1, 1], dtype=np.int32)
            bufs = np.array([], dtype=np.int32)
            size_ptr = size.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            buf_ptr = bufs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            self._lib.runFunction(self._ptr, size_ptr, buf_ptr, 0, True)
            self._has_pending = False
