"""User-defined Metal kernel support (analogous to CuPy's RawKernel)."""

from __future__ import annotations

from typing import Sequence, Union

from ._metal_backend import MetalBackend, MetalSize
from .ndarray import ndarray

__all__ = ["RawKernel"]

_SYNC_KERNEL = "\nkernel void _sync() {}\n"


class RawKernel:
    """Compile and launch a user-written Metal Shading Language kernel.

    Parameters
    ----------
    source : str
        MSL source code containing at least one kernel function.
    func_name : str
        Name of the kernel function to invoke.
    """

    def __init__(self, source: str, func_name: str) -> None:
        if "_sync" not in source:
            source = source.rstrip() + _SYNC_KERNEL
        self._source = source
        self._func_name = func_name

    def __call__(
        self,
        grid: Union[int, tuple],
        args: Sequence[ndarray],
    ) -> None:
        """Dispatch the kernel.

        Parameters
        ----------
        grid : int or tuple
            Number of threads (1-D int) or ``(width, height, depth)`` tuple.
        args : sequence of ndarray
            GPU arrays passed as ``[[buffer(0)]], [[buffer(1)]], ...``
        """
        backend = MetalBackend()
        if isinstance(grid, tuple):
            if len(grid) == 1:
                grid = MetalSize(grid[0], 1, 1)
            elif len(grid) == 2:
                grid = MetalSize(grid[0], grid[1], 1)
            elif len(grid) == 3:
                grid = MetalSize(*grid)
            else:
                raise ValueError("grid tuple must have 1-3 elements")
        for a in args:
            a._ensure_gpu()
        buffers = [a._buffer for a in args]
        backend.execute_kernel(self._source, self._func_name, grid, buffers)
        # GPU kernel may have written to any buffer; clear stale CPU data
        for a in args:
            a._np_data = None
