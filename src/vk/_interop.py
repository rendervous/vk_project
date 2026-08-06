"""Adds .torch()/.numpy()/.jax()/.cupy()/.tensorflow() convenience methods
to Buffer/Tensor, and .as_torch()/.as_numpy()/.as_jax()/.as_cupy()/
.as_tensorflow() to WrappedMemory, each lazily importing the named library
and constructing a tensor/array from this object via the DLPack protocol
(__dlpack__/__dlpack_device__, already implemented by all three classes).

None of these libraries are hard dependencies (see pyproject.toml) -- each
import is attempted only when its method is actually called, raising the
library's own ImportError if it isn't installed.
"""

import importlib

from .vk import Buffer, Tensor, WrappedMemory


def _dlpack_method(module_name: str, attr_path: str, capsule: bool = False):
    """Builds a method that imports `module_name` on first call and
    invokes the callable found by walking `attr_path` (dotted, e.g.
    "dlpack.from_dlpack") from it, on self -- passing self directly (the
    common case: torch/numpy/jax/cupy all accept a __dlpack__-supporting
    object as-is), or self.__dlpack__() if `capsule` (tensorflow's
    from_dlpack expects a raw PyCapsule, not an object).
    """

    def method(self):
        mod = importlib.import_module(module_name)
        from_dlpack = mod
        for part in attr_path.split("."):
            from_dlpack = getattr(from_dlpack, part)
        return from_dlpack(self.__dlpack__() if capsule else self)

    method.__name__ = module_name
    method.__doc__ = f"Returns this buffer/tensor as a `{module_name}` tensor/array, via the DLPack protocol."
    return method


for _cls in (Buffer, Tensor):
    _cls.torch = _dlpack_method("torch", "from_dlpack")
    _cls.numpy = _dlpack_method("numpy", "from_dlpack")
    _cls.jax = _dlpack_method("jax", "dlpack.from_dlpack")
    _cls.cupy = _dlpack_method("cupy", "from_dlpack")
    _cls.tensorflow = _dlpack_method("tensorflow", "experimental.dlpack.from_dlpack", capsule=True)

# WrappedMemory exports its own device_ptr_ directly (not the originally
# wrapped Python object) as a tensor/array backed by the very same memory --
# named as_*() rather than bare torch()/numpy()/... to make that distinction
# (e.g. from a Buffer/Tensor already wrapped: as_torch() and the original
# .torch() view the identical memory either way, since wrap() never copies
# one of this library's own resources).
WrappedMemory.as_torch = _dlpack_method("torch", "from_dlpack")
WrappedMemory.as_numpy = _dlpack_method("numpy", "from_dlpack")
WrappedMemory.as_jax = _dlpack_method("jax", "dlpack.from_dlpack")
WrappedMemory.as_cupy = _dlpack_method("cupy", "from_dlpack")
WrappedMemory.as_tensorflow = _dlpack_method("tensorflow", "experimental.dlpack.from_dlpack", capsule=True)
