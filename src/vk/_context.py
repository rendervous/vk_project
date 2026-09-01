"""Implicit "current device"/"current engine" context.

Hides :class:`~vk.Device` and :class:`~vk.Engine` objects from typical user
code. Call :func:`device`/:func:`engine` to explicitly activate one
(optionally as a context manager, restoring the previously active one on
exit), or just start calling the free functions below directly (``buffer``,
``image``, ``command_buffer``, ``submit``, ``wait``, etc.) -- the first one
called lazily activates sensible defaults: device 0, and a combined
GRAPHICS|COMPUTE engine (falling back to a plain COMPUTE engine if this
device has no queue family supporting both).

The device/engine registry and every one of these free functions are native
(C++) now -- see cpp/context.hpp/context.cpp -- this module is just a thin
Python-side shim: it re-exports most of them under their public names
as-is, and adds two pieces of logic that can't move to C++ as-is:

- buffer()'s single-instance convenience form takes a raw TypeSpec (a bare
  Type, or an arbitrarily nested ``[count, element]``/``{name: field,
  ...}`` structure of plain Python data), which has to be walked in Python
  to turn into a Layout before it can be handed to the native,
  Layout-based buffer() overload.
- device()/engine() are wrapped in a Python-side "is this already current"
  cache (see _current_device_index/_current_engine_args below) so that
  re-entering the same device()/engine() pair -- the common case at the
  top of a hot dispatch loop -- short-circuits in pure Python instead of
  paying a pybind11 call into the native registry every time.
"""

from typing import Optional, Union, overload

from .vk import Buffer, EngineType, Format, Image, Layout, LayoutRule, MemoryLocation, Type, TypeKind
from .vk import (
    ads,
    caps,
    command_buffer,
    compute,
    depth_buffer_image,
    device_infos,
    graphics,
    image,
    load_scene,
    pipeline,
    sampler,
    submit,
    tensor,
    transfer,
    wait,
    window,
    wrap,
)
from .vk import buffer as _native_buffer
from .vk import staging as _native_staging
from .vk import device as _native_device
from .vk import engine as _native_engine
from .vk import dispose as _native_dispose
from .vk import relax as _native_relax
from ._declarations import TypeSpec, _to_type_descriptor


# ---- Python-side "is this already current" cache ----
#
# Mirrors the native registry's current-device-index/current-engine-args
# state (see context.cpp), populated only by successful device()/engine()
# calls through the wrappers below and invalidated by dispose()/relax() --
# the only other entry points that can change what's current on the native
# side. Kept in sync by construction: every native mutation of "current"
# state is reachable only through these 4 wrappers, so as long as callers
# don't reach past them into vk.vk.device()/vk.vk.engine() directly, this
# cache can never disagree with the native registry it mirrors.
#
# _current_device_index starts at (and falls back to, on dispose()/relax()
# and on every device()/__exit__ that has nothing more specific to restore)
# 0 rather than an "unset" sentinel: device 0 *is* vulky's documented lazy
# default (see the module docstring), so assuming it up front is never
# wrong -- it just means the very first device(0) is itself a no-op, same
# as every other one, with the actual device created later, on demand, by
# whatever native call first needs one (current_device()'s own lazy
# resolve_device_index(0) fallback -- see context.cpp).
_current_device_index: int = 0
_current_engine_args: dict = {}  # device_index -> (engine_type, engine_index) last passed to engine()


class _NoOpContext:
    """Returned by device()/engine() when the request is already current:
    both __enter__ and __exit__ do nothing, since nothing needed to change
    (and so nothing needs restoring on exit either).
    """

    def __enter__(self) -> "_NoOpContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


class _DeviceContext:
    """Returned by device() when it actually switched devices. Wraps the
    native context object purely to also restore `_current_device_index`
    (in addition to the native side's own active-device pointer) on exit.
    """

    def __init__(self, native_context, previous_index: int):
        self._native = native_context
        self._previous_index = previous_index
        self._native.__enter__()

    def __enter__(self) -> "_DeviceContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        global _current_device_index
        result = self._native.__exit__(exc_type, exc_val, exc_tb)
        _current_device_index = self._previous_index
        return bool(result)


class _EngineContext:
    """Returned by engine() when it actually switched engines. Wraps the
    native context object purely to also restore `_current_engine_args`
    for `device_idx` (in addition to the native side's own active-engine
    pointer) on exit.
    """

    def __init__(self, native_context, device_idx: int, previous_args: Optional[tuple]):
        self._native = native_context
        self._device_idx = device_idx
        self._previous_args = previous_args

    def __enter__(self) -> "_EngineContext":
        self._native.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        result = self._native.__exit__(exc_type, exc_val, exc_tb)
        if self._previous_args is None:
            _current_engine_args.pop(self._device_idx, None)
        else:
            _current_engine_args[self._device_idx] = self._previous_args
        return bool(result)


def device(device_index: int = 0):
    """Creates/activates the device at `device_index` as the current
    device. Used as a context manager (``with vk.device(1): ...``), the
    previously active device (device 0, if device() was never called
    before) is restored on exit; used as a plain statement, the switch is
    permanent.

    Already-current requests (the common case at the top of a hot dispatch
    loop re-entering the same device every iteration -- including
    `device(0)` before any device() call at all, since 0 is the assumed
    default) are resolved entirely in Python, without calling into the
    native registry.

    :param device_index: Index of the physical Vulkan device to use.
    """
    global _current_device_index
    if _current_device_index == device_index:
        return _NoOpContext()
    previous_index = _current_device_index
    native_context = _native_device(device_index)
    _current_device_index = device_index
    return _DeviceContext(native_context, previous_index)


def device_index() -> int:
    """The current/active device's index (0, vulky's implicit default, if
    device() was never called).
    """
    return _current_device_index


def engine(engine_type: Optional[EngineType] = None, engine_index: int = 0):
    """Creates/activates an engine on the current device as the current
    engine. Used as a context manager, the previously active engine (for
    this device) is restored on exit; used as a plain statement, the
    switch is permanent.

    Already-current requests (the common case at the top of a hot dispatch
    loop re-entering the same engine every iteration) are resolved
    entirely in Python, without calling into the native registry.

    :param engine_type: Capability requested for this engine. ``None``
        (default) tries a combined GRAPHICS|COMPUTE engine first, falling
        back to a plain COMPUTE engine if unsupported.
    :param engine_index: Index selecting among multiple queues that
        support the requested capability, when more than one is available.
    """
    di = _current_device_index
    args = (engine_type, engine_index)
    if _current_engine_args.get(di) == args:
        return _NoOpContext()
    native_context = _native_engine(engine_type, engine_index)
    previous_args = _current_engine_args.get(di)
    _current_engine_args[di] = args
    return _EngineContext(native_context, di, previous_args)


def dispose() -> None:
    """Disposes the current device and drops it from the registry."""
    global _current_device_index
    _native_dispose()
    _current_engine_args.pop(_current_device_index, None)
    _current_device_index = 0


def relax() -> None:
    """Drops every reference this module keeps to created devices/engines,
    without disposing them.
    """
    global _current_device_index
    _native_relax()
    _current_device_index = 0
    _current_engine_args.clear()


def _to_type_descriptor_layout(spec: TypeSpec) -> Layout:
    """Computes a Layout for a raw type spec under LayoutRule.Scalar (the
    natural, tightly-packed rule for a GPU-visible buffer -- as opposed to
    a uniform-block field, where Std140/Std430 padding rules would apply).
    Used by buffer()'s single-instance convenience form.
    """
    from .vk import compute_layout as _vk_compute_layout
    return _vk_compute_layout(_to_type_descriptor(spec), LayoutRule.Scalar)


def _dynamic_tail_layout(layout: Layout) -> Optional[Layout]:
    """Returns the element Layout of `layout`'s trailing unsized/runtime
    array -- `layout` itself an ARRAY with count == 0, or a STRUCT whose
    last field is one (the classic SSBO "fixed header + flexible array
    member" shape) -- or None if `layout` doesn't end in one.
    """
    if layout.kind == TypeKind.ARRAY and layout.count == 0:
        return layout.element_layout
    if layout.kind == TypeKind.STRUCT and layout.fields:
        last_field_layout = layout.fields[-1].layout
        if last_field_layout.kind == TypeKind.ARRAY and last_field_layout.count == 0:
            return last_field_layout.element_layout
    return None


@overload
def buffer(elements: int, element_type: Type, location: MemoryLocation = MemoryLocation.DEVICE) -> Buffer:
    """Creates a Buffer of `elements` of type `element_type` on the
    current device. The resulting Buffer's element_layout is exactly `element_type`.
    The size is determined as elements * element_layout.aligned_size.
    """
    ...


@overload
def buffer(elements: int, format: Format, location: MemoryLocation = MemoryLocation.DEVICE) -> Buffer:
    """Creates a Buffer of `elements` texels of `format` on the current
    device, via Device.create_buffer(). The resulting Buffer's
    element_layout is an ARRAY-kind Layout of `format`'s own per-channel
    scalar type (e.g. ``Format.RGBA8_UNorm`` -> an array of 4
    ``Type.UINT8``), not `format` itself.
    """
    ...


@overload
def buffer(elements: int, layout: Layout, location: MemoryLocation = MemoryLocation.DEVICE) -> Buffer:
    """Creates a Buffer sized to hold `elements` instances of `layout` on
    the current device, via Device.create_buffer(). The resulting
    Buffer's element_layout is `layout` itself, and its byte size is
    exactly ``elements * layout.aligned_size`` (the per-element stride
    accounts for array-stride padding even when ``elements == 1``).
    """
    ...


@overload
def buffer(
    type: "TypeSpec",
    location: MemoryLocation = MemoryLocation.DEVICE,
    dynamic_size: int = 1,
) -> Buffer:
    """Creates a Buffer holding a single instance of `type` (a bare
    ``Type``, an ``[count, element]``/``{name: field, ...}`` type spec, or
    an already-computed Layout) on the current device -- a convenience
    equivalent to ``buffer(1, type if isinstance(type, Layout) else
    compute_layout(type, LayoutRule.Scalar), location)``.

    If `type`'s Layout ends in an unsized/runtime array (an ``[0,
    element]`` spec, or a struct whose last field is one -- the classic
    SSBO "fixed header + flexible array member" shape), `dynamic_size` is
    the number of elements actually reserved for that trailing array
    (default 1); ignored otherwise.
    """
    ...


def buffer(*args, **kwargs) -> Buffer:
    """Creates a Buffer on the current device. Most IDEs show this
    docstring (not the @overload stubs above) at the call site, so it
    repeats all 4 accepted forms:

    - ``buffer(elements, scalar_type, location=DEVICE)``: `elements`
      scalars of `scalar_type`; the resulting Buffer's element_layout is
      exactly `scalar_type`.
    - ``buffer(elements, format, location=DEVICE)``: `elements` texels of
      `format`; element_layout is an array of `format`'s own per-channel
      scalar type.
    - ``buffer(elements, layout, location=DEVICE)``: `elements` instances
      of `layout` (an already-computed Layout); element_layout is
      `layout` itself, byte size exactly ``elements * layout.aligned_size``.
    - ``buffer(type, location=DEVICE, dynamic_size=1)``: a single instance
      (elements=1 implied) of `type` -- a bare Type, an ``[count,
      element]``/``{name: field, ...}`` type spec, or an already-computed
      Layout. If `type`'s Layout ends in an unsized/runtime array,
      `dynamic_size` reserves that many elements for it (ignored
      otherwise).

    `location` defaults to DEVICE in every form (the native buffer()
    itself has no default for it in the first 3 forms).
    """
    if args and not isinstance(args[0], int):
        type_or_layout = args[0]
        layout = type_or_layout if isinstance(type_or_layout, Layout) else _to_type_descriptor_layout(type_or_layout)
        location = args[1] if len(args) > 1 else kwargs.get("location", MemoryLocation.DEVICE)
        dynamic_size = args[2] if len(args) > 2 else kwargs.get("dynamic_size", 1)
        tail_layout = _dynamic_tail_layout(layout)
        if tail_layout is not None:
            total_bytes = layout.aligned_size + dynamic_size * tail_layout.aligned_size
            raw = _native_buffer(total_bytes, Type.UINT8, location)
            return raw.cast(layout)
        return _native_buffer(1, layout, location)
    if "location" not in kwargs and len(args) < 3:
        kwargs["location"] = MemoryLocation.DEVICE
    return _native_buffer(*args, **kwargs)


@overload
def staging(buffer: Buffer, location: MemoryLocation = MemoryLocation.HOST) -> Buffer:
    """Creates a plain, byte-addressable staging Buffer sized to match
    `buffer`'s own ``size`` on the current device -- the correct Vulkan
    pattern for moving data between CPU and GPU memory (record a
    command_buffer().transfer() between it and `buffer` rather than
    mapping device-local memory directly). The resulting Buffer's
    element_layout is a plain ``Type.UINT8`` array (untyped raw bytes),
    regardless of `buffer`'s own element type.
    """
    ...


@overload
def staging(image: Image, location: MemoryLocation = MemoryLocation.HOST) -> Buffer:
    """Creates a plain, byte-addressable staging Buffer sized to match
    `image`'s own backing store on the current device, for staged CPU/GPU
    transfer of its texel data. The resulting Buffer's element_layout is
    a plain ``Type.UINT8`` array (untyped raw bytes).
    """
    ...


def staging(*args, **kwargs) -> Buffer:
    """Creates a plain, byte-addressable staging Buffer on the current
    device. Most IDEs show this docstring (not the @overload stubs above)
    at the call site, so it repeats both accepted forms:

    - ``staging(buffer, location=HOST)``: sized to match `buffer`'s own
      size.
    - ``staging(image, location=HOST)``: sized to match `image`'s own
      backing store; element_layout is a properly-shaped (height, width,
      texel_bytes) nested array for a plain 2D/single-mip/single-layer
      image, or a flat byte array otherwise.

    Either way, element_layout is untyped (UINT8-based) regardless of the
    source's own element type -- the correct Vulkan pattern for moving
    data between CPU and GPU memory is to record a
    ``command_buffer().transfer()`` between this buffer and the source.
    """
    return _native_staging(*args, **kwargs)
