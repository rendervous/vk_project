"""Implicit "current device"/"current engine" context.

Hides :class:`~vk.Device` and :class:`~vk.Engine` objects from typical user
code. Call :func:`device`/:func:`engine` to explicitly activate one
(optionally as a context manager, restoring the previously active one on
exit), or just start calling the free functions below directly (``buffer``,
``image``, ``command_buffer``, ``submit``, ``wait``, etc.) -- the first one
called lazily activates sensible defaults: device 0, and a combined
GRAPHICS|COMPUTE engine (falling back to a plain COMPUTE engine if this
device has no queue family supporting both).
"""

import os
from typing import Optional, Union, overload

from .vk import (
    AccelerationStructure,
    Buffer,
    CommandBuffer,
    Device,
    Engine,
    EngineType,
    Filter,
    Format,
    Image,
    Layout,
    LayoutRule,
    MemoryLocation,
    MipmapMode,
    Pipeline,
    Sampler,
    Scene,
    SubmissionTask,
    Tensor,
    Type,
    VertexResolutionMode,
    Window,
    WrapMode,
    WrappedMemory,
)
from .vk import device_infos as _query_device_infos
from ._declarations import TypeSpec, _to_type_descriptor


def _to_type_descriptor_layout(spec: TypeSpec) -> Layout:
    """Computes a Layout for a raw type spec under LayoutRule.Scalar (the
    natural, tightly-packed rule for a GPU-visible buffer -- as opposed to
    a uniform-block field, where Std140/Std430 padding rules would apply).
    Used by buffer()'s single-instance convenience form.
    """
    from .vk import compute_layout as _vk_compute_layout
    return _vk_compute_layout(_to_type_descriptor(spec), LayoutRule.Scalar)

__devices__: dict = {}  # Created Device, keyed by physical device index.
__active_device__: Optional[Device] = None  # Currently active device.
__active_engine__: dict = {}  # Currently active Engine, keyed by device index.


def _resolve_device_index(index: int) -> Device:
    """Looks `index` up in the module's device registry, creating (and
    registering) a new Device for it via Device.create_device() the first
    time it's requested. Validation layers are enabled iff the VK_DEBUG
    environment variable is set to "True"/"true"/"1" at that moment.
    """
    if index not in __devices__:
        use_debug = os.environ.get("VK_DEBUG", "False") in ("True", "true", "1")
        __devices__[index] = Device.create_device(index, enable_validation_layers=use_debug)
    return __devices__[index]


def __current_device() -> Device:
    """Returns the active device, activating device 0 first (via
    _resolve_device_index) if device() was never called.
    """
    global __active_device__
    if __active_device__ is None:
        __active_device__ = _resolve_device_index(0)
    return __active_device__


class _DeviceContext:
    """Returned by device(): restores `previous` as the active device on
    __exit__, so a `with device(...):` block only switches temporarily.
    Discarding the return value (no `with`) makes the switch permanent.
    """

    def __init__(self, previous: Optional[Device]):
        self._previous = previous

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global __active_device__
        __active_device__ = self._previous
        return False


def device(device_index: int = 0) -> _DeviceContext:
    """Creates/activates the device at `device_index` as the current
    device. If used as a context manager, the previously active device is
    restored on exit; used as a plain statement, the switch is permanent.

    For validation layers, set the ``VK_DEBUG`` environment variable to
    ``"True"`` before the device at `device_index` is first created.

    :example:

    .. code-block:: python

        vk.device(0)
        ...  # actions occur on the physical device 0
        vk.device(1)  # further actions occur on a device created for gpu 1
        ...
        vk.device(0)  # back to gpu 0 (manual)

        # Equivalently, with a context:
        vk.device(0)
        ...  # on gpu 0
        with vk.device(1):
            ...  # on gpu 1
        ...  # back on gpu 0, automatically

    :param device_index: Index of the physical Vulkan device to use.
    """
    global __active_device__
    previous = __active_device__
    __active_device__ = _resolve_device_index(device_index)
    return _DeviceContext(previous)


def device_index() -> int:
    """Returns the current/active device's index (Device.index of
    whatever device() last activated, or of the lazily-created device 0
    if device() was never called).
    """
    return __current_device().index


# GRAPHICS | COMPUTE: EngineType has no __or__ of its own (plain enum, not
# a bitmask-enabled one), and create_engine's binding requires an actual
# EngineType instance (a bare int is rejected), so the combined value has
# to be constructed explicitly -- it has no named member of its own.
_default_engine_type = EngineType(EngineType.GRAPHICS.value | EngineType.COMPUTE.value)


class _EngineContext:
    """Returned by engine(): restores `previous` as the active engine (for
    `device_idx`) on __exit__, so a `with engine(...):` block only
    switches temporarily. Discarding the return value (no `with`) makes
    the switch permanent.
    """

    def __init__(self, device_idx: int, previous: Optional[Engine]):
        self._device_idx = device_idx
        self._previous = previous

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        __active_engine__[self._device_idx] = self._previous
        return False


def engine(engine_type: Optional[Union[EngineType, int]] = None, engine_index: int = 0) -> _EngineContext:
    """Creates/activates an engine on the current device as the current
    engine. If used as a context manager, the previously active engine
    (for this device) is restored on exit; used as a plain statement, the
    switch is permanent.

    :param engine_type: Capability requested for this engine. A
        combination of engine types is valid, e.g.
        ``EngineType(EngineType.COMPUTE.value | EngineType.GRAPHICS.value)``.
        If ``None`` (default), tries a combined GRAPHICS|COMPUTE engine
        first, falling back to a plain COMPUTE engine if this device has
        no queue family supporting both.
    :param engine_index: Index selecting among multiple queues that
        support the requested capability, when more than one is available.
    """
    dev = __current_device()
    previous = __active_engine__.get(dev.index)
    if engine_type is None:
        try:
            new_engine = dev.create_engine(_default_engine_type, engine_index)
        except RuntimeError:
            new_engine = dev.create_engine(EngineType.COMPUTE, engine_index)
    else:
        new_engine = dev.create_engine(engine_type, engine_index)
    __active_engine__[dev.index] = new_engine
    return _EngineContext(dev.index, previous)


def __current_engine() -> Engine:
    """Returns the active engine for the current device, activating the
    default GRAPHICS|COMPUTE-or-COMPUTE engine first (via engine()) if
    engine() was never called for this device.
    """
    dev = __current_device()
    current = __active_engine__.get(dev.index)
    if current is None:
        engine()
        current = __active_engine__[dev.index]
    return current


class _TransferContext:
    def __init__(self, eng: Engine, cmd: CommandBuffer):
        self._eng = eng
        self._cmd = cmd

    def __enter__(self):
        return self._cmd

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cmd.close()
        self._eng.submit([self._cmd]).wait()


def transfer(engine_index: int = 0):
    """Creates a temporary command buffer on a transfer engine,
    and begins recording a transfer operation into it.
    Use as a context manager: the command buffer is closed and submitted
    automatically on exit, and the context blocks until the transfer is
    complete (via wait()).

    :param engine_index: Index selecting among multiple queues that
        support transfer operations, when more than one is available.
    """
    eng = __current_device().create_engine(EngineType.TRANSFER, engine_index)
    cmd = eng.create_command_buffer()
    return _TransferContext(eng, cmd)


class _ComputeContext:
    def __init__(self, eng: Engine, cmd: CommandBuffer):
        self._eng = eng
        self._cmd = cmd

    def __enter__(self):
        return self._cmd

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cmd.close()
        self._eng.submit([self._cmd]).wait()


def compute(engine_index: int = 0):
    """Creates a temporary command buffer on a compute | transfer engine,
    and begins recording a compute operation into it.
    Use as a context manager: the command buffer is closed and submitted
    automatically on exit, and the context blocks until the compute is
    complete (via wait())."""
    eng = __current_device().create_engine(EngineType(EngineType.COMPUTE.value | EngineType.TRANSFER.value), engine_index)
    cmd = eng.create_command_buffer()
    return _ComputeContext(eng, cmd)


class _GraphicsContext:
    def __init__(self, eng: Engine, cmd: CommandBuffer):
        self._eng = eng
        self._cmd = cmd

    def __enter__(self):
        return self._cmd

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cmd.close()
        self._eng.submit([self._cmd]).wait()


def graphics(engine_index: int = 0):
    """Creates a temporary command buffer on a graphics | compute | transfer engine,
    and begins recording a graphics operation into it.
    Use as a context manager: the command buffer is closed and submitted
    automatically on exit, and the context blocks until the graphics is
    complete (via wait())."""
    eng = __current_device().create_engine(
        EngineType(EngineType.GRAPHICS.value | EngineType.COMPUTE.value | EngineType.TRANSFER.value), engine_index
    )
    cmd = eng.create_command_buffer()
    return _GraphicsContext(eng, cmd)


# ---- Shallow wrappers for every Device method, operating on __current_device() ----

def tensor(shape: list, scalar_type: Type, location: MemoryLocation = MemoryLocation.DEVICE) -> Tensor:
    """Creates a Tensor on the current device, via Device.create_tensor().
    Unlike buffer() (whose shape/dtype for DLPack export is derived from
    its element_layout), the resulting Tensor's shape is exactly `shape`
    and its scalar type exactly `scalar_type` -- hand it straight to
    ``torch.from_dlpack()``.
    """
    return __current_device().create_tensor(shape, scalar_type, location)


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
def buffer(type: "TypeSpec", location: MemoryLocation = MemoryLocation.DEVICE) -> Buffer:
    """Creates a Buffer holding a single instance of `type` (a bare
    ``Type``, an ``[count, element]``/``{name: field, ...}`` type spec, or
    an already-computed Layout) on the current device -- a convenience
    equivalent to ``buffer(1, type if isinstance(type, Layout) else
    compute_layout(type, LayoutRule.Scalar), location)``.
    """
    ...


def buffer(*args, **kwargs) -> Buffer:
    """Creates a Buffer on the current device, via Device.create_buffer().
    Most IDEs show this docstring (not the @overload stubs above) at the
    call site, so it repeats all 4 accepted forms:

    - ``buffer(elements, scalar_type, location=DEVICE)``: `elements`
      scalars of `scalar_type`; the resulting Buffer's element_layout is
      exactly `scalar_type`.
    - ``buffer(elements, format, location=DEVICE)``: `elements` texels of
      `format`; element_layout is an array of `format`'s own per-channel
      scalar type.
    - ``buffer(elements, layout, location=DEVICE)``: `elements` instances
      of `layout` (an already-computed Layout); element_layout is
      `layout` itself, byte size exactly ``elements * layout.aligned_size``.
    - ``buffer(type, location=DEVICE)``: a single instance (elements=1
      implied) of `type` -- a bare Type, an ``[count, element]``/
      ``{name: field, ...}`` type spec, or an already-computed Layout.

    `location` defaults to DEVICE in every form (Device.create_buffer()
    itself has no default for it).
    """
    if args and not isinstance(args[0], int):
        type_or_layout = args[0]
        layout = type_or_layout if isinstance(type_or_layout, Layout) else _to_type_descriptor_layout(type_or_layout)
        location = args[1] if len(args) > 1 else kwargs.get("location", MemoryLocation.DEVICE)
        return __current_device().create_buffer(1, layout, location)
    if "location" not in kwargs and len(args) < 3:
        kwargs["location"] = MemoryLocation.DEVICE
    return __current_device().create_buffer(*args, **kwargs)


def image(
    width: int,
    height: int = 1,
    depth: int = 1,
    mip_levels: int = 1,
    array_layers: int = 1,
    format: Format = Format.RGBA8_UNorm,
    location: MemoryLocation = MemoryLocation.DEVICE,
) -> Image:
    """Creates a 1D/2D/3D Image on the current device, via
    Device.create_image(). ``depth > 1`` makes the resulting Image 3D
    (`array_layers` must then be 1); otherwise 2D (or 1D if `height` is
    also 1), optionally arrayed if `array_layers > 1`. Every Image is
    created with a broad set of usage flags (transfer source/destination,
    sampled, storage, color attachment) and is transitioned to
    ``VK_IMAGE_LAYOUT_GENERAL`` once, synchronously, before this call
    returns -- the one layout every other operation on it assumes.
    """
    return __current_device().create_image(width, height, depth, mip_levels, array_layers, format, location)


def depth_buffer_image(
    width, height, format: Format = Format.Depth32_Float, location: MemoryLocation = MemoryLocation.DEVICE
) -> Image:
    """Creates a 2D depth (or depth/stencil) attachment Image on the
    current device, via Device.create_depth_buffer_image(), for use with
    a pipeline's depth test and depth_test-aware rendering. Unlike
    image(), the resulting Image's usage is depth/stencil-attachment +
    sampled (color-attachment/storage usage is invalid for a depth
    format); it's likewise transitioned to ``VK_IMAGE_LAYOUT_GENERAL``
    once before this call returns.
    """
    return __current_device().create_depth_buffer_image(width, height, format, location)


def sampler(
    mag_filter: Filter = Filter.LINEAR,
    min_filter: Filter = Filter.LINEAR,
    mipmap_mode: MipmapMode = MipmapMode.LINEAR,
    wrap_u: WrapMode = WrapMode.REPEAT,
    wrap_v: WrapMode = WrapMode.REPEAT,
    wrap_w: WrapMode = WrapMode.REPEAT,
) -> Sampler:
    """Creates a texture Sampler on the current device, via
    Device.create_sampler(): the resulting Sampler filters magnification/
    minification per `mag_filter`/`min_filter`, blends between mip levels
    per `mipmap_mode` (every mip level an image has stays reachable --
    there's no separate LOD bias/clamp control), and wraps out-of-[0, 1]
    texture coordinates per axis per `wrap_u`/`wrap_v`/`wrap_w`
    (``CLAMP_TO_BORDER`` always reads an opaque black border).
    """
    return __current_device().create_sampler(mag_filter, min_filter, mipmap_mode, wrap_u, wrap_v, wrap_w)


def ads(declaration) -> AccelerationStructure:
    """Creates (sizes and allocates, but does not build) an acceleration
    structure on the current device, via Device.create_ads(): a
    bottom-level one (BLAS) if `declaration` is an ads_triangles()/
    ads_aabb() result, or a top-level one (TLAS) if it's an
    ads_instances() result. Build it afterwards by recording
    ``command_buffer().build_ads(result, declaration)``.
    """
    return __current_device().create_ads(declaration)


def window(width, height, title: str, format: Format, frames_on_the_fly: int = 3, vsync: bool = True) -> Window:
    """Creates a Window (an OS window, via GLFW, plus its Vulkan swapchain
    and presentation machinery) on the current device, via
    Device.create_window(). The resulting Window's swapchain (and its
    render target) is always ``Format.BGRA8_UNorm`` regardless of
    `format`, which instead applies to its image/buffer/tensor targets
    (presenting one of those blits/converts it into the swapchain image);
    `frames_on_the_fly` becomes its swapchain image count (clamped to
    whatever the surface capabilities actually grant); `vsync` selects a
    vsync'd (default) or an uncapped present mode, falling back to the
    vsync'd one if this surface supports neither uncapped mode.
    """
    return __current_device().create_window(width, height, title, format, frames_on_the_fly, vsync)


@overload
def staging(buffer: Buffer, location: MemoryLocation = MemoryLocation.HOST) -> Buffer:
    """Creates a plain, byte-addressable staging Buffer sized to match
    `buffer`'s own ``size`` on the current device, via
    Device.create_staging() -- the correct Vulkan pattern for moving data
    between CPU and GPU memory (record a command_buffer().transfer()
    between it and `buffer` rather than mapping device-local memory
    directly). The resulting Buffer's element_layout is a plain
    ``Type.UINT8`` array (untyped raw bytes), regardless of
    `buffer`'s own element type.
    """
    ...


@overload
def staging(image: Image, location: MemoryLocation = MemoryLocation.HOST) -> Buffer:
    """Creates a plain, byte-addressable staging Buffer sized to match
    `image`'s own backing store on the current device, via
    Device.create_staging(), for staged CPU/GPU transfer of its texel
    data. The resulting Buffer's element_layout is a plain
    ``Type.UINT8`` array (untyped raw bytes).
    """
    ...


def staging(*args, **kwargs) -> Buffer:
    """Creates a plain, byte-addressable staging Buffer on the current
    device, via Device.create_staging(). Most IDEs show this docstring
    (not the @overload stubs above) at the call site, so it repeats both
    accepted forms:

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
    return __current_device().create_staging(*args, **kwargs)


def pipeline(type) -> Pipeline:
    """Creates a new, empty Pipeline of the given type on the current
    device, via Device.create_pipeline(). The resulting Pipeline has no
    shader stages/descriptor layout/vertex layout/attachments yet -- build
    it up via its own ``stage()``/``layout()``/``vertex_layout()``/
    ``attach()`` methods, then finalize it with ``close()`` before using
    it (``command_buffer()``'s ``set_pipeline()``).
    """
    return __current_device().create_pipeline(type)


def wrap(obj, location: MemoryLocation = MemoryLocation.DEVICE) -> WrappedMemory:
    """Wraps an external object as a WrappedMemory exposing a Vulkan
    buffer device address, on the current device, via Device.wrap().
    The resulting WrappedMemory copies `obj`'s data lazily, on demand, or
    not at all: a Buffer/Tensor's own device address is reused directly
    (never copied); a DLPack-compatible object (e.g. a torch tensor)
    already living in one of this device's own allocations is likewise
    reused directly, otherwise copied into a new buffer at `location` on
    demand; a plain Python buffer-protocol object (e.g. a numpy array) is
    always copied into a new buffer at `location` on demand, since it has
    no Vulkan device address of its own.
    """
    return __current_device().wrap(obj, location)


def load_scene(filename: str, resolution_mode: VertexResolutionMode = VertexResolutionMode.BY_ALL_ATTRIBUTES) -> Scene:
    """Loads a scene file into device-resident Mesh buffers on the
    current device, via Device.load_scene(), dispatching on `filename`'s
    extension (only ``.obj``, via tinyobjloader, is currently supported).
    The resulting Scene's nodes: a group/shape referencing more than one
    material produces one SceneNode per material used within it (sharing
    the group's name), rather than one node with a single, arbitrarily
    chosen material; `resolution_mode` controls how vertices are welded
    across nodes.
    """
    return __current_device().load_scene(filename, resolution_mode)


def dispose() -> None:
    """Disposes the current device (Device.dispose()), drops it from the
    module's device registry (so a further device(same_index) call
    creates a fresh one), and clears the current-engine bookkeeping kept
    for it.
    """
    global __active_device__
    if __active_device__ is None:
        return  # No device has been created
    dev = __active_device__
    __devices__.pop(dev.index, None)
    __active_engine__.pop(dev.index, None)
    __active_device__ = None
    dev.dispose()


def relax() -> None:
    """Drops this module's own references to every created device/engine
    (clearing the device/engine registries and the current-device/
    current-engine pointers), without disposing them. A device whose
    Vulkan resources are still referenced by a live Buffer/Image/
    CommandBuffer/etc. stays alive (Python's own refcounting keeps its
    underlying Device object around); one with no such references left is
    freed once garbage collected. The next device()/buffer()/etc. call
    afterwards creates a brand new device (even for an index that was
    already active), since none of the previous ones are tracked anymore.
    Use dispose() instead to force the current device's teardown
    unconditionally, regardless of what still references it.
    """
    global __active_device__
    __active_device__ = None
    global __devices__
    __devices__.clear()
    global __active_engine__
    __active_engine__.clear()


__device_infos_cache__: Optional[list] = None  # Memoized result of device_infos(); physical devices don't change mid-process.


def device_infos() -> list:
    """Lists every Vulkan-visible physical device (queried once, then
    cached for the rest of the process -- physical devices don't come and
    go while this process runs). Each entry is a dict with keys "index"
    (pass to device()/vk.device() to create/activate that one), "name",
    "vendor" (a friendly name, e.g. "NVIDIA", or a "0x...." hex string for
    an unrecognized vendor), "vendor_id", "device_id", "device_type"
    ("discrete_gpu"/"integrated_gpu"/"virtual_gpu"/"cpu"/"other"),
    "api_version" (e.g. "1.3.278"), "driver_version" (raw, vendor-specific
    encoding), and "vram_bytes" (total device-local memory heap size).

    Querying doesn't create a logical Device (or activate/change the
    current one): a throwaway Vulkan instance is created, queried, and
    destroyed internally, just for this call.
    """
    global __device_infos_cache__
    if __device_infos_cache__ is None:
        __device_infos_cache__ = _query_device_infos()
    return __device_infos_cache__


# ---- Shallow wrappers for every Engine method, operating on __current_engine() ----
# (Device.create_engine() itself has no free-function equivalent here: use
# engine() instead, which both creates/reuses one and activates it.)

def command_buffer() -> CommandBuffer:
    """Creates (or recycles) a command buffer on the current engine, via
    Engine.create_command_buffer(), ready for recording.
    """
    return __current_engine().create_command_buffer()


def submit(command_buffers) -> SubmissionTask:
    """Submits one or more closed command buffers for execution on the
    current engine, via Engine.submit().
    """
    return __current_engine().submit(command_buffers)


def wait() -> None:
    """Blocks until every command previously submitted through the
    current engine has finished executing on the GPU, via Engine.wait().
    """
    __current_engine().wait()
