"""Minimal Vulkan bindings.

This module exposes a small Python surface over a native Vulkan backend:
creating logical devices, allocating host- or device-visible memory that
can be shared with tensor libraries (e.g. PyTorch) through DLPack, and
recording/submitting GPU commands through :class:`Engine`.
"""

import enum


class MemoryLocation(enum.Enum):
    """Where an allocated resource physically lives, and which side of
    the host/device boundary can access it efficiently.
    """

    HOST = 0
    """
    The memory is efficiently read/write by the host application.
    """
    DEVICE = 1
    """
    The memory is efficiently read/write by the gpu programs.
    """


class ScalarType(enum.Enum):
    """Scalar element types supported by buffers and tensors allocated
    through :class:`Device`.
    """

    UNDEFINED = 0
    """No scalar type specified."""
    BOOL = 1
    """Boolean value."""
    FLOAT16 = 2
    """IEEE-754 half precision (16-bit) floating point value."""
    FLOAT32 = 3
    """IEEE-754 single precision (32-bit) floating point value."""
    FLOAT64 = 4
    """IEEE-754 double precision (64-bit) floating point value."""
    INT8 = 5
    """Signed 8-bit integer."""
    INT16 = 6
    """Signed 16-bit integer."""
    INT32 = 7
    """Signed 32-bit integer."""
    INT64 = 8
    """Signed 64-bit integer."""
    UINT8 = 9
    """Unsigned 8-bit integer."""
    UINT16 = 10
    """Unsigned 16-bit integer."""
    UINT32 = 11
    """Unsigned 32-bit integer."""
    UINT64 = 12
    """Unsigned 64-bit integer."""


class Format(enum.Enum):
    """Texel/pixel formats usable by :meth:`Device.create_texels` and
    :meth:`Device.create_image`, and by :meth:`Buffer.cast_format`.

    Named as ``<channels><bits>_<encoding>``, e.g. ``RGBA8_UNorm`` is a
    four-channel, 8-bit-per-channel, unsigned-normalized format.
    """

    Undefined = 0
    """No format specified."""

    R8_UNorm = 1
    """Single-channel 8-bit unsigned normalized value."""
    RG8_UNorm = 2
    """Two-channel 8-bit unsigned normalized value."""
    RGB8_UNorm = 3
    """Three-channel 8-bit unsigned normalized value."""
    RGBA8_UNorm = 4
    """Four-channel 8-bit unsigned normalized value."""

    R8_SNorm = 5
    """Single-channel 8-bit signed normalized value."""
    RG8_SNorm = 6
    """Two-channel 8-bit signed normalized value."""
    RGB8_SNorm = 7
    """Three-channel 8-bit signed normalized value."""
    RGBA8_SNorm = 8
    """Four-channel 8-bit signed normalized value."""

    R8_UInt = 9
    """Single-channel 8-bit unsigned integer."""
    RG8_UInt = 10
    """Two-channel 8-bit unsigned integer."""
    RGB8_UInt = 11
    """Three-channel 8-bit unsigned integer."""
    RGBA8_UInt = 12
    """Four-channel 8-bit unsigned integer."""

    R8_SInt = 13
    """Single-channel 8-bit signed integer."""
    RG8_SInt = 14
    """Two-channel 8-bit signed integer."""
    RGB8_SInt = 15
    """Three-channel 8-bit signed integer."""
    RGBA8_SInt = 16
    """Four-channel 8-bit signed integer."""

    R16_UNorm = 17
    """Single-channel 16-bit unsigned normalized value."""
    RG16_UNorm = 18
    """Two-channel 16-bit unsigned normalized value."""
    RGB16_UNorm = 19
    """Three-channel 16-bit unsigned normalized value."""
    RGBA16_UNorm = 20
    """Four-channel 16-bit unsigned normalized value."""

    R16_SNorm = 21
    """Single-channel 16-bit signed normalized value."""
    RG16_SNorm = 22
    """Two-channel 16-bit signed normalized value."""
    RGB16_SNorm = 23
    """Three-channel 16-bit signed normalized value."""
    RGBA16_SNorm = 24
    """Four-channel 16-bit signed normalized value."""

    R16_UInt = 25
    """Single-channel 16-bit unsigned integer."""
    RG16_UInt = 26
    """Two-channel 16-bit unsigned integer."""
    RGB16_UInt = 27
    """Three-channel 16-bit unsigned integer."""
    RGBA16_UInt = 28
    """Four-channel 16-bit unsigned integer."""

    R16_SInt = 29
    """Single-channel 16-bit signed integer."""
    RG16_SInt = 30
    """Two-channel 16-bit signed integer."""
    RGB16_SInt = 31
    """Three-channel 16-bit signed integer."""
    RGBA16_SInt = 32
    """Four-channel 16-bit signed integer."""

    R32_UInt = 33
    """Single-channel 32-bit unsigned integer."""
    RG32_UInt = 34
    """Two-channel 32-bit unsigned integer."""
    RGB32_UInt = 35
    """Three-channel 32-bit unsigned integer."""
    RGBA32_UInt = 36
    """Four-channel 32-bit unsigned integer."""

    R32_SInt = 37
    """Single-channel 32-bit signed integer."""
    RG32_SInt = 38
    """Two-channel 32-bit signed integer."""
    RGB32_SInt = 39
    """Three-channel 32-bit signed integer."""
    RGBA32_SInt = 40
    """Four-channel 32-bit signed integer."""

    R64_UInt = 41
    """Single-channel 64-bit unsigned integer."""
    RG64_UInt = 42
    """Two-channel 64-bit unsigned integer."""
    RGB64_UInt = 43
    """Three-channel 64-bit unsigned integer."""
    RGBA64_UInt = 44
    """Four-channel 64-bit unsigned integer."""

    R64_SInt = 45
    """Single-channel 64-bit signed integer."""
    RG64_SInt = 46
    """Two-channel 64-bit signed integer."""
    RGB64_SInt = 47
    """Three-channel 64-bit signed integer."""
    RGBA64_SInt = 48
    """Four-channel 64-bit signed integer."""

    R16_Float = 49
    """Single-channel 16-bit floating point value."""
    RG16_Float = 50
    """Two-channel 16-bit floating point value."""
    RGB16_Float = 51
    """Three-channel 16-bit floating point value."""
    RGBA16_Float = 52
    """Four-channel 16-bit floating point value."""

    R32_Float = 53
    """Single-channel 32-bit floating point value."""
    RG32_Float = 54
    """Two-channel 32-bit floating point value."""
    RGB32_Float = 55
    """Three-channel 32-bit floating point value."""
    RGBA32_Float = 56
    """Four-channel 32-bit floating point value."""

    R64_Float = 57
    """Single-channel 64-bit floating point value."""
    RG64_Float = 58
    """Two-channel 64-bit floating point value."""
    RGB64_Float = 59
    """Three-channel 64-bit floating point value."""
    RGBA64_Float = 60
    """Four-channel 64-bit floating point value."""


class EngineType(enum.Enum):
    """Capability requested when creating an :class:`Engine` via
    ``Device.create_engine``.
    """

    NONE = 0
    """No capability requested; not a valid engine type on its own."""
    GRAPHICS = 1
    """Engine capable of graphics (rasterization) commands."""
    COMPUTE = 2
    """Engine capable of compute dispatches."""
    TRANSFER = 4
    """Engine capable of transfer (copy) operations."""


class PipelineType(enum.Enum):
    """Kind of GPU pipeline created via ``Device.create_pipeline``."""

    COMPUTE = 0
    """A compute pipeline, built around a single compute shader."""
    RASTERIZATION = 1
    """A graphics (rasterization) pipeline."""
    RAYTRACING = 2
    """A ray tracing pipeline."""


class Buffer:
    """A view over a slice of GPU-accessible memory, created through
    ``Device.create_buffer`` or ``Device.create_array``.
    """

    def dlpack(self) -> object:
        """Exports this buffer as a DLPack capsule.

        :return: A DLPack capsule wrapping this buffer's memory,
            consumable e.g. by ``torch.from_dlpack``.
        """
        ...


class CommandBuffer:
    """User-facing handle used to record GPU commands.

    Obtained from ``Engine.create_command_buffer``. Hides the fact that
    the underlying command buffer is allocated from, and recycled by,
    an internal command pool.
    """

    def transfer(self, source: Buffer, destination: Buffer, bytes: int) -> None:
        """Records a device-side copy from one buffer to another.

        Must be called while the command buffer is still recording,
        i.e. before :meth:`close`.

        :param source: Buffer to copy from.
        :param destination: Buffer to copy into.
        :param bytes: Number of bytes to copy, starting at the beginning
            of each buffer.
        """
        ...

    def close(self) -> None:
        """Ends recording and makes it ready to submit for execution
        on its engine.

        No further commands can be recorded into it afterwards. Use
        :meth:`Engine.wait` to wait for the submitted work to finish.
        """
        ...

    @property
    def is_submitted(self) -> bool:
        """Whether this command buffer is currently submitted to, and
        potentially still executing on, the GPU.
        """
        ...

    @property
    def is_executable(self) -> bool:
        """Whether this command buffer is closed and ready to be submitted."""
        ...

    @property
    def is_closed(self) -> bool:
        """Whether recording has ended, i.e. :meth:`close` was called."""
        ...

    @property
    def is_released(self) -> bool:
        """Whether this command buffer has been released back to its
        engine and can no longer be used.
        """
        ...


class SubmissionTask:
    """Handle to a batch of commands submitted through an :class:`Engine`.

    Lets the caller wait for, or poll, the GPU-side completion of a
    submission.
    """

    def wait(self) -> None:
        """Blocks the calling thread until this submission has finished
        executing on the GPU.
        """
        ...

    @property
    def is_complete(self) -> bool:
        """Whether the GPU has finished executing this submission."""
        ...


class Engine:
    """Execution context bound to a single Vulkan queue.

    Obtained from :class:`Device`. Lets you record and submit GPU
    commands without dealing with the underlying Vulkan queue or command
    pool directly.
    """

    def create_command_buffer(self) -> CommandBuffer:
        """Creates, or recycles, a command buffer ready for recording.

        :return: A new :class:`CommandBuffer`, already open for recording.
        """
        ...

    def submit(self, command_buffers: list[CommandBuffer]) -> SubmissionTask:
        """Submits one or more closed command buffers for execution on
        this engine's queue.

        :param command_buffers: Command buffers to submit. Each must
            already be closed, i.e. :meth:`CommandBuffer.close` was
            called on it.
        :return: A :class:`SubmissionTask` tracking this submission's
            GPU-side completion.
        """
        ...

    def wait(self) -> None:
        """Blocks the calling thread until every command previously
        submitted through this engine has finished executing on the GPU.
        """
        ...


class Device:
    """A logical Vulkan device.

    Owns the Vulkan instance/device handles and the memory managers used
    to allocate host- and device-visible resources, such as tensors
    shared with other libraries through DLPack.
    """

    def __init__(self) -> None:
        """Creates an uninitialized device.

        Prefer :func:`create_device` or :meth:`Device.create_device`
        over constructing a device directly.
        """
        ...

    def dispose(self) -> None:
        """Releases all Vulkan resources owned by this device."""
        ...

    @staticmethod
    def create_device(device_index: int = 0, enable_validation_layers: bool = False) -> "Device":
        """Creates and initializes a new Vulkan device.

        :param device_index: Index of the physical Vulkan device to use.
        :param enable_validation_layers: Whether to enable the Vulkan
            validation layers, useful for debugging.
        :return: A fully initialized :class:`Device`.
        """
        ...

    def allocate_tensor_dlpack(self, shape: list[int], scalar_type: ScalarType, location: MemoryLocation) -> object:
        """Allocates a memory for a tensor with specific type and shape.
        The tensor is packed as a dl_pack.

        :param shape: Shape of the tensor to allocate.
        :param scalar_type: Scalar element type of the tensor.
        :param location: Memory location where the tensor should be allocated.
        :return: A DLPack capsule wrapping the allocated tensor memory,
            consumable e.g. by ``torch.from_dlpack``.
        """
        ...

    def create_buffer(self, size: int, location: MemoryLocation) -> Buffer:
        """Creates a raw buffer of the given size, in bytes.

        :param size: Size of the buffer, in bytes.
        :param location: Memory location where the buffer should be allocated.
        :return: A new :class:`Buffer`.
        """
        ...

    def create_array(self, elements: int, scalar_type: ScalarType, location: MemoryLocation) -> Buffer:
        """Creates a buffer to store an array of elements of a given scalar type.

        :param elements: Number of elements in the array.
        :param scalar_type: Scalar element type of the array.
        :param location: Memory location where the buffer should be allocated.
        :return: A new :class:`Buffer`.
        """
        ...

    def create_engine(self, type: EngineType, index: int = 0) -> Engine:
        """Creates (or reuses) an engine to record and submit GPU commands.

        :param type: Capability requested for this engine.
        :param index: Index selecting among multiple queues that support
            the requested capability, when more than one is available.
        :return: An :class:`Engine` bound to a matching Vulkan queue.
        """
        ...


def create_device(device_index: int = 0, enable_validation_layers: bool = False) -> Device:
    """Convenience wrapper around Device.create_device.

    :param device_index: Index of the physical Vulkan device to use.
    :param enable_validation_layers: Whether to enable the Vulkan
        validation layers, useful for debugging.
    :return: A fully initialized :class:`Device`.
    """
    ...
