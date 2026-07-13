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


class ShaderStageType(enum.Enum):
    """Shader stage a :class:`ShaderSource` is compiled/attached for, via
    :meth:`Pipeline.stage`.
    """

    VERTEX = 0
    """Vertex shader stage."""
    FRAGMENT = 1
    """Fragment (pixel) shader stage."""
    GEOMETRY = 2
    """Geometry shader stage."""
    TESS_CONTROL = 3
    """Tessellation control shader stage."""
    TESS_EVAL = 4
    """Tessellation evaluation shader stage."""
    COMPUTE = 5
    """Compute shader stage."""


class DescriptorType(enum.Enum):
    """Kind of resource bound to a descriptor set binding, declared via
    :meth:`Pipeline.layout` and written via :meth:`DescriptorSet.bind`.
    """

    STORAGE_BUFFER = 0
    """A read/write buffer (GLSL ``buffer`` block)."""
    UNIFORM_BUFFER = 1
    """A read-only buffer (GLSL ``uniform`` block)."""
    STORAGE_IMAGE = 2
    """A read/write image, without a sampler."""
    SAMPLED_IMAGE = 3
    """A read-only, filterable image, without a sampler."""
    SAMPLER = 4
    """A standalone sampler. Not yet bindable via :meth:`DescriptorSet.bind`."""
    COMBINED_IMAGE_SAMPLER = 5
    """An image and sampler bound together. Not yet bindable via
    :meth:`DescriptorSet.bind`.
    """
    ACCELERATION_STRUCTURE = 6
    """A ray tracing acceleration structure. Not yet bindable via
    :meth:`DescriptorSet.bind`.
    """


class TypeKind(enum.Enum):
    """Shape of a :class:`TypeDescriptor`: which alternative of its
    payload is populated.
    """

    SCALAR = 0
    """A single scalar value."""
    VECTOR = 1
    """A vector of 2, 3 or 4 components of the same scalar type."""
    MATRIX = 2
    """A column-major matrix of scalar components."""
    ARRAY = 3
    """A fixed-size, or unsized/runtime, array of a single element type."""
    STRUCT = 4
    """An ordered collection of named, independently-typed fields."""


class LayoutRule(enum.Enum):
    """GPU buffer block layout convention used by :func:`compute_layout`
    and ``Device.create_structured_buffer``.
    """

    Std140 = 0
    """GLSL/Vulkan ``std140`` layout: vec3 aligns like vec4, and array/struct
    alignment is rounded up to 16 bytes.
    """
    Std430 = 1
    """GLSL/Vulkan ``std430`` layout: like ``Std140`` but array/struct
    alignment is not rounded up to 16 bytes.
    """
    Scalar = 2
    """Scalar block layout: every type is aligned to its base component
    size, with no vector or block padding beyond natural alignment.
    """


class TypeDescriptor:
    """Describes the shape of a GPU-buffer-resident type: scalar, vector,
    matrix, array or struct.

    Used together with :class:`LayoutRule` and :func:`compute_layout` to
    compute byte offsets, sizes and strides without touching the GPU, and
    with ``Device.create_structured_buffer`` to allocate a buffer sized
    to hold it.
    """

    @staticmethod
    def scalar(type: ScalarType) -> "TypeDescriptor":
        """Creates a scalar type descriptor.

        :param type: Scalar element type.
        :return: A new :class:`TypeDescriptor` of kind
            :attr:`TypeKind.SCALAR`.
        """
        ...

    @staticmethod
    def vector(component_type: ScalarType, components: int) -> "TypeDescriptor":
        """Creates a vector type descriptor.

        :param component_type: Scalar type of each component.
        :param components: Number of components; must be 2, 3 or 4.
        :return: A new :class:`TypeDescriptor` of kind
            :attr:`TypeKind.VECTOR`.
        """
        ...

    @staticmethod
    def matrix(component_type: ScalarType, rows: int, columns: int) -> "TypeDescriptor":
        """Creates a column-major matrix type descriptor.

        :param component_type: Scalar type of each component.
        :param rows: Number of rows; must be 2, 3 or 4.
        :param columns: Number of columns; must be 2, 3 or 4.
        :return: A new :class:`TypeDescriptor` of kind
            :attr:`TypeKind.MATRIX`.
        """
        ...

    @staticmethod
    def array_of(element_type: "TypeDescriptor", count: int) -> "TypeDescriptor":
        """Creates an array type descriptor.

        :param element_type: Type of each element.
        :param count: Number of elements. Zero denotes an unsized/runtime
            array, only meaningful as the last field of a struct.
        :return: A new :class:`TypeDescriptor` of kind
            :attr:`TypeKind.ARRAY`.
        """
        ...

    @staticmethod
    def struct_of(fields: list[tuple[str, "TypeDescriptor"]]) -> "TypeDescriptor":
        """Creates a struct type descriptor.

        :param fields: Ordered ``(name, type)`` pairs, one per field.
        :return: A new :class:`TypeDescriptor` of kind
            :attr:`TypeKind.STRUCT`.
        """
        ...

    @property
    def kind(self) -> TypeKind:
        """Which shape this type descriptor represents."""
        ...


class LayoutField:
    """A single named field of a computed struct :class:`Layout`."""

    @property
    def name(self) -> str:
        """Name of this field, as given to :meth:`TypeDescriptor.struct_of`."""
        ...

    @property
    def offset(self) -> int:
        """Byte offset of this field within the owning struct."""
        ...

    @property
    def layout(self) -> "Layout":
        """Layout of this field's own type."""
        ...

    @property
    def root(self) -> "Layout | None":
        """The outermost :class:`Layout` this field was computed as part
        of (itself, if this field belongs to a top-level struct passed to
        :func:`compute_layout`).

        Used by :meth:`Buffer.field` to resolve strides across
        instances of this field's root layout. ``None`` only if this
        field's :class:`Layout` tree no longer exists.
        """
        ...


class Layout:
    """Computed byte layout of a :class:`TypeDescriptor` under a given
    :class:`LayoutRule`, as returned by :func:`compute_layout`.
    """

    @property
    def size(self) -> int:
        """Total size of this type, in bytes."""
        ...

    @property
    def alignment(self) -> int:
        """Required byte alignment of this type."""
        ...

    @property
    def aligned_size(self) -> int:
        """Size this layout would occupy as one element of an array of
        itself, i.e. :attr:`size` rounded up to :attr:`alignment` (and,
        under :attr:`LayoutRule.Std140`, up to 16) -- the byte stride
        between consecutive elements. This is what
        :meth:`Device.create_structured_buffer` uses as the per-element
        stride when ``count > 1``.
        """
        ...

    @property
    def kind(self) -> TypeKind:
        """Shape of the type this layout was computed for."""
        ...

    @property
    def component_type(self) -> ScalarType:
        """Scalar component type of this layout.

        Only meaningful when :attr:`kind` is :attr:`TypeKind.SCALAR`,
        :attr:`TypeKind.VECTOR` or :attr:`TypeKind.MATRIX`;
        :attr:`ScalarType.UNDEFINED` otherwise.
        """
        ...

    @property
    def fields(self) -> list[LayoutField]:
        """Ordered fields with their offsets and sub-layouts.

        Only meaningful when :attr:`kind` is :attr:`TypeKind.STRUCT`;
        empty otherwise.
        """
        ...

    @property
    def element_layout(self) -> "Layout | None":
        """Layout of a single element (for an array) or column (for a
        matrix).

        Only meaningful when :attr:`kind` is :attr:`TypeKind.ARRAY` or
        :attr:`TypeKind.MATRIX`; ``None`` otherwise.
        """
        ...

    @property
    def stride(self) -> int:
        """Byte stride between consecutive elements or matrix columns.

        Only meaningful when :attr:`kind` is :attr:`TypeKind.ARRAY` or
        :attr:`TypeKind.MATRIX`; zero otherwise.
        """
        ...

    @property
    def count(self) -> int:
        """Number of elements (for an array) or columns (for a matrix).

        Only meaningful when :attr:`kind` is :attr:`TypeKind.ARRAY` or
        :attr:`TypeKind.MATRIX`; zero otherwise.
        """
        ...


def compute_layout(type: TypeDescriptor, rule: LayoutRule) -> Layout:
    """Computes the byte layout of a type under a given layout rule.

    Pure host-side arithmetic; does not require a :class:`Device`.

    :param type: Type to compute the layout of.
    :param rule: Layout convention to apply.
    :return: The computed :class:`Layout`.
    """
    ...


class Buffer:
    """A view over a slice of GPU-accessible memory, created through
    ``Device.create_buffer`` or ``Device.create_array``.

    Implements the Python DLPack protocol directly (``__dlpack__``/
    ``__dlpack_device__``), so a :class:`Buffer` can be passed straight to
    ``torch.from_dlpack(buffer)`` without an intermediate call.
    """

    @property
    def size(self) -> int:
        """Size of this buffer, in bytes."""
        ...

    def __dlpack__(self, stream: object = None) -> object:
        """Exports this buffer as a DLPack capsule (the Python DLPack
        protocol's tensor-export method).

        :param stream: Accepted for protocol compatibility (e.g. a CUDA
            stream pointer) and ignored; this export does no stream
            synchronization of its own.
        :return: A DLPack capsule wrapping this buffer's memory,
            consumable e.g. by ``torch.from_dlpack``.
        """
        ...

    def __dlpack_device__(self) -> tuple[int, int]:
        """Returns the ``(device_type, device_id)`` pair this buffer's
        memory lives on (the Python DLPack protocol's device-query method).
        """
        ...

    def field(self, field: LayoutField) -> object:
        """Exports ``field``, repeated across every instance of its root
        layout in this buffer, as a single strided DLPack tensor.

        :param field: Field to export. Must describe a scalar, vector,
            matrix, or array of those (not a struct, since a struct
            isn't representable as a single numeric tensor).
        :return: A DLPack capsule, consumable e.g. by
            ``torch.from_dlpack``, whose first dimension indexes
            instances of ``field``'s root layout.
        """
        ...


class CommandBuffer:
    """User-facing handle used to record GPU commands.

    Obtained from ``Engine.create_command_buffer``. Hides the fact that
    the underlying command buffer is allocated from, and recycled by,
    an internal command pool.
    """

    def transfer(self, source: Buffer, destination: Buffer) -> None:
        """Records a device-side copy from one buffer to another.

        Must be called while the command buffer is still recording,
        i.e. before :meth:`close`.

        :param source: Buffer to copy from.
        :param destination: Buffer to copy into.
        """
        ...

    def set_pipeline(self, pipeline: "Pipeline") -> None:
        """Binds a pipeline for subsequent :meth:`bind`/:meth:`dispatch_threads`
        (compute) or draw commands (graphics).

        Must be called while the command buffer is still recording, i.e.
        before :meth:`close`. ``pipeline`` must already be closed, i.e.
        :meth:`Pipeline.close` was called on it. Keeps a reference to
        ``pipeline`` alive for as long as this command buffer is, including
        while submitted and pending on the GPU.

        :param pipeline: Pipeline to bind.
        """
        ...

    def bind(self, initial_set: int, descriptor_sets: list["DescriptorSet"]) -> None:
        """Binds ``descriptor_sets`` at consecutive set indices starting at
        ``initial_set``, against the pipeline layout of the pipeline most
        recently bound via :meth:`set_pipeline`.

        Must be called while the command buffer is still recording, i.e.
        before :meth:`close`, and after a matching :meth:`set_pipeline`
        call. Keeps references to ``descriptor_sets`` alive for as long as
        this command buffer is, including while submitted and pending on
        the GPU.

        :param initial_set: Set index of ``descriptor_sets[0]``; each
            subsequent entry is bound at the next consecutive index.
        :param descriptor_sets: Descriptor sets to bind.
        """
        ...

    def dispatch_threads(self, x: int, y: int = 1, z: int = 1) -> None:
        """Records a compute dispatch covering at least ``(x, y, z)``
        threads, using the pipeline most recently bound via
        :meth:`set_pipeline`.

        Must be called while the command buffer is still recording, i.e.
        before :meth:`close`, and requires a
        :attr:`PipelineType.COMPUTE` pipeline to be bound.

        .. note::
            ``x``/``y``/``z`` are thread counts, not workgroup counts: the
            number of workgroups dispatched along each dimension is
            ``ceil(threads / local_size)``, using the bound pipeline's
            declared :meth:`Pipeline.local_size` (default ``(1, 1, 1)`` if
            never set).

        :param x: Minimum number of threads to cover along X.
        :param y: Minimum number of threads to cover along Y.
        :param z: Minimum number of threads to cover along Z.
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


class ShaderSource:
    """Compiled SPIR-V bytecode for a single shader stage, attached to a
    :class:`Pipeline` via :meth:`Pipeline.stage`.
    """

    @staticmethod
    def from_spirv(code: list[int], entry_point: str = "main") -> "ShaderSource":
        """Wraps already-compiled SPIR-V bytecode.

        :param code: SPIR-V words (32-bit each).
        :param entry_point: Name of this module's entry point function, as
            it actually appears in the compiled SPIR-V.
        :return: A new :class:`ShaderSource`.
        """
        ...

    @staticmethod
    def from_file(
        path: str,
        stage: ShaderStageType,
        entry_point: str = "main",
        include_dirs: list[str] = [],
    ) -> "ShaderSource":
        """Loads a shader from a file.

        If ``path`` ends in ``.spv`` it is read as a raw SPIR-V binary
        (``entry_point`` is then just recorded as-is, and ``include_dirs``
        is ignored); otherwise it is read as GLSL source text and compiled
        via :meth:`from_glsl`.

        :param path: Path to the shader file.
        :param stage: Shader stage, used to select the GLSL compiler
            profile when compiling source text (ignored for ``.spv`` files).
        :param entry_point: Name of the entry point function as written in
            the GLSL source (ignored for ``.spv`` files, where it's just
            recorded as the module's already-compiled entry point name).
        :param include_dirs: Directories searched for ``#include`` files,
            in order (ignored for ``.spv`` files).
        :return: A new :class:`ShaderSource`.
        """
        ...

    @staticmethod
    def from_glsl(
        source: str,
        stage: ShaderStageType,
        entry_point: str = "main",
        include_dirs: list[str] = [],
    ) -> "ShaderSource":
        """Compiles GLSL source text to SPIR-V by shelling out to
        ``glslangValidator``, with the source piped through its stdin.

        .. note::
            Requires ``glslangValidator`` to be available on ``PATH`` (or
            found via ``$VULKAN_SDK/Bin``). This is a runtime dependency
            only; the library itself has no build- or link-time dependency
            on the Vulkan SDK.

        :param source: GLSL source code.
        :param stage: Shader stage to compile for.
        :param entry_point: Name of the entry point function as written in
            ``source``. The compiled SPIR-V's own entry point is given this
            same name.
        :param include_dirs: Directories searched for ``#include`` files
            (requires ``#extension GL_GOOGLE_include_directive``), in order.
        :return: A new :class:`ShaderSource`.
        :raises RuntimeError: If compilation fails, or ``glslangValidator``
            could not be found/run.
        """
        ...


class Framebuffer:
    """A set of render target image views bound for one :class:`Pipeline`'s
    render pass, obtained from :meth:`Pipeline.create_framebuffer`.
    """

    @property
    def width(self) -> int:
        """Width, in pixels, shared by all attached images."""
        ...

    @property
    def height(self) -> int:
        """Height, in pixels, shared by all attached images."""
        ...


class DescriptorSet:
    """A descriptor set matching one :class:`Pipeline`'s declared layout for
    a given ``set`` index, obtained from :meth:`Pipeline.create_descriptor_set`.
    """

    def bind(self, layout_id: int, resource: "Buffer | object") -> None:
        """Writes a resource into the binding identified by ``layout_id``.

        Accepts either a :class:`Buffer` (for a binding declared with
        :attr:`DescriptorType.STORAGE_BUFFER` or
        :attr:`DescriptorType.UNIFORM_BUFFER`) or an image (for
        :attr:`DescriptorType.STORAGE_IMAGE` or
        :attr:`DescriptorType.SAMPLED_IMAGE`; sampler-based descriptor
        types are not yet supported). Image resources are not yet exposed
        to Python (see :meth:`Pipeline.create_framebuffer`), so only the
        buffer overload is currently usable from Python.

        :param layout_id: Id returned by the matching :meth:`Pipeline.layout` call.
        :param resource: Resource to bind.
        """
        ...


class Pipeline:
    """A GPU pipeline (compute or graphics/rasterization), obtained from
    :meth:`Device.create_pipeline`.

    Built up via :meth:`stage`, :meth:`layout`, :meth:`vertex_layout` and
    :meth:`attach`, then finalized with :meth:`close`; afterwards,
    :meth:`create_descriptor_set` and (for graphics pipelines)
    :meth:`create_framebuffer` become usable.
    """

    def stage(self, type: ShaderStageType, source: ShaderSource) -> None:
        """Attaches a compiled shader to one stage of this pipeline.

        Must be called before :meth:`close`.

        :param type: Shader stage this source implements.
        :param source: Compiled shader bytecode.
        """
        ...

    def layout(self, set: int, binding: int, description: DescriptorType, count: int = 1) -> int:
        """Declares one binding of the descriptor set layout for ``set``.

        Must be called before :meth:`close`.

        :param set: Descriptor set index.
        :param binding: Binding index within ``set``.
        :param description: Kind of resource this binding expects.
        :param count: Number of array elements at this binding.
        :return: An id identifying this binding, for use with
            :meth:`DescriptorSet.bind`.
        """
        ...

    def vertex_layout(self, start_location: int, layout: Layout) -> None:
        """Declares the per-vertex input consumed by the vertex shader, as
        one new vertex buffer binding starting at ``start_location``.

        Graphics pipelines only. Must be called before :meth:`close`.

        :param start_location: Shader input location of the first field of
            ``layout``. Subsequent fields (and, for a matrix field, each of
            its columns) consume consecutive locations.
        :param layout: Layout of one vertex, e.g. from
            ``compute_layout(type, rule)``. Must describe a struct whose
            fields are all scalar, vector or matrix (not array or nested
            struct).
        """
        ...

    def attach(self, slot: int, format: Format) -> int:
        """Declares one color attachment of this pipeline's render pass.

        Graphics pipelines only. Must be called before :meth:`close`.

        :param slot: Fragment shader output location for this attachment.
        :param format: Pixel format of this attachment.
        :return: An id used to bind a render target image via
            :meth:`create_framebuffer`.
        """
        ...

    def local_size(self, x: int, y: int = 1, z: int = 1) -> None:
        """Declares the compute shader's own workgroup size (its
        ``layout(local_size_x = ..., ...)`` declaration in GLSL).

        Must match the actual shader code -- there is no way to verify this
        automatically. Used by ``CommandBuffer.dispatch_threads`` to compute
        the number of workgroups needed to cover a requested thread count.

        Compute pipelines only. Must be called before :meth:`close`;
        defaults to ``(1, 1, 1)`` if never called.

        :param x: Workgroup size along X.
        :param y: Workgroup size along Y.
        :param z: Workgroup size along Z.
        """
        ...

    def close(self) -> None:
        """Finalizes this pipeline: builds the descriptor set layouts,
        pipeline layout, (for graphics pipelines) render pass and vertex
        input state, and the underlying GPU pipeline.

        No further :meth:`stage`/:meth:`layout`/:meth:`vertex_layout`/
        :meth:`attach` calls are allowed afterwards.
        """
        ...

    def create_framebuffer(self, attachments: list[tuple[int, object]]) -> Framebuffer:
        """Creates a framebuffer compatible with this pipeline's render pass.

        Pipeline must be closed first. Graphics pipelines only.

        .. note::
            Image resources are not yet exposed to Python (``Device.create_image``
            is not yet implemented), so this method cannot be used from Python
            until that lands. The underlying C++ API is already functional.

        :param attachments: One ``(slot, image)`` pair per id returned by
            :meth:`attach`, each image matching that attachment's declared
            format, and all sharing the same dimensions.
        :return: A new :class:`Framebuffer`.
        """
        ...

    def create_descriptor_set(self, set: int = 0) -> DescriptorSet:
        """Allocates a new descriptor set matching the layout declared for
        ``set`` via :meth:`layout`.

        Pipeline must be closed first.

        :param set: Descriptor set index, as passed to :meth:`layout`.
        :return: A new :class:`DescriptorSet`.
        """
        ...

    @property
    def is_closed(self) -> bool:
        """Whether :meth:`close` has already been called."""
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

    def create_structured_buffer(self, layout: Layout, location: MemoryLocation, count: int = 1) -> Buffer:
        """Creates a buffer sized to hold one, or an array of, instances
        of an element type, as already laid out by :func:`compute_layout`.

        :param layout: Layout of a single instance, e.g. from
            ``compute_layout(type, rule)``.
        :param location: Memory location where the buffer should be allocated.
        :param count: Number of instances to allocate room for. When
            greater than 1, ``layout.aligned_size`` is used as the
            per-instance stride, so array-stride padding between
            instances is accounted for.
        :return: A new :class:`Buffer`.
        """
        ...

    def create_pipeline(self, type: PipelineType) -> Pipeline:
        """Creates a new, empty pipeline of the given type.

        Build it up via :meth:`Pipeline.stage`, :meth:`Pipeline.layout`,
        :meth:`Pipeline.vertex_layout` and :meth:`Pipeline.attach`, then
        finalize it with :meth:`Pipeline.close`.

        :param type: Kind of pipeline to create. :attr:`PipelineType.RAYTRACING`
            is not yet supported and will raise on :meth:`Pipeline.close`.
        :return: A new, not-yet-closed :class:`Pipeline`.
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

    def create_staging(self, buffer: Buffer, location: MemoryLocation = MemoryLocation.HOST) -> Buffer:
        """Creates a plain buffer sized to match ``buffer``, for staged
        CPU/GPU data movement.

        Device-local memory generally can't be mapped or read directly
        by the CPU. The correct Vulkan pattern is to allocate a staging
        buffer with this method and record a
        :meth:`CommandBuffer.transfer` between it and ``buffer``, rather
        than attempting to access device memory directly.

        :param buffer: Buffer whose size the staging buffer should match.
        :param location: Memory location for the staging buffer, HOST
            by default.
        :return: A new :class:`Buffer` of the same size as ``buffer``.
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
