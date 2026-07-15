"""Minimal Vulkan bindings.

This module exposes a small Python surface over a native Vulkan backend:
creating logical devices, allocating host- or device-visible memory that
can be shared with tensor libraries (e.g. PyTorch) through DLPack, and
recording/submitting GPU commands through :class:`Engine`.
"""

import enum
from typing import overload


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


class WrapMode(enum.Enum):
    """When :meth:`Device.wrap` must copy data because the wrapped
    object isn't already directly usable in place, this controls at
    which point(s) of the :class:`WrappedMemory`'s lifetime the copy
    happens.
    """

    IN = 0
    """Copy the source data in when the wrap is created."""
    OUT = 1
    """Copy back to the source object when the :class:`WrappedMemory`
    is destroyed."""
    INOUT = 2
    """Copy in when created and copy back when destroyed."""


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
    """Texel/pixel formats usable by :meth:`Device.create_buffer`,
    :meth:`Device.create_image`, and :meth:`Buffer.cast`.

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
    BGRA8_UNorm = 61
    """Byte-swapped channel order of RGBA8_UNorm: the native swapchain
    surface format on most Windows (and many Linux) drivers. Every
    :class:`Window` swapchain is created with this format regardless of
    the ``format`` passed to :meth:`Device.create_window`, which instead
    applies to its ``image_target``/``buffer_target``/``tensor_target``.
    """

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


class Filter(enum.Enum):
    """Texel filtering used by :meth:`CommandBuffer.blit_image` when the
    source and destination extents differ.
    """

    NEAREST = 0
    """Nearest-texel sampling."""
    LINEAR = 1
    """Linearly interpolated sampling."""


class LayoutHandle:
    """Opaque handle returned by :meth:`Pipeline.layout`, identifying one
    declared descriptor set binding.

    Not constructible from Python: only meaningful when passed back to
    :meth:`DescriptorSet.bind` on a descriptor set from the same
    :class:`Pipeline`.
    """


class AttachHandle:
    """Opaque handle returned by :meth:`Pipeline.attach`, identifying one
    declared color attachment slot.

    Not constructible from Python: only meaningful when passed back to
    :meth:`Pipeline.create_framebuffer` on the same :class:`Pipeline`.
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
    and ``Device.create_buffer(elements, layout, ...)``.
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
    with ``Device.create_buffer(elements, layout, ...)`` to allocate a buffer sized
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
        ``Device.create_buffer(elements, layout, ...)`` uses as the per-element
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
    """A view over a slice of GPU-accessible memory, created through one of
    the :meth:`Device.create_buffer` overloads.

    Implements the Python DLPack protocol directly (``__dlpack__``/
    ``__dlpack_device__``), so a :class:`Buffer` can be passed straight to
    ``torch.from_dlpack(buffer)`` without an intermediate call.
    """

    @property
    def size(self) -> int:
        """Size of this buffer, in bytes."""
        ...

    @property
    def device_ptr(self) -> int:
        """Vulkan buffer device address of this buffer -- a raw pointer
        usable from within a shader (e.g. through ``GL_EXT_buffer_reference``),
        not to be confused with any external-memory-exported (e.g. CUDA)
        pointer.
        """
        ...

    @property
    def element_layout(self) -> Layout:
        """Layout of a single element of this buffer: a scalar
        (``Device.create_buffer(elements, ScalarType, ...)``), an array of a
        format's per-channel scalar (``Device.create_buffer(elements,
        Format, ...)``), or a struct (``Device.create_buffer(elements,
        layout, ...)``). Every :class:`Buffer` has one -- there is no "untyped"
        buffer.

        Its ``aligned_size`` is the per-element byte stride used by
        :meth:`slice`/:meth:`element`; its ``kind`` determines the
        shape/dtype :meth:`__dlpack__` exports.
        """
        ...

    @overload
    def cast(self, scalar: ScalarType) -> "Buffer":
        """Reinterprets this buffer's bytes (same underlying offset/size)
        as a flat array of ``scalar`` elements.

        :param scalar: New scalar element type.
        :return: A new :class:`Buffer` viewing the same bytes.
        """
        ...

    @overload
    def cast(self, format: Format) -> "Buffer":
        """Reinterprets this buffer's bytes as a flat array of ``format``
        texels (internally, an array of the format's own per-channel
        scalar type).

        :param format: New texel format.
        :return: A new :class:`Buffer` viewing the same bytes.
        """
        ...

    @overload
    def cast(self, layout: Layout) -> "Buffer":
        """Reinterprets this buffer's bytes as being laid out per
        ``layout``.

        :param layout: New element layout, e.g. from
            ``compute_layout(type, rule)``.
        :return: A new :class:`Buffer` viewing the same bytes.
        """
        ...

    def slice(self, start_element: int, count: int) -> "Buffer":
        """Slices this buffer to ``count`` elements starting at
        ``start_element``, using ``element_layout.aligned_size`` as the
        byte size of one element.

        :param start_element: Index of the first element to include.
        :param count: Number of elements to include.
        :return: A new :class:`Buffer` viewing just that range.
        :raises RuntimeError: If the requested range is out of bounds.
        """
        ...

    def element(self, index: int) -> "Buffer":
        """A single-element slice: equivalent to ``slice(index, 1)``.

        :param index: Index of the element to select.
        :return: A new :class:`Buffer` viewing just that element.
        """
        ...

    def __dlpack__(self, stream: object = None) -> object:
        """Exports this buffer as a DLPack capsule (the Python DLPack
        protocol's tensor-export method).

        Shape and dtype are derived from :attr:`element_layout`: a scalar
        or array-of-scalars layout exports as a flat (or, for a texel
        buffer, 2D ``(texels, components)``) array of that scalar type;
        a struct layout (e.g. a buffer from ``Device.create_buffer(elements,
        layout, ...)``) has no single numeric type, so it falls back to raw
        bytes (``UINT8``, shape ``(size,)``).

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

        Only valid when :attr:`element_layout`'s ``kind`` is
        :attr:`TypeKind.STRUCT` (e.g. a buffer from
        ``Device.create_buffer(elements, layout, ...)``).

        :param field: Field to export. Must describe a scalar, vector,
            matrix, or array of those (not a struct, since a struct
            isn't representable as a single numeric tensor).
        :return: A DLPack capsule, consumable e.g. by
            ``torch.from_dlpack``, whose first dimension indexes
            instances of ``field``'s root layout.
        :raises RuntimeError: If :attr:`element_layout` isn't a struct, or
            ``field`` describes a struct.
        """
        ...

    def read(self, field: LayoutField) -> object:
        """Reads ``field`` directly via ``memcpy``, bypassing DLPack --
        much cheaper than :meth:`field` for small (e.g. scalar) fields.

        Only valid when :attr:`element_layout`'s ``kind`` is
        :attr:`TypeKind.STRUCT`, on a host-visible buffer
        (``MemoryLocation.HOST``) holding exactly one instance of
        ``field``'s root layout (i.e. :attr:`size` equals
        ``field.root.aligned_size``).

        :param field: Field to read. Must describe a scalar, vector,
            matrix, or (possibly nested) array of any of those -- not a
            struct, since a struct isn't a single numeric type.
        :return: A plain Python number (``int``/``float``/``bool``) for a
            scalar field; a tightly-packed buffer object (``bytes``) for a
            vector, matrix, or array field -- e.g. wrap it with
            ``numpy.frombuffer`` or ``torch.frombuffer`` for a properly
            typed/shaped view.
        :raises RuntimeError: If :attr:`element_layout` isn't a struct,
            this buffer isn't host-visible, doesn't hold exactly one
            instance of ``field``'s root layout, or ``field`` describes a
            struct.
        """
        ...

    def write(self, field: LayoutField, value: object) -> None:
        """Writes ``value`` into ``field`` directly via ``memcpy``,
        bypassing DLPack -- much cheaper than going through :meth:`field`
        for small (e.g. scalar) fields.

        Same buffer requirements as :meth:`read`.

        :param field: Field to write. Must describe a scalar, vector,
            matrix, or (possibly nested) array of any of those -- not a
            struct, since a struct isn't a single numeric type.
        :param value: For a scalar field, a plain Python number. For a
            vector or matrix field, also a plain (possibly nested, for a
            matrix) Python list/tuple of numbers -- converted on the
            Python side to a packed array before writing. For a vector,
            matrix, or array field, otherwise, a Python object supporting
            the buffer protocol (e.g. a C-contiguous numpy array) or a
            DLPack-compatible object (e.g. a CPU torch tensor) -- both are
            read directly via ``memcpy``, so a CUDA tensor is rejected
            (there's no valid CPU pointer to copy from).
        :raises RuntimeError: If :attr:`element_layout` isn't a struct,
            this buffer isn't host-visible, doesn't hold exactly one
            instance of ``field``'s root layout, ``value``'s size doesn't
            match ``field``'s, or (for a DLPack source) it isn't
            CPU-accessible.
        """
        ...

    def load(self, source: object) -> None:
        """Copies ``source``'s data into this buffer's own memory, as a
        whole -- a shortcut for
        ``torch.from_dlpack(buffer).copy_(source)``, using the same
        DLPack/buffer-protocol handling as :meth:`Device.wrap` (so a CUDA
        source works via the interop plugin even when this buffer is
        DEVICE-located, and a strided source is handled correctly).

        Unlike :meth:`write`, which targets a single struct field, this
        always copies the buffer's entire contents.

        :param source: A DLPack-compatible object (e.g. a torch tensor,
            CPU or CUDA) or a Python buffer-protocol object (e.g. a numpy
            array). Its total byte size must equal :attr:`size`.
        :raises RuntimeError: If ``source`` is neither, its size doesn't
            match :attr:`size`, or a copy from CUDA-resident data is
            required but no interop library is available.
        """
        ...

    def save(self, target: object) -> None:
        """Copies this buffer's own memory, as a whole, into ``target``.

        Symmetric to :meth:`load`.

        :param target: A DLPack-compatible object or a writable Python
            buffer-protocol object, already sized to receive
            :attr:`size` bytes.
        :raises RuntimeError: If ``target`` is neither, is read-only, its
            size doesn't match :attr:`size`, or a copy to CUDA-resident
            data is required but no interop library is available.
        """
        ...


class Tensor:
    """A plain N-dimensional array resource of a single scalar type,
    obtained from :meth:`Device.create_tensor`.

    Unlike :class:`Buffer` (whose DLPack shape/dtype is derived from its
    element layout, falling back to raw bytes for a struct), a
    :class:`Tensor`'s shape is exactly what was requested at creation
    time. Implements the DLPack protocol (``__dlpack__``/
    ``__dlpack_device__``) directly, so it can be passed straight to
    ``torch.from_dlpack(tensor)``. A resource like any other: tracked for
    proactive disposal the same way as :class:`Buffer`/:class:`Image`,
    and directly wrappable via :meth:`Device.wrap` with no copy.
    """

    @property
    def shape(self) -> list[int]:
        """Shape of this tensor, in elements."""
        ...

    @property
    def scalar_type(self) -> ScalarType:
        """Scalar element type of this tensor."""
        ...

    @property
    def size(self) -> int:
        """Size of this tensor, in bytes."""
        ...

    @property
    def device_ptr(self) -> int:
        """Vulkan buffer device address of this tensor -- a raw pointer
        usable from within a shader, not a CUDA/host pointer.
        """
        ...

    def load(self, source: object) -> None:
        """Copies ``source``'s data into this tensor's own memory, as a
        whole -- a shortcut for
        ``torch.from_dlpack(tensor).copy_(source)``, using the same
        DLPack/buffer-protocol handling as :meth:`Device.wrap`.

        :param source: A DLPack-compatible object or a Python
            buffer-protocol object. Its total byte size must equal
            :attr:`size`.
        :raises RuntimeError: If ``source`` is neither, its size doesn't
            match :attr:`size`, or a copy from CUDA-resident data is
            required but no interop library is available.
        """
        ...

    def save(self, target: object) -> None:
        """Copies this tensor's own memory, as a whole, into ``target``.

        Symmetric to :meth:`load`.

        :param target: A DLPack-compatible object or a writable Python
            buffer-protocol object, already sized to receive
            :attr:`size` bytes.
        :raises RuntimeError: If ``target`` is neither, is read-only, its
            size doesn't match :attr:`size`, or a copy to CUDA-resident
            data is required but no interop library is available.
        """
        ...


class WrappedMemory:
    """A handle to memory owned by an external Python object (a
    :class:`Buffer`, a :class:`Tensor`, a DLPack-compatible tensor, or a
    Python buffer-protocol object), obtained through :meth:`Device.wrap`.

    Exposes a Vulkan buffer device address (``device_ptr``) usable from
    within a shader, alongside the wrapped object's ``shape`` and
    ``scalar_type``. If wrapping required an intermediate copy, this
    object copies data back to the source when :meth:`unwrap` is called
    (or, failing that, when it's garbage collected), depending on the
    :class:`WrapMode` it was created with.
    """

    def unwrap(self) -> None:
        """Performs, immediately, whatever copy-back this wrap's
        :class:`WrapMode` calls for (``OUT``/``INOUT``), instead of
        leaving it to whenever the garbage collector gets around to
        destroying this object.

        Safe to call more than once -- only the first call has any
        effect. Unlike garbage collection, exceptions raised while
        copying back propagate to the caller.
        """
        ...

    @property
    def device_ptr(self) -> int:
        """Vulkan buffer device address of the (possibly copied)
        contiguous memory -- a raw pointer usable from a shader, not a
        CUDA/host pointer.
        """
        ...

    @property
    def shape(self) -> list[int]:
        """Shape, in elements, of the wrapped object."""
        ...

    @property
    def scalar_type(self) -> ScalarType:
        """Scalar element type of the wrapped object."""
        ...


class Image:
    """A 1D/2D/3D image resource, obtained from :meth:`Device.create_image`.

    Every image is created with a broad set of usage flags (transfer
    source/destination, sampled, storage, color attachment) and is
    transitioned to ``VK_IMAGE_LAYOUT_GENERAL`` once at creation time --
    the one layout every other operation on it (descriptor binds,
    :meth:`CommandBuffer.blit_image`, use as a framebuffer attachment)
    assumes uniformly, since this library does not otherwise track
    per-image Vulkan layouts.
    """

    @property
    def format(self) -> Format:
        """Pixel format of this image."""
        ...

    @property
    def mip_count(self) -> int:
        """Number of mip levels this image (view) covers."""
        ...

    @property
    def array_count(self) -> int:
        """Number of array layers this image (view) covers."""
        ...

    def cast_format(self, format: Format) -> "Image":
        """Reinterprets this image's texel data as `format` (same
        underlying image and subresource range), for a format of the
        same texel size.
        """
        ...

    def slice(self, mip_start: int, mip_count: int, array_start: int, array_count: int) -> "Image":
        """A view over a sub-range of this image's mip levels and array
        layers, sharing the same underlying image.
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

    def set_framebuffer(self, framebuffer: "Framebuffer") -> None:
        """Begins rendering into ``framebuffer``'s render pass, ending any
        previously active one first.

        A command buffer may switch render targets multiple times while
        recording, the same way it may switch pipelines. Must be called
        while the command buffer is still recording, i.e. before
        :meth:`close`. Keeps a reference to ``framebuffer`` alive for as
        long as this command buffer is, including while submitted and
        pending on the GPU. Graphics pipelines only: :meth:`set_pipeline`
        with a :attr:`PipelineType.RASTERIZATION` pipeline requires a
        render pass to already be active.

        :param framebuffer: Framebuffer (and its render pass) to render into.
        """
        ...

    def set_viewport(self, x: float, y: float, width: float, height: float) -> None:
        """Sets the dynamic viewport and scissor rectangle (identically)
        used by subsequent draw calls.

        Must be called while the command buffer is still recording, i.e.
        before :meth:`close`. Graphics pipelines only.

        :param x: Left edge of the viewport, in pixels.
        :param y: Top edge of the viewport, in pixels.
        :param width: Viewport width, in pixels.
        :param height: Viewport height, in pixels.
        """
        ...

    def blit_image(self, src: "Image", dst: "Image", filter: Filter = Filter.LINEAR) -> None:
        """Records a device-side blit (resizing/format-converting as
        needed) from ``src`` into ``dst``, covering each image's full
        extent.

        Both images are assumed to already be in
        ``VK_IMAGE_LAYOUT_GENERAL`` -- the layout every :class:`Image` is
        transitioned to once, at creation time (see :class:`Image`). Must
        be called while the command buffer is still recording, i.e.
        before :meth:`close`, and cannot be called while a render pass is
        active (between a :meth:`set_framebuffer` call and the next
        :meth:`set_framebuffer`/:meth:`close`). Keeps references to
        ``src``/``dst`` alive for as long as this command buffer is,
        including while submitted and pending on the GPU.

        :param src: Image to blit from.
        :param dst: Image to blit into.
        :param filter: Texel filtering to use if ``src`` and ``dst``
            have different extents.
        """
        ...

    def bind_vertices(self, binding: int, vertex_buffer: Buffer) -> None:
        """Binds ``vertex_buffer`` at vertex input binding ``binding``,
        matching a :meth:`Pipeline.vertex_layout` declaration.

        Must be called while the command buffer is still recording, i.e.
        before :meth:`close`. Keeps a reference to ``vertex_buffer`` alive
        for as long as this command buffer is, including while submitted
        and pending on the GPU.

        :param binding: Vertex input binding index, matching the
            ``start_location``'s binding from :meth:`Pipeline.vertex_layout`.
        :param vertex_buffer: Buffer of per-vertex data to bind.
        """
        ...

    def bind_indices(self, index_buffer: Buffer) -> None:
        """Binds ``index_buffer`` for subsequent
        :meth:`dispatch_indexed_primitives` calls.

        ``index_buffer.element_layout`` must be a scalar
        :attr:`ScalarType.UINT16` or :attr:`ScalarType.UINT32`. Must be
        called while the command buffer is still recording, i.e. before
        :meth:`close`. Keeps a reference to ``index_buffer`` alive for as
        long as this command buffer is, including while submitted and
        pending on the GPU.

        :param index_buffer: Buffer of ``uint16``/``uint32`` indices to bind.
        """
        ...

    def dispatch_primitives(self, vertices: int, vertex_start: int = 0) -> None:
        """Records a non-indexed draw of ``vertices`` vertices, starting
        at ``vertex_start`` within the vertex buffers bound via
        :meth:`bind_vertices`.

        Must be called while the command buffer is still recording, i.e.
        before :meth:`close`, and requires a
        :attr:`PipelineType.RASTERIZATION` pipeline to be bound and an
        active render pass (see :meth:`set_framebuffer`).

        :param vertices: Number of vertices to draw.
        :param vertex_start: Index of the first vertex to draw.
        """
        ...

    def dispatch_indexed_primitives(self, indices: int, index_start: int = 0, vertex_offset: int = 0) -> None:
        """Records an indexed draw of ``indices`` indices, starting at
        ``index_start`` within the index buffer bound via
        :meth:`bind_indices`.

        Must be called while the command buffer is still recording, i.e.
        before :meth:`close`, and requires :meth:`bind_indices` to have
        been called, a :attr:`PipelineType.RASTERIZATION` pipeline to be
        bound, and an active render pass (see :meth:`set_framebuffer`).

        :param indices: Number of indices to draw.
        :param index_start: Index of the first index to read within the
            bound index buffer.
        :param vertex_offset: Added to every fetched index before it's
            used to read vertex attributes.
        """
        ...

    def close(self) -> None:
        """Ends recording (and any still-active render pass) and makes
        it ready to submit for execution on its engine.

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


class Frame:
    """One frame's worth of resources, obtained from :meth:`Window.begin_frame`.

    Exactly one of :meth:`render_target`, :meth:`image_target`,
    :meth:`buffer_target` or :meth:`tensor_target` must be called to pick
    how this frame's content will be produced -- calling one discards the
    other three (so only one kind of resource is ever actually used per
    frame), and :meth:`present` requires that one of them was called
    first. Usage::

        window = device.create_window(1280, 720, "my app", vk.Format.BGRA8_UNorm)
        while window.check_alive():
            frame = window.begin_frame()
            render_target = frame.render_target()
            # ... render into render_target (e.g. via a Pipeline/Framebuffer
            # built around it) ...
            frame.present()
    """

    def render_target(self) -> "Image":
        """Selects this frame's render target: an :class:`Image` that IS
        (aliases, with no copy) the swapchain image about to be
        presented. Render into it directly (e.g. as a
        :meth:`Pipeline.create_framebuffer` color attachment) for the
        lowest-overhead path -- :meth:`present` then only needs a layout
        transition, no blit.

        :raises RuntimeError: If a different target was already selected
            for this frame.
        """
        ...

    def image_target(self) -> "Image":
        """Selects this frame's image target: a plain owned
        :class:`Image`, matching the window's format/size, that gets
        blitted into the presented image at :meth:`present` time.

        Use this instead of :meth:`render_target` when it's more
        convenient to write into a "normal" image (e.g. from a compute
        shader) than to render directly into the swapchain image.

        :raises RuntimeError: If a different target was already selected
            for this frame.
        """
        ...

    def buffer_target(self) -> "Buffer":
        """Selects this frame's buffer target: a plain owned
        :class:`Buffer` of ``width * height * 4`` bytes (matching the
        window's format), uploaded and blitted into the presented image
        at :meth:`present` time.

        Use :meth:`Buffer.load` to fill it (e.g. from a numpy array or
        torch tensor) without needing to render or dispatch anything.

        :raises RuntimeError: If a different target was already selected
            for this frame.
        """
        ...

    def tensor_target(self) -> "Tensor":
        """Selects this frame's tensor target: a plain owned
        :class:`Tensor` of shape ``[height, width, 4]`` (matching the
        window's format), uploaded and blitted into the presented image
        at :meth:`present` time.

        Use :meth:`Tensor.load` to fill it (e.g. from a numpy array or
        torch tensor) without needing to render or dispatch anything.

        :raises RuntimeError: If a different target was already selected
            for this frame.
        """
        ...

    @property
    def index(self) -> int:
        """This frame's index: increments by one every
        :meth:`Window.begin_frame` call, starting at 0.
        """
        ...

    @property
    def is_target_acquired(self) -> bool:
        """Whether one of :meth:`render_target`/:meth:`image_target`/
        :meth:`buffer_target`/:meth:`tensor_target` has been called yet.
        """
        ...

    @property
    def is_render_target(self) -> bool:
        """Whether :meth:`render_target` was the target selected."""
        ...

    @property
    def is_image_target(self) -> bool:
        """Whether :meth:`image_target` was the target selected."""
        ...

    @property
    def is_buffer_target(self) -> bool:
        """Whether :meth:`buffer_target` was the target selected."""
        ...

    @property
    def is_tensor_target(self) -> bool:
        """Whether :meth:`tensor_target` was the target selected."""
        ...

    @property
    def presented(self) -> bool:
        """Whether :meth:`present` has already been called on this frame."""
        ...

    def present(self) -> None:
        """Presents this frame: submits the pre-recorded command buffer
        matching whichever target was selected, then presents the result.

        Frames must be presented in the same order they were obtained
        from :meth:`Window.begin_frame`.

        :raises RuntimeError: If no target was selected (see
            :attr:`is_target_acquired`), or this frame is presented out
            of order.
        """
        ...


class Stats:
    """Read-only per-frame timing info, exposed as :attr:`Window.stats`."""

    @property
    def fps(self) -> float:
        """An exponential moving average of 1/frame_time, updated once per
        :meth:`Window.begin_frame` call -- not an instantaneous
        single-frame value (too noisy to display directly).
        """
        ...


class Checkbox:
    """A persistent checkbox widget, obtained from :meth:`Window.checkbox`.

    Must be drawn via :meth:`draw` once per frame (between
    :meth:`Window.begin_frame` and :meth:`Frame.present`) to appear at
    all -- ImGui is immediate-mode, so a widget that isn't (re)drawn this
    frame simply isn't shown. Read/write :attr:`value` whenever;
    :meth:`draw` updates it in place if the user toggles it.
    """

    def draw(self) -> bool:
        """Draws this frame. :return: Whether the value changed this frame."""
        ...

    @property
    def value(self) -> bool: ...
    @value.setter
    def value(self, value: bool) -> None: ...


class SliderFloat:
    """A persistent float slider widget, obtained from
    :meth:`Window.slider_float`. See :class:`Checkbox` for the draw()/value
    usage pattern.
    """

    def draw(self) -> bool:
        """Draws this frame. :return: Whether the value changed this frame."""
        ...

    @property
    def value(self) -> float: ...
    @value.setter
    def value(self, value: float) -> None: ...


class SliderInt:
    """A persistent integer slider widget, obtained from
    :meth:`Window.slider_int`. See :class:`Checkbox` for the draw()/value
    usage pattern.
    """

    def draw(self) -> bool:
        """Draws this frame. :return: Whether the value changed this frame."""
        ...

    @property
    def value(self) -> int: ...
    @value.setter
    def value(self, value: int) -> None: ...


class Combobox:
    """A persistent combobox widget, obtained from :meth:`Window.combobox`.
    See :class:`Checkbox` for the draw()/value usage pattern.
    """

    def draw(self) -> bool:
        """Draws this frame. :return: Whether the selection changed this frame."""
        ...

    @property
    def selected_index(self) -> int: ...
    @selected_index.setter
    def selected_index(self, index: int) -> None: ...
    @property
    def selected_item(self) -> str:
        """``items[selected_index]``."""
        ...
    @property
    def items(self) -> list[str]: ...


class Window:
    """An OS window (via GLFW) plus its Vulkan swapchain and presentation
    machinery, obtained from :meth:`Device.create_window`.

    Drives a render loop via :meth:`check_alive`/:meth:`begin_frame`/
    :meth:`Frame.present` -- see :class:`Frame`'s docstring for the usage
    pattern. ``frames_on_the_fly`` (from :meth:`Device.create_window`)
    becomes the swapchain's image count (clamped to whatever the surface
    capabilities actually grant): every possible target-kind-to-presented-
    image command buffer is pre-recorded once, at window/swapchain
    creation time, so presenting never re-records anything.

    The swapchain itself is always :attr:`Format.BGRA8_UNorm`. The
    ``format`` passed to :meth:`Device.create_window` instead controls
    :meth:`Frame.image_target`/:meth:`Frame.buffer_target`/
    :meth:`Frame.tensor_target`, which may be any format -- presenting one
    of them blits it into the swapchain image, converting formats along
    the way.
    """

    def check_alive(self) -> bool:
        """Polls OS/window events and returns whether the window is still
        open (``False`` once the user has closed it).
        """
        ...

    def set_title(self, title: str) -> None:
        """Changes the window's title bar text."""
        ...

    def set_size(self, width: int, height: int) -> None:
        """Resizes the window (and its swapchain)."""
        ...

    @property
    def width(self) -> int:
        """Current width, in pixels."""
        ...

    @property
    def height(self) -> int:
        """Current height, in pixels."""
        ...

    def begin_frame(self) -> Frame:
        """Waits for a swapchain image to become available and returns a
        new :class:`Frame` for it.

        :return: A new, not-yet-presented :class:`Frame`.
        """
        ...

    @property
    def stats(self) -> Stats:
        """This window's per-frame timing stats (currently just fps)."""
        ...

    def label(self, text: str, value: float | str | None = None) -> None:
        """Displays `text` (optionally followed by `value`) via ImGui,
        drawn as an overlay on top of whatever this frame presents
        (:meth:`Frame.render_target`/:meth:`Frame.image_target`/
        :meth:`Frame.buffer_target`/:meth:`Frame.tensor_target` -- see
        this class's docstring). Direct and stateless: call it every
        frame, between :meth:`begin_frame` and the matching
        :meth:`Frame.present`.
        """
        ...

    def button(self, text: str) -> bool:
        """Draws a button labeled `text`. Direct and stateless: call it
        every frame; :return: whether it was clicked this frame.
        """
        ...

    def checkbox(self, label: str, value: bool = False) -> Checkbox:
        """Creates a persistent :class:`Checkbox` widget. Call this once;
        call the returned widget's ``draw()`` every frame afterwards.
        """
        ...

    def slider_float(self, label: str, min: float, max: float, value: float) -> SliderFloat:
        """Creates a persistent :class:`SliderFloat` widget. Call this
        once; call the returned widget's ``draw()`` every frame afterwards.
        """
        ...

    def slider_int(self, label: str, min: int, max: int, value: int) -> SliderInt:
        """Creates a persistent :class:`SliderInt` widget. Call this once;
        call the returned widget's ``draw()`` every frame afterwards.
        """
        ...

    def combobox(self, label: str, items: list[str], selected_index: int = 0) -> Combobox:
        """Creates a persistent :class:`Combobox` widget. Call this once;
        call the returned widget's ``draw()`` every frame afterwards.
        """
        ...


class DescriptorSet:
    """A descriptor set matching one :class:`Pipeline`'s declared layout for
    a given ``set`` index, obtained from :meth:`Pipeline.create_descriptor_set`.
    """

    def bind(self, layout_id: LayoutHandle, resource: "Buffer | object") -> None:
        """Writes a resource into the binding identified by ``layout_id``.

        Accepts either a :class:`Buffer` (for a binding declared with
        :attr:`DescriptorType.STORAGE_BUFFER` or
        :attr:`DescriptorType.UNIFORM_BUFFER`) or an :class:`Image` (for
        :attr:`DescriptorType.STORAGE_IMAGE` or
        :attr:`DescriptorType.SAMPLED_IMAGE`; sampler-based descriptor
        types are not yet supported).

        :param layout_id: Handle returned by the matching :meth:`Pipeline.layout` call.
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

    def layout(self, set: int, binding: int, description: DescriptorType, count: int = 1) -> LayoutHandle:
        """Declares one binding of the descriptor set layout for ``set``.

        Must be called before :meth:`close`.

        :param set: Descriptor set index.
        :param binding: Binding index within ``set``.
        :param description: Kind of resource this binding expects.
        :param count: Number of array elements at this binding.
        :return: An opaque handle identifying this binding, for use with
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

    def attach(self, slot: int, format: Format) -> AttachHandle:
        """Declares one color attachment of this pipeline's render pass.

        Graphics pipelines only. Must be called before :meth:`close`.

        :param slot: Fragment shader output location for this attachment.
        :param format: Pixel format of this attachment.
        :return: An opaque handle used to bind a render target image via
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

    def create_framebuffer(self, attachments: list[tuple[AttachHandle, "Image"]]) -> Framebuffer:
        """Creates a framebuffer compatible with this pipeline's render pass.

        Pipeline must be closed first. Graphics pipelines only.

        :param attachments: One ``(slot, image)`` pair per handle returned
            by :meth:`attach`, each image matching that attachment's
            declared format, and all sharing the same dimensions.
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

    def create_tensor(self, shape: list[int], scalar_type: ScalarType, location: MemoryLocation) -> Tensor:
        """Creates a :class:`Tensor`: a plain N-dimensional array resource
        of a single scalar type.

        Unlike :meth:`create_buffer`, whose shape/dtype for DLPack export
        is derived from its element layout, a :class:`Tensor`'s shape is
        exactly ``shape`` -- pass it straight to ``torch.from_dlpack()``.

        :param shape: Shape of the tensor to allocate.
        :param scalar_type: Scalar element type of the tensor.
        :param location: Memory location where the tensor should be allocated.
        :return: A new :class:`Tensor`.
        """
        ...

    @overload
    def create_buffer(self, elements: int, scalar_type: ScalarType, location: MemoryLocation) -> Buffer:
        """Creates a buffer to store an array of elements of a given scalar
        type. Its :attr:`Buffer.element_layout` is that scalar type.

        A raw byte buffer is just this with ``scalar_type =
        ScalarType.UINT8``: its :attr:`Buffer.element_layout` dlpack-exports
        as raw bytes, and :meth:`Buffer.slice`/:meth:`Buffer.element`
        operate byte-by-byte.

        :param elements: Number of elements in the array.
        :param scalar_type: Scalar element type of the array.
        :param location: Memory location where the buffer should be allocated.
        :return: A new :class:`Buffer`.
        """
        ...

    @overload
    def create_buffer(self, elements: int, format: Format, location: MemoryLocation) -> Buffer:
        """Creates a buffer to store an array of texels of a given format.
        Its :attr:`Buffer.element_layout` is an array of the format's own
        per-channel scalar type.

        :param elements: Number of texels in the array.
        :param format: Texel format of the array.
        :param location: Memory location where the buffer should be allocated.
        :return: A new :class:`Buffer`.
        """
        ...

    @overload
    def create_buffer(self, elements: int, layout: Layout, location: MemoryLocation) -> Buffer:
        """Creates a buffer sized to hold ``elements`` instances of the
        type already laid out by :func:`compute_layout`, e.g. for a uniform
        or storage buffer. Its :attr:`Buffer.element_layout` is ``layout``
        itself.

        :param elements: Number of instances to allocate room for.
            ``layout.aligned_size`` is used as the per-instance stride, so
            array-stride padding between instances is accounted for even
            when ``elements`` is 1.
        :param layout: Layout of a single instance, e.g. from
            ``compute_layout(type, rule)``.
        :param location: Memory location where the buffer should be allocated.
        :return: A new :class:`Buffer`.
        """
        ...

    def create_image(
        self,
        width: int,
        height: int,
        depth: int,
        mip_levels: int,
        array_layers: int,
        format: Format,
        location: MemoryLocation,
    ) -> Image:
        """Creates a new 1D/2D/3D image.

        ``depth > 1`` creates a 3D image (in which case ``array_layers``
        must be 1); otherwise a 2D image (or 1D, if ``height`` is also 1),
        optionally arrayed if ``array_layers > 1``.

        Every image is created with a broad set of usage flags (transfer
        source/destination, sampled, storage, color attachment) and is
        transitioned to ``VK_IMAGE_LAYOUT_GENERAL`` once, synchronously,
        before this call returns.

        :param width: Width in texels.
        :param height: Height in texels (1 for a 1D image).
        :param depth: Depth in texels (1 for a 1D/2D image; > 1 makes
            this a 3D image).
        :param mip_levels: Number of mip levels.
        :param array_layers: Number of array layers (must be 1 if
            ``depth > 1``).
        :param format: Pixel format.
        :param location: Memory location for the image's backing store.
        :return: A new :class:`Image`.
        """
        ...

    def create_window(
        self,
        width: int,
        height: int,
        title: str,
        format: Format,
        frames_on_the_fly: int = 3,
    ) -> Window:
        """Creates an OS window (via GLFW) with a Vulkan swapchain, ready
        to render into and present through.

        :param width: Initial window width, in pixels.
        :param height: Initial window height, in pixels.
        :param title: Initial window title.
        :param format: Pixel format for :meth:`Frame.image_target`/
            :meth:`Frame.buffer_target`/:meth:`Frame.tensor_target` --
            any format works, including ones with non-8-bit or
            floating-point channels. The swapchain (and
            :meth:`Frame.render_target`) is always
            :attr:`Format.BGRA8_UNorm` regardless of this parameter;
            presenting one of the other targets blits it into the
            swapchain image, converting formats along the way.
        :param frames_on_the_fly: Requested swapchain image count (and
            number of pre-recorded, ready-to-submit frame slots); the
            actual count is whatever the surface capabilities grant for
            this request.
        :return: A new :class:`Window`.
        :raises RuntimeError: If VK_KHR_swapchain isn't supported/enabled
            on this device, GLFW window/surface creation fails, or this
            surface doesn't support presenting BGRA8_UNorm.
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

    def wrap(
        self,
        obj: "Buffer | Tensor | object",
        mode: WrapMode,
        location: MemoryLocation = MemoryLocation.DEVICE,
    ) -> WrappedMemory:
        """Wraps an external object as a :class:`WrappedMemory` exposing a
        Vulkan buffer device address (``device_ptr``, usable from a shader),
        without necessarily copying its data.

        ``obj`` can be:

        - A :class:`Buffer` or :class:`Tensor`: its own device address is
          used directly, no copy.
        - A DLPack-compatible object (e.g. a torch tensor): if it's
          already contiguous memory that belongs to this device's own
          memory managers (e.g. it came from ``torch.from_dlpack()`` on
          one of our own buffers), its corresponding device address is
          reused directly, no copy. Otherwise a new buffer is allocated
          at ``location`` and the data is copied in (via the CUDA
          interop plugin for CUDA-resident data, or a strided host
          memcpy otherwise), respecting ``mode``.
        - A Python buffer-protocol object (e.g. a C-contiguous numpy
          array): always copied into a new buffer at ``location``, since
          a foreign host allocation has no Vulkan device address of its
          own.

        :param obj: The object to wrap.
        :param mode: Whether a required copy happens when the wrap is
            created (``IN``), when :meth:`WrappedMemory.unwrap` is
            called or this object is garbage collected (``OUT``), or
            both (``INOUT``).
        :param location: Where to allocate a copy, if one is needed.
            Ignored when the object can be used directly.
        :return: A new :class:`WrappedMemory`.
        :raises RuntimeError: If ``obj`` is none of the above, or a
            copy is required from/to CUDA-resident data but no interop
            library is available.
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
