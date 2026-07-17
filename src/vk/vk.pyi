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


class VertexAttribute(enum.Enum):
    """A single interleaved-vertex-buffer attribute kind, as produced by
    :meth:`Device.load_scene` and stored on :class:`Mesh`.
    """

    POSITION = 0
    NORMAL = 1
    TEXCOORD = 2
    TANGENT = 3
    BITANGENT = 4


class VertexResolutionMode(enum.Enum):
    """How :meth:`Device.load_scene` decides whether two face-corners
    referring to the same source data resolve to the same output vertex.
    """

    BY_POSITION = 0
    """Weld by position only: face-corners sharing a position keep
    whichever corner's normal/texcoord was seen first; any differing
    normal/texcoord on later corners is discarded."""
    BY_ALL_ATTRIBUTES = 1
    """Weld by the full (position, normal, texcoord) tuple: two
    face-corners share a vertex only if every attribute matches. Default."""
    DUPLICATE = 2
    """No welding: every face-corner gets its own vertex."""


class Type(enum.Enum):
    """Scalar, vector or matrix element type supported by buffers/tensors
    allocated through :class:`Device`, and by a type description passed to
    :func:`compute_layout` (see also :class:`TypeDescriptor`).

    Vector/matrix members match GLSL's own shapes (``matCxR``: `C` columns,
    `R` rows). ``INT8``/``UINT8`` aren't valid GLSL shader types but are
    kept for byte-level buffer/``Format``-channel use.
    """

    UNDEFINED = 0
    """No type specified."""
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
    VEC2 = 13
    """2-component FLOAT32 vector."""
    VEC3 = 14
    """3-component FLOAT32 vector."""
    VEC4 = 15
    """4-component FLOAT32 vector."""
    IVEC2 = 16
    """2-component INT32 vector."""
    IVEC3 = 17
    """3-component INT32 vector."""
    IVEC4 = 18
    """4-component INT32 vector."""
    UVEC2 = 19
    """2-component UINT32 vector."""
    UVEC3 = 20
    """3-component UINT32 vector."""
    UVEC4 = 21
    """4-component UINT32 vector."""
    BVEC2 = 22
    """2-component BOOL vector."""
    BVEC3 = 23
    """3-component BOOL vector."""
    BVEC4 = 24
    """4-component BOOL vector."""
    MAT2 = 25
    """2x2 FLOAT32 matrix."""
    MAT2x3 = 26
    """2-column, 3-row FLOAT32 matrix."""
    MAT2x4 = 27
    """2-column, 4-row FLOAT32 matrix."""
    MAT3x2 = 28
    """3-column, 2-row FLOAT32 matrix."""
    MAT3 = 29
    """3x3 FLOAT32 matrix."""
    MAT3x4 = 30
    """3-column, 4-row FLOAT32 matrix."""
    MAT4x2 = 31
    """4-column, 2-row FLOAT32 matrix."""
    MAT4x3 = 32
    """4-column, 3-row FLOAT32 matrix."""
    MAT4 = 33
    """4x4 FLOAT32 matrix."""


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

    Depth16_UNorm = 124
    """16-bit unsigned normalized depth. Only valid with
    :meth:`Device.create_depth_buffer_image`."""
    Depth32_Float = 126
    """32-bit floating point depth. Only valid with
    :meth:`Device.create_depth_buffer_image`."""
    Depth24_UNorm_Stencil8_UInt = 129
    """24-bit unsigned normalized depth plus 8-bit stencil. Only valid
    with :meth:`Device.create_depth_buffer_image`."""
    Depth32_Float_Stencil8_UInt = 130
    """32-bit floating point depth plus 8-bit stencil. Only valid with
    :meth:`Device.create_depth_buffer_image`."""


class EngineType(enum.IntFlag):
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
    """A read-only image with no sampler of its own -- pair it with a
    separate :attr:`SAMPLER` binding (GLSL ``texture2D``)."""
    SAMPLER = 4
    """A standalone sampler (:class:`Sampler`), paired with a separate
    :attr:`SAMPLED_IMAGE` binding."""
    COMBINED_IMAGE_SAMPLER = 5
    """An image and sampler bound together in one binding (GLSL ``sampler2D``)."""
    ACCELERATION_STRUCTURE = 6
    """A ray tracing acceleration structure. Not yet bindable via
    :meth:`DescriptorSet.bind`.
    """


class Filter(enum.Enum):
    """Texel filtering: used by :meth:`CommandBuffer.blit_image` when the
    source and destination extents differ, and by
    :meth:`Device.create_sampler` (``mag_filter``/``min_filter``).
    """

    NEAREST = 0
    """Nearest-texel sampling."""
    LINEAR = 1
    """Linearly interpolated sampling."""


class MipmapMode(enum.Enum):
    """How a :class:`Sampler` filters between mip levels, set via
    :meth:`Device.create_sampler`'s ``mipmap_mode``.
    """

    NEAREST = 0
    """Snaps to the nearest mip level."""
    LINEAR = 1
    """Linearly interpolates between the two nearest mip levels."""


class WrapMode(enum.Enum):
    """How a :class:`Sampler` handles texture coordinates outside
    ``[0, 1]``, set (once per axis) via :meth:`Device.create_sampler`'s
    ``wrap_u``/``wrap_v``/``wrap_w``.
    """

    REPEAT = 0
    """Tiles the texture (wraps around)."""
    MIRRORED_REPEAT = 1
    """Tiles the texture, mirroring every other tile."""
    CLAMP_TO_EDGE = 2
    """Clamps to the texture's edge texels."""
    CLAMP_TO_BORDER = 3
    """Clamps to a fixed border color (always opaque black -- there is no
    separate border-color parameter)."""


class CullMode(enum.Enum):
    """Dynamic rasterization cull mode, set via
    :meth:`CommandBuffer.set_cull_mode`. Requires Vulkan 1.3 (core
    "extended dynamic state").
    """

    NONE = 0
    """No faces are culled."""
    FRONT = 1
    """Front-facing (per :class:`FrontFace`) triangles are culled."""
    BACK = 2
    """Back-facing triangles are culled."""
    FRONT_AND_BACK = 3
    """Every triangle is culled."""


class FrontFace(enum.Enum):
    """Winding order that identifies a triangle's front face, set via
    :meth:`CommandBuffer.set_front_face`. Requires Vulkan 1.3.
    """

    COUNTER_CLOCKWISE = 0
    CLOCKWISE = 1


class CompareOp(enum.Enum):
    """Depth comparison operator, set via
    :meth:`CommandBuffer.set_depth_test`. A fragment passes the depth
    test when ``fragment_depth <compare_op> stored_depth`` is true.
    """

    NEVER = 0
    LESS = 1
    EQUAL = 2
    LESS_OR_EQUAL = 3
    GREATER = 4
    NOT_EQUAL = 5
    GREATER_OR_EQUAL = 6
    ALWAYS = 7


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

    SINGLE = 0
    """A single scalar, vector or matrix value -- see :attr:`Layout.type`
    (and :attr:`Layout.element_layout`, non-``None`` for a matrix) for
    which."""
    ARRAY = 1
    """A fixed-size, or unsized/runtime, array of a single element type."""
    STRUCT = 2
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
    """Describes the shape of a GPU-buffer-resident type: a single
    scalar/vector/matrix value, an array or a struct.

    Used together with :class:`LayoutRule` and :func:`compute_layout` to
    compute byte offsets, sizes and strides without touching the GPU, and
    with ``Device.create_buffer(elements, layout, ...)`` to allocate a buffer sized
    to hold it. Not constructed directly in typical code -- pass a plain
    type spec (a bare :class:`Type`, or a ``[count, spec]``/``{name: spec}``
    literal) straight to :func:`compute_layout` instead.
    """

    @staticmethod
    def single(type: Type) -> "TypeDescriptor":
        """Creates a scalar/vector/matrix type descriptor.

        :param type: Scalar, vector or matrix element type.
        :return: A new :class:`TypeDescriptor` of kind
            :attr:`TypeKind.SINGLE`.
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
    def type(self) -> Type:
        """The exact scalar/vector/matrix type this layout was computed
        for (e.g. :attr:`Type.VEC3`, :attr:`Type.MAT4`).

        Only meaningful when :attr:`kind` is :attr:`TypeKind.SINGLE`;
        :attr:`Type.UNDEFINED` otherwise.
        """
        ...

    @property
    def component_type(self) -> Type:
        """`type`'s own base scalar component (itself, for a scalar).

        Only meaningful when :attr:`kind` is :attr:`TypeKind.SINGLE`;
        :attr:`Type.UNDEFINED` otherwise.
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
        matrix-shaped :attr:`TypeKind.SINGLE`).

        ``None`` for a scalar/vector :attr:`TypeKind.SINGLE`, or for
        :attr:`TypeKind.STRUCT`.
        """
        ...

    @property
    def stride(self) -> int:
        """Byte stride between consecutive elements or matrix columns.

        Only meaningful when :attr:`kind` is :attr:`TypeKind.ARRAY`, or a
        matrix-shaped :attr:`TypeKind.SINGLE`; zero otherwise.
        """
        ...

    @property
    def count(self) -> int:
        """Number of elements (for an array) or columns (for a matrix).

        Only meaningful when :attr:`kind` is :attr:`TypeKind.ARRAY`, or a
        matrix-shaped :attr:`TypeKind.SINGLE`; zero otherwise.
        """
        ...

    @property
    def device_index(self) -> int:
        """Index of the Device this was created from."""
        ...

    def field(self, name: str) -> LayoutField:
        """Looks up a named field of this struct layout, for navigating a
        Layout tree by name (e.g. ``Layouts.instance().field("transform")``).

        Chain into a nested struct via the returned field's own
        ``.layout`` (itself a :class:`Layout`), or into an array/matrix
        field via ``.layout.element_layout``. Raises ``RuntimeError`` if
        this layout isn't a struct, or no field named ``name`` exists.
        """
        ...


class Layouts:
    """Tool class with commonly-needed :class:`Layout`\\ s for ray tracing
    acceleration structure buffers, precomputed once and cached. Every
    layout here is computed under scalar (tightly packed, natural
    alignment) rules, matching the corresponding Vulkan C struct's own
    byte layout exactly -- navigable field-by-field via :meth:`Layout.field`.
    """

    @staticmethod
    def index16() -> Layout:
        """A single UINT16 triangle-mesh index (:attr:`ADSTriangles.indices`)."""
        ...

    @staticmethod
    def index32() -> Layout:
        """A single UINT32 triangle-mesh index (:attr:`ADSTriangles.indices`)."""
        ...

    @staticmethod
    def position() -> Layout:
        """A single tightly-packed FLOAT32 vec3 vertex position
        (:attr:`ADSTriangles.vertices`).
        """
        ...

    @staticmethod
    def aabb() -> Layout:
        """A single ``VkAabbPositionsKHR``-shaped procedural AABB entry
        (:attr:`ADSAABB.aabbs`): fields ``min_x``/``min_y``/``min_z``/
        ``max_x``/``max_y``/``max_z``.
        """
        ...

    @staticmethod
    def instance() -> Layout:
        """A single ``VkAccelerationStructureInstanceKHR``-shaped TLAS
        instance entry (:attr:`ADSInstances.instances`): fields
        ``transform`` (the row-major 3x4 affine transform, as an array of
        3 vec4 rows), ``instance_custom_index_and_mask`` and
        ``instance_shader_binding_table_record_offset_and_flags`` (each
        packing two bitfields into one uint32 -- low 24 bits/high 8 bits
        respectively: pack via ``(mask << 24) | (custom_index & 0xFFFFFF)``),
        and ``acceleration_structure_reference`` (the referenced BLAS's
        device address, see ``AccelerationStructure.device_address``).
        """
        ...


class ADSTriangles:
    """Declares the geometry for a bottom-level acceleration structure
    (BLAS) built from a triangle mesh.
    """

    def __init__(
        self,
        vertices: Buffer,
        vertex_count: int,
        indices: Buffer | None = None,
        primitive_count: int = 0,
        opaque: bool = True,
    ) -> None:
        """`vertices` must be a DEVICE-resident buffer of tightly-packed
        FLOAT32 vec3 positions (e.g. ``Layouts.position()``) with at least
        `vertex_count` elements. `indices`, if given, must be a DEVICE-
        resident buffer whose element_layout is a scalar UINT16 or UINT32
        (non-indexed triangles are used otherwise); `primitive_count` is
        the triangle count either way.
        """
        ...

    vertices: Buffer
    vertex_count: int
    indices: Buffer | None
    primitive_count: int
    opaque: bool


class ADSAABB:
    """Declares the geometry for a bottom-level acceleration structure
    (BLAS) built from procedural AABBs.
    """

    def __init__(self, aabbs: Buffer, count: int, opaque: bool = True) -> None:
        """`aabbs` must be a DEVICE-resident buffer of
        ``Layouts.aabb()``-shaped entries, with at least `count` elements.
        """
        ...

    aabbs: Buffer
    count: int
    opaque: bool


class ADSInstances:
    """Declares a top-level acceleration structure (TLAS) built from
    instances of other, already-built bottom-level acceleration
    structures.
    """

    def __init__(self, instances: Buffer, count: int) -> None:
        """`instances` must be a DEVICE-resident buffer of
        ``Layouts.instance()``-shaped entries, with at least `count`
        elements.
        """
        ...

    instances: Buffer
    count: int


ADSDeclaration = ADSTriangles | ADSAABB | ADSInstances
"""Declares the geometry backing one acceleration structure build:
:class:`ADSTriangles`/:class:`ADSAABB` build a BLAS, :class:`ADSInstances`
builds a TLAS. Passed to ``Device.create_ads`` (to size and allocate the
acceleration structure) and ``CommandBuffer.build_ads`` (to actually
record the build).
"""


class AccelerationStructure:
    """User-facing handle to a built (or build-pending) acceleration
    structure, obtained from ``Device.create_ads``. Pass it to
    ``CommandBuffer.build_ads`` to actually record its build, and to
    ``DescriptorSet.bind`` (declared type ``DescriptorType.ACCELERATION_STRUCTURE``)
    to use it from a shader, or reference it from a TLAS instance entry
    via :attr:`device_address`.
    """

    @property
    def device_address(self) -> int:
        """This acceleration structure's Vulkan device address: for a
        BLAS, the value to place in a TLAS instance entry's
        ``acceleration_structure_reference`` field (see
        ``Layouts.instance()``).
        """
        ...

    @property
    def device_index(self) -> int:
        """Index of the Device this was created from."""
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
    def device_index(self) -> int:
        """Index of the Device this buffer was created from."""
        ...

    @property
    def element_layout(self) -> Layout:
        """Layout of a single element of this buffer: a scalar
        (``Device.create_buffer(elements, Type, ...)``), an array of a
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
    def cast(self, scalar: Type) -> "Buffer":
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

    def __dlpack__(self, stream: None = None) -> object:
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

    def torch(self) -> object:
        """Returns this buffer as a ``torch.Tensor``, via ``torch.from_dlpack``.
        Lazily imports ``torch`` (not a hard dependency); raises
        ``ImportError`` if it isn't installed.
        """
        ...

    def numpy(self) -> object:
        """Returns this buffer as a ``numpy.ndarray``, via
        ``numpy.from_dlpack``. Lazily imports ``numpy`` (not a hard
        dependency); raises ``ImportError`` if it isn't installed.
        """
        ...

    def jax(self) -> object:
        """Returns this buffer as a ``jax.Array``, via
        ``jax.dlpack.from_dlpack``. Lazily imports ``jax`` (not a hard
        dependency); raises ``ImportError`` if it isn't installed.
        """
        ...

    def cupy(self) -> object:
        """Returns this buffer as a ``cupy.ndarray``, via
        ``cupy.from_dlpack``. Lazily imports ``cupy`` (not a hard
        dependency); raises ``ImportError`` if it isn't installed.
        """
        ...

    def tensorflow(self) -> object:
        """Returns this buffer as a ``tf.Tensor``, via
        ``tf.experimental.dlpack.from_dlpack``. Lazily imports
        ``tensorflow`` (not a hard dependency); raises ``ImportError`` if
        it isn't installed.
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

    @overload
    def read(self, field: LayoutField) -> object: ...

    @overload
    def read(self, field: str) -> object:
        """Same as the ``LayoutField`` overload, but ``field`` is a dotted
        name path (e.g. ``"lights.0.P"``, field names and/or integer
        array/matrix-column indices) resolved against :attr:`element_layout`.
        A single plain name (no dot, not an integer) is exactly equivalent
        to ``read(element_layout.field(name))``; a deeper path is read via
        a byte-level ``cast``/``slice`` instead (lazily imports ``torch``),
        returning a ``torch.Tensor`` rather than a ``vk.math3d`` object.
        """
        ...

    def read(self, field) -> object:
        """Reads ``field`` directly via ``memcpy``, bypassing DLPack --
        much cheaper than :meth:`field` for small (e.g. scalar) fields.

        Only valid when :attr:`element_layout`'s ``kind`` is
        :attr:`TypeKind.STRUCT`, on a host-visible buffer
        (``MemoryLocation.HOST``) holding exactly one instance of
        ``field``'s root layout (i.e. :attr:`size` equals
        ``field.root.aligned_size``).

        :param field: A :class:`LayoutField` (from
            ``element_layout.field(name)``), or a dotted name path string
            (e.g. ``"lights.0.P"``, field names and/or integer
            array/matrix-column indices) navigated against
            :attr:`element_layout` -- a single plain name is equivalent to
            the :class:`LayoutField` form, a deeper path is instead
            resolved via a byte-level ``cast``/``slice`` (lazily imports
            ``torch``) and returns a ``torch.Tensor``. Must describe a
            scalar, vector, matrix, or (possibly nested) array of any of
            those -- not a struct, since a struct isn't a single numeric
            type.
        :return: For the :class:`LayoutField`/plain-name forms: a plain
            Python number (``int``/``float``/``bool``) for a scalar field;
            the matching ``vk.math3d`` vec*/mat* object for a vector or
            matrix field (e.g. a FLOAT32 vec3 field -> :class:`vk.math3d.vec3`,
            an INT32 vec2 field -> :class:`vk.math3d.ivec2`; a matrix field
            is always float, e.g. :class:`vk.math3d.mat4`); a
            tightly-packed buffer object (``bytes``) for an array field,
            or for a vector/matrix field with no ``vk.math3d`` equivalent.
            For a deeper dotted path: a ``torch.Tensor`` instead.
        :raises RuntimeError: If :attr:`element_layout` isn't a struct,
            this buffer isn't host-visible, doesn't hold exactly one
            instance of ``field``'s root layout, or ``field`` describes a
            struct.
        """
        ...

    @overload
    def write(self, field: LayoutField, value: object) -> None: ...

    @overload
    def write(self, field: str, value: object) -> None: ...

    @overload
    def write(self, **fields: object) -> None: ...

    def write(self, field=None, value: object = None, **fields: object) -> None:
        """Writes one or more fields of this struct-layout buffer directly
        via ``memcpy``, bypassing DLPack -- much cheaper than going through
        :meth:`field` for small (e.g. scalar) fields.

        Three forms:

        - ``write(field, value)``: ``field`` a :class:`LayoutField` (from
          ``element_layout.field(name)``) -- must describe a scalar,
          vector, matrix, or (possibly nested) array of any of those, not
          a struct.
        - ``write(path, value)``: ``path`` a dotted name string (e.g.
          ``"lights.0.P"``, field names and/or integer array/matrix-column
          indices), navigated against :attr:`element_layout`. A single
          plain name (no dot, not an integer) is exactly equivalent to the
          :class:`LayoutField` form; a deeper path is instead resolved via
          a byte-level ``cast``/``slice`` (lazily imports ``torch``).
        - ``write(name=value, ...)``: one or more top-level fields by
          name, each written independently (equivalent to calling
          ``write(name, value)`` once per keyword).

        In every form, if ``value`` is a ``dict``, the target field is
        assumed to be a struct and each ``{key: value}`` pair is written
        recursively to ``path.key``. If ``value`` is a ``list``/``tuple``
        and the target field is an array, each element is written to the
        matching numeric index (``path.0``, ``path.1``, ...); for a
        vector/matrix field (not an array), a plain (possibly nested, for
        a matrix) list/tuple of numbers is instead written element-by-element
        as that field's own components, converting to its scalar type.
        Otherwise (a vector, matrix or array field, given a value that
        isn't a dict/list/tuple matching the above), ``value`` may be a
        ``vk.math3d`` vec*/mat* object, a Python object supporting the
        buffer protocol (e.g. a C-contiguous numpy array), or a
        DLPack-compatible object (e.g. a CPU torch tensor) -- read
        directly via ``memcpy``, so a CUDA tensor is rejected (there's no
        valid CPU pointer to copy from).

        :raises TypeError: If both a positional ``field``/``path`` and
            ``name=value`` keywords are given.
        :raises RuntimeError: If :attr:`element_layout` isn't a struct,
            this buffer isn't host-visible, doesn't hold exactly one
            instance of the target field's root layout, ``value``'s size
            doesn't match the field's, or (for a DLPack source) it isn't
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
    def scalar_type(self) -> Type:
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

    @property
    def device_index(self) -> int:
        """Index of the Device this tensor was created from."""
        ...

    def torch(self) -> object:
        """Returns this tensor as a ``torch.Tensor``, via ``torch.from_dlpack``.
        Lazily imports ``torch`` (not a hard dependency); raises
        ``ImportError`` if it isn't installed.
        """
        ...

    def numpy(self) -> object:
        """Returns this tensor as a ``numpy.ndarray``, via
        ``numpy.from_dlpack``. Lazily imports ``numpy`` (not a hard
        dependency); raises ``ImportError`` if it isn't installed.
        """
        ...

    def jax(self) -> object:
        """Returns this tensor as a ``jax.Array``, via
        ``jax.dlpack.from_dlpack``. Lazily imports ``jax`` (not a hard
        dependency); raises ``ImportError`` if it isn't installed.
        """
        ...

    def cupy(self) -> object:
        """Returns this tensor as a ``cupy.ndarray``, via
        ``cupy.from_dlpack``. Lazily imports ``cupy`` (not a hard
        dependency); raises ``ImportError`` if it isn't installed.
        """
        ...

    def tensorflow(self) -> object:
        """Returns this tensor as a ``tf.Tensor``, via
        ``tf.experimental.dlpack.from_dlpack``. Lazily imports
        ``tensorflow`` (not a hard dependency); raises ``ImportError`` if
        it isn't installed.
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
    ``scalar_type``.

    If the wrapped object is directly accessible from both the CPU and
    GPU sides (a :class:`Buffer`, a :class:`Tensor`, or memory that
    already belongs to this device), ``device_ptr`` aliases it directly:
    there is only one copy of the data, so every dirty/update method
    below is a no-op.

    Otherwise a temporary buffer stands in for the GPU side, and this
    object tracks which side -- the wrapped Python object ("cpu") or the
    temporary buffer ("gpu") -- was modified more recently via a pair of
    version counters. Call :meth:`make_cpu_dirty`/:meth:`make_gpu_dirty`
    after modifying one side, then :meth:`update_cpu`/:meth:`update_gpu`
    to lazily copy the other side's changes across -- a copy only
    actually happens when the destination is stale.
    """

    def make_cpu_dirty(self) -> None:
        """Marks the CPU side (the wrapped Python object) as holding the
        freshest data, so the next :meth:`update_gpu` call will copy it
        across. No-op for a direct mapping.
        """
        ...

    def make_gpu_dirty(self) -> None:
        """Marks the GPU side (this wrap's own ``device_ptr``, e.g.
        after a shader wrote through it) as holding the freshest data,
        so the next :meth:`update_cpu` call will copy it back. No-op
        for a direct mapping.
        """
        ...

    def update_cpu(self) -> None:
        """Copies GPU -> CPU only if :meth:`make_gpu_dirty` was called
        more recently than the last :meth:`update_cpu`/creation. A
        no-op otherwise, and always a no-op for a direct mapping.
        """
        ...

    def update_gpu(self) -> None:
        """Copies CPU -> GPU only if :meth:`make_cpu_dirty` was called
        more recently than the last :meth:`update_gpu`/creation. A
        no-op otherwise, and always a no-op for a direct mapping.
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
    def scalar_type(self) -> Type:
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

    @property
    def device_index(self) -> int:
        """Index of the Device this image was created from."""
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


class Sampler:
    """A texture sampler (filtering + per-axis wrap modes), obtained from
    :meth:`Device.create_sampler`.

    Used together with an :class:`Image` via :meth:`DescriptorSet.bind`,
    either as a standalone :attr:`DescriptorType.SAMPLER` binding (paired
    with a separate :attr:`DescriptorType.SAMPLED_IMAGE` binding), or
    combined with an image in one
    :attr:`DescriptorType.COMBINED_IMAGE_SAMPLER` binding.
    """

    @property
    def device_index(self) -> int:
        """Index of the Device this sampler was created from."""
        ...


class Mesh:
    """A single interleaved vertex buffer plus index buffer, as produced
    by :meth:`Device.load_scene`.
    """

    @property
    def vertices(self) -> Buffer:
        """Interleaved vertex data, laid out per :attr:`attributes`."""
        ...

    @property
    def indices(self) -> Buffer:
        """Triangle indices (``uint32``) into :attr:`vertices`."""
        ...

    @property
    def attributes(self) -> list[VertexAttribute]:
        """Order of attributes interleaved in each :attr:`vertices` entry."""
        ...


MaterialValue = float | list[float] | str
"""A single material property's value: a scalar, an RGB triplet (as a
3-element list), or a string (e.g. a texture map path)."""


class Material:
    """A dynamic, open-ended set of named properties (e.g. ``"diffuse"``
    -> an RGB triplet, ``"diffuse_map"`` -> a texture path) -- not a fixed
    schema, so any loader can store whatever it finds.
    """

    def __init__(self) -> None: ...

    def set(self, name: str, value: MaterialValue) -> None:
        """Sets (or overwrites) a named property."""
        ...

    def get(self, name: str) -> MaterialValue | None:
        """Returns a named property's value, or ``None`` if unset."""
        ...

    @property
    def properties(self) -> dict[str, MaterialValue]:
        """All properties currently set on this material."""
        ...


class SceneNode:
    """A single named mesh instance within a :class:`Scene`, obtained
    from :meth:`Device.load_scene`.
    """

    @property
    def name(self) -> str: ...

    @property
    def material(self) -> Material | None:
        """``None`` if this node has no associated material."""
        ...

    @property
    def mesh(self) -> Mesh: ...

    @property
    def transform(self) -> list[list[float]] | None:
        """A 3x4 row-major affine transform, or ``None`` when the source
        format has no per-node transform concept (e.g. OBJ).
        """
        ...


class Scene:
    """A flat list of :class:`SceneNode`, obtained from
    :meth:`Device.load_scene`.
    """

    @property
    def nodes(self) -> list[SceneNode]: ...


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

    def set_cull_mode(self, mode: CullMode) -> None:
        """Sets the dynamic cull mode used by subsequent draw calls.

        Must be called while the command buffer is still recording.
        Graphics pipelines only.

        :raises RuntimeError: If the device doesn't support Vulkan 1.3
            (extended dynamic state).
        """
        ...

    def set_front_face(self, front_face: FrontFace) -> None:
        """Sets the dynamic front-face winding order used by subsequent
        draw calls.

        Must be called while the command buffer is still recording.
        Graphics pipelines only.

        :raises RuntimeError: If the device doesn't support Vulkan 1.3
            (extended dynamic state).
        """
        ...

    def set_depth_test(
        self, enable: bool, write_enable: bool = True, compare_op: CompareOp = CompareOp.LESS
    ) -> None:
        """Sets dynamic depth testing state used by subsequent draw calls.

        :param enable: Whether the depth test itself is performed.
        :param write_enable: Whether a fragment that passes the depth
            test writes its depth into the depth buffer.
        :param compare_op: How a fragment's depth compares against the
            depth buffer to pass the test.
        :raises RuntimeError: If the device doesn't support Vulkan 1.3
            (extended dynamic state), if no render pass is active (call
            :meth:`set_framebuffer` first), or if ``enable`` is true but
            the active framebuffer has no depth attachment (see
            :meth:`Pipeline.attach_depth`).
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
        :attr:`Type.UINT16` or :attr:`Type.UINT32`. Must be
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

    def build_ads(self, ads: AccelerationStructure, declaration: ADSDeclaration) -> None:
        """Records a build of `ads` (previously sized/allocated by
        :meth:`Device.create_ads`) from the geometry described by
        `declaration`, which must be the same kind
        (:class:`ADSTriangles`/:class:`ADSAABB` vs :class:`ADSInstances`)
        it was created with.

        Allocates a temporary scratch buffer for the build, kept alive
        (like every buffer/acceleration structure referenced here) for as
        long as this command buffer is, including while submitted and
        pending on the GPU. Requires a COMPUTE-capable engine.
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

    @property
    def device_index(self) -> int:
        """Index of the Device this command buffer was created from.
        Raises ``RuntimeError`` if already released.
        """
        ...

    @property
    def engine_type(self) -> "EngineType":
        """Capability of the engine this command buffer was created from.
        Raises ``RuntimeError`` if already released.
        """
        ...

    @property
    def engine_index(self) -> int:
        """Index of the engine this command buffer was created from.
        Raises ``RuntimeError`` if already released.
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

    @property
    def device_index(self) -> int:
        """Index of the Device this engine was created from."""
        ...

    @property
    def engine_type(self) -> "EngineType":
        """Capability this engine was created with."""
        ...

    @property
    def engine_index(self) -> int:
        """Index selecting among multiple queues of the same capability."""
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

    @property
    def device_index(self) -> int:
        """Index of the Device this framebuffer was created from."""
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

    @property
    def device_index(self) -> int:
        """Index of the Device this window was created from."""
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

    @overload
    def bind(self, layout_id: LayoutHandle, resource: Buffer) -> None:
        """Writes `resource` into the binding identified by ``layout_id``.

        :param layout_id: Handle returned by the matching :meth:`Pipeline.layout` call.
        :param resource: A :class:`Buffer`, for a binding declared with
            :attr:`DescriptorType.STORAGE_BUFFER` or
            :attr:`DescriptorType.UNIFORM_BUFFER`.
        """
        ...

    @overload
    def bind(self, layout_id: LayoutHandle, resource: "Image") -> None:
        """Writes `resource` into the binding identified by ``layout_id``.

        :param layout_id: Handle returned by the matching :meth:`Pipeline.layout` call.
        :param resource: An :class:`Image`, for a binding declared with
            :attr:`DescriptorType.STORAGE_IMAGE` (read/write, no sampler)
            or :attr:`DescriptorType.SAMPLED_IMAGE` (read-only, sampled in
            the shader via a separately-bound :class:`Sampler` at another
            binding -- see the ``(image, sampler)`` overload for a single
            combined binding instead).
        """
        ...

    @overload
    def bind(self, layout_id: LayoutHandle, resource: "Sampler") -> None:
        """Writes `resource` into the binding identified by ``layout_id``.

        :param layout_id: Handle returned by the matching :meth:`Pipeline.layout` call.
        :param resource: A :class:`Sampler`, for a binding declared with
            :attr:`DescriptorType.SAMPLER` (paired with a separate
            :attr:`DescriptorType.SAMPLED_IMAGE` binding, e.g. GLSL
            ``texture2D tex; sampler s;``).
        """
        ...

    @overload
    def bind(self, layout_id: LayoutHandle, image: "Image", sampler: "Sampler") -> None:
        """Writes `image`+`sampler` together into the binding identified by
        ``layout_id``.

        :param layout_id: Handle returned by the matching :meth:`Pipeline.layout` call.
        :param image: Image to sample.
        :param sampler: Sampler describing how to sample it. The binding's
            declared type must be :attr:`DescriptorType.COMBINED_IMAGE_SAMPLER`
            (GLSL ``sampler2D``).
        """
        ...

    @overload
    def bind(self, **bindings: object) -> None:
        """Binds one or more resources by the ``name`` given to the
        matching :meth:`Pipeline.layout` call(s) instead of its
        ``LayoutHandle`` -- e.g. ``ds.bind(img=render_target, ubo=ubo)``.
        A ``(image, sampler)`` tuple value binds a combined image/sampler.
        Only works for a descriptor set whose originating
        :class:`Pipeline` used ``name=...`` at :meth:`Pipeline.layout`.

        :raises KeyError: If a keyword doesn't match any named binding.
        """
        ...

    @property
    def device_index(self) -> int:
        """Index of the Device this descriptor set was created from."""
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
        :param description: Kind of resource this binding expects. May
            instead be passed as a single ``name=description`` keyword
            (e.g. ``pipeline.layout(0, 0, img=DescriptorType.STORAGE_IMAGE)``),
            remembering ``name`` so the matching :class:`DescriptorSet`
            can later be bound via :meth:`DescriptorSet.bind` ``(name=resource)``
            instead of the returned handle.
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
        :param format: Pixel format of this attachment. May instead be
            passed as a single ``name=format`` keyword (e.g.
            ``pipeline.attach(0, color=Format.RGBA32_Float)``), remembering
            ``name`` so :meth:`create_framebuffer` can later be passed
            ``name=image`` instead of the returned handle.
        :return: An opaque handle used to bind a render target image via
            :meth:`create_framebuffer`.
        """
        ...

    def attach_depth(self, format: Format) -> None:
        """Declares this pipeline's depth (or depth/stencil) attachment,
        enabling :meth:`CommandBuffer.set_depth_test` and depth-tested
        rendering. At most one per pipeline.

        Graphics pipelines only. Must be called before :meth:`close`.

        :param format: One of the ``Format.Depth*`` values.
        :raises RuntimeError: If ``format`` isn't a depth format, a depth
            attachment was already declared, or the device doesn't
            support Vulkan 1.3 (extended dynamic state).
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

    def create_framebuffer(
        self, attachments: "list[tuple[AttachHandle, Image]] | None" = None, depth_image: "Image | None" = None,
        **named_attachments: "Image"
    ) -> Framebuffer:
        """Creates a framebuffer compatible with this pipeline's render pass.

        Pipeline must be closed first. Graphics pipelines only.

        :param attachments: One ``(slot, image)`` pair per handle returned
            by :meth:`attach`, each image matching that attachment's
            declared format, and all sharing the same dimensions. Omit and
            use ``name=image`` keywords instead (one per :meth:`attach`
            call that used ``name=...``), e.g.
            ``pipeline.create_framebuffer(color=render_target)``.
        :param depth_image: Required (and matching :meth:`attach_depth`'s
            format/dimensions) if and only if :meth:`attach_depth` was
            called on this pipeline.
        :return: A new :class:`Framebuffer`.
        """
        ...

    def create_descriptor_set(self, set: int = 0) -> DescriptorSet:
        """Allocates a new descriptor set matching the layout declared for
        ``set`` via :meth:`layout`.

        Pipeline must be closed first.

        :param set: Descriptor set index, as passed to :meth:`layout`.
        :return: A new :class:`DescriptorSet`, whose :meth:`DescriptorSet.bind`
            accepts this pipeline's ``name=...`` bindings, if any.
        """
        ...

    @property
    def is_closed(self) -> bool:
        """Whether :meth:`close` has already been called."""
        ...

    @property
    def device_index(self) -> int:
        """Index of the Device this pipeline was created from."""
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

    @property
    def index(self) -> int:
        """Index of the physical Vulkan device this was created from
        (the same value passed to :meth:`create_device`/:func:`create_device`).
        """
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

    def create_tensor(self, shape: list[int], scalar_type: Type, location: MemoryLocation) -> Tensor:
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
    def create_buffer(self, elements: int, scalar_type: Type, location: MemoryLocation) -> Buffer:
        """Creates a buffer to store an array of elements of a given scalar
        type. Its :attr:`Buffer.element_layout` is that scalar type.

        A raw byte buffer is just this with ``scalar_type =
        Type.UINT8``: its :attr:`Buffer.element_layout` dlpack-exports
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

    def create_depth_buffer_image(
        self,
        width: int,
        height: int,
        format: Format = Format.Depth32_Float,
        location: MemoryLocation = MemoryLocation.DEVICE,
    ) -> Image:
        """Creates a 2D depth (or depth/stencil) attachment image, for use
        with :meth:`Pipeline.attach_depth`/:meth:`Pipeline.create_framebuffer`
        and :meth:`CommandBuffer.set_depth_test`.

        Unlike :meth:`create_image`, its usage is depth/stencil-attachment
        + sampled (color-attachment/storage usage is invalid for a depth
        format). Like every other image, it's transitioned once to
        ``VK_IMAGE_LAYOUT_GENERAL`` before this call returns.

        :param width: Width in texels.
        :param height: Height in texels.
        :param format: One of the ``Format.Depth*`` values.
        :param location: Memory location for the image's backing store.
        :return: A new :class:`Image`.
        :raises RuntimeError: If ``format`` isn't a depth format.
        """
        ...

    def create_sampler(
        self,
        mag_filter: Filter = Filter.LINEAR,
        min_filter: Filter = Filter.LINEAR,
        mipmap_mode: MipmapMode = MipmapMode.LINEAR,
        wrap_u: WrapMode = WrapMode.REPEAT,
        wrap_v: WrapMode = WrapMode.REPEAT,
        wrap_w: WrapMode = WrapMode.REPEAT,
    ) -> Sampler:
        """Creates a texture sampler, for use with an :class:`Image` via
        :meth:`DescriptorSet.bind`.

        :param mag_filter: Filtering used when magnifying (texture smaller
            than the sampled area).
        :param min_filter: Filtering used when minifying (texture larger
            than the sampled area).
        :param mipmap_mode: Filtering used between mip levels. There is no
            LOD bias/clamp control -- every mip level an image has is
            reachable.
        :param wrap_u: Wrap mode for the U (X) texture coordinate axis.
        :param wrap_v: Wrap mode for the V (Y) texture coordinate axis.
        :param wrap_w: Wrap mode for the W (Z) texture coordinate axis
            (3D images only).
        :return: A new :class:`Sampler`.
        """
        ...

    def create_ads(self, declaration: ADSDeclaration) -> AccelerationStructure:
        """Creates (sizes and allocates, but does not build -- see
        :meth:`CommandBuffer.build_ads`) an acceleration structure: a
        bottom-level one (BLAS) for an :class:`ADSTriangles`/
        :class:`ADSAABB` declaration, or a top-level one (TLAS) for an
        :class:`ADSInstances` declaration.

        :param declaration: Geometry description used to size the
            acceleration structure.
        :return: A new :class:`AccelerationStructure`.
        :raises RuntimeError: If ``VK_KHR_acceleration_structure`` isn't
            supported/enabled on this device.
        """
        ...

    def create_window(
        self,
        width: int,
        height: int,
        title: str,
        format: Format,
        frames_on_the_fly: int = 3,
        vsync: bool = True,
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
        :param vsync: ``True`` (default) picks the always-supported,
            vsync'd present mode (``VK_PRESENT_MODE_FIFO_KHR``). ``False``
            tries to disable vsync: ``VK_PRESENT_MODE_MAILBOX_KHR``
            (uncapped, triple-buffered, doesn't tear) if supported,
            otherwise ``VK_PRESENT_MODE_IMMEDIATE_KHR`` (uncapped, can
            tear), otherwise silently falls back to the vsync'd default
            if this surface supports neither.
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

    @overload
    def create_staging(self, buffer: Buffer, location: MemoryLocation = MemoryLocation.HOST) -> Buffer:
        """Creates a plain, byte-addressable :class:`Buffer` sized to
        match ``buffer``'s own ``size``, for staged CPU/GPU data movement.

        Device-local memory generally can't be mapped or read directly
        by the CPU. The correct Vulkan pattern is to allocate a staging
        buffer with this method and record a
        :meth:`CommandBuffer.transfer` between it and ``buffer``, rather
        than attempting to access device memory directly.

        :param buffer: Buffer whose size the staging buffer should match.
        :param location: Memory location for the staging buffer, HOST
            by default.
        :return: A new :class:`Buffer` of the same size as ``buffer``,
            with ``element_layout`` a plain ``Type.UINT8`` array
            (i.e. untyped raw bytes -- ``buffer``'s own element type is
            not preserved).
        """
        ...

    @overload
    def create_staging(self, image: "Image", location: MemoryLocation = MemoryLocation.HOST) -> Buffer:
        """Creates a plain, byte-addressable :class:`Buffer` sized to
        match ``image``'s own backing store, for staged CPU/GPU transfer
        of its texel data (e.g. via :meth:`CommandBuffer.transfer` after
        first blitting/copying into a same-size linear buffer -- see
        ``image``'s own docs for the exact layout assumptions).

        :param image: Image whose backing store size the staging buffer
            should match.
        :param location: Memory location for the staging buffer, HOST
            by default.
        :return: A new, untyped (``Type.UINT8`` element_layout)
            :class:`Buffer`.
        """
        ...

    def wrap(
        self,
        obj: "Buffer | Tensor | object",
        location: MemoryLocation = MemoryLocation.DEVICE,
    ) -> WrappedMemory:
        """Wraps an external object as a :class:`WrappedMemory` exposing a
        Vulkan buffer device address (``device_ptr``, usable from a shader),
        without necessarily copying its data.

        ``obj`` can be:

        - A :class:`Buffer` or :class:`Tensor`: its own device address is
          used directly, no copy, ever.
        - A DLPack-compatible object (e.g. a torch tensor): if it's
          already contiguous memory that belongs to this device's own
          memory managers (e.g. it came from ``torch.from_dlpack()`` on
          one of our own buffers), its corresponding device address is
          reused directly, no copy. Otherwise a new buffer is allocated
          at ``location``, and data is lazily copied in/out of it on
          demand -- see :meth:`WrappedMemory.update_cpu`/
          :meth:`WrappedMemory.update_gpu` -- via the CUDA interop
          plugin for CUDA-resident data, or a strided host memcpy
          otherwise.
        - A Python buffer-protocol object (e.g. a C-contiguous numpy
          array): always copied into a new buffer at ``location`` on
          demand, since a foreign host allocation has no Vulkan device
          address of its own.

        :param obj: The object to wrap.
        :param location: Where to allocate a copy, if one is needed.
            Ignored when the object can be used directly.
        :return: A new :class:`WrappedMemory`.
        :raises RuntimeError: If ``obj`` is none of the above, or a
            copy is required from/to CUDA-resident data but no interop
            library is available.
        """
        ...

    def load_scene(
        self,
        filename: str,
        resolution_mode: VertexResolutionMode = VertexResolutionMode.BY_ALL_ATTRIBUTES,
    ) -> Scene:
        """Loads a scene file into device-resident :class:`Mesh`
        buffers, dispatching on ``filename``'s extension. Only ``.obj``
        (via tinyobjloader) is currently supported.

        A group/shape referencing more than one material produces one
        :class:`SceneNode` per material used within it (sharing the
        group's name), rather than one node with a single, arbitrarily
        chosen material.

        :param filename: Path to the scene file.
        :param resolution_mode: How vertices are welded; see
            :class:`VertexResolutionMode`.
        :return: A new :class:`Scene`.
        :raises RuntimeError: If the extension is unsupported or the
            file fails to load.
        """
        ...


def device_infos() -> list[dict]:
    """Enumerates every Vulkan-visible physical device without creating a
    logical :class:`Device` for any of them: a throwaway Vulkan instance
    is created, queried, and destroyed within this call alone.

    Each entry is a dict with keys ``"index"`` (the value to pass to
    :meth:`Device.create_device`/:func:`create_device` for that GPU),
    ``"name"``, ``"vendor"`` (a friendly name, e.g. ``"NVIDIA"``, or a
    ``"0x...."`` hex string for an unrecognized vendor), ``"vendor_id"``,
    ``"device_id"``, ``"device_type"`` (one of ``"discrete_gpu"``/
    ``"integrated_gpu"``/``"virtual_gpu"``/``"cpu"``/``"other"``),
    ``"api_version"`` (e.g. ``"1.3.278"``), ``"driver_version"`` (raw
    ``int``, vendor-specific encoding), and ``"vram_bytes"`` (total
    device-local memory heap size, in bytes).

    :return: One dict per Vulkan-visible physical device, in the same
        order/indexing as :meth:`Device.create_device` expects.
    """
    ...
