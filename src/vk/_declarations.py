"""Plain-function/plain-data constructors for otherwise-hidden
"declaration"/value types: a type description (see compute_layout() below),
the precomputed ray-tracing layouts (:class:`~vk.vk.Layouts`),
:class:`~vk.vk.ADSTriangles`/:class:`~vk.vk.ADSAABB`/:class:`~vk.vk.ADSInstances`,
:class:`~vk.vk.Material` and :class:`~vk.vk.ShaderSource`.

None of these need a Device, so none of the functions here touch the
current-device/current-engine context (see :mod:`vk._context`) -- they're
just friendlier names for what would otherwise be ``vk.vk.SomeClass(...)``/
``vk.vk.SomeClass.static_method(...)`` calls, keeping every such class out
of the top-level ``vk`` namespace (see ``vk/__init__.py``).
"""

from typing import Optional, Union

from .vk import (
    ADSAABB,
    ADSInstances,
    ADSTriangles,
    Buffer,
    Layout,
    LayoutRule,
    Layouts,
    Material,
    ShaderSource,
    ShaderStageType,
    Type,
    TypeDescriptor,
)
from .vk import compute_layout as _compute_layout

# ---- compute_layout: a type description is plain data, no class needed --

# A "type spec" is one of:
# - a Type enum value (BOOL, VEC3, MAT4, ...): a scalar/vector/matrix.
# - [count, element_spec]: an array of `count` elements of `element_spec`
#   (count == 0 denotes an unsized/runtime array, only meaningful as a
#   struct's last field).
# - {name: field_spec, ...}: a struct, fields in (dict, so insertion) order.
# - [(name, field_spec), ...]: a struct, fields in the given order (an
#   explicit alternative to a dict, e.g. when field order matters and
#   isn't already obvious from a dict literal).
TypeSpec = Union[Type, list, dict]


def _to_type_descriptor(spec: TypeSpec) -> TypeDescriptor:
    if isinstance(spec, dict):
        return TypeDescriptor.struct_of([(name, _to_type_descriptor(field_spec)) for name, field_spec in spec.items()])
    if isinstance(spec, list):
        if len(spec) == 2 and isinstance(spec[0], int):
            count, element_spec = spec
            return TypeDescriptor.array_of(_to_type_descriptor(element_spec), count)
        # A list of (name, field_spec) pairs: a struct, given order.
        return TypeDescriptor.struct_of([(name, _to_type_descriptor(field_spec)) for name, field_spec in spec])
    return TypeDescriptor.single(spec)


def compute_layout(type: TypeSpec, rule: LayoutRule) -> Layout:
    """Computes the byte layout (size, alignment, offsets/strides) of a
    type description under `rule`. Pure host-side arithmetic; does not
    require a device.

    `type` is plain data -- no class to build up front:

    - A bare ``Type`` value (e.g. ``Type.FLOAT32``, ``Type.VEC3``,
      ``Type.MAT4``) describes a scalar/vector/matrix. The resulting
      Layout's ``type`` is exactly this value; its ``kind`` is
      ``TypeKind.SINGLE``.
    - ``[count, element]`` describes an array of `count` elements of
      `element` (itself a type spec; `count` == 0 denotes an unsized/
      runtime array, only meaningful as a struct's last field). The
      resulting Layout's ``kind`` is ``TypeKind.ARRAY``, with
      ``element_layout`` computed from `element` and ``stride`` ==
      ``element_layout.aligned_size``.
    - ``{name: field, ...}`` (or, equivalently, ``[(name, field), ...]``
      when field order must be given explicitly) describes a struct,
      fields in the given order. The resulting Layout's ``kind`` is
      ``TypeKind.STRUCT``, with one ``LayoutField`` per entry (navigable
      by name via ``Layout.field(name)``).

    Nesting is arbitrary, e.g. ``{"bones": [3, Type.MAT4]}``.
    """
    return _compute_layout(_to_type_descriptor(type), rule)


# ---- Layouts: precomputed layouts for ray tracing acceleration structures ----

def layout_index16() -> Layout:
    """The Layout of a single UINT16 triangle-mesh index (2 bytes, tightly
    packed): use as the `layout` of a buffer passed as
    ads_triangles()'s `indices`.
    """
    return Layouts.index16()


def layout_index32() -> Layout:
    """The Layout of a single UINT32 triangle-mesh index (4 bytes, tightly
    packed): use as the `layout` of a buffer passed as
    ads_triangles()'s `indices`.
    """
    return Layouts.index32()


def layout_position() -> Layout:
    """The Layout of a single tightly-packed FLOAT32 vec3 vertex position
    (12 bytes, no padding): use as the `layout` of a buffer passed as
    ads_triangles()'s `vertices`.
    """
    return Layouts.position()


def layout_aabb() -> Layout:
    """The Layout of a single ``VkAabbPositionsKHR``-shaped procedural AABB
    entry (24 bytes: 6 tightly-packed FLOAT32 fields ``min_x``/``min_y``/
    ``min_z``/``max_x``/``max_y``/``max_z``, navigable via Layout.field()):
    use as the `layout` of a buffer passed as ads_aabb()'s `aabbs`.
    """
    return Layouts.aabb()


def layout_instance() -> Layout:
    """The Layout of a single ``VkAccelerationStructureInstanceKHR``-shaped
    TLAS instance entry (64 bytes, navigable via Layout.field()): fields
    ``transform`` (row-major 3x4 affine transform, as an array of 3 vec4
    rows), ``instance_custom_index_and_mask`` and
    ``instance_shader_binding_table_record_offset_and_flags`` (each
    packing two bitfields into one uint32 -- low 24 bits/high 8 bits
    respectively: pack via ``(mask << 24) | (custom_index & 0xFFFFFF)``),
    and ``acceleration_structure_reference`` (a referenced BLAS's device
    address, from an AccelerationStructure's own ``device_address``).
    Use as the `layout` of a buffer passed as ads_instances()'s
    `instances`.
    """
    return Layouts.instance()


# ---- Acceleration structure geometry declarations ------------------------

def ads_triangles(
    vertices: Buffer,
    indices: Optional[Buffer] = None,
    transform: Optional[Buffer] = None,
    opaque: bool = True,
) -> ADSTriangles:
    """Declares the geometry for a bottom-level acceleration structure
    (BLAS) built from a triangle mesh, for ads()/CommandBuffer.build_ads().

    `vertices` must be a DEVICE-resident buffer of tightly-packed FLOAT32
    vec3 positions (e.g. layout_position()); its vertex count is
    `vertices.count`. `indices`, if given, must be a DEVICE-resident buffer
    whose element_layout is a scalar UINT16 or UINT32 (e.g. layout_index16()/
    layout_index32()) -- non-indexed triangles are used otherwise; the
    triangle count is `(indices or vertices).count // 3` either way.
    `transform`, if given, is a static per-geometry 3x4 row-major transform
    (a VkTransformMatrixKHR-shaped buffer -- e.g. layout_instance()'s
    "transform" field layout) applied at build time; no transform
    (identity) if omitted.
    """
    return ADSTriangles(vertices, indices, transform, opaque)


def ads_aabb(aabbs: Buffer, opaque: bool = True) -> ADSAABB:
    """Declares the geometry for a bottom-level acceleration structure
    (BLAS) built from procedural AABBs, for ads()/CommandBuffer.build_ads().

    `aabbs` must be a DEVICE-resident buffer of layout_aabb()-shaped
    entries; its count is `aabbs.count`.
    """
    return ADSAABB(aabbs, opaque)


def ads_instances(instances: Buffer) -> ADSInstances:
    """Declares a top-level acceleration structure (TLAS) built from
    instances of other, already-built bottom-level acceleration
    structures, for ads()/CommandBuffer.build_ads().

    `instances` must be a DEVICE-resident buffer of layout_instance()-
    shaped entries; its instance count is `instances.count`.
    """
    return ADSInstances(instances)


# ---- Material: a dynamic, open-ended property bag ------------------------

def material() -> Material:
    """Creates an empty Material: a dynamic, open-ended set of named
    properties (e.g. "diffuse" -> an RGB triplet, "diffuse_map" -> a
    texture path), populated afterwards via Material.set(name, value)
    and read back via Material.get(name)/Material.properties.
    """
    return Material()


# ---- ShaderSource: compiled SPIR-V bytecode for one shader stage --------

def shader_from_spirv(code: list[int], entry_point: str = "main") -> ShaderSource:
    """Wraps already-compiled SPIR-V bytecode (`code`, 32-bit words) as a
    ShaderSource ready to attach to a Pipeline via Pipeline.stage(),
    recording `entry_point` as its entry function name.
    """
    return ShaderSource.from_spirv(code, entry_point)


def shader_from_file(
    path: str,
    stage: ShaderStageType,
    entry_point: str = "main",
    include_dirs: Optional[list[str]] = None,
) -> ShaderSource:
    """Loads a shader from `path` as a ShaderSource ready to attach to a
    Pipeline via Pipeline.stage(). If `path` ends in ".spv" it's read as
    raw SPIR-V (`entry_point` just recorded as-is, `include_dirs`
    ignored); otherwise it's read as GLSL source text and compiled the
    same way as shader_from_glsl(), using `stage` to pick the compiler
    profile.
    """
    return ShaderSource.from_file(path, stage, entry_point, include_dirs or [])


def shader_from_glsl(
    source: str,
    stage: ShaderStageType,
    entry_point: str = "main",
    include_dirs: Optional[list[str]] = None,
) -> ShaderSource:
    """Compiles `source` (GLSL text, for `stage`) to SPIR-V by shelling out
    to glslangValidator (must be on PATH, or found via $VULKAN_SDK/Bin),
    returning a ShaderSource ready to attach to a Pipeline via
    Pipeline.stage(). The compiled SPIR-V's own entry point is named
    `entry_point`, which must match the function name actually used in
    `source`.
    """
    return ShaderSource.from_glsl(source, stage, entry_point, include_dirs or [])
