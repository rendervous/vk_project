"""Vulkan bindings with a pythonic, implicit-context surface.

Only free functions and enums are exported here -- no classes. A device
and an engine are activated implicitly (lazily defaulted, or explicitly via
:func:`device`/:func:`engine`); every other function (:func:`buffer`,
:func:`image`, :func:`command_buffer`, :func:`submit`, ...) operates on
whichever is currently active. A structured type description (see
:func:`compute_layout`) is plain data -- a bare :class:`Type` value, a
``[count, element]`` list for an array, or a ``{name: field, ...}`` dict
for a struct -- rather than a class built up call by call. Other
"declaration" value types that would otherwise require importing a class
(a precomputed ray-tracing layout, an acceleration structure geometry
declaration, a shader source, ...) are likewise obtained by calling a
function (:func:`layout_instance`, :func:`ads_triangles`,
:func:`shader_from_glsl`, ...) instead of a class constructor. The objects
these functions return (buffers, images, pipelines, ...) are still
full-featured instances -- their classes are just not part of this
package's public namespace, since typical code never needs to name them
directly.
"""

from . import math3d
from . import _interop  # noqa: F401  (side effect only: adds Buffer/Tensor.torch()/.numpy()/.jax()/...)
from . import _named_access  # noqa: F401  (side effect only: name=... bindings, and Buffer.write/read("a.b.0") paths)

from .math3d import (
    mat4,
    mat3,
    mat2,
    mat4x3,
    mat3x4,
    vec2,
    vec3,
    vec4,
    bvec2,
    bvec3,
    bvec4,
    ivec2,
    ivec3,
    ivec4,
    uvec2,
    uvec3,
    uvec4
)

from .vk import (
    CompareOp,
    CullMode,
    DescriptorType,
    EngineType,
    Filter,
    Format,
    FrontFace,
    LayoutRule,
    MemoryLocation,
    MipmapMode,
    PipelineType,
    ShaderStageType,
    Type,
    TypeKind,
    VertexAttribute,
    VertexResolutionMode,
    WrapMode
)
from ._context import (
    ads,
    buffer,
    command_buffer,
    depth_buffer_image,
    device,
    device_index,
    device_infos,
    dispose,
    engine,
    transfer,
    compute,
    graphics,
    image,
    load_scene,
    pipeline,
    relax,
    sampler,
    staging,
    submit,
    tensor,
    wait,
    window,
    wrap,
)
from ._declarations import (
    ads_aabb,
    ads_instances,
    ads_triangles,
    compute_layout,
    layout_aabb,
    layout_index16,
    layout_index32,
    layout_instance,
    layout_position,
    material,
    shader_from_file,
    shader_from_glsl,
    shader_from_spirv,
)

__version__ = "0.0.1"
__all__ = [
    # Enums
    "CompareOp",
    "CullMode",
    "DescriptorType",
    "EngineType",
    "Filter",
    "Format",
    "FrontFace",
    "LayoutRule",
    "MemoryLocation",
    "MipmapMode",
    "PipelineType",
    "ShaderStageType",
    "Type",
    "TypeKind",
    "VertexAttribute",
    "VertexResolutionMode",
    "WrapMode",
    # Device/engine context (vk._context)
    "ads",
    "buffer",
    "command_buffer",
    "depth_buffer_image",
    "device",
    "device_index",
    "device_infos",
    "dispose",
    "engine",
    "image",
    "load_scene",
    "pipeline",
    "relax",
    "sampler",
    "staging",
    "submit",
    "tensor",
    "wait",
    "window",
    "wrap",
    # Declaration/value-type constructors (vk._declarations)
    "ads_aabb",
    "ads_instances",
    "ads_triangles",
    "compute_layout",
    "layout_aabb",
    "layout_index16",
    "layout_index32",
    "layout_instance",
    "layout_position",
    "material",
    "shader_from_file",
    "shader_from_glsl",
    "shader_from_spirv",
    "transfer",
    "compute",
    "graphics",
    # Misc
    "math3d",
    "mat4",
    "mat3",
    "mat2",
    "mat4x3",
    "mat3x4",
    "vec2",
    "vec3",
    "vec4",
    "bvec2",
    "bvec3",
    "bvec4",
    "ivec2",
    "ivec3",
    "ivec4",
    "uvec2",
    "uvec3",
    "uvec4",
    "__version__",
]
