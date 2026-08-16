# %%
try:  # install vulky in colab -- it installs any missing Vulkan driver itself
    import google.colab
    import subprocess
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 08 - Basic Raytracing
# Every previous tutorial rasterized. This one traces rays instead: a
# **ray tracing pipeline** (raygen/miss/closest-hit shaders instead of
# vertex/fragment) dispatched over a **top-level acceleration structure**
# (TLAS) of instances, each referencing a **bottom-level acceleration
# structure** (BLAS) built from the same triangle mesh as tutorial 06.
# %%

import math
import os

import matplotlib.pyplot as plt
import vk

WIDTH, HEIGHT = 512, 512

# %% [markdown]
# ## The same bunny mesh as tutorial 06
# %%

data_dir = vk.download_examples_data()
bunny_path = os.path.join(data_dir, "bunny.obj")
scene = vk.load_scene(bunny_path)
mesh = scene.nodes[0].mesh

# %% [markdown]
# ## From `Mesh` to BLAS geometry
# `Mesh.vertices` is a *flat* buffer of raw floats (position, then normal,
# then texcoord, back to back, with no per-vertex structure of its own) --
# `Buffer.cast` reinterprets those same bytes as instances of
# `vertex_layout` (built from `mesh.attributes`, as in tutorial 06, so its
# byte stride matches whatever attributes this file actually has; position
# first is what a BLAS triangle geometry actually reads: only the first 12
# bytes of each element, as a position), which is also what makes
# `ads_triangles`'s per-vertex byte stride come out correctly. Vertex and
# triangle counts default to `blas_vertices.count`/`mesh.indices.count // 3`
# -- no need to pass them explicitly.
# %%

_ATTRIBUTE_FIELDS = {
    vk.VertexAttribute.POSITION: ("position", vk.Type.VEC3),
    vk.VertexAttribute.NORMAL: ("normal", vk.Type.VEC3),
    vk.VertexAttribute.TEXCOORD: ("texcoord", vk.Type.VEC2),
}
vertex_layout = vk.compute_layout([_ATTRIBUTE_FIELDS[a] for a in mesh.attributes], vk.LayoutRule.Scalar)
blas_vertices = mesh.vertices.cast(vertex_layout)

blas = vk.ads(vk.ads_triangles(blas_vertices, mesh.indices))

# %% [markdown]
# ## Instancing the BLAS into a TLAS
# Three copies of the same cube, arranged in a circle -- the same
# transforms as tutorial 06, but as `layout_instance()` entries (a 3x4
# row-major transform + a BLAS device address) instead of per-draw
# uniform buffers.
# %%

NUMBER_OF_INSTANCES = 3


def instance_rows(m):
    """A mat4's top 3 rows, as plain lists -- the row-major 3x4 shape
    layout_instance()'s "transform" field expects."""
    return [[m[c, r] for c in range(4)] for r in range(3)]


instances = vk.buffer(NUMBER_OF_INSTANCES, vk.layout_instance(), vk.MemoryLocation.HOST)
for i in range(NUMBER_OF_INSTANCES):
    angle = i * 2.0 * math.pi / NUMBER_OF_INSTANCES
    rotation = vk.math3d.rotate_y(i * math.pi / NUMBER_OF_INSTANCES)
    translation = vk.vec3(math.cos(angle) * 2.5, 0.0, math.sin(angle) * 2.5)
    world = vk.math3d.trs(translation, rotation, vk.vec3(1.0, 1.0, 1.0))
    instances.element(i).write(
        transform=instance_rows(world),
        instance_custom_index_and_mask=0xFF000000,
        instance_shader_binding_table_record_offset_and_flags=0,
        acceleration_structure_reference=blas.device_address,
    )

tlas = vk.ads(vk.ads_instances(instances))

# %% [markdown]
# ## Building both acceleration structures
# A single `vk.compute()` block: the acceleration-structure build barrier
# added after every `build_ads` call makes it safe for the TLAS build (and
# later `trace_rays`) to read the BLAS it just referenced, even though
# nothing here explicitly waits between the two.
# %%

with vk.compute() as cmd:
    cmd.build_ads(blas, vk.ads_triangles(blas_vertices, mesh.indices))
    cmd.build_ads(tlas, vk.ads_instances(instances))

# %% [markdown]
# ## Camera
# %%

global_transform = vk.buffer(dict(proj=vk.Type.MAT4, view=vk.Type.MAT4), vk.MemoryLocation.HOST)
global_transform.write(
    proj=vk.math3d.perspective_rh(math.radians(60.0), WIDTH / HEIGHT, 0.1, 100.0),
    view=vk.math3d.look_at_rh(vk.vec3(0.0, 2.0, 6.0), vk.vec3(0.0, 0.8, 0.0), vk.vec3(0.0, 1.0, 0.0)),
)

# %% [markdown]
# ## Shaders
# `traceRayEXT` in the raygen shader casts one ray per pixel, computed by
# unprojecting the pixel through the inverse view-projection matrix. On a
# hit, the closest-hit shader looks up its triangle's 3 vertices directly
# (as plain storage buffers -- no vertex-fetch stage exists for ray
# tracing), interpolates the normal via the hit's barycentric coordinates
# (`HitAttribs`), and shades it with a fixed directional light. A miss
# just paints a background color.
# %%

raygen_shader = """
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 1) uniform Globals { mat4 proj; mat4 view; } globals;
layout(set = 0, binding = 2, rgba32f) uniform image2D result_image;

layout(location = 0) rayPayloadEXT vec4 payload;

void main() {
    vec2 uv = (gl_LaunchIDEXT.xy + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy);
    vec4 near_point = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    vec4 far_point = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    mat4 inv_view_proj = inverse(globals.proj * globals.view);
    near_point = inv_view_proj * near_point;
    far_point = inv_view_proj * far_point;
    vec3 origin = near_point.xyz / near_point.w;
    vec3 direction = normalize(far_point.xyz / far_point.w - origin);
    traceRayEXT(scene, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, origin, 0.001, direction, 1000.0, 0);
    imageStore(result_image, ivec2(gl_LaunchIDEXT.xy), payload);
}
"""
miss_shader = """
#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec4 payload;
void main() { payload = vec4(0.1, 0.1, 0.3, 1.0); }
"""
closest_hit_shader = """
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) rayPayloadInEXT vec4 payload;
hitAttributeEXT vec2 hit_attribs;

struct Vertex { vec3 P; vec3 N; vec2 C; };
layout(scalar, set = 0, binding = 3) readonly buffer Vertices { Vertex data[]; } vertices;
layout(scalar, set = 0, binding = 4) readonly buffer Indices { uint data[]; } indices;

void main() {
    uint triangle_index = gl_PrimitiveID;
    Vertex v0 = vertices.data[indices.data[triangle_index * 3 + 0]];
    Vertex v1 = vertices.data[indices.data[triangle_index * 3 + 1]];
    Vertex v2 = vertices.data[indices.data[triangle_index * 3 + 2]];
    vec3 coord = vec3(1.0 - hit_attribs.x - hit_attribs.y, hit_attribs.x, hit_attribs.y);
    vec3 N = v0.N * coord.x + v1.N * coord.y + v2.N * coord.z;
    N = normalize(gl_ObjectToWorldEXT * vec4(N, 0.0));
    vec3 light = vec3(3.0, 3.0, 0.3) / 3.14159 * max(0.1, dot(N, normalize(vec3(2.0, 1.0, 5.0))));
    payload = vec4(light, 1.0);
}
"""

# %% [markdown]
# ## Building the ray tracing pipeline
# `append_raygen_group`/`append_miss_group`/`append_hit_group` turn each
# shader into a shader-binding-table entry, in the order they're created;
# `close()` both finalizes the pipeline and builds that table.
# %%

pipeline = vk.pipeline(vk.PipelineType.RAYTRACING)
pipeline.layout(
    0, 0,
    scene=vk.DescriptorType.ACCELERATION_STRUCTURE,
    global_transform=vk.DescriptorType.UNIFORM_BUFFER,
    image=vk.DescriptorType.STORAGE_IMAGE,
    vertices=vk.DescriptorType.STORAGE_BUFFER,
    indices=vk.DescriptorType.STORAGE_BUFFER,
)

raygen = pipeline.stage(vk.ShaderStageType.RAYGEN, vk.shader_from_glsl(raygen_shader, vk.ShaderStageType.RAYGEN))
miss = pipeline.stage(vk.ShaderStageType.MISS, vk.shader_from_glsl(miss_shader, vk.ShaderStageType.MISS))
chit = pipeline.stage(vk.ShaderStageType.CLOSEST_HIT, vk.shader_from_glsl(closest_hit_shader, vk.ShaderStageType.CLOSEST_HIT))
pipeline.append_raygen_group(raygen)
pipeline.append_miss_group(miss)
pipeline.append_hit_group(closest_hit=chit)
pipeline.close()

render_target = vk.image(WIDTH, HEIGHT, 1, 1, 1, vk.Format.RGBA32_Float, vk.MemoryLocation.DEVICE)
bindings = pipeline.descriptor_set(set=0)
bindings.bind(
    scene=tlas,
    global_transform=global_transform,
    image=render_target,
    vertices=blas_vertices,
    indices=mesh.indices,
)

# %% [markdown]
# ## Dispatching rays and reading the result back
# %%

with vk.compute() as cmd:
    cmd.set_pipeline(pipeline)
    cmd.bind(0, [bindings])
    cmd.trace_rays(WIDTH, HEIGHT)

staging = vk.staging(render_target)
with vk.transfer() as cmd:
    cmd.transfer(render_target, staging)

plt.imshow(staging.numpy().reshape(HEIGHT, WIDTH, -1))
plt.gca().axis("off")
plt.show()
