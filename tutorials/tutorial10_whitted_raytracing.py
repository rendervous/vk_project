# %%
try:  # install vulky in colab -- it installs any missing Vulkan driver itself
    import google.colab
    import subprocess
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 10 - Whitted Raytracing
# The classic Whitted ray tracer traces recursively: a hit doesn't just
# shade and stop (tutorial 09) -- it can also spawn *further* rays (here,
# reflections) whose contribution gets folded back into the original
# pixel. This tutorial's closest-hit shader carries a payload of
# `{depth, importance, radiance}` across those recursive calls: `depth`
# caps how many bounces are allowed, `importance` is how much the next
# bounce's radiance actually contributes (attenuated by each surface's
# reflectivity), and `radiance` is the running total ultimately written
# to the image.
#
# Like the original notebook this tutorial is based on, the scene now
# mixes three *different* meshes from `vk.download_examples_data()`
# (`bunny.obj`, `dragon.obj`, `plate.obj` -- a ground plane) instead of
# three copies of one. This project has no `GL_EXT_buffer_reference`/
# bindless-descriptor-array support, so each mesh gets its own BLAS *and*
# its own dedicated hit group (with its own closest-hit shader, reading
# that mesh's own vertex/index bindings) rather than a shared shader
# indexing a per-instance geometry pointer.
# %%

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import vk

WIDTH, HEIGHT = 512, 512

_ATTRIBUTE_FIELDS = {
    vk.VertexAttribute.POSITION: ("position", vk.Type.VEC3),
    vk.VertexAttribute.NORMAL: ("normal", vk.Type.VEC3),
    vk.VertexAttribute.TEXCOORD: ("texcoord", vk.Type.VEC2),
}

# %% [markdown]
# ## Loading three different meshes
# `bunny.obj`/`plate.obj` have texcoords, `dragon.obj` doesn't --
# `vertex_layout` is computed per mesh from its own `attributes`, so each
# one's BLAS geometry (and later, its closest-hit shader's `Vertex`
# struct) matches its real per-vertex byte stride.
# %%

data_dir = vk.download_examples_data()

MESH_NAMES = ["bunny", "dragon", "plate"]
meshes = []
vertex_layouts = []
blas_vertices_list = []
blas_list = []
for name in MESH_NAMES:
    mesh_scene = vk.load_scene(os.path.join(data_dir, f"{name}.obj"))
    mesh = mesh_scene.nodes[0].mesh
    vertex_layout = vk.compute_layout([_ATTRIBUTE_FIELDS[a] for a in mesh.attributes], vk.LayoutRule.Scalar)
    blas_vertices = mesh.vertices.cast(vertex_layout)
    blas = vk.ads(vk.ads_triangles(blas_vertices, mesh.indices))
    meshes.append(mesh)
    vertex_layouts.append(vertex_layout)
    blas_vertices_list.append(blas_vertices)
    blas_list.append(blas)

# %% [markdown]
# ## One instance per mesh
# `bunny`/`dragon` sit on top of `plate` (a large flat quad, used as the
# ground): dragon's own local Y range straddles 0, so it's shifted up to
# rest on the floor the same way bunny already does. Each instance's
# `instance_shader_binding_table_record_offset_and_flags` packs `mesh_index
# * 2` (see the pipeline section below for why 2).
# %%

NUMBER_OF_INSTANCES = len(MESH_NAMES)  # bunny, dragon, plate
INSTANCE_OFFSETS = [vk.vec3(-1.3, 0.0, 0.0), vk.vec3(1.3, 0.705, 0.0), vk.vec3(0.0, 0.0, 0.0)]


def instance_rows(m):
    """A mat4's top 3 rows, as plain lists -- the row-major 3x4 shape
    layout_instance()'s "transform" field expects."""
    return [[m[c, r] for c in range(4)] for r in range(3)]


instances = vk.buffer(NUMBER_OF_INSTANCES, vk.layout_instance(), vk.MemoryLocation.HOST)
for i in range(NUMBER_OF_INSTANCES):
    world = vk.math3d.trs(INSTANCE_OFFSETS[i], vk.math3d.mat3(), vk.vec3(1.0, 1.0, 1.0))
    instances.element(i).write(
        transform=instance_rows(world),
        instance_custom_index_and_mask=(0xFF << 24) | i,
        instance_shader_binding_table_record_offset_and_flags=i * 2,
        acceleration_structure_reference=blas_list[i].device_address,
    )
tlas = vk.ads(vk.ads_instances(instances))

with vk.compute() as cmd:
    for blas, blas_vertices, mesh in zip(blas_list, blas_vertices_list, meshes):
        cmd.build_ads(blas, vk.ads_triangles(blas_vertices, mesh.indices))
    cmd.build_ads(tlas, vk.ads_instances(instances))

# %% [markdown]
# ## Floor texture
# `plate.obj`'s texcoords range roughly -100..100 (a 200x200 tile repeat
# across its 2000x2000-unit extent) -- `vk.sampler`'s default
# `WrapMode.REPEAT` handles that automatically. A mip chain (as in
# tutorial 07) keeps such an extreme tiling ratio from aliasing into
# moire noise at a distance.
# %%

texture_data = plt.imread(os.path.join(data_dir, "whitted_texture.png")).astype(np.float32)
if texture_data.shape[-1] == 3:
    texture_data = np.concatenate([texture_data, np.ones((*texture_data.shape[:2], 1), dtype=np.float32)], axis=-1)
tex_h, tex_w = texture_data.shape[:2]
tex_mip_levels = int(math.log2(max(tex_w, tex_h))) + 1
floor_texture = vk.image(tex_w, tex_h, 1, tex_mip_levels, 1, vk.Format.RGBA32_Float, vk.MemoryLocation.DEVICE)
floor_mip0 = floor_texture.slice(0, 1, 0, 1)
floor_staging = vk.staging(floor_mip0)
floor_staging.torch()[:] = torch.from_numpy(texture_data.reshape(-1, 4))
with vk.transfer() as cmd:
    cmd.transfer(floor_staging, floor_mip0)
with vk.graphics() as cmd:  # blit_image needs a graphics-capable engine
    for i in range(1, floor_texture.mip_count):
        cmd.blit_image(floor_texture.slice(i - 1, 1, 0, 1), floor_texture.slice(i, 1, 0, 1), filter=vk.Filter.LINEAR)
floor_sampler = vk.sampler(mipmap_mode=vk.MipmapMode.LINEAR)

# %% [markdown]
# ## Per-instance materials and camera
# One material per mesh: glass bunny (refractive, via `Kt`/`ior`),
# reflective dragon, mostly-matte floor.
# %%

material_layout = vk.compute_layout(
    dict(albedo=vk.Type.VEC3, Ks=vk.Type.FLOAT32, Kt=vk.Type.FLOAT32, ior=vk.Type.FLOAT32), vk.LayoutRule.Scalar
)
materials = vk.buffer(NUMBER_OF_INSTANCES, material_layout, vk.MemoryLocation.HOST)
albedos = [vk.vec3(1.0, 1.0, 1.0), vk.vec3(0.9, 0.9, 0.9), vk.vec3(1.0, 1.0, 1.0)]
reflectivities = [0.05, 0.35, 0.05]
transmissivities = [0.9, 0.0, 0.0]
iors = [1.5, 1.0, 1.0]
for i in range(NUMBER_OF_INSTANCES):
    materials.element(i).write(albedo=albedos[i], Ks=reflectivities[i], Kt=transmissivities[i], ior=iors[i])

global_transform = vk.buffer(dict(proj=vk.Type.MAT4, view=vk.Type.MAT4, light_pos=vk.Type.VEC3), vk.MemoryLocation.HOST)
global_transform.write(
    proj=vk.math3d.perspective_rh(math.radians(60.0), WIDTH / HEIGHT, 0.1, 100.0),
    view=vk.math3d.look_at_rh(vk.vec3(0.0, 2.0, 5.0), vk.vec3(0.0, 0.7, 0.0), vk.vec3(0.0, 1.0, 0.0)),
    light_pos=vk.vec3(2.0, 5.0, 2.0),
)

# %% [markdown]
# ## Shaders
# `MAX_DEPTH` caps the reflection recursion; `pipeline.stack_size` below
# accounts for shadow rays fired from the deepest bounce too. A miss now
# *adds* a background contribution rather than overwriting the radiance
# outright, since a reflection ray's miss should only contribute *its*
# attenuated share, on top of whatever the original hit already accumulated.
#
# `make_closest_hit_shader` builds one closest-hit shader per mesh: the
# shading logic (materials, shadow ray, reflection) is identical, only the
# `Vertex` struct (with or without a `C` texcoord field, matching that
# mesh's own attributes), the vertex/index buffer bindings, and (`plate`
# only) the `floor_texture` sample differ. Reflection/shadow ray origins
# are offset along `N` by a larger epsilon (`0.01`/`0.001`) than tutorial
# 09's -- too small an offset on a dense mesh like the dragon lets a
# reflected/shadow ray immediately re-hit the same surface it left (a
# floating-point self-intersection), showing up as salt-and-pepper noise.
# %%

raygen_shader = """
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 1) uniform Globals { mat4 proj; mat4 view; vec3 light_pos; } globals;
layout(set = 0, binding = 2, rgba32f) uniform image2D result_image;

struct RayPayload { int depth; vec3 importance; vec3 radiance; };
layout(location = 0) rayPayloadEXT RayPayload payload;

void main() {
    vec2 uv = (gl_LaunchIDEXT.xy + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy);
    vec4 near_point = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    vec4 far_point = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    mat4 inv_view_proj = inverse(globals.proj * globals.view);
    near_point = inv_view_proj * near_point;
    far_point = inv_view_proj * far_point;
    vec3 origin = near_point.xyz / near_point.w;
    vec3 direction = normalize(far_point.xyz / far_point.w - origin);
    payload.depth = 0;
    payload.importance = vec3(1.0);
    payload.radiance = vec3(0.0);
    traceRayEXT(scene, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, origin, 0.001, direction, 1000.0, 0);
    imageStore(result_image, ivec2(gl_LaunchIDEXT.xy), vec4(payload.radiance, 1.0));
}
"""
miss_shader = """
#version 460
#extension GL_EXT_ray_tracing : require
struct RayPayload { int depth; vec3 importance; vec3 radiance; };
layout(location = 0) rayPayloadInEXT RayPayload payload;
void main() { payload.radiance += payload.importance * vec3(0.1, 0.1, 0.3); }
"""
shadow_miss_shader = """
#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 1) rayPayloadInEXT float shadow;
void main() { shadow = 1.0; }
"""
shadow_any_hit_shader = """
#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 1) rayPayloadInEXT float shadow;
void main() { shadow = 0.0; }
"""


FLOOR_TEXTURE_BINDING = 4 + 2 * len(MESH_NAMES)


def make_closest_hit_shader(mesh_index, has_texcoord, has_texture):
    vertices_binding = 4 + 2 * mesh_index
    indices_binding = 5 + 2 * mesh_index
    vertex_fields = "vec3 P; vec3 N;" + (" vec2 C;" if has_texcoord else "")
    texture_uniform = f"layout(set = 0, binding = {FLOOR_TEXTURE_BINDING}) uniform sampler2D floor_texture;" if has_texture else ""
    texcoord_interp = "vec2 C = v0.C * coord.x + v1.C * coord.y + v2.C * coord.z;" if has_texture else ""
    surface_albedo = "material.albedo * texture(floor_texture, C).rgb" if has_texture else "material.albedo"
    return f"""
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

#define MAX_DEPTH 6

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 1) uniform Globals {{ mat4 proj; mat4 view; vec3 light_pos; }} globals;
{texture_uniform}

struct RayPayload {{ int depth; vec3 importance; vec3 radiance; }};
layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT float shadow;
hitAttributeEXT vec2 hit_attribs;

struct Vertex {{ {vertex_fields} }};
layout(scalar, set = 0, binding = {vertices_binding}) readonly buffer Vertices {{ Vertex data[]; }} vertices;
layout(scalar, set = 0, binding = {indices_binding}) readonly buffer Indices {{ uint data[]; }} indices;

struct Material {{ vec3 albedo; float Ks; float Kt; float ior; }};
layout(scalar, set = 0, binding = 3) readonly buffer Materials {{ Material data[]; }} materials;

void main() {{
    Material material = materials.data[gl_InstanceCustomIndexEXT];

    uint triangle_index = gl_PrimitiveID;
    Vertex v0 = vertices.data[indices.data[triangle_index * 3 + 0]];
    Vertex v1 = vertices.data[indices.data[triangle_index * 3 + 1]];
    Vertex v2 = vertices.data[indices.data[triangle_index * 3 + 2]];
    vec3 coord = vec3(1.0 - hit_attribs.x - hit_attribs.y, hit_attribs.x, hit_attribs.y);
    vec3 P = v0.P * coord.x + v1.P * coord.y + v2.P * coord.z;
    P = gl_ObjectToWorldEXT * vec4(P, 1.0);
    vec3 N = v0.N * coord.x + v1.N * coord.y + v2.N * coord.z;
    N = normalize(gl_ObjectToWorldEXT * vec4(N, 0.0));
    {texcoord_interp}

    vec3 I = normalize(gl_WorldRayDirectionEXT);
    bool entering = dot(I, N) < 0.0;
    vec3 Nf = entering ? N : -N;

    vec3 to_light = globals.light_pos - P;
    float d = length(to_light);
    vec3 L = to_light / d;

    shadow = 0.0;
    if (dot(N, L) > 0.0) {{
        traceRayEXT(scene, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, 1, 0, 1, P + N * 0.001, 0.001, L, d, 1);
    }}

    vec3 surface_albedo = {surface_albedo};
    float Kd = max(0.0, 1.0 - material.Ks - material.Kt);
    vec3 direct = Kd * surface_albedo / 3.14159 * shadow * 100.0 * max(0.0, dot(N, L)) / (0.5 + d * d);
    payload.radiance += payload.importance * direct;

    int base_depth = payload.depth;
    vec3 base_importance = payload.importance;

    if (base_depth < MAX_DEPTH && material.Ks > 0.0) {{
        payload.depth = base_depth + 1;
        payload.importance = base_importance * material.Ks;
        traceRayEXT(scene, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, P + Nf * 0.01, 0.001, reflect(I, Nf), 10000.0, 0);
    }}

    if (base_depth < MAX_DEPTH && material.Kt > 0.0) {{
        float eta = entering ? (1.0 / material.ior) : material.ior;
        vec3 T = refract(I, Nf, eta);
        payload.depth = base_depth + 1;
        payload.importance = base_importance * material.Kt;
        if (dot(T, T) < 0.0001) {{
            // total internal reflection
            traceRayEXT(scene, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, P + Nf * 0.01, 0.001, reflect(I, Nf), 10000.0, 0);
        }} else {{
            traceRayEXT(scene, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, P - Nf * 0.01, 0.001, normalize(T), 10000.0, 0);
        }}
    }}
}}
"""


# %% [markdown]
# ## Pipeline: one hit group pair per mesh
# Vulkan resolves a hit's shader group as `instance's own
# instanceShaderBindingTableRecordOffset + traceRayEXT's own sbtRecordOffset
# argument` (there's exactly one geometry per BLAS here, so the "geometry
# index" term drops out). Groups are appended interleaved as
# `[bunny_hit, bunny_shadow, dragon_hit, dragon_shadow, plate_hit,
# plate_shadow]`; with each instance's own offset set to `mesh_index * 2`
# (above) and every primary/reflection `traceRayEXT` call using
# `sbtRecordOffset = 0` and every shadow call using `1`, a ray always
# resolves to *whichever mesh it actually hit*'s own pair -- regardless of
# which instance originally cast it.
# %%

pipeline = vk.pipeline(vk.PipelineType.RAYTRACING)
# Per-mesh vertex/index bindings land at consecutive indices 4, 5, 6, 7, ...
# (2 per mesh, in MESH_NAMES order) right after the 4 fixed bindings below,
# and FLOOR_TEXTURE_BINDING (== 4 + 2 * len(MESH_NAMES)) is the very next
# index after the last mesh's -- so the whole set-0 layout is one
# contiguous run and can be declared in a single .layout() call.
mesh_layout_bindings = {}
for i, name in enumerate(MESH_NAMES):
    mesh_layout_bindings[f"mesh{i}_vertices"] = vk.DescriptorType.STORAGE_BUFFER
    mesh_layout_bindings[f"mesh{i}_indices"] = vk.DescriptorType.STORAGE_BUFFER
pipeline.layout(
    0, 0,
    scene=vk.DescriptorType.ACCELERATION_STRUCTURE,
    global_transform=vk.DescriptorType.UNIFORM_BUFFER,
    image=vk.DescriptorType.STORAGE_IMAGE,
    materials=vk.DescriptorType.STORAGE_BUFFER,
    **mesh_layout_bindings,
    floor_texture=vk.DescriptorType.COMBINED_IMAGE_SAMPLER,
)
pipeline.stack_size(8)

raygen = pipeline.stage(vk.ShaderStageType.RAYGEN, vk.shader_from_glsl(raygen_shader, vk.ShaderStageType.RAYGEN))
miss = pipeline.stage(vk.ShaderStageType.MISS, vk.shader_from_glsl(miss_shader, vk.ShaderStageType.MISS))
shadow_miss = pipeline.stage(vk.ShaderStageType.MISS, vk.shader_from_glsl(shadow_miss_shader, vk.ShaderStageType.MISS))
shadow_any_hit = pipeline.stage(vk.ShaderStageType.ANY_HIT, vk.shader_from_glsl(shadow_any_hit_shader, vk.ShaderStageType.ANY_HIT))

pipeline.append_raygen_group(raygen)
pipeline.append_miss_group(miss)
pipeline.append_miss_group(shadow_miss)
for mesh_index, (name, mesh) in enumerate(zip(MESH_NAMES, meshes)):
    has_texcoord = vk.VertexAttribute.TEXCOORD in mesh.attributes
    chit_source = make_closest_hit_shader(mesh_index, has_texcoord, has_texture=(name == "plate"))
    chit = pipeline.stage(vk.ShaderStageType.CLOSEST_HIT, vk.shader_from_glsl(chit_source, vk.ShaderStageType.CLOSEST_HIT))
    pipeline.append_hit_group(closest_hit=chit)
    pipeline.append_hit_group(any_hit=shadow_any_hit)
pipeline.close()

render_target = vk.image(WIDTH, HEIGHT, 1, 1, 1, vk.Format.RGBA32_Float, vk.MemoryLocation.DEVICE)
bindings = pipeline.descriptor_set(set=0)
mesh_resource_bindings = {}
for i, name in enumerate(MESH_NAMES):
    mesh_resource_bindings[f"mesh{i}_vertices"] = blas_vertices_list[i]
    mesh_resource_bindings[f"mesh{i}_indices"] = meshes[i].indices
bindings.bind(
    scene=tlas,
    global_transform=global_transform,
    image=render_target,
    materials=materials,
    **mesh_resource_bindings,
    floor_texture=(floor_texture, floor_sampler),
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

plt.imshow((staging.numpy().reshape(HEIGHT, WIDTH, -1) ** (1.0 / 2.2)).clip(0.0, 1.0))
plt.gca().axis("off")
plt.show()
