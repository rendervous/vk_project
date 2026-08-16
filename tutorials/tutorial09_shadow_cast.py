# %%
try:  # install vulky in colab -- it installs any missing Vulkan driver itself
    import google.colab
    import subprocess
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 09 - Shadow Cast
# Tutorial 08's closest-hit shader shaded a point using only its normal and
# a fixed light direction -- it never checked whether something else
# blocks the light. This tutorial adds a *second* ray, traced from within
# the closest-hit shader itself, to answer exactly that: a shadow ray
# using its own miss/hit-group pair (SBT index 1), so a hit and a miss mean
# opposite things depending on which ray is asking.
# %%

import math
import os

import matplotlib.pyplot as plt
import vk

WIDTH, HEIGHT = 512, 512

# %% [markdown]
# ## Scene setup
# Identical to tutorial 08: the same bunny mesh, cast as BLAS geometry,
# instanced three times into a TLAS.
# %%

data_dir = vk.download_examples_data()
bunny_path = os.path.join(data_dir, "bunny.obj")
scene = vk.load_scene(bunny_path)
mesh = scene.nodes[0].mesh

_ATTRIBUTE_FIELDS = {
    vk.VertexAttribute.POSITION: ("position", vk.Type.VEC3),
    vk.VertexAttribute.NORMAL: ("normal", vk.Type.VEC3),
    vk.VertexAttribute.TEXCOORD: ("texcoord", vk.Type.VEC2),
}
vertex_layout = vk.compute_layout([_ATTRIBUTE_FIELDS[a] for a in mesh.attributes], vk.LayoutRule.Scalar)
blas_vertices = mesh.vertices.cast(vertex_layout)

blas = vk.ads(vk.ads_triangles(blas_vertices, mesh.indices))

NUMBER_OF_INSTANCES = 3


def instance_rows(m):
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

with vk.compute() as cmd:
    cmd.build_ads(blas, vk.ads_triangles(blas_vertices, mesh.indices))
    cmd.build_ads(tlas, vk.ads_instances(instances))

# %% [markdown]
# ## Camera and light
# `light_pos` joins `proj`/`view` in the same uniform buffer.
# %%

global_transform = vk.buffer(dict(proj=vk.Type.MAT4, view=vk.Type.MAT4, light_pos=vk.Type.VEC3), vk.MemoryLocation.HOST)
global_transform.write(
    proj=vk.math3d.perspective_rh(math.radians(60.0), WIDTH / HEIGHT, 0.1, 100.0),
    view=vk.math3d.look_at_rh(vk.vec3(0.0, 2.0, 6.0), vk.vec3(0.0, 0.8, 0.0), vk.vec3(0.0, 1.0, 0.0)),
    light_pos=vk.vec3(1.0, 5.0, 2.0),
)

# %% [markdown]
# ## Shaders
# The closest-hit shader traces a *second* ray towards the light, with
# `gl_RayFlagsTerminateOnFirstHitEXT` (stop at the very first intersection
# -- exactly right for a shadow test, which only cares "blocked or not",
# not *which* triangle blocks it) and its own miss/hit-group pair: the
# shadow miss shader means "nothing in the way" (`shadow = 1.0`), the
# shadow any-hit shader means "something is" (`shadow = 0.0`) -- opposite
# of what a miss/hit means for the primary ray.
# %%

raygen_shader = """
#version 460
#extension GL_EXT_ray_tracing : require
layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 1) uniform Globals { mat4 proj; mat4 view; vec3 light_pos; } globals;
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

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 1) uniform Globals { mat4 proj; mat4 view; vec3 light_pos; } globals;

layout(location = 0) rayPayloadInEXT vec4 payload;
layout(location = 1) rayPayloadEXT float shadow;
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
    vec3 P = v0.P * coord.x + v1.P * coord.y + v2.P * coord.z;
    P = gl_ObjectToWorldEXT * vec4(P, 1.0);
    vec3 N = v0.N * coord.x + v1.N * coord.y + v2.N * coord.z;
    N = normalize(gl_ObjectToWorldEXT * vec4(N, 0.0));

    vec3 to_light = globals.light_pos - P;
    float d = length(to_light);
    vec3 L = to_light / d;

    shadow = 0.0;
    if (dot(N, L) > 0.0) {
        traceRayEXT(scene, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, 1, 0, 1, P + N * 0.0001, 0.001, L, d, 1);
    }

    vec3 light_intensity = vec3(100.0);
    vec3 albedo = vec3(1.0, 1.0, 0.0);
    vec3 direct = shadow * light_intensity * albedo / 3.14159 * max(0.0, dot(N, L)) / (0.5 + d * d);
    payload = vec4(direct, 1.0);
}
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

# %% [markdown]
# ## Pipeline
# Two miss groups and two hit groups now -- `append_miss_group`/
# `append_hit_group` are called twice, and creation order *is* SBT order,
# so "regular" stays at index 0 and "shadow" lands at index 1, matching
# the `0`/`1` miss/hit-group indices used by the two `traceRayEXT` calls
# above. `stack_size(2)`: the closest-hit shader itself calls
# `traceRayEXT`, one level deeper than tutorial 08's default of 1.
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
pipeline.stack_size(2)

raygen = pipeline.stage(vk.ShaderStageType.RAYGEN, vk.shader_from_glsl(raygen_shader, vk.ShaderStageType.RAYGEN))
miss = pipeline.stage(vk.ShaderStageType.MISS, vk.shader_from_glsl(miss_shader, vk.ShaderStageType.MISS))
shadow_miss = pipeline.stage(vk.ShaderStageType.MISS, vk.shader_from_glsl(shadow_miss_shader, vk.ShaderStageType.MISS))
chit = pipeline.stage(vk.ShaderStageType.CLOSEST_HIT, vk.shader_from_glsl(closest_hit_shader, vk.ShaderStageType.CLOSEST_HIT))
shadow_any_hit = pipeline.stage(vk.ShaderStageType.ANY_HIT, vk.shader_from_glsl(shadow_any_hit_shader, vk.ShaderStageType.ANY_HIT))

pipeline.append_raygen_group(raygen)
pipeline.append_miss_group(miss)
pipeline.append_miss_group(shadow_miss)
pipeline.append_hit_group(closest_hit=chit)
pipeline.append_hit_group(any_hit=shadow_any_hit)
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

plt.imshow(staging.numpy().reshape(HEIGHT, WIDTH, -1) ** (1.0 / 2.2))
plt.gca().axis("off")
plt.show()
