# %%
try:  # install vulky in colab -- it installs any missing Vulkan driver itself
    import google.colab
    import subprocess
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 06 - OBJ Viewer
# Tutorial 05 built a vertex/index buffer by hand. `vk.load_scene` instead
# loads a full mesh (currently `.obj`, via tinyobjloader) straight into a
# device-resident, ready-to-bind vertex+index :class:`Mesh` -- and this
# tutorial draws three copies of it, each with its own transform, plus a
# real perspective camera (`vk.math3d`, introduced in tutorial 02).
# %%

import math
import os

import matplotlib.pyplot as plt
import vk

WIDTH, HEIGHT = 512, 512
render_target = vk.image(WIDTH, HEIGHT, 1, 1, 1, vk.Format.RGBA32_Float, vk.MemoryLocation.DEVICE)
depth_buffer = vk.depth_buffer_image(WIDTH, HEIGHT)

# %% [markdown]
# ## Loading the mesh
# `vk.download_examples_data()` fetches (once -- later calls skip it) the
# shared examples/tutorials data bundle and extracts it into `./data`;
# `bunny.obj` is the classic Stanford bunny. Swap `bunny_path` below for a
# path to your own `.obj` to view something else.
# %%

data_dir = vk.download_examples_data()
bunny_path = os.path.join(data_dir, "bunny.obj")
scene = vk.load_scene(bunny_path)

mesh = scene.nodes[0].mesh
print("attributes:", mesh.attributes)

# %% [markdown]
# ## Vertex layout from `Mesh.attributes`
# A `Mesh`'s vertex buffer is interleaved in the order `attributes` lists
# (always starting with `POSITION` when present); `compute_layout` mirrors
# that order so `pipeline.vertex_layout` matches the buffer's real byte
# layout exactly, whatever attributes this particular file happened to have.
# %%

_ATTRIBUTE_FIELDS = {
    vk.VertexAttribute.POSITION: ("position", vk.Type.VEC3),
    vk.VertexAttribute.NORMAL: ("normal", vk.Type.VEC3),
    vk.VertexAttribute.TEXCOORD: ("texcoord", vk.Type.VEC2),
    vk.VertexAttribute.TANGENT: ("tangent", vk.Type.VEC3),
    vk.VertexAttribute.BITANGENT: ("bitangent", vk.Type.VEC3),
}
vertex_layout = vk.compute_layout([_ATTRIBUTE_FIELDS[a] for a in mesh.attributes], vk.LayoutRule.Scalar)

# %% [markdown]
# ## Camera and per-instance transforms
# `global_transform` (projection + view) is shared by every draw;
# `local_transforms` holds one `World` matrix per instance -- three copies
# of the cube arranged in a circle, via `vk.math3d.trs`/`rotate_y`.
# %%

NUMBER_OF_INSTANCES = 3

global_transform = vk.buffer(dict(Proj=vk.Type.MAT4, View=vk.Type.MAT4), vk.MemoryLocation.HOST)
global_transform.write(
    Proj=vk.math3d.perspective_rh(math.radians(60.0), WIDTH / HEIGHT, 0.1, 100.0),
    View=vk.math3d.look_at_rh(vk.vec3(0.0, 2.0, 6.0), vk.vec3(0.0, 0.8, 0.0), vk.vec3(0.0, 1.0, 0.0)),
)

local_transforms = []
for i in range(NUMBER_OF_INSTANCES):
    angle = i * 2.0 * math.pi / NUMBER_OF_INSTANCES
    rotation = vk.math3d.rotate_y(i * math.pi / NUMBER_OF_INSTANCES)
    translation = vk.vec3(math.cos(angle) * 2.5, 0.0, math.sin(angle) * 2.5)
    local = vk.buffer(dict(World=vk.Type.MAT4), vk.MemoryLocation.HOST)
    local.write(World=vk.math3d.trs(translation, rotation, vk.vec3(1.0, 1.0, 1.0)))
    local_transforms.append(local)

# %% [markdown]
# ## Shaders
# `Globals` (set 0) is bound once; `Locals` (set 1) is rebound between
# draws -- one `World` matrix per instance -- so the same pipeline/vertex
# data draws each copy in its own place. Normal-as-color is the same debug
# visualization used informally in earlier examples.
# %%

vertex_shader = """
#version 450
#extension GL_EXT_scalar_block_layout: require

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 0) out vec3 out_normal;

layout(scalar, set = 0, binding = 0) uniform Globals { mat4 Proj; mat4 View; };
layout(scalar, set = 1, binding = 0) uniform Locals { mat4 World; };

void main() {
    vec4 P = World * vec4(in_position, 1.0);
    gl_Position = Proj * (View * P);
    out_normal = mat3(World) * in_normal;
}
"""
fragment_shader = """
#version 450
layout(location = 0) in vec3 in_normal;
layout(location = 0) out vec4 out_color;
void main() { out_color = vec4(normalize(in_normal) * 0.5 + 0.5, 1.0); }
"""

# %% [markdown]
# ## Pipeline and descriptor sets
# Two declared sets: `0` for globals (one shared `descriptor_set`) and `1`
# for locals -- `descriptor_set_collection(set=1, count=N)` allocates `N`
# independent descriptor sets against that same declared layout in one
# call, one per instance to draw.
# %%

pipeline = vk.pipeline(vk.PipelineType.RASTERIZATION)
pipeline.attach(0, color=vk.Format.RGBA32_Float)
pipeline.attach_depth(vk.Format.Depth32_Float)
pipeline.layout(0, 0, global_transform=vk.DescriptorType.UNIFORM_BUFFER)
pipeline.layout(1, 0, local_transform=vk.DescriptorType.UNIFORM_BUFFER)
pipeline.vertex_layout(0, vertex_layout)
pipeline.stage(vk.ShaderStageType.VERTEX, vk.shader_from_glsl(vertex_shader, vk.ShaderStageType.VERTEX))
pipeline.stage(vk.ShaderStageType.FRAGMENT, vk.shader_from_glsl(fragment_shader, vk.ShaderStageType.FRAGMENT))
pipeline.close()

framebuffer = pipeline.create_framebuffer(depth_image=depth_buffer, color=render_target)

global_bindings = pipeline.descriptor_set(set=0)
global_bindings.bind(global_transform=global_transform)
local_bindings = pipeline.descriptor_set_collection(set=1, count=NUMBER_OF_INSTANCES)
for ds, local in zip(local_bindings, local_transforms):
    ds.bind(local_transform=local)

# %% [markdown]
# ## Rendering
# One `bind_vertices`/`bind_indices` pair (set once), then one
# `bind`+`dispatch_indexed_primitives` pair per instance, rebinding only
# set 1 each time -- set 0 stays bound from before the loop.
# %%

index_count = mesh.indices.count

with vk.graphics() as cmd:
    cmd.set_framebuffer(framebuffer)
    cmd.set_pipeline(pipeline)
    cmd.set_viewport(0, HEIGHT, WIDTH, -HEIGHT)
    cmd.set_depth_test(True)
    cmd.bind(0, [global_bindings])
    cmd.bind_vertices(0, mesh.vertices.cast(vertex_layout))
    cmd.bind_indices(mesh.indices)
    for ds in local_bindings:
        cmd.bind(1, [ds])
        cmd.dispatch_indexed_primitives(index_count)

# %% [markdown]
# ## Reading the result back
# %%

staging = vk.staging(render_target)
with vk.transfer() as cmd:
    cmd.transfer(render_target, staging)

plt.imshow(staging.numpy().reshape(HEIGHT, WIDTH, -1))
plt.gca().axis("off")
plt.show()
