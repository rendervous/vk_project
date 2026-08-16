# %%
try:  # install vulky in colab -- it installs any missing Vulkan driver itself
    import google.colab
    import subprocess
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 05 - Vertex Buffers
# Tutorial 04's vertex shader invented its own geometry from
# `gl_VertexIndex` alone. Real geometry instead comes from a **vertex
# buffer** -- one entry per vertex, read automatically by the fixed-function
# vertex-fetch stage before the vertex shader runs -- plus, optionally, an
# **index buffer** so vertices can be shared between triangles instead of
# duplicated. This tutorial draws a single textured-look quad this way.
# %%

import matplotlib.pyplot as plt
import torch
import vk

WIDTH, HEIGHT = 512, 512
render_target = vk.image(WIDTH, HEIGHT, 1, 1, 1, vk.Format.RGBA32_Float, vk.MemoryLocation.DEVICE)
depth_buffer = vk.depth_buffer_image(WIDTH, HEIGHT)

# %% [markdown]
# ## Vertex and index data
# `vertex_layout` describes one vertex (`P`: position, `C`: a 2D
# coordinate); `vk.buffer(4, vertex_layout, ...)` allocates 4 of them, and
# `Buffer.element(i).write(...)` fills each in -- exactly like tutorial 04's
# per-sprite struct buffer, just interpreted as *vertices* now instead of
# *instances*. The index buffer lists 6 indices (two triangles) into those
# 4 vertices, avoiding a duplicated 5th/6th vertex.
# %%

vertex_layout = vk.compute_layout(dict(P=vk.Type.VEC3, C=vk.Type.VEC2), vk.LayoutRule.Scalar)
vertices = vk.buffer(4, vertex_layout, vk.MemoryLocation.HOST)
vertices.element(0).write(P=vk.vec3(-0.9, -0.9, 0.5), C=vk.vec2(0.0, 0.0))
vertices.element(1).write(P=vk.vec3(0.9, -0.9, 0.5), C=vk.vec2(1.0, 0.0))
vertices.element(2).write(P=vk.vec3(0.9, 0.9, 0.5), C=vk.vec2(1.0, 1.0))
vertices.element(3).write(P=vk.vec3(-0.9, 0.9, 0.5), C=vk.vec2(0.0, 1.0))

indices = vk.buffer(6, vk.Type.UINT32, vk.MemoryLocation.HOST)
indices.torch()[:] = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.int32)

# %% [markdown]
# ## Shaders
# No storage buffer this time: `in_position`/`in_coordinates` are ordinary
# vertex-input attributes, fetched automatically per vertex.
# %%

vertex_shader = """
#version 450
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_coordinates;
layout(location = 0) out vec2 out_coordinates;
void main() {
    gl_Position = vec4(in_position, 1.0);
    out_coordinates = in_coordinates;
}
"""
fragment_shader = """
#version 450
layout(location = 0) in vec2 in_coordinates;
layout(location = 0) out vec4 out_color;
void main() { out_color = vec4(in_coordinates.x, in_coordinates.y, 1.0, 1.0); }
"""

# %% [markdown]
# ## Pipeline
# `vertex_layout(start_location, layout)` maps `vertex_layout`'s fields
# (`P`, `C`, in that order) to consecutive shader input locations starting
# at `start_location` -- location 0 for `in_position`, location 1 for
# `in_coordinates`, matching the shader above.
# %%

pipeline = vk.pipeline(vk.PipelineType.RASTERIZATION)
pipeline.attach(0, color=vk.Format.RGBA32_Float)
pipeline.attach_depth(vk.Format.Depth32_Float)
pipeline.vertex_layout(0, vertex_layout)
pipeline.stage(vk.ShaderStageType.VERTEX, vk.shader_from_glsl(vertex_shader, vk.ShaderStageType.VERTEX))
pipeline.stage(vk.ShaderStageType.FRAGMENT, vk.shader_from_glsl(fragment_shader, vk.ShaderStageType.FRAGMENT))
pipeline.close()

framebuffer = pipeline.create_framebuffer(depth_image=depth_buffer, color=render_target)

# %% [markdown]
# ## NDC vs. image Y orientation
# Vulkan's normalized device coordinates put `(-1, -1)` at the viewport's
# *first* corner and `(1, 1)` at its last -- by default that maps to
# `(0, 0)`/`(width, height)` in pixels, so Y increases downward (matching
# image conventions, but flipped compared to OpenGL). Giving `set_viewport`
# a negative height (and an origin at the top of that flipped range) undoes
# this, without touching the projection: `y = height`, `height = -height`.
# %%

with vk.graphics() as cmd:
    cmd.set_framebuffer(framebuffer)
    cmd.set_pipeline(pipeline)
    cmd.set_viewport(0, HEIGHT, WIDTH, -HEIGHT)
    cmd.set_depth_test(True)
    cmd.bind_vertices(0, vertices)
    cmd.bind_indices(indices)
    cmd.dispatch_indexed_primitives(6)

# %% [markdown]
# ## Reading the result back
# %%

staging = vk.staging(render_target)
with vk.transfer() as cmd:
    cmd.transfer(render_target, staging)

plt.imshow(staging.numpy().reshape(HEIGHT, WIDTH, -1))
plt.gca().axis("off")
plt.show()
