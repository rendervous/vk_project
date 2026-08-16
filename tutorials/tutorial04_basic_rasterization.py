# %%
try:  # install vulky in colab -- it installs any missing Vulkan driver itself
    import google.colab
    import subprocess
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 04 - Basic Rasterization
# Tutorial 03 rendered by dispatching a *compute* shader over an image,
# writing wherever it liked. A rasterization (graphics) pipeline works
# differently: it dispatches *primitives* (triangles), and an implicit
# stage between the vertex and fragment shader interpolates outputs and
# decides, pixel by pixel, which primitive is currently the closest one --
# consuming color/depth values and writing them into whichever images a
# **framebuffer** attaches, without the fragment shader ever choosing a
# pixel coordinate itself.
#
# This tutorial renders a batch of independently-colored quads ("sprites")
# from a single storage buffer, to show how a graphics pipeline's
# attachments, depth test and per-primitive data fit together.
# %%

import random

import matplotlib.pyplot as plt
import vk

# %% [markdown]
# ## Render targets
# A rasterization pipeline needs a color attachment (`render_target`) to
# write into and, if depth testing is wanted, a separate depth attachment
# (`depth_buffer`) -- Vulkan keeps these as distinct images, unlike a
# compute shader's single output image. `vk.depth_buffer_image` creates
# one already set up for `Pipeline.attach_depth`/`CommandBuffer.set_depth_test`.
# %%

WIDTH, HEIGHT = 512, 512
NUMBER_OF_SPRITES = 30

render_target = vk.image(WIDTH, HEIGHT, 1, 1, 1, vk.Format.RGBA32_Float, vk.MemoryLocation.DEVICE)
depth_buffer = vk.depth_buffer_image(WIDTH, HEIGHT)

# %% [markdown]
# ## Per-sprite data
# `vk.compute_layout` turns a struct spec into a `Layout` usable as a
# multi-element buffer's own element type -- here, `NUMBER_OF_SPRITES`
# instances of `{offset, size, color}`, one per sprite. `Buffer.element(i)`
# slices out a single instance, whose `write(name=value, ...)` (from
# tutorial 03) then fills in its fields.
# %%

sprite_layout = vk.compute_layout(
    dict(offset=vk.Type.VEC3, size=vk.Type.FLOAT32, color=vk.Type.VEC4),
    vk.LayoutRule.Scalar,
)
sprites = vk.buffer(NUMBER_OF_SPRITES, sprite_layout, vk.MemoryLocation.HOST)

random.seed(0)
for i in range(NUMBER_OF_SPRITES):
    sprites.element(i).write(
        offset=vk.vec3(random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8), random.uniform(0.1, 0.9)),
        size=random.uniform(0.03, 0.12),
        color=vk.vec4(random.uniform(0.4, 1.0), random.uniform(0.4, 1.0), random.uniform(0.4, 1.0), 1.0),
    )

# %% [markdown]
# ## Shaders
# There's no vertex buffer at all here: the vertex shader looks up its own
# sprite via `gl_VertexIndex / 6` (this pipeline dispatches 6 vertices per
# sprite -- 2 triangles making a quad) and picks a corner via
# `gl_VertexIndex % 6` from a small constant array, entirely on its own.
# This is the same trick real instancing would normally handle via
# `gl_InstanceIndex` and a per-instance vertex/storage buffer; without
# hardware instancing dispatched in one call, indexing through
# `gl_VertexIndex` gets the same result with one non-instanced draw.
# %%

vertex_shader = """
#version 450
#extension GL_EXT_scalar_block_layout: require

layout(location = 0) out vec4 out_color;

struct SpriteInfo { vec3 offset; float size; vec4 color; };
layout(scalar, set = 0, binding = 0) readonly buffer Sprites { SpriteInfo data[]; } sprites;

vec2 quad[6] = vec2[](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
);

void main() {
    int sprite_index = gl_VertexIndex / 6;
    int corner_index = gl_VertexIndex % 6;
    SpriteInfo info = sprites.data[sprite_index];
    vec2 q = quad[corner_index];
    gl_Position = vec4(vec3(q, 0.0) * info.size + info.offset, 1.0);
    out_color = info.color;
}
"""
fragment_shader = """
#version 450
layout(location = 0) in vec4 in_color;
layout(location = 0) out vec4 out_color;
void main() { out_color = in_color; }
"""

# %% [markdown]
# ## Pipeline, framebuffer, descriptor set
# `attach`/`layout` are name-based: each keyword argument's name becomes
# the name used later to fill it in. `attach(0, color=...)` declares an
# attachment named `color`, to be passed by that name into
# `create_framebuffer` later, and `layout(0, 0, sprites=...)` declares a
# binding named `sprites`, to be passed by that name into `ds.bind` the
# same way.
# %%

pipeline = vk.pipeline(vk.PipelineType.RASTERIZATION)
pipeline.attach(0, color=vk.Format.RGBA32_Float)
pipeline.attach_depth(vk.Format.Depth32_Float)
pipeline.layout(0, 0, sprites=vk.DescriptorType.STORAGE_BUFFER)
pipeline.stage(vk.ShaderStageType.VERTEX, vk.shader_from_glsl(vertex_shader, vk.ShaderStageType.VERTEX))
pipeline.stage(vk.ShaderStageType.FRAGMENT, vk.shader_from_glsl(fragment_shader, vk.ShaderStageType.FRAGMENT))
pipeline.close()

framebuffer = pipeline.create_framebuffer(depth_image=depth_buffer, color=render_target)

bindings = pipeline.descriptor_set(set=0)
bindings.bind(sprites=sprites)

# %% [markdown]
# ## Rendering
# `set_framebuffer` begins the render pass (and always clears every
# attachment -- color to opaque black, depth to 1.0 -- there's no
# per-attachment clear color yet). `set_depth_test` enables the depth
# comparison so nearer sprites (smaller `offset.z`) draw over farther ones,
# regardless of draw order. `dispatch_primitives(6 * N)` issues one
# non-indexed draw covering every sprite's 6 corner-vertices at once.
# %%

with vk.graphics() as cmd:
    cmd.set_framebuffer(framebuffer)
    cmd.set_pipeline(pipeline)
    cmd.set_viewport(0, 0, WIDTH, HEIGHT)
    cmd.set_depth_test(True)
    cmd.bind(0, [bindings])
    cmd.dispatch_primitives(6 * NUMBER_OF_SPRITES)

# %% [markdown]
# ## Reading the result back
# Same staging/transfer pattern as tutorial 03.
# %%

staging = vk.staging(render_target)
with vk.transfer() as cmd:
    cmd.transfer(render_target, staging)

plt.imshow(staging.numpy().reshape(HEIGHT, WIDTH, -1))
plt.gca().axis("off")
plt.show()
