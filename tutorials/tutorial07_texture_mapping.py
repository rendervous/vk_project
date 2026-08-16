# %%
try:  # install vulky in colab -- it installs any missing Vulkan driver itself
    import google.colab
    import subprocess
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 07 - Texture Mapping
# So far, every fragment shader has computed its own color. This tutorial
# samples one from an `Image` instead, through a `Sampler` -- and builds a
# mip chain for it, to compare how the two `MipmapMode`s blend between
# levels on a ground plane seen at a shallow angle (where minification is
# most visible).
# %%

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import vk

WIDTH, HEIGHT = 512, 512
render_target = vk.image(WIDTH, HEIGHT, 1, 1, 1, vk.Format.RGBA32_Float, vk.MemoryLocation.DEVICE)
depth_buffer = vk.depth_buffer_image(WIDTH, HEIGHT)

# %% [markdown]
# ## Building a checkerboard texture
# `Image` upload works the same way `Buffer` upload does for anything
# device-local: fill a host-visible staging buffer (`vk.staging`, sized/
# laid out to match the image), then `cmd.transfer(staging, image)` --
# tutorial 03's readback direction, reversed. `Format.RGBA32_Float` is used
# instead of a `_UNorm` format so the staging buffer's DLPack view is a
# plain `(texels, 4)` float array with no extra byte-packing to worry about.
# %%

TEX_SIZE = 64
checker = np.zeros((TEX_SIZE, TEX_SIZE, 4), dtype=np.float32)
checker[..., 3] = 1.0
cell = np.indices((TEX_SIZE, TEX_SIZE)).sum(axis=0) // 8 % 2
checker[cell == 0, :3] = 1.0

mip_levels = int(math.log2(TEX_SIZE)) + 1
texture = vk.image(TEX_SIZE, TEX_SIZE, 1, mip_levels, 1, vk.Format.RGBA32_Float, vk.MemoryLocation.DEVICE)
mip0 = texture.slice(0, 1, 0, 1)
tex_staging = vk.staging(mip0)
tex_staging.torch()[:] = torch.from_numpy(checker.reshape(-1, 4))
with vk.transfer() as cmd:
    cmd.transfer(tex_staging, mip0)

# %% [markdown]
# ## Building the mip chain
# `blit_image` resizes (and, if formats differ, converts) `src` into `dst`;
# blitting each mip level from the previous one -- via `Image.slice`, a
# view over a mip/array sub-range of the same underlying image -- builds
# the whole chain one level at a time.
# %%

with vk.graphics() as cmd:  # blit_image needs a graphics-capable engine
    for i in range(1, texture.mip_count):
        cmd.blit_image(texture.slice(i - 1, 1, 0, 1), texture.slice(i, 1, 0, 1), filter=vk.Filter.LINEAR)

# %% [markdown]
# ## Camera and shaders
# The vertex shader invents a large ground-plane quad from `gl_VertexIndex`
# alone (tutorial 04's trick again), scaled up and viewed from an angle so
# minification (and thus mip selection) is clearly visible.
# %%

transforms = vk.buffer(dict(projection=vk.Type.MAT4, view=vk.Type.MAT4), vk.MemoryLocation.HOST)
transforms.write(
    projection=vk.math3d.perspective_rh(math.radians(60.0), WIDTH / HEIGHT, 0.05, 200.0),
    view=vk.math3d.look_at_rh(vk.vec3(0.0, 1.5, 2.0), vk.vec3(0.0, 0.0, 8.0), vk.vec3(0.0, 1.0, 0.0)),
)

vertex_shader = """
#version 450
#extension GL_EXT_scalar_block_layout: require
layout(location = 0) out vec2 out_coordinates;
layout(scalar, set = 0, binding = 0) uniform Globals { mat4 projection; mat4 view; };
vec2 quad[6] = vec2[](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
);
void main() {
    vec2 q = quad[gl_VertexIndex] * 20.0;
    vec4 P = vec4(q.x, 0.0, q.y * 0.5 + 10.0, 1.0);
    gl_Position = projection * (view * P);
    out_coordinates = q;
}
"""
fragment_shader = """
#version 450
layout(location = 0) in vec2 in_coordinates;
layout(location = 0) out vec4 out_color;
layout(set = 0, binding = 1) uniform sampler2D image;
void main() { out_color = texture(image, in_coordinates); }
"""

# %% [markdown]
# ## Pipeline and descriptor set
# `image` is a `COMBINED_IMAGE_SAMPLER` binding: `bind` takes an
# `(image, sampler)` tuple as its value for it.
# %%

pipeline = vk.pipeline(vk.PipelineType.RASTERIZATION)
pipeline.attach(0, color=vk.Format.RGBA32_Float)
pipeline.attach_depth(vk.Format.Depth32_Float)
pipeline.layout(0, 0, transforms=vk.DescriptorType.UNIFORM_BUFFER, image=vk.DescriptorType.COMBINED_IMAGE_SAMPLER)
pipeline.stage(vk.ShaderStageType.VERTEX, vk.shader_from_glsl(vertex_shader, vk.ShaderStageType.VERTEX))
pipeline.stage(vk.ShaderStageType.FRAGMENT, vk.shader_from_glsl(fragment_shader, vk.ShaderStageType.FRAGMENT))
pipeline.close()

framebuffer = pipeline.create_framebuffer(depth_image=depth_buffer, color=render_target)
bindings = pipeline.descriptor_set(set=0)
bindings.bind(transforms=transforms)


def render_with(sampler):
    bindings.bind(image=(texture, sampler))
    with vk.graphics() as cmd:
        cmd.set_framebuffer(framebuffer)
        cmd.set_pipeline(pipeline)
        cmd.set_viewport(0, HEIGHT, WIDTH, -HEIGHT)
        cmd.set_depth_test(True)
        cmd.bind(0, [bindings])
        cmd.dispatch_primitives(6)
    staging = vk.staging(render_target)
    with vk.transfer() as cmd:
        cmd.transfer(render_target, staging)
    return staging.numpy().reshape(HEIGHT, WIDTH, -1).copy()


# %% [markdown]
# ## Comparing mipmap filtering
# Both samplers use the full mip chain (there's no per-sampler LOD clamp);
# the difference is how they blend *between* levels --
# `MipmapMode.NEAREST` snaps to one level per pixel (visible seams where
# the chosen level changes), `MipmapMode.LINEAR` blends smoothly.
# %%

nearest_sampler = vk.sampler(mag_filter=vk.Filter.NEAREST, min_filter=vk.Filter.NEAREST, mipmap_mode=vk.MipmapMode.NEAREST)
linear_sampler = vk.sampler(mag_filter=vk.Filter.LINEAR, min_filter=vk.Filter.LINEAR, mipmap_mode=vk.MipmapMode.LINEAR)

pixels_nearest = render_with(nearest_sampler)
pixels_linear = render_with(linear_sampler)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(pixels_nearest)
axes[0].set_title("MipmapMode.NEAREST")
axes[0].axis("off")
axes[1].imshow(pixels_linear)
axes[1].set_title("MipmapMode.LINEAR")
axes[1].axis("off")
plt.show()
