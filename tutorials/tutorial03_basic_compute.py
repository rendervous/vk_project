# %%
try:  # install vulky in colab -- it installs any missing Vulkan driver itself
    import google.colab
    import subprocess
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 03 - Basic Compute
# In this example a compute is used to generate the image
# of the famous Mandelbrot set.
#
# This tutorial ties together everything from the previous two: an
# image and a uniform buffer (device resources), a GLSL compute shader
# operating on them, and a command buffer that dispatches the shader
# and then transfers the result back to the CPU for display.
# %%
from matplotlib import pyplot as plt
import numpy as np
import vk

# %% [markdown]
# ## The compute shader
# A GLSL compute shader is just a string, compiled at runtime by
# ``vk.shader_from_glsl``. This one reads a center point ``C`` and a
# zoom width ``W`` from a uniform buffer, maps each invocation's thread
# id to a point in the complex plane, iterates the Mandelbrot recurrence
# `Z = Z^2 + c`, and stores a color derived from the escape magnitude
# into the output image. ``local_size_x/y = 32`` matches the
# ``pipeline.local_size(32, 32)`` call below -- the two must agree.
# %%


compute_code = """
#version 460
#extension GL_EXT_scalar_block_layout: require

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0, rgba32f) uniform image2D render_target;
layout(binding = 1, scalar) uniform Parameters { vec2 C;  float W; };

vec4 get_color(float m) {
    m = min(m, 100.0);
    float s = 2*(1.0 / (1 + exp(-m)) - 0.5);
    return vec4(1 - s*s, 1 - s, mod(s*s + 0.5, 1.0), 1.0);
}

void main() {
    ivec3 thread_id = ivec3(gl_GlobalInvocationID);
    ivec2 dim = imageSize(render_target);
    vec2 c = C + W * vec2((thread_id.x + 0.5) / dim.x, (thread_id.y + 0.5) / dim.y) - W/2.0;
    vec2 Z = vec2(0., 0.);
    for (int i=0; i<256; i++)
        Z = vec2(Z.x*Z.x - Z.y*Z.y, 2*Z.x*Z.y) + c;
    float m = sqrt(dot(Z, Z));

    imageStore(render_target, thread_id.xy, get_color(m));    
}
"""

# %% [markdown]
# ## Device resources
# ``render_target`` is a device-local image the shader writes into
# directly (bound as a storage image, ``rgba32f`` to match the GLSL
# side). ``ubo`` is a *struct* buffer -- built from a ``{name: type}``
# dict, exactly like ``layout(...) uniform Parameters { vec2 C; float W; }``
# in the shader above -- and lives in host-visible memory so the CPU
# can fill it in with plain ``.write()`` calls before dispatch.
# %%

WIDTH, HEIGHT = 512, 512
# Image to store the Mandelbrot set
render_target = vk.image(WIDTH, HEIGHT, 1, 1, 1, vk.Format.RGBA32_Float, vk.MemoryLocation.DEVICE)
# ubo to store the parameters accessible by cpu memory.
ubo = vk.buffer(dict(
    C=vk.Type.VEC2,
    W=vk.Type.FLOAT32
), vk.MemoryLocation.HOST)

# %% [markdown]
# ## Filling the uniform buffer
# ``element_layout.field(name)`` resolves a struct field by name into a
# ``LayoutField``, and ``write`` copies a value into it directly (no
# DLPack round-trip) -- here, the view center ``C`` and zoom width ``W``
# the shader reads every invocation.
# %%

ubo.write(
    C=vk.vec2(-0.6, 0.0),
    W=2.0
)

# %% [markdown]
# ## Building the pipeline
# A compute pipeline needs: the workgroup size (must match the GLSL
# ``local_size_x/y/z``), a descriptor slot per resource the shader
# binds (set 0, binding 0 for the image; set 0, binding 1 for the
# uniform buffer, matching the ``layout(binding = ...)`` declarations
# above), and the compiled shader stage. ``close()`` finalizes the
# pipeline so it can be bound and dispatched.
# %%

pipeline = vk.pipeline(vk.PipelineType.COMPUTE)
pipeline.local_size(32, 32)
img_slot = pipeline.layout(0, 0, vk.DescriptorType.STORAGE_IMAGE)
ubo_slot = pipeline.layout(0, 1, vk.DescriptorType.UNIFORM_BUFFER)
pipeline.stage(vk.ShaderStageType.COMPUTE, vk.shader_from_glsl(compute_code, vk.ShaderStageType.COMPUTE))
pipeline.close()

# %% [markdown]
# ## Binding resources and dispatching
# A descriptor set connects each pipeline slot to a concrete resource.
# ``vk.compute()`` opens a temporary compute-capable command buffer as a
# context manager: on exit it closes and submits the recorded commands
# and waits for completion, so by the time the ``with`` block ends the
# GPU has finished writing ``render_target``.
# ``dispatch_threads(WIDTH, HEIGHT)`` dispatches enough workgroups to
# cover one thread per pixel, given the ``local_size(32, 32)`` set above.
# %%

bindings = pipeline.descriptor_set(set=0)
bindings.bind(ubo_slot, ubo)
bindings.bind(img_slot, render_target)

with vk.compute() as cmd:
    cmd.set_pipeline(pipeline)
    cmd.bind(0, [bindings])
    cmd.dispatch_threads(WIDTH, HEIGHT)

# %% [markdown]
# ## Reading the result back
# ``render_target`` is device-local, so it isn't directly readable from
# Python -- ``vk.staging(render_target)`` allocates a host-visible
# buffer sized and laid out for that image's own format, and
# ``cmd.transfer(image, buffer)`` (inside a transfer-capable command
# buffer) copies the GPU image into it. ``staging.numpy()`` then
# exposes that memory as a flat ``(texels, components)`` array via
# DLPack -- reshaped here into ``(HEIGHT, WIDTH, components)`` for
# ``imshow``.
# %%

staging = vk.staging(render_target)

with vk.transfer() as cmd:
    cmd.transfer(render_target, staging)

plt.imshow(staging.numpy().reshape(HEIGHT, WIDTH, -1))
plt.gca().axis('off')
plt.show()