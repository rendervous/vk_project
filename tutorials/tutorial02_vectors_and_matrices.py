# %%
try:  # install vulky in colab -- it installs any missing Vulkan driver itself
    import google.colab
    import subprocess
    subprocess.run(["pip", "install", "vulky"], check=True)
except ImportError:
    print("Executing locally")

# %% [markdown]
# # Tutorial 02 - Vectors and Matrices
# Vulky presents a standalone library for 3d math, called math3d.
# It is a simple library that provides basic operations for vectors and matrices.
# It is designed to be used with vulky, but it can be used independently as well.
# %%

import vk
import numpy as np
import math

# %% [markdown]
# Types in math3d mimic those existing in GLSL. For example, a 3D vector is represented by the class `m3.vec3`, and a 4x4 matrix is represented by the class `m3.mat4`.
# Those vectors and matrices can be initialized with the sequence of
# values, or with a dlpack tensor satisfying the dimensions of the vector/matrix.
# If initialized with vectors, the dl device must always be the cpu.
# %%

v1 = vk.vec3(1.0, 0.2, 0.3)
v2 = vk.vec3(np.array([1.0, 0.2, 0.3]))

print(v1 == v2)

# %% [markdown]
# ## Vector and Matrix Operations
# Matrices are stored column-major, exactly like GLSL/GLM: `matCxR` has
# `C` columns and `R` rows, so `mat3x4` maps a 3-component vector to a
# 4-component one. `m * v` is a matrix-vector product (`v` treated as a
# column), and `m1 * m2` composes transforms so that `m2` is applied
# first -- the same convention as GLSL's `*` operator, not numpy's
# row-major broadcasting rules.
# %%

m = vk.mat3x4(
    0.0, 1.0, 2.0, 3.0,
    4.0, 5.0, 6.0, 7.0,
    8.0, 9.0, 10.0, 11.0,
)
v3 = vk.math3d.mul(m, v1)
print(v3)
v3 = m * v1
print(v3)

# %% [markdown]
# ## Component access and swizzling
# Vector components are readable/writable by name (`.x`, `.y`, `.z`,
# `.w`) as well as by index, exactly like GLSL. There is no full GLSL
# swizzling (`v.xyz`, `v.zyx`, ...) -- component-by-component access
# covers most needs while keeping the library tiny.
# %%

p = vk.vec4(2.0, 4.0, 6.0, 1.0)
print("p.x =", p.x, " p[1] =", p[1])
p.z = 0.0
print("after p.z = 0:", p)

# %% [markdown]
# ## Geometric operations: dot, length, normalize, cross
# These are the building blocks used constantly in graphics code: `dot`
# for projections and angles, `length`/`normalize` for unit directions,
# and `cross` for a vector perpendicular to two others (e.g. deriving a
# surface normal from two edges of a triangle).
# %%

a = vk.vec3(1.0, 0.0, 0.0)
b = vk.vec3(0.0, 1.0, 0.0)

print("dot(a, b) =", vk.math3d.dot(a, b))          # 0: perpendicular vectors
print("cross(a, b) =", vk.math3d.cross(a, b))      # (0, 0, 1): right-hand rule
print("length(vec3(3,4,0)) =", vk.math3d.length(vk.vec3(3.0, 4.0, 0.0)))  # 5 (3-4-5 triangle)

diagonal = vk.vec3(1.0, 1.0, 1.0)
print("normalize(diagonal) =", vk.math3d.normalize(diagonal))  # unit length, same direction

# %% [markdown]
# `lerp` linearly interpolates between two vectors (or two matrices),
# handy for animating a value or blending two states over a parameter
# `t` in `[0, 1]`.
# %%

start = vk.vec3(0.0, 0.0, 0.0)
end = vk.vec3(10.0, 0.0, 0.0)
mid = vk.math3d.lerp(start, end, 0.5)
print("lerp at t=0.5:", mid)  # (5, 0, 0)

# %% [markdown]
# ## Building and composing transforms
# `translate`, `scale`, `rotate_x/y/z`, `rotate_axis` and `trs` build
# the mat4/mat3 transforms used to place objects in a scene. Because
# `m1 * m2` applies `m2` first, "rotate then translate" (rotate around
# the object's own origin, then move it into place) is written
# `translate(t) * mat4(rotation)`, matching how you'd read it in GLSL.
# `trs` builds the same translate * rotate * scale matrix directly,
# without the intermediate `mat4(rotation)` embedding step.
# %%

rotation = vk.math3d.rotate_y(math.radians(90.0))
translation = vk.vec3(5.0, 0.0, 0.0)

composed = vk.math3d.translate(translation) * vk.mat4(rotation)
direct = vk.math3d.trs(translation, rotation, vk.vec3(1.0, 1.0, 1.0))
print("rotate-then-translate == trs:", composed == direct)

# A point originally on +Z ends up on +X after the 90-degree rotation,
# then shifted by `translation`.
point = vk.vec4(0.0, 0.0, 1.0, 1.0)
print("transformed point:", composed * point)

# %% [markdown]
# ## Embedding, transpose and inverse
# `mat4(a_mat3)` embeds a smaller matrix into a larger identity matrix
# (typical when promoting a pure rotation to a full 4x4 transform).
# `transpose` swaps rows/columns (turning a `matCxR` into a `matRxC`),
# and `inverse` solves for the matrix that undoes a transform -- useful
# e.g. to move a vector from world space back into local/object space.
# %%

embedded = vk.mat4(rotation)  # 3x3 rotation embedded into a 4x4 identity
print("embedded rotation:\n", embedded)

m2x3 = vk.mat3x4(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
print("transpose shape check (mat3x4 -> mat4x3):", type(vk.math3d.transpose(m2x3)).__name__)

world_to_local = vk.math3d.inverse(composed)
back_to_origin_ish = world_to_local * (composed * point)
print("round-trip through inverse:", back_to_origin_ish)  # back to the original `point`

# %% [markdown]
# ## A camera: look_at + perspective
# `look_at_rh`/`look_at_lh` build a view matrix from an eye position, a
# target to look at, and an up vector; `perspective_rh`/`perspective_lh`
# build the projection matrix. Multiplying them gives the combined
# view-projection matrix typically uploaded to a shader's uniform buffer.
# %%

view = vk.math3d.look_at_rh(
    eye=vk.vec3(0.0, 2.0, 5.0),
    center=vk.vec3(0.0, 0.0, 0.0),
    up=vk.vec3(0.0, 1.0, 0.0),
)
proj = vk.math3d.perspective_rh(math.radians(60.0), 16.0 / 9.0, 0.1, 100.0)
view_proj = proj * view
print("view-projection matrix:\n", view_proj)

# %% [markdown]
# ## Interop with buffers
# `Buffer.write` accepts a `vk.math3d` vec*/mat* object directly (as
# well as a plain nested list, or anything buffer-protocol/DLPack
# compatible) -- this is how a transform computed with math3d ends up
# in a uniform buffer read by a shader, with no manual packing step.
# `to_array()` (a tightly-packed, column-major `array.array`) is there
# for the cases that do want the raw bytes explicitly, e.g. handing
# them to a third-party library.
# %%

ubo = vk.buffer(dict(VP=vk.Type.MAT4), vk.MemoryLocation.HOST)
ubo.write(ubo.element_layout.field("VP"), view_proj)
print("uploaded view-projection matrix to a uniform buffer")
print("raw column-major floats:", view_proj.to_array())
