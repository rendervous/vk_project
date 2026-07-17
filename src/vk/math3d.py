"""Minimal, dependency-free 3D math: vec2/vec3/vec4 and mat2/mat3/mat3x4/
mat4x3/mat4, plus the usual free functions (dot, cross, length, normalize,
transpose, inverse, look_at_lh/rh, perspective_lh/rh, trs, ...).

Conventions (matching GLSL/GLM, and this project's own TypeDescriptor.matrix
docstring: "a column-major matrix"):

- Matrices are stored column-major. ``matCxR`` has ``C`` columns and ``R``
  rows -- ``mat3x4`` maps a vec3 to a vec4, ``mat4x3`` maps a vec4 to a
  vec3. ``m[i]`` is the i-th column (a vector); ``m[i, j]`` (or ``m[i][j]``)
  is the element at column ``i``, row ``j``.
- ``mat * vec`` is a matrix-vector product (vector is a column); ``mat *
  mat`` is standard matrix composition, ``m1 * m2`` applies ``m2`` first.
- perspective_lh/rh use Vulkan's [0, 1] clip-space depth range (unlike
  OpenGL's [-1, 1]) -- GLM's ``*_ZO`` variants. Vulkan's default clip
  space also has Y pointing down; these functions don't flip it for you
  (negate row 1, e.g. ``m[1, 1] *= -1``, if your pipeline needs that).
- ``vec*``/``mat*`` components are ``float`` (GLSL ``float``); ``ivec*``/
  ``uvec*``/``bvec*`` mirror GLSL's ``int``/``uint``/``bool`` vectors
  (``uvec`` doesn't emulate 32-bit unsigned wraparound -- it's a plain
  Python ``int`` expected to stay non-negative). Like GLSL, ``bvec*``
  doesn't support arithmetic operators.
- ``to_array()`` returns a tightly-packed, column-major ``array.array``
  (typecode ``'f'``/``'i'``/``'I'``/``'B'`` matching the component type)
  -- usable directly with :meth:`Buffer.write`/:meth:`Buffer.load` (the
  ``array`` module supports the buffer protocol). :meth:`Buffer.write`
  also accepts any of these objects directly.
"""

import math
from array import array as _array
from collections.abc import Iterable

__all__ = [
    "vec2", "vec3", "vec4",
    "ivec2", "ivec3", "ivec4",
    "uvec2", "uvec3", "uvec4",
    "bvec2", "bvec3", "bvec4",
    "mat2", "mat3", "mat3x4", "mat4x3", "mat4",
    "dot", "length", "normalize", "cross", "lerp", "mul",
    "transpose", "inverse",
    "look_at_lh", "look_at_rh", "perspective_lh", "perspective_rh",
    "translate", "scale", "rotate_x", "rotate_y", "rotate_z", "rotate_axis", "trs",
]


class _VecBase:
    __slots__ = ("_data",)
    _SIZE = 0
    _CAST = float
    _ARRAY_TYPECODE = 'f'

    def __init__(self, *args):
        n = self._SIZE
        cast = self._CAST
        if len(args) == 0:
            self._data = [cast(0)] * n
            return
        if len(args) == 1:
            a = args[0]
            if isinstance(a, _VecBase):
                if len(a) != n:
                    raise ValueError(f"{type(self).__name__} cannot be constructed from a {type(a).__name__}")
                self._data = [cast(c) for c in a._data]
                return
            if isinstance(a, (int, float)):
                self._data = [cast(a)] * n
                return
            if isinstance(a, Iterable):
                data = [cast(c) for c in a]
                if len(data) != n:
                    raise ValueError(f"{type(self).__name__} expects {n} components, got {len(data)}")
                self._data = data
                return
            raise TypeError(f"Cannot construct {type(self).__name__} from {a!r}")
        # Multiple arguments: flatten (each may itself be a shorter vector,
        # e.g. vec3(a_vec2, z) or vec4(a_vec3, w)).
        data = []
        for a in args:
            if isinstance(a, _VecBase):
                data.extend(cast(c) for c in a._data)
            elif isinstance(a, (int, float)):
                data.append(cast(a))
            else:
                raise TypeError(f"Cannot construct {type(self).__name__} from {a!r}")
        if len(data) != n:
            raise ValueError(f"{type(self).__name__} expects {n} components, got {len(data)}")
        self._data = data

    def __len__(self):
        return self._SIZE

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __setitem__(self, i, value):
        self._data[i] = self._CAST(value)

    def __eq__(self, other):
        if not isinstance(other, _VecBase) or len(other) != len(self):
            return NotImplemented
        return self._data == other._data

    def __repr__(self):
        comps = ", ".join(f"{c:g}" if isinstance(c, float) else repr(c) for c in self._data)
        return f"{type(self).__name__}({comps})"

    def __neg__(self):
        return type(self)(*(-c for c in self._data))

    def __add__(self, other):
        if isinstance(other, _VecBase) and len(other) == len(self):
            return type(self)(*(a + b for a, b in zip(self._data, other._data)))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, _VecBase) and len(other) == len(self):
            return type(self)(*(a - b for a, b in zip(self._data, other._data)))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(*(a * other for a in self._data))
        if isinstance(other, _VecBase) and len(other) == len(self):
            return type(self)(*(a * b for a, b in zip(self._data, other._data)))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(*(a * other for a in self._data))
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(*(a / other for a in self._data))
        if isinstance(other, _VecBase) and len(other) == len(self):
            return type(self)(*(a / b for a, b in zip(self._data, other._data)))
        return NotImplemented

    def to_array(self):
        """A tightly-packed ``array.array(...)`` copy of this vector's
        components -- buffer-protocol compatible (see module docstring)."""
        return _array(self._ARRAY_TYPECODE, self._data)


class _BoolVecBase(_VecBase):
    """Base for bvec2/3/4: like GLSL's bvec, no arithmetic operators."""
    __slots__ = ()
    _CAST = bool
    _ARRAY_TYPECODE = 'B'

    def __neg__(self): raise TypeError("bvec does not support arithmetic operators")
    def __add__(self, other): raise TypeError("bvec does not support arithmetic operators")
    def __sub__(self, other): raise TypeError("bvec does not support arithmetic operators")
    def __mul__(self, other): raise TypeError("bvec does not support arithmetic operators")
    def __rmul__(self, other): raise TypeError("bvec does not support arithmetic operators")
    def __truediv__(self, other): raise TypeError("bvec does not support arithmetic operators")


class _XY:
    __slots__ = ()

    @property
    def x(self): return self._data[0]
    @x.setter
    def x(self, v): self._data[0] = self._CAST(v)

    @property
    def y(self): return self._data[1]
    @y.setter
    def y(self, v): self._data[1] = self._CAST(v)


class _XYZ(_XY):
    __slots__ = ()

    @property
    def z(self): return self._data[2]
    @z.setter
    def z(self, v): self._data[2] = self._CAST(v)


class _XYZW(_XYZ):
    __slots__ = ()

    @property
    def w(self): return self._data[3]
    @w.setter
    def w(self, v): self._data[3] = self._CAST(v)


class vec2(_XY, _VecBase):
    """A 2-component ``float`` vector (``x``, ``y``); GLSL ``vec2``."""
    __slots__ = ()
    _SIZE = 2


class vec3(_XYZ, _VecBase):
    """A 3-component ``float`` vector (``x``, ``y``, ``z``); GLSL ``vec3``."""
    __slots__ = ()
    _SIZE = 3


class vec4(_XYZW, _VecBase):
    """A 4-component ``float`` vector (``x``, ``y``, ``z``, ``w``); GLSL ``vec4``."""
    __slots__ = ()
    _SIZE = 4


class ivec2(_XY, _VecBase):
    """A 2-component ``int`` vector; GLSL ``ivec2``."""
    __slots__ = ()
    _SIZE = 2
    _CAST = int
    _ARRAY_TYPECODE = 'i'


class ivec3(_XYZ, _VecBase):
    """A 3-component ``int`` vector; GLSL ``ivec3``."""
    __slots__ = ()
    _SIZE = 3
    _CAST = int
    _ARRAY_TYPECODE = 'i'


class ivec4(_XYZW, _VecBase):
    """A 4-component ``int`` vector; GLSL ``ivec4``."""
    __slots__ = ()
    _SIZE = 4
    _CAST = int
    _ARRAY_TYPECODE = 'i'


class uvec2(_XY, _VecBase):
    """A 2-component ``uint`` vector; GLSL ``uvec2`` (no 32-bit wraparound emulation)."""
    __slots__ = ()
    _SIZE = 2
    _CAST = int
    _ARRAY_TYPECODE = 'I'


class uvec3(_XYZ, _VecBase):
    """A 3-component ``uint`` vector; GLSL ``uvec3`` (no 32-bit wraparound emulation)."""
    __slots__ = ()
    _SIZE = 3
    _CAST = int
    _ARRAY_TYPECODE = 'I'


class uvec4(_XYZW, _VecBase):
    """A 4-component ``uint`` vector; GLSL ``uvec4`` (no 32-bit wraparound emulation)."""
    __slots__ = ()
    _SIZE = 4
    _CAST = int
    _ARRAY_TYPECODE = 'I'


class bvec2(_XY, _BoolVecBase):
    """A 2-component ``bool`` vector; GLSL ``bvec2``."""
    __slots__ = ()
    _SIZE = 2


class bvec3(_XYZ, _BoolVecBase):
    """A 3-component ``bool`` vector; GLSL ``bvec3``."""
    __slots__ = ()
    _SIZE = 3


class bvec4(_XYZW, _BoolVecBase):
    """A 4-component ``bool`` vector; GLSL ``bvec4``."""
    __slots__ = ()
    _SIZE = 4


_VEC_TYPES = {2: vec2, 3: vec3, 4: vec4}


def dot(a, b):
    """Dot (inner) product of two same-size vectors."""
    if len(a) != len(b):
        raise ValueError("dot: vectors must have the same size")
    return sum(x * y for x, y in zip(a, b))


def length(v):
    """Euclidean length (magnitude) of a float vector (vec2/vec3/vec4)."""
    if type(v)._CAST is not float:
        raise TypeError("length: only defined for a float vector (vec2/vec3/vec4)")
    return math.sqrt(dot(v, v))


def normalize(v):
    """``v`` scaled to unit length. Raises ValueError for a zero-length vector."""
    l = length(v)
    if l == 0.0:
        raise ValueError("normalize: cannot normalize a zero-length vector")
    return v / l


def cross(a, b):
    """3D cross product (vec3/ivec3/uvec3 x same-type -> same type), or the
    2D "perp dot" product (vec2/ivec2/uvec2 x same-type -> a scalar, the Z
    component of the equivalent 3D cross product)."""
    if not (isinstance(a, _VecBase) and isinstance(b, _VecBase) and len(a) == len(b)):
        raise TypeError("cross: only defined for two same-size vectors")
    if len(a) == 3:
        return type(a)(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    if len(a) == 2:
        return a.x * b.y - a.y * b.x
    raise TypeError("cross: only defined for 2- or 3-component vectors")


def lerp(a, b, t):
    """Linear interpolation: ``a + (b - a) * t``."""
    return a + (b - a) * t


def mul(a, b):
    """Explicit multiplication (HLSL-style ``mul``), equivalent to ``a *
    b``: componentwise for two same-size vectors, scaling for a
    vector/matrix and a plain number, or a matrix-vector/matrix-matrix
    product for a matrix."""
    return a * b


class _MatBase:
    __slots__ = ("_columns",)
    _COLS = 0
    _ROWS = 0
    _COLUMN_TYPE = None

    def __init__(self, *args):
        c_count, r_count, vt = self._COLS, self._ROWS, self._COLUMN_TYPE
        if len(args) == 0:
            # Identity (1s on the leading diagonal, 0s elsewhere) -- matches
            # GLM's default matrix constructor, generalized to non-square
            # shapes.
            self._columns = [vt(*(1.0 if r == c else 0.0 for r in range(r_count))) for c in range(c_count)]
            return
        if len(args) == 1 and isinstance(args[0], (int, float)):
            value = float(args[0])
            self._columns = [vt(*(value if r == c else 0.0 for r in range(r_count))) for c in range(c_count)]
            return
        if len(args) == 1 and isinstance(args[0], _MatBase):
            # Embeds (or crops) another matrix into this shape, identity on
            # the leading diagonal elsewhere -- e.g. mat4(a_mat3) embeds a
            # rotation into the top-left 3x3 of an otherwise-identity mat4.
            other = args[0]
            self._columns = [
                vt(*(
                    (other._columns[c][r] if c < other._COLS and r < other._ROWS else (1.0 if r == c else 0.0))
                    for r in range(r_count)
                ))
                for c in range(c_count)
            ]
            return
        if len(args) == c_count:
            self._columns = [a if isinstance(a, vt) else vt(a) for a in args]
            return
        if len(args) == c_count * r_count:
            self._columns = [vt(*args[c * r_count:(c + 1) * r_count]) for c in range(c_count)]
            return
        raise TypeError(
            f"{type(self).__name__} expects 0 args (identity), 1 scalar (diagonal fill), "
            f"{c_count} column vectors, or {c_count * r_count} flat scalars"
        )

    def __len__(self):
        return self._COLS

    def __iter__(self):
        return iter(self._columns)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            c, r = key
            return self._columns[c][r]
        return self._columns[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            c, r = key
            self._columns[c][r] = float(value)
        else:
            self._columns[key] = value if isinstance(value, self._COLUMN_TYPE) else self._COLUMN_TYPE(value)

    def __eq__(self, other):
        if not isinstance(other, _MatBase) or other._COLS != self._COLS or other._ROWS != self._ROWS:
            return NotImplemented
        return self._columns == other._columns

    def __repr__(self):
        rows = []
        for r in range(self._ROWS):
            rows.append("[" + ", ".join(f"{self._columns[c][r]:g}" for c in range(self._COLS)) + "]")
        return f"{type(self).__name__}(\n  " + ",\n  ".join(rows) + "\n)"

    def __neg__(self):
        return type(self)(*(-col for col in self._columns))

    def __add__(self, other):
        if isinstance(other, _MatBase) and other._COLS == self._COLS and other._ROWS == self._ROWS:
            return type(self)(*(a + b for a, b in zip(self._columns, other._columns)))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, _MatBase) and other._COLS == self._COLS and other._ROWS == self._ROWS:
            return type(self)(*(a - b for a, b in zip(self._columns, other._columns)))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(*(col * other for col in self._columns))
        if isinstance(other, _VecBase):
            if len(other) != self._COLS:
                raise ValueError(f"Cannot multiply {type(self).__name__} by a length-{len(other)} vector")
            result_type = _VEC_TYPES[self._ROWS]
            return result_type(*(
                sum(self._columns[c][r] * other[c] for c in range(self._COLS)) for r in range(self._ROWS)
            ))
        if isinstance(other, _MatBase):
            if other._ROWS != self._COLS:
                raise ValueError(f"Cannot multiply {type(self).__name__} by {type(other).__name__}: dimension mismatch")
            result_type = _MAT_TYPES[(other._COLS, self._ROWS)]
            row_vec_type = _VEC_TYPES[self._ROWS]
            new_columns = [
                row_vec_type(*(
                    sum(self._columns[c][r] * other._columns[j][c] for c in range(self._COLS))
                    for r in range(self._ROWS)
                ))
                for j in range(other._COLS)
            ]
            return result_type(*new_columns)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return type(self)(*(col * other for col in self._columns))
        return NotImplemented

    def to_array(self):
        """A tightly-packed, column-major ``array.array('f', ...)`` copy of
        this matrix -- buffer-protocol compatible (see module docstring)."""
        flat = []
        for col in self._columns:
            flat.extend(col)
        return _array('f', flat)


class mat2(_MatBase):
    """A 2x2 matrix (2 columns, 2 rows)."""
    __slots__ = ()
    _COLS = 2
    _ROWS = 2
    _COLUMN_TYPE = vec2


class mat3(_MatBase):
    """A 3x3 matrix (3 columns, 3 rows)."""
    __slots__ = ()
    _COLS = 3
    _ROWS = 3
    _COLUMN_TYPE = vec3


class mat3x4(_MatBase):
    """A matrix with 3 columns and 4 rows (maps a vec3 to a vec4)."""
    __slots__ = ()
    _COLS = 3
    _ROWS = 4
    _COLUMN_TYPE = vec4


class mat4x3(_MatBase):
    """A matrix with 4 columns and 3 rows (maps a vec4 to a vec3)."""
    __slots__ = ()
    _COLS = 4
    _ROWS = 3
    _COLUMN_TYPE = vec3


class mat4(_MatBase):
    """A 4x4 matrix (4 columns, 4 rows)."""
    __slots__ = ()
    _COLS = 4
    _ROWS = 4
    _COLUMN_TYPE = vec4


_MAT_TYPES = {
    (2, 2): mat2,
    (3, 3): mat3,
    (3, 4): mat3x4,
    (4, 3): mat4x3,
    (4, 4): mat4,
}


def transpose(m):
    """Transpose of ``m`` (a ``matCxR`` becomes a ``matRxC``)."""
    result_type = _MAT_TYPES[(m._ROWS, m._COLS)]
    row_vec_type = _VEC_TYPES[m._COLS]
    new_columns = [row_vec_type(*(m._columns[c][r] for c in range(m._COLS))) for r in range(m._ROWS)]
    return result_type(*new_columns)


def inverse(m):
    """Inverse of a square matrix (mat2/mat3/mat4), via Gauss-Jordan
    elimination with partial pivoting. Raises ValueError if ``m`` isn't
    square or is singular."""
    n = m._COLS
    if m._ROWS != n:
        raise ValueError("inverse: only defined for square matrices")

    a = [[m._columns[c][r] for c in range(n)] for r in range(n)]
    inv = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot_row][col]) < 1e-12:
            raise ValueError("inverse: matrix is singular")
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            inv[col], inv[pivot_row] = inv[pivot_row], inv[col]
        pivot = a[col][col]
        a[col] = [v / pivot for v in a[col]]
        inv[col] = [v / pivot for v in inv[col]]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor == 0.0:
                continue
            a[r] = [av - factor * cv for av, cv in zip(a[r], a[col])]
            inv[r] = [iv - factor * cv for iv, cv in zip(inv[r], inv[col])]

    result_type = _MAT_TYPES[(n, n)]
    vt = _VEC_TYPES[n]
    return result_type(*(vt(*(inv[r][c] for r in range(n))) for c in range(n)))


def look_at_rh(eye, center, up):
    """Right-handed view matrix looking from ``eye`` towards ``center``."""
    f = normalize(center - eye)
    s = normalize(cross(f, up))
    u = cross(s, f)
    result = mat4()
    result[0, 0], result[1, 0], result[2, 0] = s.x, s.y, s.z
    result[0, 1], result[1, 1], result[2, 1] = u.x, u.y, u.z
    result[0, 2], result[1, 2], result[2, 2] = -f.x, -f.y, -f.z
    result[3, 0] = -dot(s, eye)
    result[3, 1] = -dot(u, eye)
    result[3, 2] = dot(f, eye)
    return result


def look_at_lh(eye, center, up):
    """Left-handed view matrix looking from ``eye`` towards ``center``."""
    f = normalize(center - eye)
    s = normalize(cross(up, f))
    u = cross(f, s)
    result = mat4()
    result[0, 0], result[1, 0], result[2, 0] = s.x, s.y, s.z
    result[0, 1], result[1, 1], result[2, 1] = u.x, u.y, u.z
    result[0, 2], result[1, 2], result[2, 2] = f.x, f.y, f.z
    result[3, 0] = -dot(s, eye)
    result[3, 1] = -dot(u, eye)
    result[3, 2] = -dot(f, eye)
    return result


def perspective_rh(fovy, aspect, z_near, z_far):
    """Right-handed perspective projection, Vulkan/D3D-style [0, 1] depth
    range. ``fovy`` is the vertical field of view, in radians."""
    tan_half_fovy = math.tan(fovy / 2.0)
    result = mat4(0.0)
    result[0, 0] = 1.0 / (aspect * tan_half_fovy)
    result[1, 1] = 1.0 / tan_half_fovy
    result[2, 2] = z_far / (z_near - z_far)
    result[2, 3] = -1.0
    result[3, 2] = -(z_far * z_near) / (z_far - z_near)
    return result


def perspective_lh(fovy, aspect, z_near, z_far):
    """Left-handed perspective projection, Vulkan/D3D-style [0, 1] depth
    range. ``fovy`` is the vertical field of view, in radians."""
    tan_half_fovy = math.tan(fovy / 2.0)
    result = mat4(0.0)
    result[0, 0] = 1.0 / (aspect * tan_half_fovy)
    result[1, 1] = 1.0 / tan_half_fovy
    result[2, 2] = z_far / (z_far - z_near)
    result[2, 3] = 1.0
    result[3, 2] = -(z_far * z_near) / (z_far - z_near)
    return result


def translate(v):
    """A mat4 translating by vec3 ``v``."""
    result = mat4()
    result[3, 0], result[3, 1], result[3, 2] = v.x, v.y, v.z
    return result


def scale(v):
    """A mat4 scaling by vec3 ``v`` along each axis."""
    result = mat4()
    result[0, 0], result[1, 1], result[2, 2] = v.x, v.y, v.z
    return result


def rotate_x(angle):
    """A mat3 rotating ``angle`` radians about the X axis."""
    c, s = math.cos(angle), math.sin(angle)
    result = mat3()
    result[1, 1], result[2, 1] = c, -s
    result[1, 2], result[2, 2] = s, c
    return result


def rotate_y(angle):
    """A mat3 rotating ``angle`` radians about the Y axis."""
    c, s = math.cos(angle), math.sin(angle)
    result = mat3()
    result[0, 0], result[2, 0] = c, s
    result[0, 2], result[2, 2] = -s, c
    return result


def rotate_z(angle):
    """A mat3 rotating ``angle`` radians about the Z axis."""
    c, s = math.cos(angle), math.sin(angle)
    result = mat3()
    result[0, 0], result[1, 0] = c, -s
    result[0, 1], result[1, 1] = s, c
    return result


def rotate_axis(axis, angle):
    """A mat3 rotating ``angle`` radians about an arbitrary (not
    necessarily unit-length) ``axis``, via Rodrigues' rotation formula."""
    a = normalize(axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    x, y, z = a.x, a.y, a.z
    return mat3(
        vec3(t * x * x + c, t * x * y + s * z, t * x * z - s * y),
        vec3(t * x * y - s * z, t * y * y + c, t * y * z + s * x),
        vec3(t * x * z + s * y, t * y * z - s * x, t * z * z + c),
    )


def trs(translation, rotation, scale_):
    """A mat4 combining translation (vec3), rotation (mat3) and scale
    (vec3) into one transform -- equivalent to ``translate(translation) *
    mat4(rotation) * scale(scale_)`` but built directly."""
    r0 = rotation[0] * scale_.x
    r1 = rotation[1] * scale_.y
    r2 = rotation[2] * scale_.z
    return mat4(
        vec4(r0.x, r0.y, r0.z, 0.0),
        vec4(r1.x, r1.y, r1.z, 0.0),
        vec4(r2.x, r2.y, r2.z, 0.0),
        vec4(translation.x, translation.y, translation.z, 1.0),
    )
