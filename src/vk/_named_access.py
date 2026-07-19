"""Two Python-only conveniences layered on top of the raw pybind11 classes
(no changes to the C++ bindings):

- Named descriptor/attachment bindings: :meth:`Pipeline.layout`/
  :meth:`Pipeline.attach` accept a single ``name=DescriptorType``/
  ``name=Format`` keyword instead of a positional type, remembering that
  name; :meth:`DescriptorSet.bind` and :meth:`Pipeline.create_framebuffer`
  then accept ``name=resource`` keywords instead of the raw opaque handle,
  resolved back to the matching handle.

- Name-path Buffer field access: :meth:`Buffer.write`/:meth:`Buffer.read`
  accept a dotted string path (e.g. ``"lights.0.P"``) instead of a
  ``LayoutField``, navigating the struct/array/matrix Layout tree by field
  name and integer index. A single, plain field name (no dot, not an
  integer) resolves to a real ``LayoutField`` and is forwarded to the
  original implementation unchanged; a deeper path is resolved to a raw
  byte offset and read/written through a byte-level ``Buffer.cast``/
  ``slice`` (this step lazily imports numpy -- there's no dependency-free
  way to poke arbitrary bytes into a mapped buffer from pure Python).
"""

import weakref

from .vk import Buffer, DescriptorSet, Pipeline, Type, TypeKind

# ---------------------------------------------------------------------------
# Named descriptor/attachment bindings.
# ---------------------------------------------------------------------------

_layout_names = weakref.WeakKeyDictionary()  # Pipeline -> {name: LayoutHandle}
_attach_names = weakref.WeakKeyDictionary()  # Pipeline -> {name: AttachHandle}
_descriptor_set_names = weakref.WeakKeyDictionary()  # DescriptorSet -> {name: LayoutHandle}

_orig_pipeline_layout = Pipeline.layout
_orig_pipeline_attach = Pipeline.attach
_orig_descriptor_set = Pipeline.descriptor_set
_orig_descriptor_set_collection = Pipeline.descriptor_set_collection
_orig_create_framebuffer = Pipeline.create_framebuffer
_orig_descriptor_set_bind = DescriptorSet.bind


def _one_name_kwarg(kwargs, positional, what):
    if not kwargs:
        return None, positional
    if positional is not None or len(kwargs) != 1:
        raise TypeError(f"{what}() takes either a positional value or a single name=... keyword")
    name, value = next(iter(kwargs.items()))
    return name, value


def _layout(self, set, binding, description=None, count=1, **kwargs):
    name, description = _one_name_kwarg(kwargs, description, "layout")
    handle = _orig_pipeline_layout(self, set, binding, description, count)
    if name is not None:
        _layout_names.setdefault(self, {})[name] = handle
    return handle


def _attach(self, slot, format=None, **kwargs):
    name, format = _one_name_kwarg(kwargs, format, "attach")
    handle = _orig_pipeline_attach(self, slot, format)
    if name is not None:
        _attach_names.setdefault(self, {})[name] = handle
    return handle


def _descriptor_set(self, set=0):
    ds = _orig_descriptor_set(self, set)
    names = _layout_names.get(self)
    if names:
        _descriptor_set_names[ds] = names
    return ds


def _descriptor_set_collection(self, set=0, count=1):
    collection = _orig_descriptor_set_collection(self, set, count)
    names = _layout_names.get(self)
    if names:
        for ds in collection:
            _descriptor_set_names[ds] = names
    return collection


def _create_framebuffer(self, attachments=None, depth_image=None, **kwargs):
    if kwargs:
        if attachments is not None:
            raise TypeError("create_framebuffer() takes either an attachments list or name=image keywords, not both")
        names = _attach_names.get(self, {})
        attachments = []
        for name, image in kwargs.items():
            if name not in names:
                raise KeyError(f"No attachment named {name!r} on this pipeline")
            attachments.append((names[name], image))
    return _orig_create_framebuffer(self, attachments or [], depth_image)


def _descriptor_set_bind(self, layout_id=None, *args, **kwargs):
    if layout_id is not None:
        return _orig_descriptor_set_bind(self, layout_id, *args)
    names = _descriptor_set_names.get(self, {})
    for name, resource in kwargs.items():
        if name not in names:
            raise KeyError(f"No layout binding named {name!r} on this pipeline")
        handle = names[name]
        if isinstance(resource, tuple):
            _orig_descriptor_set_bind(self, handle, *resource)
        else:
            _orig_descriptor_set_bind(self, handle, resource)


Pipeline.layout = _layout
Pipeline.attach = _attach
Pipeline.descriptor_set = _descriptor_set
Pipeline.descriptor_set_collection = _descriptor_set_collection
Pipeline.create_framebuffer = _create_framebuffer
DescriptorSet.bind = _descriptor_set_bind


# ---------------------------------------------------------------------------
# Name-path Buffer field access.
# ---------------------------------------------------------------------------

_orig_buffer_write = Buffer.write
_orig_buffer_read = Buffer.read


def _is_simple_name(path):
    return "." not in path and not path.lstrip("-").isdigit()


def _walk(layout, path):
    """(byte_offset, leaf_layout) for dotted `path` (field names and/or
    integer array/matrix-column indices), navigated from `layout`."""
    offset = 0
    current = layout
    for token in path.split("."):
        if token.lstrip("-").isdigit():
            if current.element_layout is None:
                raise RuntimeError(f"'{token}' in path {path!r}: {current.type} isn't indexable")
            offset += int(token) * current.stride
            current = current.element_layout
        else:
            field = current.field(token)
            offset += field.offset
            current = field.layout
    return offset, current


def _leaf_buffer(self, path):
    offset, leaf = _walk(self.element_layout, path)
    return self.cast(Type.UINT8).slice(offset, leaf.aligned_size).cast(leaf)


def _is_array_path(self, path):
    _, leaf = _walk(self.element_layout, path)
    return leaf.kind == TypeKind.ARRAY


def _write_leaf(self, path, value):
    """Writes a single, already-resolved (no dict/list expansion) path."""
    if _is_simple_name(path):
        return _orig_buffer_write(self, self.element_layout.field(path), value)
    # numpy.from_dlpack always comes back read-only here (no writable flag
    # is negotiated over the legacy DLPack capsule); torch.from_dlpack
    # doesn't, so it's used as the mutation backend for a nested path.
    import torch as _torch

    flat = _leaf_buffer(self, path).torch().reshape(-1)
    data = value.to_array() if hasattr(value, "to_array") else value
    flat[:] = _torch.as_tensor(data, dtype=flat.dtype).reshape(-1)


def _write_path(self, path, value):
    if isinstance(value, dict):
        for name, item in value.items():
            _write_path(self, f"{path}.{name}", item)
        return
    if isinstance(value, (list, tuple)) and _is_array_path(self, path):
        for index, item in enumerate(value):
            _write_path(self, f"{path}.{index}", item)
        return
    _write_leaf(self, path, value)


def _write(self, field=None, value=None, **fields):
    """Writes one or more fields of a struct-layout buffer directly (no
    DLPack round-trip).

    Three forms:

    - ``buf.write(field, value)``: ``field`` a ``LayoutField`` (as returned
      by ``buf.element_layout.field(name)``) -- forwarded to the original
      binding unchanged.
    - ``buf.write(path, value)``: ``path`` a dotted name string (e.g.
      ``"lights.0.P"``, field names and/or integer array/matrix-column
      indices). A single plain name is equivalent to the ``LayoutField``
      form; a deeper path is resolved via a byte-level ``cast``/``slice``
      (lazily imports ``torch``).
    - ``buf.write(name=value, ...)``: one or more top-level fields by
      name, each written independently -- equivalent to calling
      ``buf.write(name, value)`` per keyword.

    In every form, if ``value`` is a ``dict``, the target field is assumed
    to be a struct and each ``{key: value}`` pair is written recursively
    to ``path.key``. If ``value`` is a ``list``/``tuple`` and the target
    field is an array, each element is written to the matching numeric
    index (``path.0``, ``path.1``, ...) -- for a vector/matrix field
    (not an array), a list/tuple is instead passed straight through as its
    raw components, as before.

    :raises TypeError: If both a positional field and keywords are given.
    """
    if fields:
        if field is not None or value is not None:
            raise TypeError("write() takes either (field, value) or name=value keywords, not both")
        for name, item in fields.items():
            _write_path(self, name, item)
        return
    if isinstance(field, str):
        _write_path(self, field, value)
        return
    return _orig_buffer_write(self, field, value)


def _read(self, field):
    """Reads a single field of a struct-layout buffer directly (no DLPack
    round-trip).

    ``field`` is either a ``LayoutField`` (as returned by
    ``buf.element_layout.field(name)``, forwarded to the original binding
    unchanged), or a dotted name path string (e.g. ``"lights.0.P"``, field
    names and/or integer array/matrix-column indices). A single plain name
    is equivalent to the ``LayoutField`` form and returns a plain number or
    the matching ``vk.math3d`` vec*/mat* object; a deeper path is resolved
    via a byte-level ``cast``/``slice`` (lazily imports ``torch``) and
    returns a ``torch.Tensor`` instead.
    """
    if isinstance(field, str):
        if _is_simple_name(field):
            return _orig_buffer_read(self, self.element_layout.field(field))
        return _leaf_buffer(self, field).torch()[0]
    return _orig_buffer_read(self, field)


Buffer.write = _write
Buffer.read = _read
