#pragma once
#include <vulkan/vulkan.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <deque>
#include <string>
#include <variant>
#include <chrono>

namespace pybind11 { class object; }

/*  ============== PROTOTYPES ============== */

// Memory Management
class MemoryPage;
class MemorySlice;
class MemoryManager;
class ExternalInteropLibraryImpl;
class MemoryAllocator;

// Resources
class Resource;
class Buffer;
class Tensor;
class Image;
class WrappedMemory;

// Device
class Device;

// Structured buffer layout
class TypeDescriptor;
class Layout;

// Execution
class vk_Engine;
class Engine;
class vk_CommandBuffer;
class CommandBuffer;
class vk_Pipeline;
class Pipeline;
class vk_Framebuffer;
class Framebuffer;
class vk_DescriptorSet;
class DescriptorSet;

// GUI
class vk_Window;
class Window;
class Frame;
class Stats;
class Checkbox;
class SliderFloat;
class SliderInt;
class Combobox;

/*  ============== ENUMS ============== */

enum class MemoryLocation : int {
    HOST = 0,
    DEVICE = 1,
};

using ExternalImportFn = std::uint64_t (*)(vk::Device device, int device_index, vk::DeviceMemory memory, std::uint64_t size);

enum class ScalarType : int {
    UNDEFINED = 0,
    BOOL = 1,
    FLOAT16 = 2,
    FLOAT32 = 3,
    FLOAT64 = 4,
    INT8 = 5,
    INT16 = 6,
    INT32 = 7,
    INT64 = 8,
    UINT8 = 9,
    UINT16 = 10,
    UINT32 = 11,
    UINT64 = 12,
};

enum class Format : int {
    Undefined = static_cast<int>(vk::Format::eUndefined),

    // 8-bit unsigned normalized
    R8_UNorm    = static_cast<int>(vk::Format::eR8Unorm),
    RG8_UNorm   = static_cast<int>(vk::Format::eR8G8Unorm),
    RGB8_UNorm  = static_cast<int>(vk::Format::eR8G8B8Unorm),
    RGBA8_UNorm = static_cast<int>(vk::Format::eR8G8B8A8Unorm),
    // Byte-swapped channel order of RGBA8_UNorm: the native swapchain
    // surface format on most Windows (and many Linux) drivers.
    BGRA8_UNorm = static_cast<int>(vk::Format::eB8G8R8A8Unorm),

    // 8-bit signed normalized
    R8_SNorm    = static_cast<int>(vk::Format::eR8Snorm),
    RG8_SNorm   = static_cast<int>(vk::Format::eR8G8Snorm),
    RGB8_SNorm  = static_cast<int>(vk::Format::eR8G8B8Snorm),
    RGBA8_SNorm = static_cast<int>(vk::Format::eR8G8B8A8Snorm),

    // 8-bit unsigned integer
    R8_UInt     = static_cast<int>(vk::Format::eR8Uint),
    RG8_UInt    = static_cast<int>(vk::Format::eR8G8Uint),
    RGB8_UInt   = static_cast<int>(vk::Format::eR8G8B8Uint),
    RGBA8_UInt  = static_cast<int>(vk::Format::eR8G8B8A8Uint),

    // 8-bit signed integer
    R8_SInt     = static_cast<int>(vk::Format::eR8Sint),
    RG8_SInt    = static_cast<int>(vk::Format::eR8G8Sint),
    RGB8_SInt   = static_cast<int>(vk::Format::eR8G8B8Sint),
    RGBA8_SInt  = static_cast<int>(vk::Format::eR8G8B8A8Sint),

    // 16-bit unsigned normalized
    R16_UNorm    = static_cast<int>(vk::Format::eR16Unorm),
    RG16_UNorm   = static_cast<int>(vk::Format::eR16G16Unorm),
    RGB16_UNorm  = static_cast<int>(vk::Format::eR16G16B16Unorm),
    RGBA16_UNorm = static_cast<int>(vk::Format::eR16G16B16A16Unorm),

    // 16-bit signed normalized
    R16_SNorm    = static_cast<int>(vk::Format::eR16Snorm),
    RG16_SNorm   = static_cast<int>(vk::Format::eR16G16Snorm),
    RGB16_SNorm  = static_cast<int>(vk::Format::eR16G16B16Snorm),
    RGBA16_SNorm = static_cast<int>(vk::Format::eR16G16B16A16Snorm),

    // 16-bit unsigned integer
    R16_UInt     = static_cast<int>(vk::Format::eR16Uint),
    RG16_UInt    = static_cast<int>(vk::Format::eR16G16Uint),
    RGB16_UInt   = static_cast<int>(vk::Format::eR16G16B16Uint),
    RGBA16_UInt  = static_cast<int>(vk::Format::eR16G16B16A16Uint),

    // 16-bit signed integer
    R16_SInt     = static_cast<int>(vk::Format::eR16Sint),
    RG16_SInt    = static_cast<int>(vk::Format::eR16G16Sint),
    RGB16_SInt   = static_cast<int>(vk::Format::eR16G16B16Sint),
    RGBA16_SInt  = static_cast<int>(vk::Format::eR16G16B16A16Sint),

    // 32-bit unsigned integer
    R32_UInt     = static_cast<int>(vk::Format::eR32Uint),
    RG32_UInt    = static_cast<int>(vk::Format::eR32G32Uint),
    RGB32_UInt   = static_cast<int>(vk::Format::eR32G32B32Uint),
    RGBA32_UInt  = static_cast<int>(vk::Format::eR32G32B32A32Uint),

    // 32-bit signed integer
    R32_SInt     = static_cast<int>(vk::Format::eR32Sint),
    RG32_SInt    = static_cast<int>(vk::Format::eR32G32Sint),
    RGB32_SInt   = static_cast<int>(vk::Format::eR32G32B32Sint),
    RGBA32_SInt  = static_cast<int>(vk::Format::eR32G32B32A32Sint),

    // 64-bit unsigned integer
    R64_UInt     = static_cast<int>(vk::Format::eR64Uint),
    RG64_UInt    = static_cast<int>(vk::Format::eR64G64Uint),
    RGB64_UInt   = static_cast<int>(vk::Format::eR64G64B64Uint),
    RGBA64_UInt  = static_cast<int>(vk::Format::eR64G64B64A64Uint),

    // 64-bit signed integer
    R64_SInt     = static_cast<int>(vk::Format::eR64Sint),
    RG64_SInt    = static_cast<int>(vk::Format::eR64G64Sint),
    RGB64_SInt   = static_cast<int>(vk::Format::eR64G64B64Sint),
    RGBA64_SInt  = static_cast<int>(vk::Format::eR64G64B64A64Sint),

    // 16-bit float
    R16_Float    = static_cast<int>(vk::Format::eR16Sfloat),
    RG16_Float   = static_cast<int>(vk::Format::eR16G16Sfloat),
    RGB16_Float  = static_cast<int>(vk::Format::eR16G16B16Sfloat),
    RGBA16_Float = static_cast<int>(vk::Format::eR16G16B16A16Sfloat),

    // 32-bit float
    R32_Float    = static_cast<int>(vk::Format::eR32Sfloat),
    RG32_Float   = static_cast<int>(vk::Format::eR32G32Sfloat),
    RGB32_Float  = static_cast<int>(vk::Format::eR32G32B32Sfloat),
    RGBA32_Float = static_cast<int>(vk::Format::eR32G32B32A32Sfloat),

    // 64-bit float
    R64_Float    = static_cast<int>(vk::Format::eR64Sfloat),
    RG64_Float   = static_cast<int>(vk::Format::eR64G64Sfloat),
    RGB64_Float  = static_cast<int>(vk::Format::eR64G64B64Sfloat),
    RGBA64_Float = static_cast<int>(vk::Format::eR64G64B64A64Sfloat),
};

enum class ResourceType : int {
    UNDEFINED = 0,
    // Buffer used for vertex, indices, instances, structs, uniforms, etc.
    BUFFER = 1,
    // Images 1D, 2D or 3D with a specific format.
    IMAGE = 2
};

enum class vk_CommandBufferInternalState {
    CREATED = 0,
    RECORDING = 1,
    EXECUTABLE = 2,
    SUBMITTED = 3
};

enum class EngineType : int {
    NONE = 0,
    GRAPHICS = 1,
    COMPUTE = 2,
    TRANSFER = 4,
};

enum class PipelineType : int {
    COMPUTE = 0,
    RASTERIZATION = 1,
    RAYTRACING = 2
};

enum class ShaderStageType : int {
    VERTEX = 0,
    FRAGMENT = 1,
    GEOMETRY = 2,
    TESS_CONTROL = 3,
    TESS_EVAL = 4,
    COMPUTE = 5,
};

enum class DescriptorType : int {
    STORAGE_BUFFER = 0,
    UNIFORM_BUFFER = 1,
    STORAGE_IMAGE = 2,
    SAMPLED_IMAGE = 3,
    SAMPLER = 4,
    COMBINED_IMAGE_SAMPLER = 5,
    // Not yet bindable via DescriptorSet::bind (no acceleration structure
    // wrapper exists yet); declaring a binding of this type is allowed for
    // forward-compatibility but writing to it will throw.
    ACCELERATION_STRUCTURE = 6,
};

// Texel filtering used by CommandBuffer::blit_image when src/dst extents
// differ. Values match vk::Filter's own, so a plain static_cast suffices.
enum class Filter : int {
    NEAREST = 0,
    LINEAR = 1,
};

// Opaque handle returned by Pipeline::layout(), identifying one declared
// descriptor set binding. Not constructible from Python: only meaningful
// when passed back to DescriptorSet::bind() on a descriptor set from the
// same Pipeline.
class LayoutHandle {
public:
    explicit LayoutHandle(int id) noexcept : id_(id) {}
    [[nodiscard]] int vk_id() const noexcept { return id_; }
private:
    int id_;
};

// Opaque handle returned by Pipeline::attach(), identifying one declared
// color attachment slot. Not constructible from Python: only meaningful
// when passed back to Pipeline::create_framebuffer() on the same Pipeline.
class AttachHandle {
public:
    explicit AttachHandle(int slot) noexcept : slot_(slot) {}
    [[nodiscard]] int vk_slot() const noexcept { return slot_; }
private:
    int slot_;
};

enum class TypeKind : int {
    SCALAR = 0,
    VECTOR = 1,
    MATRIX = 2,
    ARRAY = 3,
    STRUCT = 4
};

enum class LayoutRule : int {
    Std140 = 0,
    Std430 = 1,
    Scalar = 2
};

// When Device::wrap() must copy the wrapped object's data through a
// temporary buffer (see WrappedMemory), determines when that copy happens.
// Enumerator names avoid the literal IN/OUT (Windows headers #define these
// as empty parameter-annotation macros); exposed to Python as IN/OUT/INOUT
// regardless (see bindings.cpp).
enum class WrapMode : int {
    CopyIn = 0,    // Copy the source in at wrap() time; never copied back.
    CopyOut = 1,   // Never copied in; copied back to the source when the WrappedMemory is destroyed.
    CopyInOut = 2, // Both.
};

/*  ============== STRUCTS ============== */

struct DLDevice {
    std::int32_t device_type;
    std::int32_t device_id;
};

struct ResourceSlice{
    ResourceType type;
    union {
        struct {
            std::uint64_t offset;
            std::uint64_t size;
        } buffer;
        struct {
            Format format;
            int mip_start;
            int mip_count;
            int array_start;
            int array_count;
        } image;
    };
};

// Payload structs for TypeDescriptor's flat hierarchy (see TypeDescriptor below).
struct ScalarDesc {
    ScalarType type;
};

struct VectorDesc {
    ScalarType component_type;
    int components; // in {2, 3, 4}
};

struct MatrixDesc {
    ScalarType component_type;
    int rows;    // in {2, 3, 4}
    int columns; // in {2, 3, 4}; column-major
};

struct ArrayDesc {
    std::shared_ptr<TypeDescriptor> element_type;
    std::uint64_t count; // 0 => unsized/runtime array
};

struct StructField {
    std::string name;
    std::shared_ptr<TypeDescriptor> type;
};

struct StructDesc {
    std::vector<StructField> fields; // ordered
};

// A single field of a computed struct Layout: its name, byte offset within
// the struct, and the Layout of its own type.
struct LayoutField {
    std::string name;
    std::uint64_t offset;
    std::shared_ptr<Layout> layout;
    // The outermost Layout this field was computed as part of (itself, if this
    // field belongs to a top-level struct passed to compute_layout()). Lets
    // Buffer::field(field) resolve strides across array elements without the
    // caller having to separately track which Layout a field came from.
    std::weak_ptr<Layout> root;
};

// One binding declared via Pipeline::layout(). The id returned by that
// method is the index of this entry in the owning vk_Pipeline's bindings.
struct vk_DescriptorBinding {
    int set;
    int binding;
    DescriptorType type;
    int count;
};

// One color attachment declared via Pipeline::attach(), identified by the
// fragment shader output location (`slot`), not by render-pass attachment
// index (attachments may be declared out of order).
struct vk_AttachmentDesc {
    int slot;
    Format format;
};

/*  ============== FUNCTIONS ============== */

int scalar_type_size(ScalarType type);

// Computes the byte size/alignment layout of `type` under `rule` (std140,
// std430 or scalar block layout rules). Pure host-side arithmetic, does not
// require a Device.
std::shared_ptr<Layout> compute_layout(const TypeDescriptor& type, LayoutRule rule);

/*  ============== CLASSES ============== */

/**
 * Describes the shape of a GPU-buffer-resident type (scalar, vector, matrix,
 * array or struct) as a flat hierarchy via std::variant. Used together with
 * LayoutRule and compute_layout() to determine byte offsets/sizes/strides
 * without touching the GPU.
 */
class TypeDescriptor {
public:
    using Payload = std::variant<ScalarDesc, VectorDesc, MatrixDesc, ArrayDesc, StructDesc>;

    static TypeDescriptor scalar(ScalarType type);
    // Throws std::runtime_error if components is not in [2, 4].
    static TypeDescriptor vector(ScalarType component_type, int components);
    // Throws std::runtime_error if rows or columns is not in [2, 4].
    static TypeDescriptor matrix(ScalarType component_type, int rows, int columns);
    // count == 0 denotes an unsized/runtime array.
    static TypeDescriptor array_of(std::shared_ptr<TypeDescriptor> element_type, std::uint64_t count);
    static TypeDescriptor struct_of(std::vector<StructField> fields);

    [[nodiscard]] TypeKind kind() const noexcept;
    [[nodiscard]] const Payload& payload() const noexcept { return payload_; }
private:
    Payload payload_;
};

/**
 * Computed byte layout of a TypeDescriptor under a given LayoutRule: size,
 * alignment and, depending on the type's kind, per-field offsets (STRUCT) or
 * element stride/count (ARRAY/MATRIX).
 */
class Layout {
public:
    std::uint64_t size = 0;
    std::uint64_t alignment = 0;
    // Size this layout would occupy as one element of an array of itself,
    // i.e. size rounded up to alignment (and, under Std140, up to 16) --
    // the byte stride between consecutive elements. Used both to size
    // ARRAY/MATRIX strides and by Device::create_buffer(layout, ..., count > 1).
    std::uint64_t aligned_size = 0;
    TypeKind kind = TypeKind::SCALAR;
    // SCALAR/VECTOR/MATRIX only: the scalar component type (UNDEFINED for
    // ARRAY/STRUCT, since those don't have one leaf type of their own).
    ScalarType component_type = ScalarType::UNDEFINED;

    // STRUCT only: ordered fields with their offsets and sub-layouts.
    std::vector<LayoutField> fields;

    // ARRAY/MATRIX only: layout of a single element (ARRAY) or column (MATRIX),
    // the byte stride between consecutive elements/columns, and their count.
    std::shared_ptr<Layout> element_layout;
    std::uint64_t stride = 0;
    std::uint64_t count = 0;
};

/**
 * Wraps a vulkan memory slice and a resource associated with it.
 */
class vk_ResourceData : public std::enable_shared_from_this<vk_ResourceData>{
public:
    // `owns_image` is false only for a swapchain image (see vk_Window):
    // its vk::Image is owned/destroyed by the swapchain itself, so
    // dispose() must not call vkDestroyImage on it (and there is no
    // dedicated_image_memory to free either, in that case).
    vk_ResourceData(std::shared_ptr<Device> device, const std::shared_ptr<MemorySlice>& memory,
        vk::Image image, vk::ImageCreateInfo image_info, vk::DeviceMemory dedicated_image_memory = nullptr,
        bool owns_image = true);
    vk_ResourceData(const vk_ResourceData&) = delete;
    vk_ResourceData& operator=(const vk_ResourceData&) = delete;
    vk_ResourceData(vk_ResourceData&& other) noexcept = delete;
    vk_ResourceData& operator=(vk_ResourceData&& other) noexcept = delete;
    ~vk_ResourceData()  noexcept;
    void dispose();
    std::shared_ptr<Device> device() const noexcept { return device_; }
    vk::Buffer get_buffer() const noexcept { return buffer_; }
    vk::Image get_image() const noexcept { return image_; }
    vk::ImageCreateInfo get_image_info() const noexcept { return image_info_; }
    ResourceType resource_type() const noexcept { return resource_type_; }
    std::uint64_t device_ptr() const;
    std::uint64_t external_ptr() const;
    bool is_cpu() const noexcept;
    std::shared_ptr<MemorySlice> get_memory() const noexcept { return memory_; }
private:
    std::shared_ptr<Device> device_;
    vk::Buffer buffer_;
    vk::Image image_;
    vk::ImageCreateInfo image_info_;
    std::shared_ptr<MemorySlice> memory_;
    ResourceType resource_type_;
    // Only set for an IMAGE resource: images get their own dedicated
    // vk::DeviceMemory allocation (unlike buffers, which sub-allocate from a
    // shared MemoryPage's single big vk::Buffer via `memory_`), freed here
    // in dispose() since MemorySlice has no notion of it.
    vk::DeviceMemory dedicated_image_memory_;
    // False only for a swapchain image: dispose() must not destroy image_
    // (or dedicated_image_memory_, which is null in that case anyway),
    // since the swapchain owns it.
    bool owns_image_ = true;
};

class Resource : public std::enable_shared_from_this<Resource>{
public:
    Resource(std::shared_ptr<vk_ResourceData> resource_data, ResourceSlice view_slice);
    Resource() = delete;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&& other) noexcept = delete;
    Resource& operator=(Resource&& other) noexcept = delete;
    virtual ~Resource() noexcept;
    void dispose();
protected:
    std::shared_ptr<vk_ResourceData> data_;
    ResourceSlice slice_{};
};

class Buffer: public Resource {
public:
    // `layout` describes a single element of this buffer: the scalar type
    // (Device::create_buffer(elements, ScalarType, ...)), an array of a
    // format's per-channel scalar (Device::create_buffer(elements, Format,
    // ...)), or the passed-in Layout as-is (Device::create_buffer(layout,
    // ...)). Every Buffer has one -- there is no "untyped" buffer. `format`
    // is only set (non-Undefined) when constructed via a Format-based path;
    // it exists solely for get_view() (a real texel vk::BufferView needs the
    // exact Format, e.g. UNorm vs UInt, which element_layout() alone can't
    // recover once reduced to a per-channel scalar type).
    Buffer(std::shared_ptr<vk_ResourceData> resource_data, ResourceSlice view_slice, std::shared_ptr<Layout> layout, Format format = Format::Undefined);
    Buffer() = delete;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept = delete;
    Buffer& operator=(Buffer&& other) noexcept = delete;
    ~Buffer() noexcept override;
    std::uint64_t size() const noexcept { return slice_.buffer.size; }
    // Layout of a single element of this buffer (see the constructor).
    // aligned_size is the per-element byte stride used by slice()/element();
    // kind determines how vk_dlpack() exports this buffer's shape/dtype.
    [[nodiscard]] const std::shared_ptr<Layout>& element_layout() const noexcept { return layout_; }

    // Reinterprets this buffer's bytes (same underlying offset/size) as a
    // flat array of `scalar` elements.
    std::shared_ptr<Buffer> cast(ScalarType scalar) const;
    // Reinterprets this buffer's bytes as a flat array of `format` texels
    // (internally, an array of the format's own per-channel scalar type).
    std::shared_ptr<Buffer> cast(Format format) const;
    // Reinterprets this buffer's bytes as being laid out per `layout`.
    std::shared_ptr<Buffer> cast(const std::shared_ptr<Layout>& layout) const;

    // Slices this buffer to `count` elements starting at `start_element`,
    // using element_layout()->aligned_size as the byte size of one element.
    // Throws if the requested range is out of bounds.
    std::shared_ptr<Buffer> slice(std::uint64_t start_element, std::uint64_t count) const;
    // A single-element slice: equivalent to slice(index, 1).
    std::shared_ptr<Buffer> element(std::uint64_t index) const;

    // Exports this buffer as a dlpack tensor, using element_layout() to
    // determine its shape/dtype (falling back to raw bytes if
    // element_layout()->kind is STRUCT, since a struct isn't a single
    // numeric type).
    pybind11::object vk_dlpack() const;
    DLDevice vk_dlpack_device() const noexcept;

    // Exports `field`, repeated across every instance of its root layout in
    // this buffer, as a single strided DLPack tensor. Only valid if
    // element_layout()->kind is STRUCT. Throws std::runtime_error if
    // `field`'s type is (or contains) a STRUCT, since a struct isn't
    // representable as a single numeric tensor.
    pybind11::object field(const LayoutField& field) const;

    // Reads a scalar/vector/matrix/array-of-scalars `field` directly via
    // memcpy, bypassing DLPack -- much cheaper for small (e.g. scalar)
    // fields. Only valid if element_layout()->kind is STRUCT, on
    // host-visible memory holding exactly one instance of `field`'s root
    // layout (buffer size must equal `field.root.aligned_size`). Returns a
    // plain Python number for a scalar field, or a tightly-packed buffer
    // object (bytes) for a vector/matrix/array-of-scalars field.
    pybind11::object read(const LayoutField& field) const;

    // Writes into a scalar/vector/matrix/array-of-scalars `field` directly
    // via memcpy, bypassing DLPack. Same buffer requirements as read().
    // `value` must be a plain Python number for a scalar field; for
    // vector/matrix/array-of-scalars fields it must be a Python object
    // supporting the buffer protocol (e.g. a numpy array) or the DLPack
    // protocol (e.g. a CPU torch tensor).
    void write(const LayoutField& field, const pybind11::object& value) const;

    // Copies `source`'s data into this buffer's own memory as a whole
    // (unlike write(), which targets a single struct field): a shortcut
    // for the "torch.from_dlpack(buffer).copy_(source)" pattern, using the
    // same DLPack/buffer-protocol handling as Device::wrap() (so a CUDA
    // source works via the interop plugin even when this buffer is
    // DEVICE-located). `source`'s total byte size must equal size().
    void load(const pybind11::object& source) const;
    // Symmetric to load(): copies this buffer's own memory, as a whole,
    // into `target` (a DLPack-compatible or Python buffer-protocol object,
    // already sized to receive size() bytes).
    void save(const pybind11::object& target) const;

    vk::BufferView get_view();
    std::uint64_t device_ptr() const;
    std::uint64_t external_ptr() const;

    // Underlying vk::Buffer this view slices into. Used internally by
    // CommandBuffer::transfer, not exposed to Python.
    [[nodiscard]] vk::Buffer vk_buffer() const noexcept { return data_->get_buffer(); }
    // Byte offset of this view within vk_buffer(). Used internally by
    // CommandBuffer::transfer, not exposed to Python.
    [[nodiscard]] std::uint64_t vk_buffer_offset() const noexcept;
    [[nodiscard]] std::uint64_t vk_buffer_size() const noexcept;
private:
    // Throws unless element_layout()->kind is STRUCT: shared by field(),
    // read() and write(), which only make sense on a buffer whose own
    // element layout is a struct (e.g. from Device::create_buffer(layout, ...)).
    void vk_require_struct_layout(const char* method_name) const;
    // Validates that this buffer is usable with read()/write() (STRUCT
    // element layout, host-visible memory, holding exactly one instance of
    // `field`'s root layout) and resolves `field`'s absolute host pointer.
    // Throws otherwise.
    char* vk_resolve_field_ptr(const LayoutField& field) const;

    vk::BufferView buffer_view_;
    std::shared_ptr<Layout> layout_;
    Format format_ = Format::Undefined;
};

/**
 * A plain N-dimensional array resource of a single scalar type, obtained
 * from Device::create_tensor. Unlike Buffer (whose shape/dtype is derived
 * from element_layout(), falling back to raw bytes for a struct), a
 * Tensor's shape is exactly what was requested, always exported as a
 * single typed DLPack tensor -- this is the type meant to be hand off
 * directly to torch.from_dlpack(). A resource like any other: tracked for
 * proactive disposal the same way as Buffer, and directly wrappable via
 * Device::wrap() with no copy.
 */
class Tensor: public Resource {
public:
    Tensor(std::shared_ptr<vk_ResourceData> resource_data, ResourceSlice view_slice,
        std::vector<std::uint64_t> shape, ScalarType scalar_type);
    Tensor() = delete;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept = delete;
    Tensor& operator=(Tensor&& other) noexcept = delete;
    ~Tensor() noexcept override = default;

    [[nodiscard]] std::uint64_t size() const noexcept { return slice_.buffer.size; }
    [[nodiscard]] const std::vector<std::uint64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] ScalarType scalar_type() const noexcept { return scalar_type_; }

    // Exports this tensor as a dlpack tensor, using shape()/scalar_type()
    // directly (no element_layout() struct-fallback logic, unlike
    // Buffer::vk_dlpack() -- a Tensor's shape is always exactly what was
    // requested at creation time).
    pybind11::object vk_dlpack() const;
    DLDevice vk_dlpack_device() const noexcept;

    // Same idea as Buffer::load()/save(): copies a whole DLPack-compatible
    // or Python buffer-protocol object into/out of this tensor's own
    // memory, using the same handling as Device::wrap(). `source`'s/
    // `target`'s total byte size must equal size().
    void load(const pybind11::object& source) const;
    void save(const pybind11::object& target) const;

    std::uint64_t device_ptr() const noexcept { return data_->device_ptr() + slice_.buffer.offset; }
    std::uint64_t external_ptr() const noexcept { return data_->external_ptr() + slice_.buffer.offset; }

    // Underlying vk::Buffer this view slices into. Used internally by
    // Device::wrap, not exposed to Python.
    [[nodiscard]] vk::Buffer vk_buffer() const noexcept { return data_->get_buffer(); }
    [[nodiscard]] std::uint64_t vk_buffer_offset() const noexcept;
private:
    std::vector<std::uint64_t> shape_;
    ScalarType scalar_type_;
};

/**
 * A device-usable view over externally-owned memory, obtained from
 * Device::wrap(). If the wrapped object could be used directly (one of
 * this library's own Buffers; an already-contiguous DLPack tensor whose
 * memory happens to already be one of this Device's own allocations; or a
 * host-visible Python buffer-protocol object when the requested location
 * is HOST), this simply forwards its pointer -- no copy is made, and none
 * is made on destruction either. Otherwise it owns a temporary Buffer:
 * WrapMode::CopyIn/CopyInOut copies the source into it at wrap() time;
 * WrapMode::CopyOut/CopyInOut copies it back into the source when this
 * WrappedMemory is destroyed.
 */
class WrappedMemory {
public:
    // Only meaningful internally, to know how (or whether) to copy back on
    // destruction; not exposed to Python.
    enum class SourceKind : int {
        NONE = 0,            // Directly mapped: no copy-back.
        DLPACK = 1,           // Copy back via `source`'s __dlpack__() export.
        BUFFER_PROTOCOL = 2,  // Copy back via `source`'s buffer-protocol memory.
    };

    WrappedMemory(
        std::weak_ptr<Device> device,
        std::uint64_t device_ptr,
        std::vector<std::uint64_t> shape,
        ScalarType scalar_type,
        std::shared_ptr<Buffer> owned_buffer,
        MemoryLocation owned_location,
        WrapMode mode,
        SourceKind source_kind,
        pybind11::object source) noexcept;
    WrappedMemory() = delete;
    WrappedMemory(const WrappedMemory&) = delete;
    WrappedMemory& operator=(const WrappedMemory&) = delete;
    WrappedMemory(WrappedMemory&& other) noexcept = delete;
    WrappedMemory& operator=(WrappedMemory&& other) noexcept = delete;
    ~WrappedMemory() noexcept;

    // Performs, immediately, whatever copy-back this wrap's WrapMode calls
    // for (OUT/INOUT), rather than leaving it to whenever the Python
    // garbage collector gets around to destroying this object. Safe to
    // call more than once -- only the first call has any effect. Unlike
    // the destructor, exceptions raised while copying back propagate to
    // the caller instead of being swallowed.
    void unwrap();

    // Vulkan buffer device address of this wrapped memory -- a raw
    // pointer usable from within a shader (e.g. through
    // GL_EXT_buffer_reference), not to be confused with any
    // external-memory-exported (e.g. CUDA) pointer.
    [[nodiscard]] std::uint64_t device_ptr() const noexcept { return device_ptr_; }
    [[nodiscard]] const std::vector<std::uint64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] ScalarType scalar_type() const noexcept { return scalar_type_; }
private:
    std::weak_ptr<Device> device_;
    std::uint64_t device_ptr_;
    std::vector<std::uint64_t> shape_;
    ScalarType scalar_type_;
    // Non-null only when a temporary allocation backs device_ptr_ (i.e. the
    // source couldn't be used directly); keeps it alive and is the source
    // of the OUT/INOUT copy-back.
    std::shared_ptr<Buffer> owned_buffer_;
    // Where owned_buffer_ was allocated (HOST or DEVICE); irrelevant when
    // owned_buffer_ is null. Needed at copy-back time since a wrap()
    // request can ask for either location regardless of the source's own.
    MemoryLocation owned_location_;
    WrapMode mode_;
    SourceKind source_kind_;
    // The original wrapped Python object; kept alive so its __dlpack__()/
    // buffer-protocol export can be revisited for the copy-back. Heap
    // allocated since pybind11::object (only forward-declared here) can't
    // be a direct member of a class defined in this header.
    std::unique_ptr<pybind11::object> source_;
    // Set by unwrap() so it (and the destructor, which calls it) only
    // performs the copy-back/release once.
    bool unwrapped_ = false;
};

class Image: public Resource {
public:
    Image(std::shared_ptr<vk_ResourceData> resource_data, ResourceSlice view_slice);
    Image() = delete;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept = delete;
    Image& operator=(Image&& other) noexcept = delete;
    ~Image() noexcept override;
    Format format() const noexcept { return slice_.image.format; }
    int mip_count() const noexcept { return slice_.image.mip_count; }
    int array_count() const noexcept { return slice_.image.array_count; }
    std::shared_ptr<Image> cast_format(Format new_format) const;
    std::shared_ptr<Image> slice(int mip_start, int mip_count, int array_start, int array_count) const;
    vk::ImageView get_view();
    // Full creation-time description of the backing image (extent, mip/array
    // counts, usage, etc). Used internally by Pipeline::create_framebuffer to
    // recover this image's dimensions, not exposed to Python.
    [[nodiscard]] vk::ImageCreateInfo vk_image_info() const noexcept { return data_->get_image_info(); }
    // Underlying vk::Image and this view's base mip/array layer. Used
    // internally by CommandBuffer::blit_image, not exposed to Python.
    [[nodiscard]] vk::Image vk_image() const noexcept { return data_->get_image(); }
    [[nodiscard]] int vk_mip_start() const noexcept { return slice_.image.mip_start; }
    [[nodiscard]] int vk_array_start() const noexcept { return slice_.image.array_start; }
private:
    vk::ImageView image_view_;
};

class SubmittedTask {
public:
    SubmittedTask(std::weak_ptr<vk_Engine> engine, std::uint64_t submission_id);
    SubmittedTask() = delete;
    SubmittedTask(const SubmittedTask&) = delete;
    SubmittedTask& operator=(const SubmittedTask&) = delete;
    SubmittedTask(SubmittedTask&& other) noexcept = delete;
    SubmittedTask& operator=(SubmittedTask&& other) noexcept = delete;
    ~SubmittedTask() noexcept = default;
    void wait();
    bool is_complete();
    void vk_notify_completion();
    std::uint64_t vk_submission_id() const noexcept { return submission_id_; }
    void vk_attach_command_buffers(std::vector<std::shared_ptr<CommandBuffer>> command_buffers) {
        submitted_ = std::move(command_buffers);
    }
private:
    std::weak_ptr<vk_Engine> engine_;
    std::vector<std::shared_ptr<CommandBuffer>> submitted_;
    std::uint64_t submission_id_;
};

/**
 * Wraps a single vk::CommandBuffer allocated from a QueueWrapper.
 * Not exposed to users directly: they interact with it through a
 * CommandBuffer, obtained from a Queue.
 */
class vk_CommandBuffer : public std::enable_shared_from_this<vk_CommandBuffer> {
public:
    vk_CommandBuffer(vk::CommandBuffer command_buffer) noexcept;
    vk_CommandBuffer() = delete;
    vk_CommandBuffer(const vk_CommandBuffer&) = delete;
    vk_CommandBuffer& operator=(const vk_CommandBuffer&) = delete;
    vk_CommandBuffer(vk_CommandBuffer&& other) noexcept = delete;
    vk_CommandBuffer& operator=(vk_CommandBuffer&& other) noexcept = delete;
    ~vk_CommandBuffer() noexcept { state = vk_CommandBufferInternalState::CREATED; command_buffer = nullptr; }

    vk_CommandBufferInternalState state;
    vk::CommandBuffer command_buffer;
    void begin(); // CREATED -> RECORDING (Begin) | EXECUTABLE -> RECORDING (Reset)
    void end(); // RECORDING -> EXECUTABLE (End)
    void notify_submitted(); // EXECUTABLE -> SUBMITTED
    void notify_executed(); // SUBMITTED -> EXECUTABLE
};

/**
 * User-facing handle to a CommandBuffer. This is the only command-recording
 * type users are meant to hold: it hides the fact that the underlying
 * vk::CommandBuffer is allocated from, and recycled by, a CommandPool.
 */
class CommandBuffer : public std::enable_shared_from_this<CommandBuffer> {
public:
    CommandBuffer(std::shared_ptr<Device> device, std::shared_ptr<Engine> engine, std::shared_ptr<vk_CommandBuffer> command_buffer) noexcept : device_(std::move(device)), engine_(std::move(engine)), command_buffer_(std::move(command_buffer)) {}
    CommandBuffer() = delete;
    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;
    CommandBuffer(CommandBuffer&& other) noexcept = delete;
    CommandBuffer& operator=(CommandBuffer&& other) noexcept = delete;
    ~CommandBuffer() noexcept { release(); }

    // Records a device-side copy from `source` into `destination`.
    void transfer(const std::shared_ptr<Buffer>& source, const std::shared_ptr<Buffer>& destination);

    // Binds `pipeline` for subsequent bind()/dispatch_threads() (compute) or
    // draw commands (graphics). `pipeline` must already be closed. Keeps a
    // reference to `pipeline` alive for as long as this command buffer is,
    // including while submitted and pending on the GPU.
    void set_pipeline(const std::shared_ptr<Pipeline>& pipeline);

    // Binds `descriptor_sets` at consecutive set indices starting at
    // `initial_set`, against the pipeline layout of the pipeline most
    // recently bound via set_pipeline(). Keeps references to
    // `descriptor_sets` alive for as long as this command buffer is,
    // including while submitted and pending on the GPU.
    void bind(int initial_set, const std::vector<std::shared_ptr<DescriptorSet>>& descriptor_sets);

    // Records a compute dispatch covering at least (x, y, z) threads, using
    // the pipeline most recently bound via set_pipeline(): the number of
    // workgroups along each dimension is ceil(threads / local_size), using
    // that pipeline's declared Pipeline::local_size() (default (1, 1, 1) if
    // never set). Requires a COMPUTE pipeline to be bound.
    void dispatch_threads(std::uint32_t x, std::uint32_t y = 1, std::uint32_t z = 1);

    // Begins rendering into `framebuffer`'s render pass, ending any
    // previously active one first -- a command buffer may switch render
    // targets multiple times while recording, same as pipelines/descriptor
    // sets. Keeps a reference to `framebuffer` alive for as long as this
    // command buffer is, including while submitted and pending on the GPU.
    // Graphics pipelines only.
    void set_framebuffer(const std::shared_ptr<Framebuffer>& framebuffer);

    // Sets the dynamic viewport and scissor rectangle (identically) used by
    // subsequent draw calls. Graphics pipelines only.
    void set_viewport(float x, float y, float width, float height);

    // Records a device-side blit (resizing/format-converting as needed,
    // using `filter` for magnification/minification) from `src` into `dst`.
    // Both images are assumed to be in VK_IMAGE_LAYOUT_GENERAL (the layout
    // every Image is transitioned to at creation -- see Image's
    // docstring). Cannot be called while a render pass is active (i.e.
    // between set_framebuffer() and the next set_framebuffer()/close()).
    void blit_image(const std::shared_ptr<Image>& src, const std::shared_ptr<Image>& dst, Filter filter = Filter::LINEAR);

    // Binds `vertex_buffer` at vertex input binding `binding`, matching a
    // Pipeline::vertex_layout() declaration. Keeps a reference to
    // `vertex_buffer` alive for as long as this command buffer is,
    // including while submitted and pending on the GPU.
    void bind_vertices(int binding, const std::shared_ptr<Buffer>& vertex_buffer);

    // Binds `index_buffer` for subsequent dispatch_indexed_primitives()
    // calls. Its element_layout() must be a scalar UINT16 or UINT32. Keeps
    // a reference to `index_buffer` alive for as long as this command
    // buffer is, including while submitted and pending on the GPU.
    void bind_indices(const std::shared_ptr<Buffer>& index_buffer);

    // Records a non-indexed draw of `vertices` vertices, starting at
    // `vertex_start` within the vertex buffers bound via bind_vertices().
    // Requires a RASTERIZATION pipeline to be bound and an active render
    // pass (set_framebuffer()).
    void dispatch_primitives(std::uint32_t vertices, std::uint32_t vertex_start = 0);

    // Records an indexed draw of `indices` indices, starting at
    // `index_start` within the index buffer bound via bind_indices();
    // `vertex_offset` is added to every fetched index before it's used to
    // read vertex attributes. Requires bind_indices() to have been called,
    // a RASTERIZATION pipeline to be bound, and an active render pass
    // (set_framebuffer()).
    void dispatch_indexed_primitives(std::uint32_t indices, std::uint32_t index_start = 0, std::int32_t vertex_offset = 0);

    // ends recording (and any still-active render pass) and submits the command buffer for execution on its engine.
    void close();
    // the command buffer is not going to be used for submission anymore. It is safe to destroy or reuse.
    void release() ;

    /**
     * Determines if the object is locked by the gpu.
     * @return True if the underlying command buffer is assumed to be used on the gpu. SubmissionTask object can be used to check termination.
     */
    [[nodiscard]] bool is_submitted() const noexcept {
        if (is_released()) return false;
        return command_buffer_->state == vk_CommandBufferInternalState::SUBMITTED;
    }

    /**
     * Determines if the object is ready for submission.
     * @return True if the underlying command buffer is granted to be on executable state.
     */
    [[nodiscard]] bool is_executable() const noexcept {
        if (is_released()) return false;
        return command_buffer_->state == vk_CommandBufferInternalState::EXECUTABLE;
    }

    /**
     * Determines if the object is already disposed.
     * @return True if the underlying command buffer is disposed.
     */
    [[nodiscard]] bool is_released() const noexcept { return device_ == nullptr; }

    [[nodiscard]] bool is_closed() const noexcept {
        if (is_released()) return true;
        return command_buffer_->state != vk_CommandBufferInternalState::RECORDING;
    }

    [[nodiscard]] std::shared_ptr<vk_CommandBuffer> vk_command_buffer() const {
        return command_buffer_;
    }
private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<Engine> engine_;
    std::shared_ptr<vk_CommandBuffer> command_buffer_;
    // Every Pipeline/DescriptorSet ever bound via set_pipeline()/bind()
    // during this recording, kept alive for as long as this command buffer
    // is (including while submitted and pending on the GPU), since the
    // already-recorded commands reference their raw Vulkan handles directly
    // -- a command buffer may switch pipelines multiple times while
    // recording, so a single slot isn't enough. bound_pipelines_.back() is
    // the currently active pipeline (used by bind()/dispatch_threads()).
    std::vector<std::shared_ptr<Pipeline>> bound_pipelines_;
    std::vector<std::shared_ptr<DescriptorSet>> bound_descriptor_sets_;
    // Framebuffers (set_framebuffer), vertex/index buffers (bind_vertices/
    // bind_indices) and images (blit_image) ever referenced during this
    // recording, kept alive the same way and for the same reason as
    // bound_pipelines_/bound_descriptor_sets_ above.
    std::vector<std::shared_ptr<Framebuffer>> bound_framebuffers_;
    std::vector<std::shared_ptr<Buffer>> bound_vertex_index_buffers_;
    std::vector<std::shared_ptr<Image>> bound_images_;
    // Whether a render pass begun by set_framebuffer() is still active
    // (i.e. hasn't been ended by a later set_framebuffer() or close()).
    bool render_pass_active_ = false;
};

/**
 * Owns a vk::CommandPool and the command buffers allocated from it.
 * Not exposed to users directly: Device creates and owns it, and hands out
 * a CommandPoolManager to interact with it.
 */
class vk_Engine : public std::enable_shared_from_this<vk_Engine> {
public:
    vk_Engine(std::weak_ptr<Device> device, vk::Queue queue, vk::CommandPool pool, EngineType type);
    vk_Engine() = delete;
    vk_Engine(const vk_Engine&) = delete;
    vk_Engine& operator=(const vk_Engine&) = delete;
    vk_Engine(vk_Engine&& other) noexcept = delete;
    vk_Engine& operator=(vk_Engine&& other) noexcept = delete;
    ~vk_Engine() noexcept { dispose(); }
    void dispose() noexcept;
    // Tears down using `dev` directly instead of device_.lock(). Used by
    // Device::dispose() -- see vk_Window::vk_dispose_with's docstring for
    // why this can't simply be device_.lock()->logical_device().
    void vk_dispose_with(vk::Device dev) noexcept;
    // creates/reuse a command buffer
    std::shared_ptr<vk_CommandBuffer> create_command_buffer();
    // sets a command buffer to reusable list after completion
    void release_command_buffer(std::shared_ptr<vk_CommandBuffer> command_buffer);
    // fast shallow submit wrapper to the queue
    std::shared_ptr<SubmittedTask> submit(std::uint32_t count, vk::CommandBuffer* command_buffers);
    void wait(); // waits for all submissions in this queue to finish
    // waits CPU until a signal with submission_id is reach. Then collect all completed to that point
    void vk_wait_for(std::uint64_t submission_id);
    // check if timeline on gpu and collect all completed.
    std::uint64_t vk_check_completion();
private:
    // iterate all submitted tasks with submission_id <= submission_id and collect them as completed. This is called automatically by vk_wait_for and vk_is_completed when the timeline semaphore has reached the submission_id.
    // notify completion.
    void vk_collect_all_completed(std::uint64_t submission_id);

    vk::CommandPool command_pool_; // vk command pool for the family of this engine.
    vk::Queue queue_; // vk queue associated to this engine.
    vk::Semaphore timeline_semaphore_;
    std::uint64_t current_submission_id_;
    EngineType engine_type_; // determines the type of commands valid for this engine
    std::weak_ptr<Device> device_; // device wrapper to access all vk device functions
    // all command buffers that are attached to this pool
    std::vector<vk::CommandBuffer> command_buffers_;
    // command buffers that have been submitted and finished and can be reused for new commands
    std::vector<std::shared_ptr<vk_CommandBuffer>> reusable_command_buffers_;
    // all pending submitted task not explicitly finished
    std::deque<std::shared_ptr<SubmittedTask>> pending_submitted_tasks_;
};

/**
 * Wraps the execution context of a Device: queue for submission and command pool for command buffer allocation.
 * Internally reuse command buffers if possible.
 */
class Engine : public std::enable_shared_from_this<Engine> {
public:
    Engine(std::shared_ptr<Device> device, std::shared_ptr<vk_Engine> engine, uint32_t index) noexcept;
    Engine() = delete;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&& other) noexcept = delete;
    Engine& operator=(Engine&& other) noexcept = delete;
    ~Engine() noexcept { device_.reset(); engine_.reset(); }

    std::uint32_t engine_index() const noexcept { return engine_index_; }

    void vk_release_command_buffer(const std::shared_ptr<vk_CommandBuffer>& command_buffer) {
        engine_->release_command_buffer(command_buffer);
    }

    std::shared_ptr<SubmittedTask> submit(std::vector<std::shared_ptr<CommandBuffer>> command_buffers) {
        std::shared_ptr<SubmittedTask> task;
        if (command_buffers.empty()) {
            task = engine_->submit(0, nullptr);
        } else
            if (command_buffers.size() == 1) // special case, try to make it faster
            {
                auto cb = command_buffers[0]->vk_command_buffer();
                task = engine_->submit(1, &cb->command_buffer);
                cb->notify_submitted();
            }
            else {
                std::vector<vk::CommandBuffer> vk_command_buffers(command_buffers.size());
                for (size_t i = 0; i < command_buffers.size(); ++i) {
                    auto cb = command_buffers[i]->vk_command_buffer();
                    vk_command_buffers[i] = cb->command_buffer;
                }
                task = engine_->submit((std::uint32_t)vk_command_buffers.size(), vk_command_buffers.data());
                for (size_t i = 0; i < command_buffers.size(); ++i) {
                    auto cb = command_buffers[i]->vk_command_buffer();
                    cb->notify_submitted();
                }
            }
        task->vk_attach_command_buffers(std::move(command_buffers));
        return task;
    }

    std::shared_ptr<CommandBuffer> create_command_buffer() {
        auto cbw = engine_->create_command_buffer();
        return std::make_shared<CommandBuffer>(device_, shared_from_this(), cbw);
    }

    // Blocks until every command submitted through this engine has finished executing on the GPU.
    void wait() { engine_->wait(); }
private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<vk_Engine> engine_;
    uint32_t engine_index_;
};

/**
 * Compiled SPIR-V bytecode for a single shader stage, obtained from
 * precompiled binary words, a file (raw ".spv" binary, or GLSL source
 * compiled at load time), or GLSL source text compiled on the spot by
 * shelling out to glslangValidator (must be on PATH, or found via
 * $VULKAN_SDK/Bin). There is no build- or link-time Vulkan SDK dependency
 * for this -- only this runtime one, and only when compiling from GLSL.
 */
class ShaderSource {
public:
    static ShaderSource from_spirv(std::vector<std::uint32_t> code, const std::string& entry_point = "main");
    // `include_dirs` are passed to glslangValidator as `-I<dir>`, only
    // consulted when `path` doesn't end in ".spv" (i.e. is compiled as GLSL).
    static ShaderSource from_file(
        const std::string& path,
        ShaderStageType stage,
        const std::string& entry_point = "main",
        const std::vector<std::string>& include_dirs = {});
    // `entry_point` is the name of the entry function as written in
    // `source` (glslangValidator's --source-entrypoint); the compiled
    // SPIR-V's own entry point is always named the same. `include_dirs`
    // are passed to glslangValidator as `-I<dir>`.
    static ShaderSource from_glsl(
        const std::string& source,
        ShaderStageType stage,
        const std::string& entry_point = "main",
        const std::vector<std::string>& include_dirs = {});

    [[nodiscard]] const std::vector<std::uint32_t>& vk_code() const noexcept { return code_; }
    // Name of this shader's entry point function, matching the compiled
    // SPIR-V's own OpEntryPoint name. Used internally as the `pName` of its
    // vk::PipelineShaderStageCreateInfo, not exposed to Python.
    [[nodiscard]] const std::string& vk_entry_point() const noexcept { return entry_point_; }
private:
    std::vector<std::uint32_t> code_;
    std::string entry_point_ = "main";
};

/**
 * Owns every raw Vulkan object backing a Pipeline (shader modules while
 * being built, descriptor set layouts, pipeline layout, render pass and the
 * pipeline itself) plus the builder-time state accumulated before close().
 * Not exposed to users directly: they interact with it through Pipeline.
 */
class vk_Pipeline : public std::enable_shared_from_this<vk_Pipeline> {
public:
    vk_Pipeline(std::weak_ptr<Device> device, PipelineType type);
    vk_Pipeline() = delete;
    vk_Pipeline(const vk_Pipeline&) = delete;
    vk_Pipeline& operator=(const vk_Pipeline&) = delete;
    vk_Pipeline(vk_Pipeline&& other) noexcept = delete;
    vk_Pipeline& operator=(vk_Pipeline&& other) noexcept = delete;
    ~vk_Pipeline() noexcept;

    void vk_stage(ShaderStageType type, const ShaderSource& source);
    int vk_layout(int set, int binding, DescriptorType description, int count);
    void vk_vertex_layout(int start_location, const Layout& layout);
    int vk_attach(int slot, Format format);
    // Declares the compute shader's own workgroup size (its
    // `layout(local_size_x = ..., ...)` declaration in GLSL). Must match the
    // actual shader code -- there is no way to verify this from the compiled
    // SPIR-V alone. Used by CommandBuffer::dispatch_threads to compute the
    // number of workgroups covering a requested thread count. Compute
    // pipelines only; defaults to (1, 1, 1) if never called.
    void vk_set_local_size(std::uint32_t x, std::uint32_t y, std::uint32_t z);
    void vk_close();

    [[nodiscard]] vk::DescriptorSet vk_allocate_descriptor_set(int set);
    [[nodiscard]] const vk_DescriptorBinding& vk_binding(int layout_id) const;
    [[nodiscard]] vk::RenderPass vk_render_pass() const noexcept { return render_pass_; }
    [[nodiscard]] const std::vector<vk_AttachmentDesc>& vk_attachments() const noexcept { return attachments_; }
    // Underlying vk::Pipeline and vk::PipelineLayout. Used internally by
    // CommandBuffer::set_pipeline, not exposed to Python.
    [[nodiscard]] vk::Pipeline vk_handle() const noexcept { return pipeline_; }
    [[nodiscard]] vk::PipelineLayout vk_pipeline_layout() const noexcept { return pipeline_layout_; }
    [[nodiscard]] std::uint32_t vk_local_size_x() const noexcept { return local_size_x_; }
    [[nodiscard]] std::uint32_t vk_local_size_y() const noexcept { return local_size_y_; }
    [[nodiscard]] std::uint32_t vk_local_size_z() const noexcept { return local_size_z_; }
    [[nodiscard]] bool is_closed() const noexcept { return closed_; }
    [[nodiscard]] PipelineType type() const noexcept { return type_; }

private:
    struct StageInfo {
        ShaderStageType type;
        vk::ShaderModule module;
        std::string entry_point;
    };
    struct VertexField {
        int location;
        vk::Format format;
        std::uint64_t offset;
    };
    struct VertexBinding {
        std::uint64_t stride;
        std::vector<VertexField> fields;
    };

    std::weak_ptr<Device> device_;
    PipelineType type_;
    bool closed_ = false;
    std::uint32_t local_size_x_ = 1;
    std::uint32_t local_size_y_ = 1;
    std::uint32_t local_size_z_ = 1;

    std::vector<StageInfo> stages_;
    std::vector<vk_DescriptorBinding> bindings_;
    std::vector<VertexBinding> vertex_bindings_;
    std::vector<vk_AttachmentDesc> attachments_;

    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts_;
    vk::PipelineLayout pipeline_layout_;
    vk::RenderPass render_pass_;
    vk::Pipeline pipeline_;
    vk::DescriptorPool descriptor_pool_;
};

class Framebuffer;
class DescriptorSet;

/**
 * User-facing handle to a GPU pipeline (compute, graphics/rasterization, or
 * ray tracing). Obtained from Device::create_pipeline. Hides the underlying
 * shader modules, descriptor set layouts, pipeline layout and render pass.
 */
class Pipeline : public std::enable_shared_from_this<Pipeline> {
public:
    Pipeline(std::shared_ptr<Device> device, std::shared_ptr<vk_Pipeline> pipeline) noexcept
        : device_(std::move(device)), pipeline_(std::move(pipeline)) {}
    Pipeline() = delete;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&& other) noexcept = delete;
    Pipeline& operator=(Pipeline&& other) noexcept = delete;
    ~Pipeline() noexcept = default;

    // Attaches a compiled shader to one stage of this pipeline. Must be
    // called before close().
    void stage(ShaderStageType type, const ShaderSource& source) { pipeline_->vk_stage(type, source); }

    // Declares one binding of the descriptor set layout for `set`. Returns
    // an opaque handle to later reference this binding from
    // DescriptorSet::bind(). Must be called before close().
    LayoutHandle layout(int set, int binding, DescriptorType description, int count = 1) {
        return LayoutHandle(pipeline_->vk_layout(set, binding, description, count));
    }

    // Declares the per-vertex input consumed by the vertex shader, as one
    // new vertex buffer binding starting at `start_location`. `layout` must
    // describe a struct (e.g. from compute_layout()); array/struct fields
    // are not valid vertex attributes. Graphics pipelines only. Must be
    // called before close().
    void vertex_layout(int start_location, const Layout& layout) { pipeline_->vk_vertex_layout(start_location, layout); }

    // Declares one color attachment of this pipeline's render pass, at
    // `slot` (the fragment shader output location) with `format`. Returns
    // an opaque handle used to bind a render target Image via
    // create_framebuffer(). Graphics pipelines only. Must be called before
    // close().
    AttachHandle attach(int slot, Format format) { return AttachHandle(pipeline_->vk_attach(slot, format)); }

    // Declares the compute shader's own workgroup size (its
    // `layout(local_size_x = ..., ...)` declaration in GLSL) so that
    // CommandBuffer::dispatch_threads can compute the number of workgroups
    // needed to cover a requested thread count. Must match the actual
    // shader code. Compute pipelines only; defaults to (1, 1, 1) if never
    // called. Must be called before close().
    void local_size(int x, int y = 1, int z = 1) {
        pipeline_->vk_set_local_size(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(z));
    }

    // Finalizes this pipeline: builds the descriptor set layouts, pipeline
    // layout, (for graphics pipelines) render pass and vertex input state,
    // and the underlying vk::Pipeline. No further stage()/layout()/
    // vertex_layout()/attach() calls are allowed afterwards.
    void close() { pipeline_->vk_close(); }

    // Creates a framebuffer compatible with this (closed) pipeline's render
    // pass. `attachments` must provide exactly one image per slot declared
    // via attach(), matching format and dimensions. Graphics pipelines only.
    [[nodiscard]] std::shared_ptr<Framebuffer> create_framebuffer(std::vector<std::pair<AttachHandle, std::shared_ptr<Image>>> attachments);

    // Allocates a new descriptor set matching the layout declared for `set`
    // via layout(). Pipeline must be closed first.
    [[nodiscard]] std::shared_ptr<DescriptorSet> create_descriptor_set(int set = 0);

    [[nodiscard]] bool is_closed() const noexcept { return pipeline_->is_closed(); }

    [[nodiscard]] std::shared_ptr<vk_Pipeline> vk_pipeline() const noexcept { return pipeline_; }
private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<vk_Pipeline> pipeline_;
};

/**
 * Owns the vk::Framebuffer created by Pipeline::create_framebuffer. Not
 * exposed to users directly: they interact with it through Framebuffer.
 */
class vk_Framebuffer {
public:
    vk_Framebuffer(std::weak_ptr<Device> device, vk::Framebuffer framebuffer, vk::RenderPass render_pass,
        std::uint32_t attachment_count, std::uint32_t width, std::uint32_t height) noexcept;
    vk_Framebuffer() = delete;
    vk_Framebuffer(const vk_Framebuffer&) = delete;
    vk_Framebuffer& operator=(const vk_Framebuffer&) = delete;
    vk_Framebuffer(vk_Framebuffer&& other) noexcept = delete;
    vk_Framebuffer& operator=(vk_Framebuffer&& other) noexcept = delete;
    ~vk_Framebuffer() noexcept { dispose(); }
    // Destroys the underlying vk::Framebuffer now, if not already done.
    // Idempotent. Called proactively by Device::dispose() (a framebuffer
    // references image views whose images may otherwise already be
    // destroyed, or outlive the device/instance themselves, by the time
    // this object's own C++ destructor eventually runs), and by the
    // destructor as a fallback.
    void dispose() noexcept;

    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }
    [[nodiscard]] std::uint32_t height() const noexcept { return height_; }
    [[nodiscard]] vk::Framebuffer vk_handle() const noexcept { return framebuffer_; }
    // Render pass this framebuffer was created against (Pipeline::create_framebuffer
    // uses its owning pipeline's render pass). Used by
    // CommandBuffer::set_framebuffer to begin a compatible render pass
    // instance; not exposed to Python.
    [[nodiscard]] vk::RenderPass vk_render_pass() const noexcept { return render_pass_; }
    // Number of attachments this framebuffer's render pass declares -- the
    // number of clear values CommandBuffer::set_framebuffer must provide at
    // vkCmdBeginRenderPass time.
    [[nodiscard]] std::uint32_t attachment_count() const noexcept { return attachment_count_; }
private:
    std::weak_ptr<Device> device_;
    vk::Framebuffer framebuffer_;
    vk::RenderPass render_pass_;
    std::uint32_t attachment_count_;
    std::uint32_t width_;
    std::uint32_t height_;
};

/**
 * User-facing handle to a set of render target image views bound for one
 * Pipeline's render pass. Obtained from Pipeline::create_framebuffer.
 */
class Framebuffer {
public:
    explicit Framebuffer(std::shared_ptr<vk_Framebuffer> framebuffer) noexcept : framebuffer_(std::move(framebuffer)) {}
    Framebuffer() = delete;
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&& other) noexcept = delete;
    Framebuffer& operator=(Framebuffer&& other) noexcept = delete;
    ~Framebuffer() noexcept = default;

    [[nodiscard]] std::uint32_t width() const noexcept { return framebuffer_->width(); }
    [[nodiscard]] std::uint32_t height() const noexcept { return framebuffer_->height(); }
    [[nodiscard]] vk::Framebuffer vk_framebuffer() const noexcept { return framebuffer_->vk_handle(); }
    [[nodiscard]] vk::RenderPass vk_render_pass() const noexcept { return framebuffer_->vk_render_pass(); }
    [[nodiscard]] std::uint32_t vk_attachment_count() const noexcept { return framebuffer_->attachment_count(); }
private:
    std::shared_ptr<vk_Framebuffer> framebuffer_;
};


/**
 * Owns the GLFW window, Vulkan surface, swapchain, and per-slot ("frame on
 * the fly") resources/synchronization for a Window. Not exposed to users
 * directly: they interact with it through Window.
 *
 * The number of frames on the fly IS the swapchain image count (whatever
 * the surface capabilities/driver actually grant for the requested count),
 * so a given frame slot always corresponds to the same acquired swapchain
 * image index -- there is never any ambiguity between "which frame is
 * this" and "which swapchain image did we get back". Each slot
 * pre-allocates its own render target (the swapchain image itself, wrapped
 * as a non-owning Image -- see vk_ResourceData's owns_image), image target
 * (a real owned Image, doubling as the common upload/blit landing pad for
 * the image/buffer/tensor target kinds), buffer target and tensor target,
 * plus its own semaphores/fence; every possible target-kind-to-presented-
 * image command buffer is pre-recorded once, at (re)creation time --
 * vk_present_frame only ever *submits* one of them, never records
 * anything at frame time.
 *
 * The swapchain (and thus render_target) is always Format::BGRA8_UNorm,
 * regardless of the `format` the Window was created with -- that's the
 * only format broadly guaranteed to be supported for presentation.
 * `format` instead applies to image_target/buffer_target/tensor_target;
 * presenting one of these blits it into the BGRA8_UNorm swapchain image,
 * which converts formats natively as part of the blit.
 */
class vk_Window : public std::enable_shared_from_this<vk_Window> {
public:
    vk_Window(std::shared_ptr<Device> device, std::uint32_t width, std::uint32_t height,
        const std::string& title, Format format, std::uint32_t frames_on_the_fly);
    vk_Window() = delete;
    vk_Window(const vk_Window&) = delete;
    vk_Window& operator=(const vk_Window&) = delete;
    vk_Window(vk_Window&& other) noexcept = delete;
    vk_Window& operator=(vk_Window&& other) noexcept = delete;
    ~vk_Window() noexcept;
    void dispose() noexcept;
    // Tears down every Vulkan object this Window owns (swapchain, surface,
    // per-slot resources, semaphores/fences/command pool) using `dev`/
    // `instance` directly, instead of re-deriving them via device_.lock().
    // Used by Device::dispose(), which calls this with its own still-valid
    // handles: by the time Device::dispose() runs, device_ (only a
    // weak_ptr<Device>, to avoid a reference cycle) may already be
    // expired even though the real vk::Device/vk::Instance haven't been
    // destroyed yet (e.g. Device::dispose() invoked from ~Device(), whose
    // body runs only once the owning shared_ptr's strong count has
    // already hit zero -- at which point every weak_ptr<Device>,
    // including this one, reports expired()). dispose() itself now just
    // derives dev/instance via device_.lock() and forwards here, for the
    // fallback/standalone path (~vk_Window() running with Device still
    // otherwise alive).
    void vk_dispose_with(vk::Device dev, vk::Instance instance) noexcept;

    bool check_alive();
    void set_title(const std::string& title);
    void set_size(std::uint32_t width, std::uint32_t height);
    [[nodiscard]] std::uint32_t width() const noexcept { return extent_.width; }
    [[nodiscard]] std::uint32_t height() const noexcept { return extent_.height; }

    // One slot's resources, for Window::begin_frame to wrap into a Frame.
    struct FrameResources {
        std::shared_ptr<Image> render_target;
        std::shared_ptr<Image> image_target;
        std::shared_ptr<Buffer> buffer_target;
        std::shared_ptr<Tensor> tensor_target;
        std::int64_t index;
    };
    // Waits for the next slot's fence (i.e. that this slot's previous
    // frame, if any, has finished presenting), (re)creates the swapchain
    // first if it's out of date (e.g. after a resize), and acquires the
    // next image.
    FrameResources vk_begin_frame();

    // Submits the pre-recorded command buffer matching whichever of
    // image_target/buffer_target/tensor_target is non-null (or, if all
    // three are null, the render-target-direct one: just a layout
    // transition, since the caller rendered into it directly), waiting
    // (GPU-side, via Device's interop semaphore -- see vk_Engine::submit)
    // for any Vulkan work submitted since this frame began before that
    // command buffer runs, then presents.
    void vk_present_frame(std::int64_t index,
        const std::shared_ptr<Image>& render_target, const std::shared_ptr<Image>& image_target,
        const std::shared_ptr<Buffer>& buffer_target, const std::shared_ptr<Tensor>& tensor_target);

    // ---- ImGui-backed immediate-mode GUI, drawn as a final overlay pass
    // on top of the presented image (whichever target kind the caller
    // used) -- see vk_present_frame's implementation for how this is
    // submitted as a second, freshly-recorded-every-frame command buffer
    // chained after the pre-recorded one, since ImGui's draw data is
    // inherently per-frame and cannot be pre-recorded. Every vk_imgui_*
    // call below must happen between vk_begin_frame() and
    // vk_present_frame() (i.e. ImGui::NewFrame() has been called but
    // ImGui::Render() hasn't yet).
    [[nodiscard]] std::uint64_t vk_next_widget_id() noexcept { return next_widget_id_++; }
    [[nodiscard]] double vk_fps() const noexcept { return fps_; }
    void vk_imgui_text(const std::string& text);
    [[nodiscard]] bool vk_imgui_button(const std::string& text);
    [[nodiscard]] bool vk_imgui_checkbox(const std::string& label, bool& value);
    [[nodiscard]] bool vk_imgui_slider_float(const std::string& label, float& value, float min, float max);
    [[nodiscard]] bool vk_imgui_slider_int(const std::string& label, int& value, int min, int max);
    [[nodiscard]] bool vk_imgui_combobox(const std::string& label, const std::vector<std::string>& items, int& selected_index);

private:
    // Per-swapchain-image resources: which image is acquired is decided by
    // the driver (vkAcquireNextImageKHR), not by a simple round-robin
    // counter, so these are indexed by the *acquired image index*.
    struct Slot {
        std::shared_ptr<Image> render_target;
        std::shared_ptr<Image> image_target;
        std::shared_ptr<Buffer> buffer_target;
        std::shared_ptr<Tensor> tensor_target;
        vk::CommandBuffer cmd_from_render_target;
        vk::CommandBuffer cmd_from_image_target;
        vk::CommandBuffer cmd_from_buffer_target;
        vk::CommandBuffer cmd_from_tensor_target;
    };
    // Per-frame-in-flight synchronization: cycled by frame counter
    // (current_frame_index_ % frames_on_the_fly_), independent of which
    // swapchain image a given frame happens to acquire -- the standard
    // Vulkan swapchain pattern (e.g. vulkan-tutorial.com's "Rendering and
    // presentation" chapter), needed because the semaphore passed to
    // vkAcquireNextImageKHR must be unsignaled, which a plain per-image
    // index can't guarantee if the driver ever returns images out of
    // strict round-robin order (e.g. under VK_PRESENT_MODE_MAILBOX_KHR).
    struct SyncGroup {
        vk::Semaphore image_available_semaphore;
        // Signaled when the pre-recorded target-to-render_target command
        // buffer finishes; waited on by the ImGui overlay command buffer
        // before it draws on top of the same image (see vk_present_frame).
        vk::Semaphore content_ready_semaphore;
        vk::Semaphore render_finished_semaphore;
        vk::Fence fence;
        bool fence_submitted = false; // false until this group's fence has ever been submitted
    };

    void create_swapchain();
    void destroy_swapchain_resources_with(vk::Device dev) noexcept;
    void record_command_buffers(std::size_t slot_index);

    void vk_imgui_init(const std::shared_ptr<Device>& device);
    void vk_imgui_shutdown(vk::Device dev) noexcept;
    void vk_imgui_ensure_current() const noexcept;

    std::weak_ptr<Device> device_;
    void* glfw_window_ = nullptr; // GLFWwindow*; kept opaque so device.hpp needs no GLFW headers
    vk::SurfaceKHR surface_;
    vk::SwapchainKHR swapchain_;
    vk::Format vk_format_ = vk::Format::eUndefined;
    Format format_ = Format::Undefined;
    vk::Extent2D extent_{};
    vk::Queue present_queue_;
    std::uint32_t present_queue_family_ = 0;
    vk::CommandPool command_pool_;
    std::vector<vk::Image> swapchain_images_;
    std::vector<Slot> slots_; // size == actual swapchain image count
    std::vector<SyncGroup> sync_groups_; // size == frames_on_the_fly_ (== slots_.size() once negotiated)
    // frame_to_image_index_[frame_index % frames_on_the_fly_] = the image
    // index acquired for that frame, recorded at vk_begin_frame() time and
    // read back at vk_present_frame() time (frames are presented strictly
    // in order, so this is never overwritten before its matching present).
    std::vector<std::uint32_t> frame_to_image_index_;
    // images_in_flight_[image_index] = the fence (non-owning, a raw
    // reference into some SyncGroup's fence) currently guarding whichever
    // slot resources (command buffers, ImGui framebuffer) belong to that
    // swapchain image, or null if never used. frames_on_the_fly_ ==
    // slots_.size() does NOT guarantee vkAcquireNextImageKHR returns
    // images in stable round-robin order, so a given image can come back
    // under a *different* sync group than last time -- without this,
    // vk_begin_frame's fence wait (keyed by sync group) would not
    // protect that image's command buffer from being resubmitted while
    // still pending from its previous (different sync group's) use. See
    // vk_begin_frame.
    std::vector<vk::Fence> images_in_flight_;
    std::uint32_t frames_on_the_fly_ = 0;
    std::int64_t current_frame_index_ = 0;
    std::int64_t last_enqueue_frame_index_ = -1;
    bool alive_ = true;

    // ImGui state. imgui_context_ is an opaque ImGuiContext* (kept opaque
    // so device.hpp needs no imgui headers, mirroring glfw_window_).
    // imgui_render_pass_/imgui_descriptor_pool_ are created once (they
    // don't depend on the swapchain's extent/image count); imgui_framebuffers_
    // (one per slot, wrapping each slot's render_target view) and
    // imgui_command_buffers_/content_ready_semaphore (one per sync group)
    // are (re)created in create_swapchain(), alongside everything else
    // that depends on the negotiated image count/extent.
    void* imgui_context_ = nullptr;
    vk::DescriptorPool imgui_descriptor_pool_;
    vk::RenderPass imgui_render_pass_;
    std::vector<vk::Framebuffer> imgui_framebuffers_; // size == slots_.size()
    std::vector<vk::CommandBuffer> imgui_command_buffers_; // size == sync_groups_.size()
    std::uint64_t next_widget_id_ = 0; // disambiguates same-label widgets via ImGui's "label##id" convention
    std::chrono::steady_clock::time_point last_frame_time_point_;
    double fps_ = 0.0;
};

/**
 * Read-only per-frame timing info, exposed as Window::stats. fps() is an
 * exponential moving average of 1/frame_time updated once per
 * Window::begin_frame() call, not an instantaneous single-frame value
 * (which is too noisy to display directly).
 */
class Stats {
public:
    explicit Stats(std::weak_ptr<vk_Window> window) noexcept : window_(std::move(window)) {}
    Stats() = delete;
    [[nodiscard]] double fps() const;
private:
    std::weak_ptr<vk_Window> window_;
};

// Disambiguates same-label widgets via ImGui's "label##id" ID convention
// (the part after "##" affects identity but is never displayed). Shared
// by every widget class below. Returns 0 if `window` has expired --
// harmless (just risks an ID collision on an already-unusable widget).
std::uint64_t vk_next_widget_id(const std::weak_ptr<vk_Window>& window);

/**
 * A persistent checkbox widget (see Window::checkbox). Must be drawn via
 * draw() once per frame (between Window::begin_frame() and
 * Frame::present()) for it to appear at all -- ImGui is immediate-mode,
 * so a widget that isn't (re)drawn this frame simply isn't shown.
 */
class Checkbox {
public:
    Checkbox(std::weak_ptr<vk_Window> window, std::string label, bool value) noexcept
        : window_(window), label_(std::move(label)), value_(value), id_(vk_next_widget_id(window)) {}
    Checkbox() = delete;
    // Draws this frame; returns whether the value changed this frame.
    bool draw();
    [[nodiscard]] bool value() const noexcept { return value_; }
    void set_value(bool value) noexcept { value_ = value; }
private:
    std::weak_ptr<vk_Window> window_;
    std::string label_;
    bool value_;
    std::uint64_t id_;
};

/** A persistent float slider widget (see Window::slider_float). */
class SliderFloat {
public:
    SliderFloat(std::weak_ptr<vk_Window> window, std::string label, float min, float max, float value) noexcept
        : window_(window), label_(std::move(label)), min_(min), max_(max), value_(value), id_(vk_next_widget_id(window)) {}
    SliderFloat() = delete;
    bool draw();
    [[nodiscard]] float value() const noexcept { return value_; }
    void set_value(float value) noexcept { value_ = value; }
private:
    std::weak_ptr<vk_Window> window_;
    std::string label_;
    float min_;
    float max_;
    float value_;
    std::uint64_t id_;
};

/** A persistent integer slider widget (see Window::slider_int). */
class SliderInt {
public:
    SliderInt(std::weak_ptr<vk_Window> window, std::string label, int min, int max, int value) noexcept
        : window_(window), label_(std::move(label)), min_(min), max_(max), value_(value), id_(vk_next_widget_id(window)) {}
    SliderInt() = delete;
    bool draw();
    [[nodiscard]] int value() const noexcept { return value_; }
    void set_value(int value) noexcept { value_ = value; }
private:
    std::weak_ptr<vk_Window> window_;
    std::string label_;
    int min_;
    int max_;
    int value_;
    std::uint64_t id_;
};

/** A persistent combobox widget (see Window::combobox). */
class Combobox {
public:
    Combobox(std::weak_ptr<vk_Window> window, std::string label, std::vector<std::string> items, int selected_index) noexcept
        : window_(window), label_(std::move(label)), items_(std::move(items)), selected_index_(selected_index), id_(vk_next_widget_id(window)) {}
    Combobox() = delete;
    bool draw();
    [[nodiscard]] int selected_index() const noexcept { return selected_index_; }
    void set_selected_index(int index) noexcept { selected_index_ = index; }
    [[nodiscard]] const std::string& selected_item() const { return items_.at(static_cast<std::size_t>(selected_index_)); }
    [[nodiscard]] const std::vector<std::string>& items() const noexcept { return items_; }
private:
    std::weak_ptr<vk_Window> window_;
    std::string label_;
    std::vector<std::string> items_;
    int selected_index_;
    std::uint64_t id_;
};

/**
 * Usage:
 * window = Window(....);
 * while (window.check_alive()) { // poll events etc.
 *  auto frame = window.frame();
 *  frame_index = frame.index(); // gives the index of this frame
 *  auto render_target = frame.render_target();
 *  render(render_target, frame_index);
 *  frame.present(); // should be called at the end of the frame to present the rendered image. The number of frames that can be 'live' concurrently will depend on the frames_on_the_fly attribute of the window, but frame.present should be in order.
 * }
 *
 */
class Frame {
public:
    Frame(
        std::shared_ptr<Window> window,
        std::shared_ptr<Image> render_target,
        std::shared_ptr<Image> image_target,
        std::shared_ptr<Buffer> buffer_target,
        std::shared_ptr<Tensor> tensor_target,
        std::int64_t index) noexcept;
    Frame() = delete;
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    Frame(Frame&& other) noexcept = delete;
    Frame& operator=(Frame&& other) noexcept = delete;
    ~Frame() noexcept = default;

    [[nodiscard]] std::shared_ptr<Image> render_target() {
        if (!render_target_) {
            throw std::runtime_error("Can not select render_target because another target was acquired previously.");
        }
        image_target_.reset();
        buffer_target_.reset();
        tensor_target_.reset();
        return render_target_;
    }

    [[nodiscard]] std::shared_ptr<Image> image_target() {
        if (!image_target_) {
            throw std::runtime_error("Can not select image_target because another target was acquired previously.");
        }
        render_target_.reset();
        buffer_target_.reset();
        tensor_target_.reset();
        return image_target_;
    }

    [[nodiscard]] std::shared_ptr<Buffer> buffer_target() {
        if (!buffer_target_) {
            throw std::runtime_error("Can not select buffer_target because another target was acquired previously.");
        }
        render_target_.reset();
        image_target_.reset();
        tensor_target_.reset();
        return buffer_target_;
    }

    [[nodiscard]] std::shared_ptr<Tensor> tensor_target() {
        if (!tensor_target_) {
            throw std::runtime_error("Can not select tensor_target because another target was acquired previously.");
        }
        render_target_.reset();
        image_target_.reset();
        buffer_target_.reset();
        return tensor_target_;
    }

    [[nodiscard]] std::int64_t index() const noexcept { return index_; }

    [[nodiscard]] bool is_target_acquired() const noexcept { return window_ && (!render_target_ || !image_target_ || !buffer_target_ || !tensor_target_); }

    [[nodiscard]] bool is_render_target() const noexcept { return render_target_ && !buffer_target_; }
    [[nodiscard]] bool is_image_target() const noexcept { return image_target_ && !buffer_target_; }
    [[nodiscard]] bool is_buffer_target() const noexcept { return buffer_target_ && !image_target_; }
    [[nodiscard]] bool is_tensor_target() const noexcept { return tensor_target_ && !buffer_target_; }

    [[nodiscard]] bool presented() const noexcept { return !window_; }

    void present();
private:
    std::shared_ptr<Window> window_;
    std::shared_ptr<Image> render_target_;
    std::shared_ptr<Image> image_target_;
    std::shared_ptr<Buffer> buffer_target_;
    std::shared_ptr<Tensor> tensor_target_;
    std::int64_t index_;
};


/**
 * Encapsulates a swapchain and window management.
 * Provides a render target, image, buffer or tensor for each frame on the fly, and handles presentation of the rendered image to the window.
 * Usage:
 * window = Window(....);
 * while (window.check_alive()) { // poll events etc.
 *  auto frame = window.frame();
 *  frame_index = frame.index(); // gives the index of this frame
 *  auto render_target = frame.render_target();
 *  render(render_target, frame_index);
 *  frame.present(); // should be called at the end of the frame to present the rendered image. The number of frames that can be 'live' concurrently will depend on the frames_on_the_fly attribute of the window, but frame.present should be in order.
 * }
 */
class Window : public std::enable_shared_from_this<Window> {
public:
    // `frames_on_the_fly` becomes the swapchain's image count (clamped to
    // whatever the surface capabilities actually grant); see vk_Window's
    // docstring for why the two are one and the same here.
    Window(std::shared_ptr<Device> device, std::uint32_t width, std::uint32_t height, const std::string& title,
        Format format, std::uint32_t frames_on_the_fly = 3);
    Window() = delete;
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    void vk_present_frame(std::int64_t index, const std::shared_ptr<Image>& render_target, const std::shared_ptr<Image>& image_target, const std::shared_ptr<Buffer>& buffer_target, const std::shared_ptr<Tensor>& tensor_target) {
        window_->vk_present_frame(index, render_target, image_target, buffer_target, tensor_target);
    }

    Window(Window&& other) noexcept = delete;
    Window& operator=(Window&& other) noexcept = delete;
    ~Window() noexcept = default;

    /**
     * Poll events and returns if the window is still opened.
     * @return True if the window is not closed.
     */
    bool check_alive() { return window_->check_alive(); }
    void set_title(const std::string& title) { window_->set_title(title); }
    void set_size(std::uint32_t width, std::uint32_t height) { window_->set_size(width, height); }
    [[nodiscard]] std::uint32_t width() const noexcept { return window_->width(); }
    [[nodiscard]] std::uint32_t height() const noexcept { return window_->height(); }

    // ---- ImGui-backed GUI. label()/button() are direct, stateless,
    // immediate calls (call every frame; nothing to persist between
    // frames). checkbox()/slider_float()/slider_int()/combobox() instead
    // return a persistent widget object (create once, call its draw()
    // every frame, read/write its value whenever) since those need to
    // remember their current value across frames -- see their classes'
    // docstrings. All must be called between begin_frame() and the
    // matching Frame::present().
    [[nodiscard]] std::shared_ptr<Stats> stats() const { return std::make_shared<Stats>(window_); }
    void label(const std::string& text) { window_->vk_imgui_text(text); }
    void label(const std::string& text, double value) { window_->vk_imgui_text(text + std::to_string(value)); }
    void label(const std::string& text, const std::string& value) { window_->vk_imgui_text(text + value); }
    [[nodiscard]] bool button(const std::string& text) { return window_->vk_imgui_button(text); }
    [[nodiscard]] std::shared_ptr<Checkbox> checkbox(const std::string& label, bool value = false) {
        return std::make_shared<Checkbox>(window_, label, value);
    }
    [[nodiscard]] std::shared_ptr<SliderFloat> slider_float(const std::string& label, float min, float max, float value) {
        return std::make_shared<SliderFloat>(window_, label, min, max, value);
    }
    [[nodiscard]] std::shared_ptr<SliderInt> slider_int(const std::string& label, int min, int max, int value) {
        return std::make_shared<SliderInt>(window_, label, min, max, value);
    }
    [[nodiscard]] std::shared_ptr<Combobox> combobox(const std::string& label, std::vector<std::string> items, int selected_index = 0) {
        return std::make_shared<Combobox>(window_, label, std::move(items), selected_index);
    }

    // Underlying vk_Window. Used internally by Device::create_window (to
    // register it for proactive disposal -- see Device::vk_register_window),
    // not exposed to Python.
    [[nodiscard]] const std::shared_ptr<vk_Window>& vk_window() const noexcept { return window_; }

    std::shared_ptr<Frame> begin_frame();
private:
    std::shared_ptr<vk_Window> window_;
};


/**
 * Owns the vk::DescriptorSet allocated by Pipeline::create_descriptor_set,
 * and resolves layout ids (from Pipeline::layout()) into their (set,
 * binding, type) for DescriptorSet::bind. Not exposed to users directly:
 * they interact with it through DescriptorSet.
 */
class vk_DescriptorSet {
public:
    vk_DescriptorSet(std::weak_ptr<Device> device, std::shared_ptr<vk_Pipeline> pipeline, int set, vk::DescriptorSet descriptor_set) noexcept;
    vk_DescriptorSet() = delete;
    vk_DescriptorSet(const vk_DescriptorSet&) = delete;
    vk_DescriptorSet& operator=(const vk_DescriptorSet&) = delete;
    vk_DescriptorSet(vk_DescriptorSet&& other) noexcept = delete;
    vk_DescriptorSet& operator=(vk_DescriptorSet&& other) noexcept = delete;
    ~vk_DescriptorSet() noexcept = default;

    void vk_bind_buffer(LayoutHandle layout_id, const std::shared_ptr<Buffer>& buffer);
    void vk_bind_image(LayoutHandle layout_id, const std::shared_ptr<Image>& image);
    // Underlying vk::DescriptorSet. Used internally by
    // CommandBuffer::set_pipeline, not exposed to Python.
    [[nodiscard]] vk::DescriptorSet vk_handle() const noexcept { return descriptor_set_; }
private:
    std::weak_ptr<Device> device_;
    std::shared_ptr<vk_Pipeline> pipeline_;
    int set_;
    vk::DescriptorSet descriptor_set_;
};

/**
 * User-facing handle to a descriptor set matching one Pipeline's declared
 * layout for a given `set` index. Obtained from Pipeline::create_descriptor_set.
 */
class DescriptorSet {
public:
    explicit DescriptorSet(std::shared_ptr<vk_DescriptorSet> descriptor_set) noexcept : descriptor_set_(std::move(descriptor_set)) {}
    DescriptorSet() = delete;
    DescriptorSet(const DescriptorSet&) = delete;
    DescriptorSet& operator=(const DescriptorSet&) = delete;
    DescriptorSet(DescriptorSet&& other) noexcept = delete;
    DescriptorSet& operator=(DescriptorSet&& other) noexcept = delete;
    ~DescriptorSet() noexcept = default;

    // Writes `buffer` into the binding identified by `layout_id`. The
    // binding's declared type must be STORAGE_BUFFER or UNIFORM_BUFFER.
    void bind(LayoutHandle layout_id, const std::shared_ptr<Buffer>& buffer) { descriptor_set_->vk_bind_buffer(layout_id, buffer); }
    // Writes `image` into the binding identified by `layout_id`. The
    // binding's declared type must be STORAGE_IMAGE or SAMPLED_IMAGE (types
    // requiring a sampler are not yet supported).
    void bind(LayoutHandle layout_id, const std::shared_ptr<Image>& image) { descriptor_set_->vk_bind_image(layout_id, image); }

    // Underlying vk_DescriptorSet. Used internally by
    // CommandBuffer::set_pipeline, not exposed to Python.
    [[nodiscard]] std::shared_ptr<vk_DescriptorSet> vk_descriptor_set() const noexcept { return descriptor_set_; }
private:
    std::shared_ptr<vk_DescriptorSet> descriptor_set_;
};

/**
 * Wraps a vk::Instance, a vk::Device and its associated resources, memory managers and engines.
 */
class Device : public std::enable_shared_from_this<Device> {
public:
    ~Device() noexcept;
    static std::shared_ptr<Device> create_device(uint32_t device_index, bool enable_validation_layers);
    void dispose() noexcept;
    Device() = delete;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&& other) noexcept = delete;
    Device& operator=(Device&& other) noexcept = delete;
    bool is_disposed() const noexcept { return device_ == nullptr; }
    [[nodiscard]] vk::PhysicalDevice physical_device() const noexcept;
    [[nodiscard]] vk::Device logical_device() const noexcept;
    [[nodiscard]] uint32_t device_index() const noexcept;
    // Creates an engine to create and submit commands. Hides the underlying vk::Queue and vk::CommandPool.
    [[nodiscard]] std::shared_ptr<Engine> create_engine(EngineType type, uint32_t index = 0);
    // Creates a new pipeline for a specific type of gpu process.
    [[nodiscard]] std::shared_ptr<Pipeline> create_pipeline(PipelineType type);
    // Creates a buffer to store an array of `elements` of specific scalar
    // type. Its element layout is that scalar type. A raw byte buffer is
    // just this with type == ScalarType::UINT8.
    [[nodiscard]] std::shared_ptr<Buffer> create_buffer(std::uint64_t elements, ScalarType type, MemoryLocation location);
    // Creates a buffer to store an array of `elements` texels with specific
    // format. Its element layout is an array of the format's own
    // per-channel scalar type.
    [[nodiscard]] std::shared_ptr<Buffer> create_buffer(std::uint64_t elements, Format format, MemoryLocation location);
    // Creates a buffer sized to hold `elements` instances of the type
    // described by `layout` (using layout->aligned_size as the per-element
    // stride). Its element layout is `layout` itself. Throws
    // std::runtime_error if `elements` is 0 or the computed size is 0.
    [[nodiscard]] std::shared_ptr<Buffer> create_buffer(std::uint64_t elements, const std::shared_ptr<Layout>& layout, MemoryLocation location);
    // Creates the resource data for an image of specific dimensions and format
    [[nodiscard]] std::shared_ptr<Image> create_image(int width, int height, int depth, int mip_levels, int array_layers, Format format, MemoryLocation location);
    // Creates a Window: an GLFW-backed OS window plus its Vulkan swapchain
    // and presentation machinery. See Window's docstring for the render
    // loop this is meant to drive. The swapchain itself is always created
    // as Format::BGRA8_UNorm (the only format broadly guaranteed to be
    // supported); `format` instead controls image_target/buffer_target/
    // tensor_target, which may be any format -- presenting one of them
    // blits it into the BGRA8_UNorm swapchain image, converting formats
    // along the way.
    [[nodiscard]] std::shared_ptr<Window> create_window(std::uint32_t width, std::uint32_t height,
        const std::string& title, Format format, std::uint32_t frames_on_the_fly = 3);
    // Creates a plain buffer sized to match `buffer`, in `location` (default HOST).
    // This is the correct Vulkan pattern to move data between CPU and GPU memory:
    // record a CommandBuffer::transfer between the staging buffer and `buffer`
    // rather than attempting to map device-local memory directly.
    [[nodiscard]] std::shared_ptr<Buffer> create_staging(const std::shared_ptr<Buffer>& buffer, MemoryLocation location = MemoryLocation::HOST);
    // Creates a plain buffer sized to match `image`'s backing store, in `location`
    // (default HOST), for staged CPU<->GPU transfers of image contents.
    [[nodiscard]] std::shared_ptr<Buffer> create_staging(const std::shared_ptr<Image>& image, MemoryLocation location = MemoryLocation::HOST);
    // Creates a Tensor: a plain N-dimensional array resource of a single
    // scalar type, tracked and disposed the same way as Buffer/Image, and
    // directly exportable via torch.from_dlpack() (Tensor implements
    // __dlpack__/__dlpack_device__).
    [[nodiscard]] std::shared_ptr<Tensor> create_tensor(const std::vector<std::uint64_t>& shape, ScalarType type, MemoryLocation location);

    // Wraps an externally-owned Python object as device-usable memory.
    // `obj` may be:
    //   - a Buffer or Tensor: used directly (its own external_ptr()), no copy ever.
    //   - a DLPack-compatible object (e.g. a torch tensor): used directly,
    //     with no copy, if it's contiguous and its memory already belongs
    //     to one of this Device's own memory managers (e.g. it came from
    //     torch.from_dlpack() on one of our own Buffers); otherwise a
    //     temporary buffer is allocated at `location` and the data is
    //     copied in/out per `mode`, using the loaded interop library for
    //     CUDA-resident tensors or a strided/plain host copy otherwise.
    //   - a Python buffer-protocol object (e.g. a numpy array): used
    //     directly (its own pointer) with no copy if `location` is HOST;
    //     otherwise copied into/out of a temporary DEVICE buffer per `mode`.
    // Throws std::runtime_error if `obj` is none of these.
    [[nodiscard]] std::shared_ptr<WrappedMemory> wrap(pybind11::object obj, WrapMode mode, MemoryLocation location = MemoryLocation::DEVICE);

    // Copies `total_bytes`/shape/strides FROM an external source (`src_data`,
    // host- or CUDA-resident per `source_is_cuda`) INTO `dst_external_ptr`
    // (a contiguous buffer at `dst_location`). Uses the loaded interop
    // library when either side needs it (a CUDA source, or a DEVICE
    // destination); otherwise a plain host-side strided copy. Throws if a
    // CUDA-aware copy is required but no interop library provides one. Used
    // internally by wrap()/WrappedMemory, not exposed to Python.
    void vk_copy_in(
        std::uint64_t dst_external_ptr, MemoryLocation dst_location,
        void* src_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
        std::uint64_t itemsize, bool source_is_cuda);
    // Symmetric to vk_copy_in: copies `total_bytes` FROM `src_external_ptr`
    // (a contiguous buffer at `src_location`) INTO an external destination
    // (`dst_data`, host- or CUDA-resident per `dst_is_cuda`). Used
    // internally by WrappedMemory's destructor, not exposed to Python.
    void vk_copy_out(
        std::uint64_t src_external_ptr, MemoryLocation src_location,
        void* dst_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
        std::uint64_t itemsize, bool dst_is_cuda);

    vk::Device vk_device() const noexcept { return device_; }
    vk::Instance vk_instance() const noexcept { return instance_; }
    [[nodiscard]] bool vk_swapchain_supported() const noexcept { return swapchain_supported_; }

    // Tracks `framebuffer` (a weak reference) so Device::dispose() can
    // proactively destroy it -- and the render pass/image views it
    // references -- before destroying the images those views point into,
    // and before destroying the device/instance themselves. Used
    // internally by Pipeline::create_framebuffer, not exposed to Python.
    void vk_register_framebuffer(const std::shared_ptr<vk_Framebuffer>& framebuffer) {
        framebuffers_.push_back(framebuffer);
    }

    // Tracks `window` (a weak reference) so Device::dispose() can
    // proactively destroy its swapchain/surface/semaphores/command pool
    // -- and every per-slot render/image/buffer/tensor target -- before
    // destroying the device/instance themselves. Without this, vk_Window
    // only holds a weak_ptr<Device> (to avoid a reference cycle with
    // Device), so if the last Python reference to a Window disappears
    // after the last reference to its Device (an unspecified, GC-
    // dependent order when both are only module-level globals -- e.g.
    // scripts that never call device.dispose()/window.dispose()
    // explicitly), vk_Window::dispose() would find device_.lock() already
    // null and skip destroying its own Vulkan objects entirely, leaking
    // them and tripping vkDestroyDevice's child-object validation checks.
    // Used internally by Device::create_window, not exposed to Python.
    void vk_register_window(const std::shared_ptr<vk_Window>& window) {
        windows_.push_back(window);
    }

    // Vulkan timeline semaphore, exported (if VK_KHR_external_semaphore
    // plus the platform win32/fd extension are supported -- see vk_init())
    // for cross-API synchronization: every engine submission additionally
    // signals this on top of its own private timeline semaphore (see
    // vk_Engine::submit), so a CUDA-side memcpy can wait on the exact
    // value corresponding to the most recent GPU work from ANY engine, via
    // a real GPU-side semaphore wait. Unlike a host-side
    // vkWaitSemaphores/queue-idle wait (which only guarantees the CPU
    // knows the work is done), this is what actually guarantees cross-API
    // memory visibility for a Vulkan write later read via CUDA -- see
    // MemoryManager::vk_copy_to_dlpack/vk_copy_from_dlpack. Null/0 if
    // unsupported; callers must handle that (falling back to a host-side
    // copy without the extra cross-API wait).
    [[nodiscard]] vk::Semaphore vk_interop_semaphore() const noexcept { return interop_semaphore_; }
    [[nodiscard]] std::uint64_t vk_interop_semaphore_value() const noexcept { return interop_semaphore_value_; }
    // Called by vk_Engine::submit() right before building its SubmitInfo:
    // returns the next value to signal interop_semaphore_ with, and
    // remembers it as the new "current" value for
    // vk_interop_semaphore_value(). A no-op (always returns 0) if
    // interop_semaphore_ wasn't created (unsupported).
    [[nodiscard]] std::uint64_t vk_bump_interop_semaphore_value() noexcept {
        return interop_semaphore_ ? ++interop_semaphore_value_ : 0;
    }

    static constexpr size_t kMaxEngineTypes = 8;

private:
    Device(uint32_t device_index, bool enable_validation_layers);
    void vk_init();
    // Allocates `size` bytes of memory in `location` and wraps it in a fresh
    // vk_ResourceData, tracked in resources_. Shared by every create_buffer()
    // overload; callers build their own ResourceSlice (view/element_size)
    // around the result.
    std::shared_ptr<vk_ResourceData> vk_allocate_buffer_data(std::uint64_t size, MemoryLocation location);
    // Records and submits (blocking until complete) a one-off pipeline
    // barrier transitioning `image` from `old_layout` to `new_layout`. Used
    // once, right after create_image(), to move a freshly created image out
    // of eUndefined (which isn't valid for sampling/storage/blit access)
    // into eGeneral -- the one layout every other use site (descriptor image
    // binds, blit_image) assumes uniformly, since this codebase does not
    // otherwise track per-image Vulkan layouts.
    void vk_transition_image_layout(vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
        std::uint32_t mip_levels, std::uint32_t array_layers);

    uint32_t device_index_ = 0;
    vk::Instance instance_;
    vk::PhysicalDevice physical_;
    vk::Device device_;
    // Whether VK_KHR_external_semaphore plus the platform win32/fd
    // extension were both supported and enabled; gates whether
    // interop_semaphore_ is created at all in vk_init().
    bool external_semaphore_supported_ = false;
    // Whether VK_KHR_swapchain was supported and enabled; gates
    // Device::create_window.
    bool swapchain_supported_ = false;
    vk::Semaphore interop_semaphore_;
    std::uint64_t interop_semaphore_value_ = 0;
    std::shared_ptr<MemoryManager> host_memory_manager_;
    std::shared_ptr<MemoryManager> device_memory_manager_;
    std::vector<vk::CommandPool> command_pools_; // A command pool for each family index
    std::array<int, kMaxEngineTypes> engine_type_index_ = {-1, -1, -1, -1, -1, -1, -1, -1};
    std::array<std::uint32_t, kMaxEngineTypes> engine_queue_flags_ = { ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    std::array<std::uint32_t, kMaxEngineTypes> engine_queue_count_ = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<std::vector<std::shared_ptr<vk_Engine>>> engines_;
    std::vector<std::weak_ptr<vk_ResourceData>> resources_; // created resources
    std::vector<std::weak_ptr<vk_Framebuffer>> framebuffers_; // created framebuffers
    std::vector<std::weak_ptr<vk_Window>> windows_; // created windows
};

/**
 * Represents a memory as a slice (offset, size) of a page.
 */
class MemorySlice : public std::enable_shared_from_this<MemorySlice> {
public:
    MemorySlice(
        std::shared_ptr<Device> device, // as long as this memory live, the device should
        std::shared_ptr<MemoryPage> page,
        std::uint64_t allocated_offset,
        std::uint64_t allocated_size,
        std::uint64_t offset, std::uint64_t size) noexcept;
    ~MemorySlice() noexcept;
    MemorySlice() = default;
    MemorySlice(const MemorySlice&) = delete;
    MemorySlice& operator=(const MemorySlice&) = delete;
    MemorySlice(MemorySlice&& other) = delete;
    MemorySlice& operator=(MemorySlice&& other) = delete;
    // Gets the offset of the allocated memory, not necessarily aligned.
    [[nodiscard]] std::uint64_t allocated_offset() const noexcept { return allocated_offset_; }
    // Gets the size of the allocated memory, always greater or equals to the requested.
    [[nodiscard]] std::uint64_t allocated_size() const noexcept { return allocated_size_; }
    [[nodiscard]] std::uint64_t offset() const noexcept { return offset_; }
    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
    [[nodiscard]] vk::Buffer page_buffer() const noexcept;
    // This is the pointer in vulkan device address space, which can be used for shader addressing
    [[nodiscard]] std::uint64_t device_ptr() const noexcept;
    // This is the pointer in external manager, e.g.: cuda, rocm or LevelZero
    [[nodiscard]] std::uint64_t external_ptr() const noexcept;
    [[nodiscard]] bool host_visible() const noexcept ;
    [[nodiscard]] DLDevice dl_device() const noexcept;
    void release() noexcept;

private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<MemoryPage> page_;
    std::uint64_t allocated_offset_ = 0;
    std::uint64_t allocated_size_ = 0;
    std::uint64_t offset_ = 0;
    std::uint64_t size_ = 0;
};

/**
 * Represents an allocator/deallocator logic within an array of specific capacity.
 */
class MemoryAllocator {
public:
    MemoryAllocator(std::uint64_t capacity);
    ~MemoryAllocator() noexcept;

    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;
    MemoryAllocator(MemoryAllocator&&) = delete;
    MemoryAllocator& operator=(MemoryAllocator&&) = delete;
    /*
     * Gives the start offset of the allocated range.
     */
    std::uint64_t allocate(std::uint64_t size);
    [[nodiscard]] bool can_allocate(std::uint64_t size) const;
    /*
     * Free the range allocated starting in offset
     */
    void free_memory(std::uint64_t offset) noexcept;

private:
    struct Range {
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        Range* prev = nullptr;
        Range* next = nullptr;
        int bin = 0;
    };

    std::uint64_t capacity_ = 0;
    std::vector<std::unique_ptr<Range>> nodes_;
    Range* head_ = nullptr;
    std::vector<std::vector<Range*>> bins_;
    std::unordered_map<std::uint64_t, std::uint64_t> allocations_;

    [[nodiscard]] int bin_for_size(std::uint64_t size) const;
    void clear_bins();
    void rebuild_bins();
    void remove_node(Range* node) noexcept;
};

/**
 * Represents a manager of memory for a specific memory type.
 * Memory is reserved by pages and suballocations on them.
 */
class MemoryManager {
public:
    MemoryManager(std::weak_ptr<Device> device, uint32_t memory_type_index, bool host_visible);
    ~MemoryManager() noexcept;

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = delete;
    MemoryManager& operator=(MemoryManager&&) = delete;

    std::shared_ptr<MemorySlice> allocate(std::uint64_t size, int alignment);

    [[nodiscard]] std::uint64_t external_to_device(std::uint64_t external_ptr) const noexcept;
    [[nodiscard]] std::uint64_t device_to_external(std::uint64_t device_ptr) const noexcept;

    // Copies a (possibly strided, possibly host- or CUDA-resident) tensor
    // into/out of a contiguous external pointer, via the loaded interop
    // library's copy_from_dlpack_to_ptr/copy_ptr_to_dlpack symbols (see
    // Device::wrap). shape/strides follow the DLPack convention (`ndim`
    // entries, strides in units of `itemsize`-byte elements). Returns false
    // if no interop library implementing these is loaded -- not
    // necessarily an error, callers may have a host-only fallback.
    [[nodiscard]] bool vk_copy_from_dlpack(
        void* tensor_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
        std::uint64_t itemsize, std::uint64_t dst_ptr, std::uint64_t total_bytes) const;
    [[nodiscard]] bool vk_copy_to_dlpack(
        std::uint64_t src_ptr, std::uint64_t total_bytes,
        void* tensor_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
        std::uint64_t itemsize) const;

    // Vulkan memory type index this manager allocates from. Used by
    // Device::create_image to allocate a dedicated vk::DeviceMemory of the
    // same kind (HOST- or DEVICE-visible) as this manager's own pages.
    [[nodiscard]] uint32_t vk_memory_type_index() const noexcept { return memory_type_index_; }

    // Destroys every page's underlying vk::Buffer/vk::DeviceMemory using
    // `dev` directly instead of each page's own device_.lock(). Used by
    // Device::dispose(), right before resetting device_memory_manager_/
    // host_memory_manager_ -- see vk_Window::vk_dispose_with's docstring
    // for why this can't simply be page->device_.lock()->vk_device().
    void vk_dispose_with(vk::Device dev) noexcept;

private:
    // Imports Device::vk_interop_semaphore() into CUDA the first time it's
    // needed (memoized in interop_semaphore_handle_/interop_semaphore_import_attempted_,
    // since a failed/unsupported import should not be retried on every
    // copy). Returns 0 if unavailable for any reason.
    [[nodiscard]] std::uint64_t vk_interop_semaphore_handle() const;

    std::weak_ptr<Device> device_;
    uint32_t memory_type_index_ = 0;
    bool host_visible_ = false;
    std::uint64_t next_page_capacity_ = 1 << 30; // Start with 1 GiB pages.
    std::vector<std::shared_ptr<MemoryPage>> pages_;
    std::shared_ptr<ExternalInteropLibraryImpl> interop_library_;
    ExternalImportFn try_import_memory_ = nullptr;
    mutable bool interop_semaphore_import_attempted_ = false;
    mutable std::uint64_t interop_semaphore_handle_ = 0;
};

/**
 * Represents a page of memory allocated on the device. It can be suballocated by MemoryAllocator.
 */
class MemoryPage : public std::enable_shared_from_this<MemoryPage> {
public:

    MemoryPage(std::weak_ptr<Device> device, uint32_t memory_type_index, bool host_visible, std::uint64_t capacity, ExternalImportFn try_import_memory);
    ~MemoryPage() noexcept;
    // Tears down using `dev` directly instead of device_.lock(). Used by
    // MemoryManager::vk_dispose_with (in turn called from Device::dispose())
    // -- see vk_Window::vk_dispose_with's docstring for why this can't
    // simply be device_.lock()->vk_device() unconditionally.
    void vk_dispose_with(vk::Device dev) noexcept;

    MemoryPage(const MemoryPage&) = delete;
    MemoryPage& operator=(const MemoryPage&) = delete;
    MemoryPage(MemoryPage&&) = delete;
    MemoryPage& operator=(MemoryPage&&) = delete;

    bool can_allocate(std::uint64_t size, std::uint64_t alignment) const;
    std::shared_ptr<MemorySlice> allocate(std::uint64_t size, std::uint64_t alignment);
    void free_memory(std::uint64_t allocated_offset) noexcept;

    // Full capacity of this page.
    [[nodiscard]] std::uint64_t capacity() const noexcept;
    // Buffer for the full allocated space. Used for retrieving the device address.
    [[nodiscard]] vk::Buffer buffer() const noexcept;
    // Memory object for the full allocated space. Used for creating the full buffer.
    [[nodiscard]] vk::DeviceMemory memory() const noexcept;
    // Returns if this page is on CPU.
    [[nodiscard]] bool host_visible() const noexcept { return host_visible_; }
    // Pointer to the device address.
    [[nodiscard]] std::uint64_t device_ptr() const noexcept;
    // Represents the pointer of the memory visible outside the device, i.e., CPU, CUDA, ROCm, etc.
    [[nodiscard]] std::uint64_t external_ptr() const noexcept;

    std::uint64_t external_to_device(std::uint64_t external_ptr) const noexcept;
    std::uint64_t device_to_external(std::uint64_t device_ptr) const noexcept;

    [[nodiscard]] DLDevice dl_device() const noexcept { return dl_device_; }
private:
    std::weak_ptr<Device> device_;
    uint32_t memory_type_index_ = 0;
    bool host_visible_ = false;
    std::uint64_t capacity_ = 0;
    vk::Buffer buffer_{};
    vk::DeviceMemory memory_{};

    std::uint64_t device_ptr_ = 0;
    std::uint64_t external_ptr_ = 0;
    std::unique_ptr<MemoryAllocator> allocator_;
    DLDevice dl_device_;
    ExternalImportFn try_import_memory_ = nullptr;
};
