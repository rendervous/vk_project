#pragma once
#include <vulkan/vulkan.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <deque>
#include <string>
#include <variant>
#include <array>
#include <optional>
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
class vk_Sampler;
class Sampler;
class vk_AccelerationStructure;
class AccelerationStructure;
class Layouts;

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

// Every value a GPU-buffer-resident TypeDescriptor::single() can hold:
// scalars, plus every GLSL vector/matrix shape (matCxR: C columns, R rows).
// INT8/UINT8 aren't valid GLSL shader types but are kept for byte-level
// buffer/Format-channel use (Buffer::cast, create_buffer, format_scalar_type).
enum class Type : int {
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

    VEC2 = 13, VEC3 = 14, VEC4 = 15,
    IVEC2 = 16, IVEC3 = 17, IVEC4 = 18,
    UVEC2 = 19, UVEC3 = 20, UVEC4 = 21,
    BVEC2 = 22, BVEC3 = 23, BVEC4 = 24,

    // matCxR: C columns, R rows (GLSL convention).
    MAT2 = 25,   // 2x2
    MAT2x3 = 26,
    MAT2x4 = 27,
    MAT3x2 = 28,
    MAT3 = 29,   // 3x3
    MAT3x4 = 30,
    MAT4x2 = 31,
    MAT4x3 = 32,
    MAT4 = 33,   // 4x4
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

    // Depth/stencil (for Device::create_depth_buffer_image only -- not
    // valid as a color attachment/sampled/storage format).
    Depth16_UNorm             = static_cast<int>(vk::Format::eD16Unorm),
    Depth32_Float             = static_cast<int>(vk::Format::eD32Sfloat),
    Depth24_UNorm_Stencil8_UInt = static_cast<int>(vk::Format::eD24UnormS8Uint),
    Depth32_Float_Stencil8_UInt = static_cast<int>(vk::Format::eD32SfloatS8Uint),
};

// Whether `format` is one of the Format::Depth* values above.
[[nodiscard]] inline bool is_depth_format(Format format) noexcept {
    return format == Format::Depth16_UNorm || format == Format::Depth32_Float
        || format == Format::Depth24_UNorm_Stencil8_UInt || format == Format::Depth32_Float_Stencil8_UInt;
}

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
    // Ray tracing pipeline stages (PipelineType::RAYTRACING only). A
    // RAYGEN/MISS shader is turned into its own shader group via
    // Pipeline::append_raygen_group()/append_miss_group(); a CLOSEST_HIT/
    // ANY_HIT shader is combined into a "triangles hit group" via
    // Pipeline::append_hit_group().
    RAYGEN = 6,
    MISS = 7,
    CLOSEST_HIT = 8,
    ANY_HIT = 9,
};

enum class DescriptorType : int {
    STORAGE_BUFFER = 0,
    UNIFORM_BUFFER = 1,
    STORAGE_IMAGE = 2,
    SAMPLED_IMAGE = 3,
    SAMPLER = 4,
    COMBINED_IMAGE_SAMPLER = 5,
    // A ray tracing acceleration structure (top-level, typically), bound
    // via DescriptorSet::bind(layout_id, AccelerationStructure).
    ACCELERATION_STRUCTURE = 6,
};

// Texel filtering used by CommandBuffer::blit_image when src/dst extents
// differ. Values match vk::Filter's own, so a plain static_cast suffices.
enum class Filter : int {
    NEAREST = 0,
    LINEAR = 1,
};

// How a Sampler filters between mip levels (Sampler::mipmap_mode, set via
// Device::create_sampler). Values match vk::SamplerMipmapMode's own, so a
// plain static_cast suffices.
enum class MipmapMode : int {
    NEAREST = 0,
    LINEAR = 1,
};

// How a Sampler handles texture coordinates outside [0, 1] (one per axis:
// Device::create_sampler's wrap_u/wrap_v/wrap_w). Values match
// vk::SamplerAddressMode's own, so a plain static_cast suffices.
enum class WrapMode : int {
    REPEAT = 0,
    MIRRORED_REPEAT = 1,
    CLAMP_TO_EDGE = 2,
    // Texels outside [0, 1] read a fixed border color (opaque black --
    // Device::create_sampler has no separate border-color parameter).
    CLAMP_TO_BORDER = 3,
};

// Dynamic rasterization state -- see CommandBuffer::set_cull_mode/
// set_front_face/set_depth_test. Requires Vulkan 1.3 (core "extended
// dynamic state"); see Device::vk_extended_dynamic_state_supported().
enum class CullMode : int {
    NONE = 0,
    FRONT = 1,
    BACK = 2,
    FRONT_AND_BACK = 3,
};

enum class FrontFace : int {
    COUNTER_CLOCKWISE = 0,
    CLOCKWISE = 1,
};

enum class CompareOp : int {
    NEVER = 0,
    LESS = 1,
    EQUAL = 2,
    LESS_OR_EQUAL = 3,
    GREATER = 4,
    NOT_EQUAL = 5,
    GREATER_OR_EQUAL = 6,
    ALWAYS = 7,
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

// Opaque handle returned by Pipeline::stage() for a ray tracing stage
// (RAYGEN/MISS/CLOSEST_HIT/ANY_HIT), identifying that shader module for use
// with Pipeline::append_raygen_group()/append_miss_group()/append_hit_group().
// Not constructible from Python.
class ShaderHandle {
public:
    explicit ShaderHandle(int index) noexcept : index_(index) {}
    [[nodiscard]] int vk_index() const noexcept { return index_; }
private:
    int index_;
};

// Opaque handle returned by Pipeline::append_raygen_group()/
// append_miss_group()/append_hit_group(), identifying one shader group of a
// PipelineType::RAYTRACING pipeline. Not constructible from Python: purely
// a build-time acknowledgement (the shader binding table is built
// automatically, in creation order, at Pipeline::close()).
class ShaderGroupHandle {
public:
    explicit ShaderGroupHandle(int index) noexcept : index_(index) {}
    [[nodiscard]] int vk_index() const noexcept { return index_; }
private:
    int index_;
};

enum class TypeKind : int {
    // A scalar, vector or matrix value (see Layout::type for which):
    // distinguished from ARRAY/STRUCT, but no longer from each other --
    // Type already encodes that distinction on its own.
    SINGLE = 0,
    ARRAY = 1,
    STRUCT = 2
};

enum class LayoutRule : int {
    Std140 = 0,
    Std430 = 1,
    Scalar = 2
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
struct SingleDesc {
    Type type; // any scalar, vector or matrix Type value
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

// One shader group declared via Pipeline::append_raygen_group()/
// append_miss_group()/append_hit_group(), for a PipelineType::RAYTRACING
// pipeline. `general`/`closest_hit`/`any_hit` are indices into vk_Pipeline's
// own stages_ (VK_SHADER_UNUSED_KHR, i.e. -1 here, when not applicable to
// this group's kind).
struct vk_ShaderGroup {
    bool is_hit_group; // false: general (raygen/miss); true: triangles hit group
    int general = -1;
    int closest_hit = -1;
    int any_hit = -1;
};

/*  ============== FUNCTIONS ============== */

int scalar_type_size(Type type);

// Component type (itself, for a scalar), row count (components, for a
// vector) and column count (1 for scalar/vector) of any Type value.
struct TypeTraits {
    Type component_type;
    int rows;
    int columns;
};
TypeTraits type_traits(Type type);

// Computes the byte size/alignment layout of `type` under `rule` (std140,
// std430 or scalar block layout rules). Pure host-side arithmetic, does not
// require a Device.
std::shared_ptr<Layout> compute_layout(const TypeDescriptor& type, LayoutRule rule);

/*  ============== CLASSES ============== */

/**
 * Describes the shape of a GPU-buffer-resident type (a single scalar/
 * vector/matrix Type value, an array or a struct) as a flat hierarchy via
 * std::variant. Used together with LayoutRule and compute_layout() to
 * determine byte offsets/sizes/strides without touching the GPU.
 */
class TypeDescriptor {
public:
    using Payload = std::variant<SingleDesc, ArrayDesc, StructDesc>;

    static TypeDescriptor single(Type type);
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
 * element stride/count (ARRAY, or a matrix-shaped SINGLE).
 */
class Layout {
public:
    std::uint64_t size = 0;
    std::uint64_t alignment = 0;
    // Size this layout would occupy as one element of an array of itself,
    // i.e. size rounded up to alignment (and, under Std140, up to 16) --
    // the byte stride between consecutive elements. Used both to size
    // ARRAY/matrix strides and by Device::create_buffer(layout, ..., count > 1).
    std::uint64_t aligned_size = 0;
    TypeKind kind = TypeKind::SINGLE;
    // SINGLE only: the exact scalar/vector/matrix Type this Layout was
    // computed for (UNDEFINED for ARRAY/STRUCT).
    Type type = Type::UNDEFINED;
    // SINGLE only: `type`'s own base scalar component (itself, for a
    // scalar) -- UNDEFINED for ARRAY/STRUCT, since those don't have one
    // leaf type of their own.
    Type component_type = Type::UNDEFINED;

    // STRUCT only: ordered fields with their offsets and sub-layouts.
    std::vector<LayoutField> fields;

    // ARRAY, or a matrix-shaped SINGLE, only: layout of a single element
    // (ARRAY) or column (matrix), the byte stride between consecutive
    // elements/columns, and their count. Null for a scalar/vector SINGLE.
    std::shared_ptr<Layout> element_layout;
    std::uint64_t stride = 0;
    std::uint64_t count = 0;

    // Looks up a named field of this STRUCT layout (as produced by
    // compute_layout() on a TypeDescriptor::struct_of()), for navigating a
    // Layout tree by name (e.g. Layouts::instance()->field("transform")).
    // Chain into a nested struct via the returned field's own `.layout`
    // (itself a Layout, so `.field(...)` applies again), or into an
    // ARRAY/MATRIX field via `.layout->element_layout`. Throws
    // std::runtime_error if this layout isn't a STRUCT, or no field named
    // `name` exists.
    [[nodiscard]] const LayoutField& field(const std::string& name) const;
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
    // Index of the Device this resource was created from (Device::index in
    // Python) -- lets Python-level code hidden behind an implicit "current
    // device" still identify which device a returned object belongs to.
    // Defined out-of-line in device.cpp: Device isn't a complete type yet here.
    [[nodiscard]] std::uint32_t device_index() const noexcept;
protected:
    std::shared_ptr<vk_ResourceData> data_;
    ResourceSlice slice_{};
};

class Buffer: public Resource {
public:
    // `layout` describes a single element of this buffer: the scalar type
    // (Device::create_buffer(elements, Type, ...)), an array of a
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
    // Number of elements this buffer holds: size() / element_layout()->aligned_size.
    [[nodiscard]] std::uint64_t count() const noexcept { return size() / layout_->aligned_size; }

    // Reinterprets this buffer's bytes (same underlying offset/size) as a
    // flat array of `scalar` elements.
    std::shared_ptr<Buffer> cast(Type scalar) const;
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
        std::vector<std::uint64_t> shape, Type scalar_type);
    Tensor() = delete;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept = delete;
    Tensor& operator=(Tensor&& other) noexcept = delete;
    ~Tensor() noexcept override = default;

    [[nodiscard]] std::uint64_t size() const noexcept { return slice_.buffer.size; }
    [[nodiscard]] const std::vector<std::uint64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] Type scalar_type() const noexcept { return scalar_type_; }

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
    Type scalar_type_;
};

/**
 * A device-usable view over externally-owned memory, obtained from
 * Device::wrap(). If the wrapped object is directly accessible from both
 * the CPU and GPU sides (one of this library's own Buffers or Tensors; or
 * an already-contiguous DLPack tensor whose memory happens to already be
 * one of this Device's own allocations), this simply forwards its pointer
 * -- no copy is ever made, and every dirty/update method below is a no-op.
 *
 * Otherwise it owns a temporary Buffer standing in for the GPU side, and
 * tracks which side (the wrapped object, "cpu", or the temporary buffer,
 * "gpu") most recently changed via a pair of version counters. Call
 * make_cpu_dirty()/make_gpu_dirty() after modifying one side, and
 * update_cpu()/update_gpu() to lazily pull the other side's changes across
 * -- a copy only actually happens when the destination is stale.
 */
class WrappedMemory {
public:
    // Only meaningful internally, to know how (or whether) to copy across;
    // not exposed to Python.
    enum class SourceKind : int {
        NONE = 0,             // Directly mapped: no copy, ever.
        DLPACK = 1,           // Copy via `source`'s __dlpack__() export.
        BUFFER_PROTOCOL = 2,  // Copy via `source`'s buffer-protocol memory.
    };

    WrappedMemory(
        std::weak_ptr<Device> device,
        std::uint64_t device_ptr,
        std::vector<std::uint64_t> shape,
        Type scalar_type,
        std::shared_ptr<Buffer> owned_buffer,
        MemoryLocation owned_location,
        SourceKind source_kind,
        pybind11::object source) noexcept;
    WrappedMemory() = delete;
    WrappedMemory(const WrappedMemory&) = delete;
    WrappedMemory& operator=(const WrappedMemory&) = delete;
    WrappedMemory(WrappedMemory&& other) noexcept = delete;
    WrappedMemory& operator=(WrappedMemory&& other) noexcept = delete;
    ~WrappedMemory() noexcept;

    // Marks the CPU side (the wrapped Python object) as holding the
    // freshest data, so the next update_gpu() call will copy it across.
    // No-op for a direct mapping (see class comment).
    void make_cpu_dirty() noexcept;
    // Symmetric to make_cpu_dirty(): marks the GPU side (this wrap's own
    // device_ptr, e.g. after a shader wrote through it) as holding the
    // freshest data, so the next update_cpu() call will copy it back.
    void make_gpu_dirty() noexcept;
    // Copies GPU -> CPU (this wrap's temporary buffer back into the
    // wrapped Python object) only if the GPU side is currently the fresher
    // one; a no-op otherwise, and always a no-op for a direct mapping.
    // Exceptions raised while copying propagate to the caller.
    void update_cpu();
    // Symmetric to update_cpu(): copies CPU -> GPU (the wrapped Python
    // object into this wrap's temporary buffer) only if the CPU side is
    // currently the fresher one.
    void update_gpu();

    // Vulkan buffer device address of this wrapped memory -- a raw
    // pointer usable from within a shader (e.g. through
    // GL_EXT_buffer_reference), not to be confused with any
    // external-memory-exported (e.g. CUDA) pointer.
    [[nodiscard]] std::uint64_t device_ptr() const noexcept { return device_ptr_; }
    [[nodiscard]] const std::vector<std::uint64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] Type scalar_type() const noexcept { return scalar_type_; }
private:
    // True when device_ptr_ aliases the wrapped object's own memory
    // directly: there is only one copy of the data, so dirty/update calls
    // have nothing to do.
    [[nodiscard]] bool is_direct() const noexcept { return source_kind_ == SourceKind::NONE; }

    std::weak_ptr<Device> device_;
    std::uint64_t device_ptr_;
    std::vector<std::uint64_t> shape_;
    Type scalar_type_;
    // Non-null only when a temporary allocation backs device_ptr_ (i.e. the
    // source couldn't be used directly); keeps it alive and is the source/
    // destination of update_gpu()/update_cpu().
    std::shared_ptr<Buffer> owned_buffer_;
    // Where owned_buffer_ was allocated (HOST or DEVICE); irrelevant when
    // owned_buffer_ is null. Needed at copy time since a wrap() request can
    // ask for either location regardless of the source's own.
    MemoryLocation owned_location_;
    SourceKind source_kind_;
    // The original wrapped Python object; kept alive so its __dlpack__()/
    // buffer-protocol export can be revisited on every update_cpu()/
    // update_gpu() call. Heap allocated since pybind11::object (only
    // forward-declared here) can't be a direct member of a class defined
    // in this header.
    std::unique_ptr<pybind11::object> source_;
    // Version counters implementing lazy CPU/GPU synchronization: whichever
    // side has the strictly higher counter is the one holding fresh data.
    // update_cpu()/update_gpu() copy from the fresher side into the stale
    // one and then equalize the two counters; make_cpu_dirty()/
    // make_gpu_dirty() bump one strictly above the other. Meaningless (and
    // never touched) when is_direct() is true.
    std::uint64_t cpu_version_;
    std::uint64_t gpu_version_;
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

/**
 * Tool class with commonly-needed Layouts for ray tracing acceleration
 * structure buffers, precomputed once and cached: an ADSTriangles::indices
 * buffer (index16()/index32()), an ADSTriangles::vertices buffer
 * (position()), an ADSAABB::aabbs buffer (aabb(), VkAabbPositionsKHR-
 * shaped), and an ADSInstances::instances buffer (instance(),
 * VkAccelerationStructureInstanceKHR-shaped). Every layout here is computed
 * under LayoutRule::Scalar (tightly packed, natural alignment), matching
 * the corresponding Vulkan C struct's own byte layout exactly -- navigable
 * field-by-field via Layout::field(), e.g.
 * Layouts::instance()->field("transform").
 */
class Layouts {
public:
    // A single UINT16 triangle-mesh index (ADSTriangles::indices).
    [[nodiscard]] static const std::shared_ptr<Layout>& index16();
    // A single UINT32 triangle-mesh index (ADSTriangles::indices).
    [[nodiscard]] static const std::shared_ptr<Layout>& index32();
    // A single tightly-packed FLOAT32 vec3 vertex position
    // (ADSTriangles::vertices).
    [[nodiscard]] static const std::shared_ptr<Layout>& position();
    // A single VkAabbPositionsKHR-shaped procedural AABB entry
    // (ADSAABB::aabbs): fields "min_x"/"min_y"/"min_z"/"max_x"/"max_y"/"max_z".
    [[nodiscard]] static const std::shared_ptr<Layout>& aabb();
    // A single VkAccelerationStructureInstanceKHR-shaped TLAS instance entry
    // (ADSInstances::instances): fields "transform" (the row-major 3x4
    // affine transform, as an array of 3 vec4 rows), "instance_custom_index_
    // and_mask" and "instance_shader_binding_table_record_offset_and_flags"
    // (each packing two bitfields into one uint32 -- low 24 bits/high 8
    // bits respectively, since Layout has no bitfield notion of its own:
    // pack via `(mask << 24) | (custom_index & 0xFFFFFF)`), and
    // "acceleration_structure_reference" (the referenced BLAS's device
    // address, see AccelerationStructure::device_address()).
    [[nodiscard]] static const std::shared_ptr<Layout>& instance();
private:
    Layouts() = delete;
};

// Declares the geometry for a bottom-level acceleration structure (BLAS)
// built from a triangle mesh. `vertices` must be a DEVICE-resident buffer
// of tightly-packed FLOAT32 vec3 positions (e.g. Layouts::position());
// `vertex_count` defaults to vertices->count(). `indices`, if given, must
// be a DEVICE-resident buffer whose element_layout() is a scalar UINT16 or
// UINT32 (non-indexed triangles are used otherwise); `primitive_count`
// defaults to (indices ? indices : vertices)->count() / 3. `transform`, if
// given, is a static per-geometry 3x4 row-major transform (a
// VkTransformMatrixKHR-shaped buffer, e.g. Layouts::instance()'s
// "transform" field layout) applied at build time; no transform (identity)
// if null.
struct ADSTriangles {
    std::shared_ptr<Buffer> vertices;
    std::uint32_t vertex_count = 0;
    std::shared_ptr<Buffer> indices;
    std::uint32_t primitive_count = 0;
    std::shared_ptr<Buffer> transform;
    bool opaque = true;
};

// Declares the geometry for a bottom-level acceleration structure (BLAS)
// built from procedural AABBs. `aabbs` must be a DEVICE-resident buffer of
// Layouts::aabb()-shaped entries; `count` defaults to aabbs->count().
struct ADSAABB {
    std::shared_ptr<Buffer> aabbs;
    std::uint32_t count = 0;
    bool opaque = true;
};

// Declares a top-level acceleration structure (TLAS) built from instances
// of other, already-built bottom-level acceleration structures. `instances`
// must be a DEVICE-resident buffer of Layouts::instance()-shaped entries;
// `count` defaults to instances->count().
struct ADSInstances {
    std::shared_ptr<Buffer> instances;
    std::uint32_t count = 0;
};

// Declares the geometry backing one acceleration structure build:
// ADSTriangles/ADSAABB build a BLAS, ADSInstances builds a TLAS. Passed to
// Device::create_ads() (to size and allocate the acceleration structure)
// and CommandBuffer::build_ads() (to actually record the build).
using ADSDeclaration = std::variant<ADSTriangles, ADSAABB, ADSInstances>;

/**
 * Owns a vk::AccelerationStructureKHR and its dedicated backing buffer.
 * Not exposed to users directly: they interact with it through
 * AccelerationStructure.
 */
class vk_AccelerationStructure {
public:
    vk_AccelerationStructure(std::weak_ptr<Device> device, vk::AccelerationStructureKHR handle,
        vk::Buffer storage_buffer, vk::DeviceMemory storage_memory,
        vk::AccelerationStructureTypeKHR type, std::uint64_t build_scratch_size) noexcept;
    vk_AccelerationStructure() = delete;
    vk_AccelerationStructure(const vk_AccelerationStructure&) = delete;
    vk_AccelerationStructure& operator=(const vk_AccelerationStructure&) = delete;
    vk_AccelerationStructure(vk_AccelerationStructure&& other) noexcept = delete;
    vk_AccelerationStructure& operator=(vk_AccelerationStructure&& other) noexcept = delete;
    ~vk_AccelerationStructure() noexcept { dispose(); }
    // Destroys the underlying vk::AccelerationStructureKHR and its backing
    // buffer/memory now, if not already done. Idempotent. Called
    // proactively by Device::dispose() (see vk_Sampler::dispose's rationale),
    // and by the destructor as a fallback.
    void dispose() noexcept;
    [[nodiscard]] vk::AccelerationStructureKHR vk_handle() const noexcept { return handle_; }
    [[nodiscard]] vk::AccelerationStructureTypeKHR vk_type() const noexcept { return type_; }
    [[nodiscard]] std::uint64_t vk_build_scratch_size() const noexcept { return build_scratch_size_; }
    // Vulkan device address of this acceleration structure, queried lazily
    // (and memoized) the first time it's needed -- e.g. to fill in a TLAS
    // instance entry's "acceleration_structure_reference" field (see
    // Layouts::instance()).
    [[nodiscard]] std::uint64_t vk_device_address() const;
    // Defined out-of-line in device.cpp: Device isn't a complete type yet here.
    [[nodiscard]] std::uint32_t device_index() const;
private:
    std::weak_ptr<Device> device_;
    vk::AccelerationStructureKHR handle_;
    vk::Buffer storage_buffer_;
    vk::DeviceMemory storage_memory_;
    vk::AccelerationStructureTypeKHR type_;
    std::uint64_t build_scratch_size_;
    mutable std::uint64_t device_address_ = 0;
};

/**
 * User-facing handle to a built (or build-pending) acceleration structure,
 * obtained from Device::create_ads(). Pass it to CommandBuffer::build_ads()
 * to actually record its build, and to DescriptorSet::bind() (declared type
 * ACCELERATION_STRUCTURE) to use it from a shader, or reference it from a
 * TLAS instance entry via device_address().
 */
class AccelerationStructure {
public:
    explicit AccelerationStructure(std::shared_ptr<vk_AccelerationStructure> ads) noexcept : ads_(std::move(ads)) {}
    AccelerationStructure() = delete;
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure(AccelerationStructure&& other) noexcept = delete;
    AccelerationStructure& operator=(AccelerationStructure&& other) noexcept = delete;
    ~AccelerationStructure() noexcept = default;

    // This acceleration structure's Vulkan device address: for a BLAS, the
    // value to place in a TLAS instance entry's
    // "acceleration_structure_reference" field (see Layouts::instance()).
    [[nodiscard]] std::uint64_t device_address() const { return ads_->vk_device_address(); }
    [[nodiscard]] std::uint32_t device_index() const { return ads_->device_index(); }

    // Underlying vk_AccelerationStructure. Used internally by
    // CommandBuffer::build_ads, not exposed to Python.
    [[nodiscard]] const std::shared_ptr<vk_AccelerationStructure>& vk_ads() const noexcept { return ads_; }
private:
    std::shared_ptr<vk_AccelerationStructure> ads_;
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
    // Records a device-side copy of every mip level/array layer `source`
    // covers into `destination` (tightly packed, mip-by-mip, no row/slice
    // padding -- matching Device::create_staging(image)'s own sizing).
    // `source` is assumed to be in VK_IMAGE_LAYOUT_GENERAL (true of every
    // Image -- see its docstring). `destination` must be exactly
    // `source`'s byte size (see Device::create_staging(image)).
    void transfer(const std::shared_ptr<Image>& source, const std::shared_ptr<Buffer>& destination);
    // Symmetric to transfer(Image, Buffer): copies `source`'s bytes into
    // every mip level/array layer `destination` covers, in the same
    // tightly-packed layout.
    void transfer(const std::shared_ptr<Buffer>& source, const std::shared_ptr<Image>& destination);

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

    // Sets the dynamic cull mode used by subsequent draw calls. Graphics
    // pipelines only. Requires Device::vk_extended_dynamic_state_supported().
    void set_cull_mode(CullMode mode);

    // Sets the dynamic front-face winding order used by subsequent draw
    // calls. Graphics pipelines only. Requires
    // Device::vk_extended_dynamic_state_supported().
    void set_front_face(FrontFace front_face);

    // Sets dynamic depth testing state used by subsequent draw calls:
    // `enable` toggles the depth test itself, `write_enable` whether a
    // passing fragment writes its depth, and `compare_op` how the
    // fragment's depth compares against the depth buffer. Requires an
    // active render pass (set_framebuffer()) whose pipeline declared a
    // depth attachment via Pipeline::attach_depth(), and
    // Device::vk_extended_dynamic_state_supported(). Graphics pipelines
    // only.
    void set_depth_test(bool enable, bool write_enable = true, CompareOp compare_op = CompareOp::LESS);

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

    // Records a build of `ads` (previously sized/allocated by
    // Device::create_ads()) from the geometry described by `declaration`,
    // which must be the same kind (ADSTriangles/ADSAABB vs ADSInstances) it
    // was created with -- allocates a temporary scratch buffer for the
    // build, kept alive (like every buffer/acceleration structure
    // referenced here) for as long as this command buffer is, including
    // while submitted and pending on the GPU. Requires a COMPUTE-capable
    // engine.
    void build_ads(const std::shared_ptr<AccelerationStructure>& ads, const ADSDeclaration& declaration);

    // Records a ray tracing dispatch (vkCmdTraceRaysKHR) of `width` x
    // `height` x `depth` rays, using the shader binding table built by the
    // bound pipeline's close() (see Pipeline::append_raygen_group()/
    // append_miss_group()/append_hit_group()). Requires a
    // PipelineType::RAYTRACING pipeline to be bound (set_pipeline()).
    void trace_rays(std::uint32_t width, std::uint32_t height, std::uint32_t depth = 1);

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

    // Index of the Device/EngineType/engine this command buffer was created
    // from (Engine::create_command_buffer) -- lets Python-level code hidden
    // behind an implicit "current device"/"current engine" still identify
    // which one produced a given command buffer. Throw if already
    // release()d, since device_/engine_ are reset then. Defined out-of-line
    // in device.cpp: Device/Engine aren't complete types yet here.
    [[nodiscard]] std::uint32_t device_index() const;
    [[nodiscard]] EngineType engine_type() const;
    [[nodiscard]] std::uint32_t engine_index() const;
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
    // Framebuffers (set_framebuffer), buffers (bind_vertices/bind_indices/
    // transfer) and images (blit_image/transfer) ever referenced during
    // this recording, kept alive the same way and for the same reason as
    // bound_pipelines_/bound_descriptor_sets_ above.
    std::vector<std::shared_ptr<Framebuffer>> bound_framebuffers_;
    std::vector<std::shared_ptr<Buffer>> bound_vertex_index_buffers_;
    std::vector<std::shared_ptr<Image>> bound_images_;
    // Acceleration structures built (build_ads), the geometry input buffers
    // they were built from, and their temporary scratch buffers, kept alive
    // the same way and for the same reason as bound_pipelines_ above.
    std::vector<std::shared_ptr<AccelerationStructure>> bound_acceleration_structures_;
    std::vector<std::shared_ptr<Buffer>> bound_ads_input_buffers_;
    std::vector<std::shared_ptr<Buffer>> bound_scratch_buffers_;
    // Whether a render pass begun by set_framebuffer() is still active
    // (i.e. hasn't been ended by a later set_framebuffer() or close()).
    bool render_pass_active_ = false;
    // Whether the framebuffer bound by the most recent set_framebuffer()
    // has a depth attachment -- gates set_depth_test(true, ...).
    bool active_framebuffer_has_depth_ = false;
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
    [[nodiscard]] EngineType engine_type() const noexcept { return engine_type_; }
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
    // Defined out-of-line in device.cpp: Device isn't a complete type yet here.
    [[nodiscard]] std::uint32_t device_index() const noexcept;
    [[nodiscard]] EngineType engine_type() const noexcept { return engine_->engine_type(); }

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
 * compiled at load time), or GLSL source text compiled on the spot
 * in-process via glslang (statically linked into this extension). There is
 * no build- or runtime dependency on a separately installed shader compiler.
 */
class ShaderSource {
public:
    static ShaderSource from_spirv(std::vector<std::uint32_t> code, const std::string& entry_point = "main");
    // `include_dirs` are searched, in order, for `#include "..."` targets,
    // only consulted when `path` doesn't end in ".spv" (i.e. is compiled as GLSL).
    static ShaderSource from_file(
        const std::string& path,
        ShaderStageType stage,
        const std::string& entry_point = "main",
        const std::vector<std::string>& include_dirs = {});
    // `entry_point` is the name of the entry function as written in
    // `source`; the compiled SPIR-V's own entry point is always named the
    // same. `include_dirs` are searched, in order, for `#include "..."` targets.
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

    // Returns the index of the newly appended stage within stages_ -- only
    // meaningful for a ray tracing stage (RAYGEN/MISS/CLOSEST_HIT/ANY_HIT),
    // to later reference it from append_raygen_group()/append_miss_group()/
    // append_hit_group().
    int vk_stage(ShaderStageType type, const ShaderSource& source);
    int vk_layout(int set, int binding, DescriptorType description, int count);
    void vk_vertex_layout(int start_location, const Layout& layout);
    int vk_attach(int slot, Format format);
    // Declares this pipeline's depth/stencil attachment (at most one).
    // `format` must satisfy is_depth_format(). Requires
    // Device::vk_extended_dynamic_state_supported() (depth test/write/
    // compare op are only ever set dynamically -- see
    // CommandBuffer::set_depth_test() -- never baked into the pipeline).
    // Graphics pipelines only. Must be called before close().
    void vk_attach_depth(Format format);
    // Declares the compute shader's own workgroup size (its
    // `layout(local_size_x = ..., ...)` declaration in GLSL). Must match the
    // actual shader code -- there is no way to verify this from the compiled
    // SPIR-V alone. Used by CommandBuffer::dispatch_threads to compute the
    // number of workgroups covering a requested thread count. Compute
    // pipelines only; defaults to (1, 1, 1) if never called.
    void vk_set_local_size(std::uint32_t x, std::uint32_t y, std::uint32_t z);
    // Ray tracing pipelines only. `shader_index` (a ShaderHandle from
    // stage(RAYGEN, ...)/stage(MISS, ...)) becomes its own shader group;
    // returns the group's SBT index within its own category (raygen groups
    // and miss groups are each numbered independently, starting at 0, in
    // creation order) -- exactly one raygen group is required.
    int vk_append_raygen_group(int shader_index);
    int vk_append_miss_group(int shader_index);
    // Ray tracing pipelines only. Combines up to one CLOSEST_HIT and one
    // ANY_HIT shader (-1 for unused) into a triangles hit group; returns
    // its SBT index among hit groups, in creation order.
    int vk_append_hit_group(int closest_hit_index, int any_hit_index);
    // Ray tracing pipelines only. Maximum nested traceRayEXT() recursion
    // depth to build the pipeline for (VkRayTracingPipelineCreateInfoKHR::
    // maxPipelineRayRecursionDepth). Must be called before close(); defaults
    // to 1 (a raygen shader may trace, but a hit/miss shader may not trace
    // again) if never called.
    void vk_stack_size(int depth);
    void vk_close();

    [[nodiscard]] vk::DescriptorSet vk_allocate_descriptor_set(int set);
    [[nodiscard]] const vk_DescriptorBinding& vk_binding(int layout_id) const;
    [[nodiscard]] vk::RenderPass vk_render_pass() const noexcept { return render_pass_; }
    [[nodiscard]] const std::vector<vk_AttachmentDesc>& vk_attachments() const noexcept { return attachments_; }
    [[nodiscard]] bool vk_has_depth_attachment() const noexcept { return depth_attachment_format_.has_value(); }
    [[nodiscard]] Format vk_depth_attachment_format() const noexcept { return *depth_attachment_format_; }
    // Underlying vk::Pipeline and vk::PipelineLayout. Used internally by
    // CommandBuffer::set_pipeline, not exposed to Python.
    [[nodiscard]] vk::Pipeline vk_handle() const noexcept { return pipeline_; }
    [[nodiscard]] vk::PipelineLayout vk_pipeline_layout() const noexcept { return pipeline_layout_; }
    [[nodiscard]] std::uint32_t vk_local_size_x() const noexcept { return local_size_x_; }
    [[nodiscard]] std::uint32_t vk_local_size_y() const noexcept { return local_size_y_; }
    [[nodiscard]] std::uint32_t vk_local_size_z() const noexcept { return local_size_z_; }
    [[nodiscard]] bool is_closed() const noexcept { return closed_; }
    [[nodiscard]] PipelineType type() const noexcept { return type_; }

    // Shader binding table regions built by close() (RAYTRACING only), for
    // CommandBuffer::trace_rays(). Used internally, not exposed to Python.
    [[nodiscard]] const vk::StridedDeviceAddressRegionKHR& vk_raygen_region() const noexcept { return raygen_region_; }
    [[nodiscard]] const vk::StridedDeviceAddressRegionKHR& vk_miss_region() const noexcept { return miss_region_; }
    [[nodiscard]] const vk::StridedDeviceAddressRegionKHR& vk_hit_region() const noexcept { return hit_region_; }
    [[nodiscard]] const vk::StridedDeviceAddressRegionKHR& vk_callable_region() const noexcept { return callable_region_; }

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
    std::optional<Format> depth_attachment_format_;

    // Ray tracing pipelines only: one vk_ShaderGroup per append_raygen_group/
    // append_miss_group/append_hit_group call, in each category's own
    // creation order -- concatenated as [raygen..., miss..., hit...] to
    // build VkRayTracingPipelineCreateInfoKHR::pGroups at close(), and to
    // retrieve shader group handles (vkGetRayTracingShaderGroupHandlesKHR)
    // in that same, matching order for the shader binding table.
    std::vector<vk_ShaderGroup> raygen_groups_;
    std::vector<vk_ShaderGroup> miss_groups_;
    std::vector<vk_ShaderGroup> hit_groups_;
    std::uint32_t max_ray_recursion_depth_ = 1;

    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts_;
    vk::PipelineLayout pipeline_layout_;
    vk::RenderPass render_pass_;
    vk::Pipeline pipeline_;
    vk::DescriptorPool descriptor_pool_;

    // Shader binding table (RAYTRACING only): one buffer backing all three
    // regions (raygen/miss/hit), built once at close() from the shader
    // group handles the driver returns.
    vk::Buffer sbt_buffer_;
    vk::DeviceMemory sbt_memory_;
    vk::StridedDeviceAddressRegionKHR raygen_region_{};
    vk::StridedDeviceAddressRegionKHR miss_region_{};
    vk::StridedDeviceAddressRegionKHR hit_region_{};
    vk::StridedDeviceAddressRegionKHR callable_region_{};
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
    // called before close(). Returns a handle identifying this shader
    // module, meaningful only for a ray tracing stage (RAYGEN/MISS/
    // CLOSEST_HIT/ANY_HIT) -- pass it to append_raygen_group()/
    // append_miss_group()/append_hit_group().
    ShaderHandle stage(ShaderStageType type, const ShaderSource& source) {
        return ShaderHandle(pipeline_->vk_stage(type, source));
    }

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

    // Declares this pipeline's depth (or depth/stencil) attachment, enabling
    // CommandBuffer::set_depth_test() and depth-buffer-aware rendering.
    // `format` must satisfy is_depth_format(). At most one per pipeline.
    // Requires Device::vk_extended_dynamic_state_supported() (Vulkan 1.3).
    // Graphics pipelines only. Must be called before close().
    void attach_depth(Format format) { pipeline_->vk_attach_depth(format); }

    // Declares the compute shader's own workgroup size (its
    // `layout(local_size_x = ..., ...)` declaration in GLSL) so that
    // CommandBuffer::dispatch_threads can compute the number of workgroups
    // needed to cover a requested thread count. Must match the actual
    // shader code. Compute pipelines only; defaults to (1, 1, 1) if never
    // called. Must be called before close().
    void local_size(int x, int y = 1, int z = 1) {
        pipeline_->vk_set_local_size(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(z));
    }

    // Turns a RAYGEN shader (from stage()) into its own shader group.
    // Exactly one is required. Ray tracing pipelines only. Must be called
    // before close().
    ShaderGroupHandle append_raygen_group(ShaderHandle shader) {
        return ShaderGroupHandle(pipeline_->vk_append_raygen_group(shader.vk_index()));
    }
    // Turns a MISS shader (from stage()) into its own shader group. Ray
    // tracing pipelines only. Must be called before close().
    ShaderGroupHandle append_miss_group(ShaderHandle shader) {
        return ShaderGroupHandle(pipeline_->vk_append_miss_group(shader.vk_index()));
    }
    // Combines up to one CLOSEST_HIT and/or one ANY_HIT shader (from
    // stage()) into a triangles hit group. Ray tracing pipelines only.
    // Must be called before close().
    ShaderGroupHandle append_hit_group(std::optional<ShaderHandle> closest_hit = std::nullopt, std::optional<ShaderHandle> any_hit = std::nullopt) {
        return ShaderGroupHandle(pipeline_->vk_append_hit_group(
            closest_hit ? closest_hit->vk_index() : -1,
            any_hit ? any_hit->vk_index() : -1));
    }
    // Maximum nested traceRayEXT() recursion depth (default 1 if never
    // called). Ray tracing pipelines only. Must be called before close().
    void stack_size(int depth) { pipeline_->vk_stack_size(depth); }

    // Finalizes this pipeline: builds the descriptor set layouts, pipeline
    // layout, (for graphics pipelines) render pass and vertex input state,
    // and the underlying vk::Pipeline. No further stage()/layout()/
    // vertex_layout()/attach() calls are allowed afterwards.
    void close() { pipeline_->vk_close(); }

    // Creates a framebuffer compatible with this (closed) pipeline's render
    // pass. `attachments` must provide exactly one image per slot declared
    // via attach(), matching format and dimensions. `depth_image` must be
    // provided (matching attach_depth()'s format and attachments' dimensions)
    // if and only if attach_depth() was called on this pipeline. Graphics
    // pipelines only.
    [[nodiscard]] std::shared_ptr<Framebuffer> create_framebuffer(
        std::vector<std::pair<AttachHandle, std::shared_ptr<Image>>> attachments,
        std::shared_ptr<Image> depth_image = nullptr);

    // Allocates a new descriptor set matching the layout declared for `set`
    // via layout(). Pipeline must be closed first.
    [[nodiscard]] std::shared_ptr<DescriptorSet> descriptor_set(int set = 0);
    // Allocates `count` independent descriptor sets matching the layout
    // declared for `set` -- e.g. one per object to draw/dispatch with its
    // own bindings, sharing the same Pipeline layout. Pipeline must be
    // closed first.
    [[nodiscard]] std::vector<std::shared_ptr<DescriptorSet>> descriptor_set_collection(int set = 0, int count = 1);

    [[nodiscard]] bool is_closed() const noexcept { return pipeline_->is_closed(); }

    [[nodiscard]] std::shared_ptr<vk_Pipeline> vk_pipeline() const noexcept { return pipeline_; }
    // Defined out-of-line in device.cpp: Device isn't a complete type yet here.
    [[nodiscard]] std::uint32_t device_index() const noexcept;
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
        std::uint32_t attachment_count, std::uint32_t width, std::uint32_t height, bool has_depth) noexcept;
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
    // Whether the last attachment (see attachment_count()) is a depth/
    // stencil attachment (Pipeline::attach_depth() was called) -- used by
    // CommandBuffer::set_framebuffer (an extra depth/stencil clear value)
    // and set_depth_test (only valid while such a framebuffer is active).
    [[nodiscard]] bool has_depth() const noexcept { return has_depth_; }
    // Defined out-of-line in device.cpp: Device isn't a complete type yet here.
    [[nodiscard]] std::uint32_t device_index() const;
private:
    std::weak_ptr<Device> device_;
    vk::Framebuffer framebuffer_;
    vk::RenderPass render_pass_;
    std::uint32_t attachment_count_;
    std::uint32_t width_;
    std::uint32_t height_;
    bool has_depth_;
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
    [[nodiscard]] bool vk_has_depth() const noexcept { return framebuffer_->has_depth(); }
    [[nodiscard]] std::uint32_t device_index() const { return framebuffer_->device_index(); }
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
        const std::string& title, Format format, std::uint32_t frames_on_the_fly, bool vsync = true);
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
    // Defined out-of-line in device.cpp: Device isn't a complete type yet here.
    [[nodiscard]] std::uint32_t device_index() const;

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
    // Whether create_swapchain() picks the vsync'd Fifo present mode
    // (always supported; the default) or tries to disable vsync (Mailbox,
    // falling back to Immediate, falling back to Fifo if this surface
    // supports neither) -- see Device::create_window's `vsync` parameter.
    bool vsync_ = true;
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
    // docstring for why the two are one and the same here. `vsync` selects
    // the vsync'd (Fifo, default) or uncapped (Mailbox/Immediate, falling
    // back to Fifo if unsupported) present mode.
    Window(std::shared_ptr<Device> device, std::uint32_t width, std::uint32_t height, const std::string& title,
        Format format, std::uint32_t frames_on_the_fly = 3, bool vsync = true);
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
    [[nodiscard]] std::uint32_t device_index() const { return window_->device_index(); }

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


enum class VertexAttribute {
    POSITION,
    NORMAL,
    TEXCOORD,
    TANGENT,
    BITANGENT
};

// How a loader (see load_scene) decides whether two face-corners referring
// to the same source data resolve to the same output vertex.
enum class VertexResolutionMode {
    // Weld by position only: face-corners sharing a position index become
    // one output vertex, keeping whichever corner's normal/texcoord was
    // encountered first (later corners' differing normal/texcoord, if any,
    // are discarded).
    ByPosition,
    // Weld by the full (position, normal, texcoord) tuple: two face-corners
    // share a vertex only if all of their attributes match. Default.
    ByAllAttributes,
    // No welding: every face-corner gets its own vertex, even if identical
    // to another one.
    Duplicate,
};

class Mesh {
public:
    Mesh(std::shared_ptr<Buffer> vertices, std::shared_ptr<Buffer> indices, std::vector<VertexAttribute> attributes)
        : vertices_(std::move(vertices)), indices_(std::move(indices)), attributes_(std::move(attributes)) {}
    Mesh() = delete;
    Mesh(const Mesh&) = delete;
    Mesh(Mesh&&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh& operator=(Mesh&&) = delete;
    virtual ~Mesh() = default;

    [[nodiscard]] const std::shared_ptr<Buffer>& vertices() const noexcept { return vertices_; }
    [[nodiscard]] const std::shared_ptr<Buffer>& indices() const noexcept { return indices_; }
    [[nodiscard]] const std::vector<VertexAttribute>& attributes() const noexcept { return attributes_; }
private:
    std::shared_ptr<Buffer> vertices_;
    std::shared_ptr<Buffer> indices_;
    std::vector<VertexAttribute> attributes_;
};

// A single named material property. Deliberately a closed set of plain
// value kinds (not a fixed schema of named fields) so any loader can stash
// whatever it finds -- e.g. ("diffuse", vec3), ("shininess", 42.0f),
// ("diffuse_map", "wood.png") -- without Material needing to know about
// every possible property up front.
using MaterialValue = std::variant<float, std::array<float, 3>, std::string>;

class Material {
public:
    Material() = default;
    Material(const Material&) = delete;
    Material& operator=(const Material&) = delete;
    Material(Material&&) = delete;
    Material& operator=(Material&&) = delete;

    void set(std::string name, MaterialValue value) { properties_[std::move(name)] = std::move(value); }
    [[nodiscard]] const MaterialValue* find(const std::string& name) const noexcept {
        auto it = properties_.find(name);
        return it == properties_.end() ? nullptr : &it->second;
    }
    [[nodiscard]] const std::unordered_map<std::string, MaterialValue>& properties() const noexcept { return properties_; }
private:
    std::unordered_map<std::string, MaterialValue> properties_;
};

class SceneNode {
public:
    SceneNode(std::string name, std::shared_ptr<Material> material, std::shared_ptr<Mesh> mesh, std::shared_ptr<float[3][4]> transform)
        : name_(std::move(name)), material_(std::move(material)), mesh_(std::move(mesh)), transform_(std::move(transform)) {}

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const std::shared_ptr<Material>& material() const noexcept { return material_; }
    [[nodiscard]] const std::shared_ptr<Mesh>& mesh() const noexcept { return mesh_; }
    // Null when the source format has no per-node transform concept (e.g. OBJ).
    [[nodiscard]] const std::shared_ptr<float[3][4]>& transform() const noexcept { return transform_; }
private:
    std::string name_;
    std::shared_ptr<Material> material_;
    std::shared_ptr<Mesh> mesh_;
    std::shared_ptr<float[3][4]> transform_;
};

class Scene {
public:
    Scene() = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(Scene&&) = delete;

    void add_node(SceneNode node) { nodes_.push_back(std::move(node)); }
    [[nodiscard]] const std::vector<SceneNode>& nodes() const noexcept { return nodes_; }
private:
    std::vector<SceneNode> nodes_;
};

// Loads a scene file into device-resident Meshes, dispatching on
// `filename`'s extension. Only ".obj" (via tinyobjloader) is implemented;
// any other extension throws std::runtime_error. `resolution_mode`
// controls vertex welding (see VertexResolutionMode). A group/shape that
// references more than one material produces one SceneNode per material
// used within it (all sharing the group's name).
std::shared_ptr<Scene> load_scene(
    const std::shared_ptr<Device>& device,
    const std::string& filename,
    VertexResolutionMode resolution_mode = VertexResolutionMode::ByAllAttributes);


/**
 * Owns the vk::Sampler created by Device::create_sampler. Not exposed to
 * users directly: they interact with it through Sampler.
 */
class vk_Sampler {
public:
    vk_Sampler(std::weak_ptr<Device> device, vk::Sampler sampler) noexcept;
    vk_Sampler() = delete;
    vk_Sampler(const vk_Sampler&) = delete;
    vk_Sampler& operator=(const vk_Sampler&) = delete;
    vk_Sampler(vk_Sampler&& other) noexcept = delete;
    vk_Sampler& operator=(vk_Sampler&& other) noexcept = delete;
    ~vk_Sampler() noexcept { dispose(); }
    // Destroys the underlying vk::Sampler now, if not already done.
    // Idempotent. Called proactively by Device::dispose() (see
    // vk_Window::vk_dispose_with's docstring for why this can't simply be
    // left to ~vk_Sampler() running whenever Python happens to drop the
    // last reference), and by the destructor as a fallback.
    void dispose() noexcept;
    [[nodiscard]] vk::Sampler vk_handle() const noexcept { return sampler_; }
    // Defined out-of-line in device.cpp: Device isn't a complete type yet here.
    [[nodiscard]] std::uint32_t device_index() const;
private:
    std::weak_ptr<Device> device_;
    vk::Sampler sampler_;
};

/**
 * User-facing handle to a texture sampler (filtering + wrap modes),
 * obtained from Device::create_sampler. Used together with an Image via
 * DescriptorSet::bind, either as a standalone DescriptorType::SAMPLER
 * binding (paired with a separate DescriptorType::SAMPLED_IMAGE binding,
 * e.g. GLSL `texture2D`/`sampler` combined manually into a `sampler2D` in
 * the shader), or directly combined with an image in one
 * DescriptorType::COMBINED_IMAGE_SAMPLER binding (GLSL `sampler2D`).
 */
class Sampler {
public:
    explicit Sampler(std::shared_ptr<vk_Sampler> sampler) noexcept : sampler_(std::move(sampler)) {}
    Sampler() = delete;
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&& other) noexcept = delete;
    Sampler& operator=(Sampler&& other) noexcept = delete;
    ~Sampler() noexcept = default;

    // Underlying vk::Sampler. Used internally by DescriptorSet::bind, not
    // exposed to Python.
    [[nodiscard]] vk::Sampler vk_sampler() const noexcept { return sampler_->vk_handle(); }
    [[nodiscard]] std::uint32_t device_index() const { return sampler_->device_index(); }
private:
    std::shared_ptr<vk_Sampler> sampler_;
};

/**
 * Owns the vk::DescriptorSet allocated by Pipeline::descriptor_set,
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
    // Binding's declared type must be SAMPLER.
    void vk_bind_sampler(LayoutHandle layout_id, const std::shared_ptr<Sampler>& sampler);
    // Binding's declared type must be COMBINED_IMAGE_SAMPLER.
    void vk_bind_combined(LayoutHandle layout_id, const std::shared_ptr<Image>& image, const std::shared_ptr<Sampler>& sampler);
    // Binding's declared type must be ACCELERATION_STRUCTURE.
    void vk_bind_ads(LayoutHandle layout_id, const std::shared_ptr<AccelerationStructure>& ads);
    // Underlying vk::DescriptorSet. Used internally by
    // CommandBuffer::set_pipeline, not exposed to Python.
    [[nodiscard]] vk::DescriptorSet vk_handle() const noexcept { return descriptor_set_; }
    // Defined out-of-line in device.cpp: Device isn't a complete type yet here.
    [[nodiscard]] std::uint32_t device_index() const;
private:
    std::weak_ptr<Device> device_;
    std::shared_ptr<vk_Pipeline> pipeline_;
    int set_;
    vk::DescriptorSet descriptor_set_;
};

/**
 * User-facing handle to a descriptor set matching one Pipeline's declared
 * layout for a given `set` index. Obtained from Pipeline::descriptor_set.
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
    // binding's declared type must be STORAGE_IMAGE (read/write, no
    // sampler) or SAMPLED_IMAGE (read-only, sampled in the shader via a
    // separately-bound SAMPLER at another binding -- see the other
    // overload for a single combined binding instead).
    void bind(LayoutHandle layout_id, const std::shared_ptr<Image>& image) { descriptor_set_->vk_bind_image(layout_id, image); }
    // Writes `sampler` into the binding identified by `layout_id`. The
    // binding's declared type must be SAMPLER (paired with a separate
    // SAMPLED_IMAGE binding, e.g. GLSL `texture2D tex; sampler s;`).
    void bind(LayoutHandle layout_id, const std::shared_ptr<Sampler>& sampler) { descriptor_set_->vk_bind_sampler(layout_id, sampler); }
    // Writes `image`+`sampler` together into the binding identified by
    // `layout_id`. The binding's declared type must be
    // COMBINED_IMAGE_SAMPLER (GLSL `sampler2D`).
    void bind(LayoutHandle layout_id, const std::shared_ptr<Image>& image, const std::shared_ptr<Sampler>& sampler) {
        descriptor_set_->vk_bind_combined(layout_id, image, sampler);
    }
    // Writes `ads` into the binding identified by `layout_id`. The
    // binding's declared type must be ACCELERATION_STRUCTURE.
    void bind(LayoutHandle layout_id, const std::shared_ptr<AccelerationStructure>& ads) {
        descriptor_set_->vk_bind_ads(layout_id, ads);
    }

    // Underlying vk_DescriptorSet. Used internally by
    // CommandBuffer::set_pipeline, not exposed to Python.
    [[nodiscard]] std::shared_ptr<vk_DescriptorSet> vk_descriptor_set() const noexcept { return descriptor_set_; }
    [[nodiscard]] std::uint32_t device_index() const { return descriptor_set_->device_index(); }
private:
    std::shared_ptr<vk_DescriptorSet> descriptor_set_;
};

// Read-only Vulkan physical device info, gathered without creating any
// logical Device -- see query_device_infos(). Not exposed to Python as a
// class: bindings.cpp converts each one into a plain dict.
struct DeviceInfo {
    std::uint32_t index;
    std::string name;
    std::string vendor;
    std::uint32_t vendor_id;
    std::uint32_t device_id;
    std::string device_type;
    std::string api_version;
    std::uint32_t driver_version;
    std::uint64_t vram_bytes;
};

// Enumerates every Vulkan-visible physical device without creating a
// logical Device for any of them: a throwaway vk::Instance is created,
// queried, and destroyed within this call alone. `index` in each result
// is what Device::create_device()/create_device() expects.
std::vector<DeviceInfo> query_device_infos();

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
    // just this with type == Type::UINT8.
    [[nodiscard]] std::shared_ptr<Buffer> create_buffer(std::uint64_t elements, Type type, MemoryLocation location);
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
    // Creates a 2D depth (or depth/stencil) attachment image for use with
    // Pipeline::attach_depth()/create_framebuffer() and
    // CommandBuffer::set_depth_test(). `format` must be one of the
    // Format::Depth* values (see is_depth_format()); unlike create_image(),
    // its usage is depth/stencil-attachment + sampled (not color-attachment/
    // storage, which are invalid for a depth format). Like every other
    // Image, it's transitioned once to VK_IMAGE_LAYOUT_GENERAL at creation.
    [[nodiscard]] std::shared_ptr<Image> create_depth_buffer_image(int width, int height, Format format = Format::Depth32_Float, MemoryLocation location = MemoryLocation::DEVICE);
    // Creates a texture sampler (filtering + per-axis wrap modes), for use
    // with an Image via DescriptorSet::bind (either a standalone SAMPLER
    // binding, or combined with an image in one COMBINED_IMAGE_SAMPLER
    // binding). CLAMP_TO_BORDER always uses an opaque black border color.
    // Mip levels beyond an image's own are simply never sampled (minLod=0,
    // maxLod effectively unbounded); there is no LOD bias/clamp control.
    [[nodiscard]] std::shared_ptr<Sampler> create_sampler(
        Filter mag_filter = Filter::LINEAR,
        Filter min_filter = Filter::LINEAR,
        MipmapMode mipmap_mode = MipmapMode::LINEAR,
        WrapMode wrap_u = WrapMode::REPEAT,
        WrapMode wrap_v = WrapMode::REPEAT,
        WrapMode wrap_w = WrapMode::REPEAT);
    // Creates a Window: an GLFW-backed OS window plus its Vulkan swapchain
    // and presentation machinery. See Window's docstring for the render
    // loop this is meant to drive. The swapchain itself is always created
    // as Format::BGRA8_UNorm (the only format broadly guaranteed to be
    // supported); `format` instead controls image_target/buffer_target/
    // tensor_target, which may be any format -- presenting one of them
    // blits it into the BGRA8_UNorm swapchain image, converting formats
    // along the way. `vsync` (default true) selects the present mode:
    // true picks the always-supported, vsync'd Fifo; false tries Mailbox
    // (uncapped, no tearing), falling back to Immediate (uncapped, can
    // tear), falling back to Fifo if this surface supports neither.
    [[nodiscard]] std::shared_ptr<Window> create_window(std::uint32_t width, std::uint32_t height,
        const std::string& title, Format format, std::uint32_t frames_on_the_fly = 3, bool vsync = true);
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
    [[nodiscard]] std::shared_ptr<Tensor> create_tensor(const std::vector<std::uint64_t>& shape, Type type, MemoryLocation location);

    // Creates (sizes and allocates, but does not build -- see
    // CommandBuffer::build_ads()) an acceleration structure: a bottom-level
    // one (BLAS) for an ADSTriangles/ADSAABB declaration, or a top-level one
    // (TLAS) for an ADSInstances declaration. Throws std::runtime_error if
    // VK_KHR_acceleration_structure isn't supported/enabled on this device
    // (see vk_acceleration_structure_supported()).
    [[nodiscard]] std::shared_ptr<AccelerationStructure> create_ads(const ADSDeclaration& declaration);

    // Wraps an externally-owned Python object as device-usable memory.
    // `obj` may be:
    //   - a Buffer or Tensor: used directly (its own external_ptr()), no copy ever.
    //   - a DLPack-compatible object (e.g. a torch tensor): used directly,
    //     with no copy, if it's contiguous and its memory already belongs
    //     to one of this Device's own memory managers (e.g. it came from
    //     torch.from_dlpack() on one of our own Buffers); otherwise a
    //     temporary buffer is allocated at `location`, and the returned
    //     WrappedMemory lazily copies data in/out of it on demand (see
    //     WrappedMemory::update_cpu()/update_gpu()), using the loaded
    //     interop library for CUDA-resident tensors or a strided/plain
    //     host copy otherwise.
    //   - a Python buffer-protocol object (e.g. a numpy array): always
    //     copied into/out of a temporary buffer at `location` on demand,
    //     since a foreign host allocation has no Vulkan device address of
    //     its own.
    // Throws std::runtime_error if `obj` is none of these.
    [[nodiscard]] std::shared_ptr<WrappedMemory> wrap(pybind11::object obj, MemoryLocation location = MemoryLocation::DEVICE);

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
    // internally by WrappedMemory::update_cpu(), not exposed to Python.
    void vk_copy_out(
        std::uint64_t src_external_ptr, MemoryLocation src_location,
        void* dst_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
        std::uint64_t itemsize, bool dst_is_cuda);

    vk::Device vk_device() const noexcept { return device_; }
    vk::Instance vk_instance() const noexcept { return instance_; }
    // Dynamically-loaded function pointer table used only for
    // VK_KHR_acceleration_structure calls: unlike core/VK_KHR_swapchain
    // functions (exported as trampolines by the Vulkan loader's own import
    // library), acceleration structure entry points aren't guaranteed to be
    // linkable that way, so they're resolved at runtime via
    // vkGetInstanceProcAddr/vkGetDeviceProcAddr instead (see vk_init()).
    // Used internally by Device::create_ads/CommandBuffer::build_ads/
    // vk_AccelerationStructure, not exposed to Python.
    [[nodiscard]] const vk::DispatchLoaderDynamic& vk_dynamic_dispatch() const noexcept { return dynamic_dispatch_; }
    [[nodiscard]] bool vk_swapchain_supported() const noexcept { return swapchain_supported_; }
    // Whether the negotiated Vulkan API version is >= 1.3, where "extended
    // dynamic state" (dynamic cull mode/front face/depth test/depth write/
    // depth compare op) is core, unconditional functionality -- gates
    // Pipeline::attach_depth() and CommandBuffer::set_cull_mode()/
    // set_front_face()/set_depth_test().
    [[nodiscard]] bool vk_extended_dynamic_state_supported() const noexcept { return extended_dynamic_state_supported_; }
    // Whether VK_KHR_acceleration_structure (plus its VK_KHR_
    // deferred_host_operations dependency) was supported and enabled; gates
    // Device::create_ads() and the two acceleration-structure-related
    // buffer usage flags (storage/build-input) added to every buffer.
    [[nodiscard]] bool vk_acceleration_structure_supported() const noexcept { return acceleration_structure_supported_; }
    // Whether VK_KHR_ray_tracing_pipeline was supported and enabled; gates
    // Pipeline::close() for a PipelineType::RAYTRACING pipeline. Requires
    // vk_acceleration_structure_supported() as well (a ray tracing pipeline
    // is only useful together with a TLAS to trace against).
    [[nodiscard]] bool vk_ray_tracing_pipeline_supported() const noexcept { return ray_tracing_pipeline_supported_; }
    // Shader group handle size/alignment and shader binding table base
    // alignment (VkPhysicalDeviceRayTracingPipelinePropertiesKHR), needed to
    // lay out the shader binding table buffer. Only meaningful when
    // vk_ray_tracing_pipeline_supported() is true.
    [[nodiscard]] std::uint32_t vk_shader_group_handle_size() const noexcept { return shader_group_handle_size_; }
    [[nodiscard]] std::uint32_t vk_shader_group_handle_alignment() const noexcept { return shader_group_handle_alignment_; }
    [[nodiscard]] std::uint32_t vk_shader_group_base_alignment() const noexcept { return shader_group_base_alignment_; }

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

    // Tracks `sampler` (a weak reference) so Device::dispose() can
    // proactively destroy it before destroying the device/instance
    // themselves (same rationale as vk_register_window's). Used internally
    // by Device::create_sampler, not exposed to Python.
    void vk_register_sampler(const std::shared_ptr<vk_Sampler>& sampler) {
        samplers_.push_back(sampler);
    }

    // Tracks `ads` (a weak reference) so Device::dispose() can proactively
    // destroy it -- and its dedicated backing buffer/memory -- before
    // destroying the device/instance themselves (same rationale as
    // windows_/samplers_ above). Used internally by Device::create_ads, not
    // exposed to Python.
    void vk_register_acceleration_structure(const std::shared_ptr<vk_AccelerationStructure>& ads) {
        acceleration_structures_.push_back(ads);
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
        std::uint32_t mip_levels, std::uint32_t array_layers, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor);

    uint32_t device_index_ = 0;
    vk::Instance instance_;
    vk::PhysicalDevice physical_;
    vk::Device device_;
    // See vk_dynamic_dispatch()'s comment; populated in vk_init().
    vk::DispatchLoaderDynamic dynamic_dispatch_;
    // Whether VK_KHR_external_semaphore plus the platform win32/fd
    // extension were both supported and enabled; gates whether
    // interop_semaphore_ is created at all in vk_init().
    bool external_semaphore_supported_ = false;
    // Whether VK_KHR_swapchain was supported and enabled; gates
    // Device::create_window.
    bool swapchain_supported_ = false;
    // Whether the negotiated Vulkan API version is >= 1.3; see
    // vk_extended_dynamic_state_supported().
    bool extended_dynamic_state_supported_ = false;
    // Whether VK_KHR_acceleration_structure was supported and enabled; see
    // vk_acceleration_structure_supported().
    bool acceleration_structure_supported_ = false;
    // Whether VK_KHR_ray_tracing_pipeline was supported and enabled; see
    // vk_ray_tracing_pipeline_supported().
    bool ray_tracing_pipeline_supported_ = false;
    // See vk_shader_group_handle_size()/vk_shader_group_handle_alignment()/
    // vk_shader_group_base_alignment().
    std::uint32_t shader_group_handle_size_ = 0;
    std::uint32_t shader_group_handle_alignment_ = 0;
    std::uint32_t shader_group_base_alignment_ = 0;
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
    std::vector<std::weak_ptr<vk_Sampler>> samplers_; // created samplers
    std::vector<std::weak_ptr<vk_AccelerationStructure>> acceleration_structures_; // created acceleration structures
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
    // Represents the pointer of the memory visible outside the device, i.e.,
    // CPU, CUDA, ROCm, etc. For a host_visible_ page this is just the
    // (eagerly, cheaply) mapped host pointer. For a DEVICE page, this is a
    // CUDA-interop-imported pointer, obtained lazily -- see
    // ensure_external_ptr_imported() -- the first time this or dl_device()
    // is actually called, not at page-allocation time.
    [[nodiscard]] std::uint64_t external_ptr() const noexcept;

    std::uint64_t external_to_device(std::uint64_t external_ptr) const noexcept;
    std::uint64_t device_to_external(std::uint64_t device_ptr) const noexcept;

    // Whether this page is CPU- or CUDA/ROCm-accessible for DLPack export
    // purposes. Like external_ptr(), triggers the lazy CUDA import for a
    // DEVICE page (answering "is this CUDA-accessible" honestly requires
    // actually attempting the import).
    [[nodiscard]] DLDevice dl_device() const noexcept;
private:
    // Imports this page's memory into CUDA via try_import_memory_ the first
    // time external_ptr()/dl_device() is called on a non-host_visible_
    // page, memoized in external_ptr_/external_ptr_import_attempted_ so a
    // failed/unsupported import isn't retried on every call. Deliberately
    // NOT done eagerly in the constructor: try_import_memory_ silently
    // initializes a CUDA context (with all its process-wide side effects,
    // e.g. background driver threads) as a side effect, and
    // Device::create_buffer(..., DEVICE) alone -- with no interop ever
    // used -- must not pay that cost.
    void ensure_external_ptr_imported() const noexcept;

    std::weak_ptr<Device> device_;
    uint32_t memory_type_index_ = 0;
    bool host_visible_ = false;
    std::uint64_t capacity_ = 0;
    vk::Buffer buffer_{};
    vk::DeviceMemory memory_{};

    std::uint64_t device_ptr_ = 0;
    mutable std::uint64_t external_ptr_ = 0;
    mutable bool external_ptr_import_attempted_ = false;
    std::unique_ptr<MemoryAllocator> allocator_;
    ExternalImportFn try_import_memory_ = nullptr;
};
