#pragma once
#include <vulkan/vulkan.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <deque>
#include <string>
#include <variant>

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
class Image;

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

/*  ============== STRUCTS ============== */

struct DLDevice {
    std::int32_t device_type;
    std::int32_t device_id;
};

struct ResourceSlice{
    ResourceType type;
    union {
        struct {
            Format format;
            std::uint64_t offset;
            std::uint64_t size;
            ScalarType scalar;
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
    // ARRAY/MATRIX strides and by Device::create_structured_buffer(count > 1).
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
    vk_ResourceData(std::shared_ptr<Device> device, const std::shared_ptr<MemorySlice>& memory,
        vk::Image image, vk::ImageCreateInfo image_info);
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
    Buffer(std::shared_ptr<vk_ResourceData> resource_data, ResourceSlice view_slice);
    Buffer() = delete;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept = delete;
    Buffer& operator=(Buffer&& other) noexcept = delete;
    ~Buffer() noexcept override;
    Format format() const noexcept { return slice_.buffer.format; }
    std::uint64_t size() const noexcept { return slice_.buffer.size; }
    std::shared_ptr<Buffer> cast_format(Format new_format) const;
    std::shared_ptr<Buffer> cast_scalar(ScalarType new_scalar) const;
    std::shared_ptr<Buffer> slice(std::uint64_t offset, std::uint64_t size) const;
    std::shared_ptr<Buffer> slice_array(std::uint64_t start, std::uint64_t count) const;

    // Exports this buffer as a dlpack tensor. Elements type is defined by the scalar info.
    pybind11::object vk_dlpack() const;
    DLDevice vk_dlpack_device() const noexcept;

    // Exports `field`, repeated across every instance of its root layout in
    // this buffer, as a single strided DLPack tensor. Throws
    // std::runtime_error if `field`'s type is (or contains) a STRUCT, since a
    // struct isn't representable as a single numeric tensor.
    pybind11::object field(const LayoutField& field) const;

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
    vk::BufferView buffer_view_;
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

    // ends recording and submits the command buffer for execution on its engine.
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
    // an id to later reference this binding from DescriptorSet::bind().
    // Must be called before close().
    int layout(int set, int binding, DescriptorType description, int count = 1) {
        return pipeline_->vk_layout(set, binding, description, count);
    }

    // Declares the per-vertex input consumed by the vertex shader, as one
    // new vertex buffer binding starting at `start_location`. `layout` must
    // describe a struct (e.g. from compute_layout()); array/struct fields
    // are not valid vertex attributes. Graphics pipelines only. Must be
    // called before close().
    void vertex_layout(int start_location, const Layout& layout) { pipeline_->vk_vertex_layout(start_location, layout); }

    // Declares one color attachment of this pipeline's render pass, at
    // `slot` (the fragment shader output location) with `format`. Returns
    // an id used to bind a render target Image via create_framebuffer().
    // Graphics pipelines only. Must be called before close().
    int attach(int slot, Format format) { return pipeline_->vk_attach(slot, format); }

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
    [[nodiscard]] std::shared_ptr<Framebuffer> create_framebuffer(std::vector<std::pair<int, std::shared_ptr<Image>>> attachments);

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
    vk_Framebuffer(std::weak_ptr<Device> device, vk::Framebuffer framebuffer, std::uint32_t width, std::uint32_t height) noexcept;
    vk_Framebuffer() = delete;
    vk_Framebuffer(const vk_Framebuffer&) = delete;
    vk_Framebuffer& operator=(const vk_Framebuffer&) = delete;
    vk_Framebuffer(vk_Framebuffer&& other) noexcept = delete;
    vk_Framebuffer& operator=(vk_Framebuffer&& other) noexcept = delete;
    ~vk_Framebuffer() noexcept;

    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }
    [[nodiscard]] std::uint32_t height() const noexcept { return height_; }
    [[nodiscard]] vk::Framebuffer vk_handle() const noexcept { return framebuffer_; }
private:
    std::weak_ptr<Device> device_;
    vk::Framebuffer framebuffer_;
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
private:
    std::shared_ptr<vk_Framebuffer> framebuffer_;
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

    void vk_bind_buffer(int layout_id, const std::shared_ptr<Buffer>& buffer);
    void vk_bind_image(int layout_id, const std::shared_ptr<Image>& image);
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
    void bind(int layout_id, const std::shared_ptr<Buffer>& buffer) { descriptor_set_->vk_bind_buffer(layout_id, buffer); }
    // Writes `image` into the binding identified by `layout_id`. The
    // binding's declared type must be STORAGE_IMAGE or SAMPLED_IMAGE (types
    // requiring a sampler are not yet supported).
    void bind(int layout_id, const std::shared_ptr<Image>& image) { descriptor_set_->vk_bind_image(layout_id, image); }

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
    // Creates a buffer to store an array of elements with specific scalar type.
    [[nodiscard]] std::shared_ptr<Buffer> create_array(std::uint64_t elements, ScalarType type, MemoryLocation location);
    // Creates a buffer to store an array of texels with specific format.
    [[nodiscard]] std::shared_ptr<Buffer> create_texels(std::uint64_t elements, Format format, MemoryLocation location);
    // Creates a buffer of specific size in bytes
    [[nodiscard]] std::shared_ptr<Buffer> create_buffer(std::uint64_t size, MemoryLocation location);
    // Creates a buffer sized to hold `count` instances of the element described by `layout`
    // (a single instance if count == 1, using layout.aligned_size as the per-element
    // stride otherwise). Throws std::runtime_error if the computed size is 0.
    [[nodiscard]] std::shared_ptr<Buffer> create_structured_buffer(const Layout& layout, MemoryLocation location, int count = 1);
    // Creates the resource data for an image of specific dimensions and format
    [[nodiscard]] std::shared_ptr<Image> create_image(int width, int height, int depth, int mip_levels, int array_layers, Format format, MemoryLocation location);
    // Creates a plain buffer sized to match `buffer`, in `location` (default HOST).
    // This is the correct Vulkan pattern to move data between CPU and GPU memory:
    // record a CommandBuffer::transfer between the staging buffer and `buffer`
    // rather than attempting to map device-local memory directly.
    [[nodiscard]] std::shared_ptr<Buffer> create_staging(const std::shared_ptr<Buffer>& buffer, MemoryLocation location = MemoryLocation::HOST);
    // Creates a plain buffer sized to match `image`'s backing store, in `location`
    // (default HOST), for staged CPU<->GPU transfers of image contents.
    [[nodiscard]] std::shared_ptr<Buffer> create_staging(const std::shared_ptr<Image>& image, MemoryLocation location = MemoryLocation::HOST);
    // Allocates a memory for a tensor with specific type and shape. The tensor is pack as a dl_pack
    [[nodiscard]] pybind11::object create_tensor_dlpack(const std::vector<std::uint64_t>& shape, ScalarType type, MemoryLocation location);

    vk::Device vk_device() const noexcept { return device_; }

    static constexpr size_t kMaxEngineTypes = 8;

private:
    Device(uint32_t device_index, bool enable_validation_layers);
    void vk_init();

    uint32_t device_index_ = 0;
    vk::Instance instance_;
    vk::PhysicalDevice physical_;
    vk::Device device_;
    std::shared_ptr<MemoryManager> host_memory_manager_;
    std::shared_ptr<MemoryManager> device_memory_manager_;
    std::vector<vk::CommandPool> command_pools_; // A command pool for each family index
    std::array<int, kMaxEngineTypes> engine_type_index_ = {-1, -1, -1, -1, -1, -1, -1, -1};
    std::array<std::uint32_t, kMaxEngineTypes> engine_queue_flags_ = { ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    std::array<std::uint32_t, kMaxEngineTypes> engine_queue_count_ = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<std::vector<std::shared_ptr<vk_Engine>>> engines_;
    std::vector<std::weak_ptr<vk_ResourceData>> resources_; // created resources
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

private:
    std::weak_ptr<Device> device_;
    uint32_t memory_type_index_ = 0;
    bool host_visible_ = false;
    std::uint64_t next_page_capacity_ = 1 << 30; // Start with 1 GiB pages.
    std::vector<std::shared_ptr<MemoryPage>> pages_;
    std::shared_ptr<ExternalInteropLibraryImpl> interop_library_;
    ExternalImportFn try_import_memory_ = nullptr;
};

/**
 * Represents a page of memory allocated on the device. It can be suballocated by MemoryAllocator.
 */
class MemoryPage : public std::enable_shared_from_this<MemoryPage> {
public:

    MemoryPage(std::weak_ptr<Device> device, uint32_t memory_type_index, bool host_visible, std::uint64_t capacity, ExternalImportFn try_import_memory);
    ~MemoryPage() noexcept;

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
