#pragma once
#include <vulkan/vulkan.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <deque>

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

enum vk_CommandBufferInternalState {
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

/*  ============== STRUCTS ============== */

struct DLDevice {
    std::int32_t device_type;
    std::int32_t device_id;
};

union ResourceSlice{
    ResourceType type;
    struct {
        vk::Format format;
        std::uint64_t offset;
        std::uint64_t size;
        ScalarType scalar;
    } buffer;
    struct {
        vk::Format format;
        int mip_start;
        int mip_count;
        int array_start;
        int array_count;
    } image;
};

/*  ============== FUNCTIONS ============== */

int scalar_type_size(ScalarType type);

/*  ============== CLASSES ============== */

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
    bool is_cpu() const noexcept ;
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
    vk::Format format() const noexcept { return slice_.buffer.format; }
    int size() const noexcept { return slice_.buffer.size; }
    std::shared_ptr<Buffer> cast_format(Format new_format) const;
    std::shared_ptr<Buffer> cast_scalar(ScalarType new_scalar) const;
    std::shared_ptr<Buffer> slice(int offset, int size) const;
    std::shared_ptr<Buffer> slice_array(int start, int count) const;

    pybind11::object dlpack() const;

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
    vk::Format format() const noexcept { return slice_.image.format; }
    int mip_count() const noexcept { return slice_.image.mip_count; }
    int array_count() const noexcept { return slice_.image.array_count; }
    std::shared_ptr<Image> cast_format(vk::Format new_format) const;
    std::shared_ptr<Image> slice(int mip_start, int mip_count, int array_start, int array_count) const;
    vk::ImageView get_view();
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
    ~SubmittedTask() noexcept {
        engine_.reset();
        submitted_.clear(); // should be empty already...
    };
    void wait();
    bool is_complete() ;
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
    // Returns this command buffer to its owning pool for reuse.
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

    // Records a device-side copy of `bytes` bytes from `source` into `destination`.
    // Must be called while the command buffer is still recording (i.e. before close()).
    void transfer(const std::shared_ptr<Buffer>& source, const std::shared_ptr<Buffer>& destination, int bytes);

    // ends recording and submits the command buffer for execution on its engine.
    void close();
    // the command buffer is not going to be used for submission anymore. It is safe to destroy or reuse.
    void release() ;

    /**
     * Determines if the object is locked by the gpu.
     * @return True if the underlying command buffer is assumed to be used on the gpu. SubmissionTask object can be used to check termination.
     */
    [[nodiscard]] bool is_submitted() {
        return command_buffer_->state == SUBMITTED;
    }

    /**
     * Determines if the object is ready for submission.
     * @return True if the underlying command buffer is granted to be on executable state.
     */
    [[nodiscard]] bool is_executable() {
        return command_buffer_->state == EXECUTABLE;
    }

    [[nodiscard]] bool is_released() const noexcept { return device_ == nullptr; }

    [[nodiscard]] bool is_closed() const noexcept {
        return command_buffer_->state != RECORDING;
    }

    [[nodiscard]] std::shared_ptr<vk_CommandBuffer> vk_command_buffer() const {
        return command_buffer_;
    }
private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<Engine> engine_;
    std::shared_ptr<vk_CommandBuffer> command_buffer_;
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
    // sets a command buffer to reusable list after completation
    void release_command_buffer(std::shared_ptr<vk_CommandBuffer> command_buffer);
    // fast shallow submit wrapper to the queue
    std::shared_ptr<SubmittedTask> submit(std::uint32_t count, vk::CommandBuffer* command_buffers);
    void wait(); // waits for all submissions in this queue to finish
    // waits CPU until a signal with submission_id is reach. Then collect all completed to that point
    void vk_wait_for(std::uint64_t submission_id);
    // check if timeline on gpu and collect all completed.
    std::uint64_t vk_check_completation();
private:
    // iterate all submitted tasks with submission_id <= submission_id and collect them as completed. This is called automatically by vk_wait_for and vk_is_completed when the timeline semaphore has reached the submission_id.
    // notify completation.
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

    int engine_index() const noexcept { return engine_index_; }

    void vk_release_command_buffer(std::shared_ptr<vk_CommandBuffer> command_buffer) {
        engine_->release_command_buffer(command_buffer);
    }

    std::shared_ptr<SubmittedTask> submit(std::vector<std::shared_ptr<CommandBuffer>> command_buffers) {
        std::shared_ptr<SubmittedTask> task;
        if (command_buffers.size() == 1) // special case, try to make it faster
        {
            auto cb = command_buffers[0]->vk_command_buffer();
            cb->notify_submitted();
            task = std::move(engine_->submit(1, &cb->command_buffer));
        }
        else {
            std::vector<vk::CommandBuffer> vk_command_buffers(command_buffers.size());
            for (size_t i = 0; i < command_buffers.size(); ++i) {
                auto cb = command_buffers[i]->vk_command_buffer();
                cb->notify_submitted();
                vk_command_buffers[i] = cb->command_buffer;
            }
            task = std::move(engine_->submit(vk_command_buffers.size(), vk_command_buffers.data()));
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
 * Wraps a vk::Instance, a vk::Device and its associated resources, memory managers and engines.
 */
class Device : public std::enable_shared_from_this<Device> {
public:
    ~Device() noexcept;
    static std::shared_ptr<Device> create_device(uint32_t device_index, bool enable_validation_layers);
    void dispose() noexcept;
    Device() = default;
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
    // Creates the resource data for an image of specific dimensions and format
    [[nodiscard]] std::shared_ptr<Image> create_image(int width, int height, int depth, int mip_levels, int array_layers, vk::Format format, MemoryLocation location);
    // Allocates a memory for a tensor with specific type and shape. The tensor is pack as a dl_pack
    [[nodiscard]] pybind11::object create_tensor_dlpack(const std::vector<std::uint64_t>& shape, ScalarType type, MemoryLocation location);

    vk::Device vk_device() const noexcept { return device_; }

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
    int engine_type_index_[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    std::uint32_t engine_queue_flags_[8] = { ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    uint32_t engine_queue_count_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<std::vector<std::shared_ptr<vk_Engine>>> engines_;
    std::vector<std::weak_ptr<vk_ResourceData>> resources_; // created resources
};

/**
 * Represents a memory as a slice (offset, size) of a page.
 */
class MemorySlice : public std::enable_shared_from_this<MemorySlice> {
public:
    MemorySlice();
    MemorySlice(
        std::shared_ptr<Device> device, // as long as this memory live, the device should
        std::shared_ptr<MemoryPage> page,
        std::uint64_t allocated_offset,
        std::uint64_t allocated_size,
        std::uint64_t offset, std::uint64_t size) noexcept;
    ~MemorySlice() noexcept;
    MemorySlice(const MemorySlice&) = delete;
    MemorySlice& operator=(const MemorySlice&) = delete;
    MemorySlice(MemorySlice&& other) noexcept;
    MemorySlice& operator=(MemorySlice&& other) noexcept;
    // Gets the offset of the allocated memory, not necessarily aligned.
    [[nodiscard]] std::uint64_t allocated_offset() const noexcept { return allocated_offset_; }
    // Gets the size of the allocated memory, allways greater or equals to the requested.
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
    std::unordered_map<int, int> allocations_;

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

    std::shared_ptr<MemorySlice> allocate(int size, int alignment);

    [[nodiscard]] std::uint64_t external_to_device(std::uint64_t external_ptr) const noexcept;
    [[nodiscard]] std::uint64_t device_to_external(std::uint64_t device_ptr) const noexcept;

private:
    std::weak_ptr<Device> device_;
    uint32_t memory_type_index_ = 0;
    bool host_visible_ = false;
    int next_page_capacity_ = 1 << 30; // Start with 1 GiB pages.
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
    [[nodiscard]] int capacity() const noexcept;
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
