#pragma once
#include <vulkan/vulkan.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace pybind11 { class object; }

/*  ============== PROTOTYPES ============== */

class MemoryPage;
class MemorySlice;
class MemoryManager;
class ExternalInteropLibraryImpl;
class MemoryAllocator;
class Resource;
class Buffer;
class Image;
class Device;

/*  ============== ENUMS ============== */

enum class MemoryLocation : int {
    HOST = 0,
    DEVICE = 1,
};

using ExternalImportFn = std::uint64_t (*)(vk::Device device, int device_index, VkDeviceMemory memory, long long size);

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

/*  ============== STRUCTS ============== */

struct DLDevice {
    std::int32_t device_type;
    std::int32_t device_id;
};

union ResourceSlice{
    struct {
        vk::Format format;
        int offset;
        int size;
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
class ResourceData : std::enable_shared_from_this<ResourceData>{
public:
    ResourceData(std::shared_ptr<Device> device, std::shared_ptr<MemorySlice> memory,
        vk::Image image, vk::ImageCreateInfo image_info);
    ResourceData(const ResourceData&) = delete;
    ResourceData& operator=(const ResourceData&) = delete;
    ResourceData(ResourceData&& other) noexcept = delete;
    ResourceData& operator=(ResourceData&& other) noexcept = delete;
    ~ResourceData()  noexcept;
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

class Resource : std::enable_shared_from_this<ResourceData>{
public:
    Resource(std::shared_ptr<ResourceData> resource_data, ResourceSlice view_slice);
    Resource() = delete;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&& other) noexcept = delete;
    Resource& operator=(Resource&& other) noexcept = delete;
    virtual ~Resource() noexcept;
    void dispose();
protected:
    std::shared_ptr<ResourceData> data_;
    ResourceSlice slice_;
};

class Buffer: public Resource {
public:
    Buffer(std::shared_ptr<ResourceData> resource_data, ResourceSlice view_slice);
    Buffer() = delete;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept = delete;
    Buffer& operator=(Buffer&& other) noexcept = delete;
    ~Buffer() noexcept override;
    vk::Format format() const noexcept { return slice_.buffer.format; }
    int size() const noexcept { return slice_.buffer.size; }
    std::shared_ptr<Buffer> cast_format(vk::Format new_format) const;
    std::shared_ptr<Buffer> cast_scalar(ScalarType new_scalar) const;
    std::shared_ptr<Buffer> slice(int offset, int size) const;
    std::shared_ptr<Buffer> slice_array(int start, int count) const;

    pybind11::object dlpack() const;

    vk::BufferView get_view() noexcept;
    std::uint64_t device_ptr() const;
    std::uint64_t external_ptr() const;
private:
    vk::BufferView buffer_view_;
};

class Image: public Resource {
public:
    Image(std::shared_ptr<ResourceData> resource_data, ResourceSlice view_slice);
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
    vk::ImageView get_view() noexcept;
private:
    vk::ImageView image_view_;
};

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

    [[nodiscard]] vk::PhysicalDevice physical_device() const noexcept;
    [[nodiscard]] vk::Device logical_device() const noexcept;
    [[nodiscard]] uint32_t device_index() const noexcept;

    void init();

    // Manager to allocate memory visible by the host application (cpu)
    [[nodiscard]] MemoryManager& host_memory_manager();
    // Manager to allocate memory visible by the device (gpu). This may or may not be visible by the host application, depending on the device capabilities.
    [[nodiscard]] MemoryManager& device_memory_manager();
    // Creates a buffer to store an array of elements with specific scalar type.
    [[nodiscard]] std::shared_ptr<Buffer> create_array(int elements, ScalarType type, MemoryLocation location);
    // Creates a buffer to store an array of texels with specific format.
    [[nodiscard]] std::shared_ptr<Buffer> create_texels(int elements, vk::Format format, MemoryLocation location);
    // Creates a buffer of specific size in bytes
    [[nodiscard]] std::shared_ptr<Buffer> create_buffer(int size, MemoryLocation location);
    // Creates the resource data for an image of specific dimensions and format
    [[nodiscard]] std::shared_ptr<Image> create_image(int width, int height, int depth, int mip_levels, int array_layers, vk::Format format, MemoryLocation location);
    // Destroy the underling resource associated to this resource data
    void destroy_data(std::shared_ptr<ResourceData> data) noexcept;

    // Allocates a memory for a tensor with specific type and shape. The tensor is pack as a dl_pack
    [[nodiscard]] pybind11::object allocate_tensor_dlpack(const std::vector<int>& shape, ScalarType type, MemoryLocation location);

    vk::Device vk_device() const noexcept { return device_; }

private:
    Device(uint32_t device_index, bool enable_validation_layers);

    uint32_t device_index_ = 0;
    vk::Instance instance_;
    vk::PhysicalDevice physical_;
    vk::Device device_;
    std::vector<std::vector<vk::Queue>> queues_;
    std::unique_ptr<MemoryManager> host_memory_manager_;
    std::unique_ptr<MemoryManager> device_memory_manager_;
    std::vector<std::shared_ptr<ResourceData>> resources_; // created resources
};

/**
 * Represents a memory as a slice (offset, size) of a page.
 */
class MemorySlice : std::enable_shared_from_this<MemorySlice> {
public:
    MemorySlice();
    MemorySlice(
        std::shared_ptr<MemoryPage> page,
        int allocated_offset,
        int allocated_size,
        int offset, int size) noexcept;
    ~MemorySlice() noexcept;
    MemorySlice(const MemorySlice&) = delete;
    MemorySlice& operator=(const MemorySlice&) = delete;
    MemorySlice(MemorySlice&& other) noexcept;
    MemorySlice& operator=(MemorySlice&& other) noexcept;
    // Gets the offset of the allocated memory, not necessarily aligned.
    [[nodiscard]] int allocated_offset() const noexcept { return allocated_offset_; }
    // Gets the size of the allocated memory, allways greater or equals to the requested.
    [[nodiscard]] int allocated_size() const noexcept { return allocated_size_; }
    [[nodiscard]] int offset() const noexcept { return offset_; }
    [[nodiscard]] int size() const noexcept { return size_; }
    [[nodiscard]] vk::Buffer page_buffer() const noexcept;
    // This is the pointer in vulkan device address space, which can be used for shader addressing
    [[nodiscard]] std::uint64_t device_ptr() const noexcept;
    // This is the pointer in external manager, e.g.: cuda, rocm or LevelZero
    [[nodiscard]] std::uint64_t external_ptr() const noexcept;
    [[nodiscard]] bool host_visible() const noexcept ;
    [[nodiscard]] DLDevice dl_device() const noexcept;
    void release() noexcept;

private:
    std::shared_ptr<MemoryPage> page_;
    int allocated_offset_ = 0;
    int allocated_size_ = 0;
    int offset_ = 0;
    int size_ = 0;
};

/**
 * Represents an allocator/deallocator logic within an array of specific capacity.
 */
class MemoryAllocator {
public:
    MemoryAllocator(int capacity);
    ~MemoryAllocator() noexcept;

    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;
    MemoryAllocator(MemoryAllocator&&) = delete;
    MemoryAllocator& operator=(MemoryAllocator&&) = delete;
    /*
     * Gives the start offset of the allocated range.
     */
    int allocate(int size);
    [[nodiscard]] bool can_allocate(int size) const;
    /*
     * Free the range allocated starting in offset
     */
    void free_memory(int offset) noexcept;

private:
    struct Range {
        int offset = 0;
        int size = 0;
        Range* prev = nullptr;
        Range* next = nullptr;
        int bin = 0;
    };

    int capacity_ = 0;
    std::vector<std::unique_ptr<Range>> nodes_;
    Range* head_ = nullptr;
    std::vector<std::vector<Range*>> bins_;
    std::unordered_map<int, int> allocations_;

    [[nodiscard]] int bin_for_size(int size) const;
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
    MemoryManager(std::shared_ptr<Device> device, uint32_t memory_type_index, bool host_visible);
    ~MemoryManager() noexcept;

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = delete;
    MemoryManager& operator=(MemoryManager&&) = delete;

    std::shared_ptr<MemorySlice> allocate(int size, int alignment);

    [[nodiscard]] std::uint64_t external_to_device(std::uint64_t external_ptr) const noexcept;
    [[nodiscard]] std::uint64_t device_to_external(std::uint64_t device_ptr) const noexcept;

private:
    std::shared_ptr<Device> device_;
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

    MemoryPage(std::shared_ptr<Device> device, uint32_t memory_type_index, bool host_visible, int capacity, ExternalImportFn try_import_memory);
    ~MemoryPage() noexcept;

    MemoryPage(const MemoryPage&) = delete;
    MemoryPage& operator=(const MemoryPage&) = delete;
    MemoryPage(MemoryPage&&) = delete;
    MemoryPage& operator=(MemoryPage&&) = delete;

    bool can_allocate(int size, int alignment) const;
    std::shared_ptr<MemorySlice> allocate(int size, int alignment);
    void free_memory(int allocated_offset) noexcept;

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
    std::shared_ptr<Device> device_;
    uint32_t memory_type_index_ = 0;
    bool host_visible_ = false;
    int capacity_ = 0;
    vk::Buffer buffer_{};
    vk::DeviceMemory memory_{};
    std::uint64_t device_ptr_ = 0;
    std::uint64_t external_ptr_ = 0;
    std::unique_ptr<MemoryAllocator> allocator_;
    DLDevice dl_device_;
    ExternalImportFn try_import_memory_ = nullptr;
};
