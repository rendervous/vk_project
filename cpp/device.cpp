#include "device.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#if defined(_WIN32) && defined(VK_USE_PLATFORM_WIN32_KHR)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

std::vector<VkLayerProperties> enumerate_instance_layers() {
    uint32_t count = 0;
    VkResult result = vkEnumerateInstanceLayerProperties(&count, nullptr);
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        throw std::runtime_error("Failed to enumerate Vulkan instance layers");
    }

    std::vector<VkLayerProperties> layers(count);
    do {
        result = vkEnumerateInstanceLayerProperties(&count, layers.data());
        if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
            throw std::runtime_error("Failed to enumerate Vulkan instance layers");
        }
        layers.resize(count);
    } while (result == VK_INCOMPLETE);

    return layers;
}

bool has_validation_suffix(const char* layer_name) {
    constexpr std::string_view suffix{"_validation"};
    std::string_view name{layer_name};
    return name.size() >= suffix.size() && name.substr(name.size() - suffix.size()) == suffix;
}

bool supports_extension(const std::vector<vk::ExtensionProperties>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const vk::ExtensionProperties& ext) {
        return std::strcmp(ext.extensionName, name) == 0;
    });
}

uint32_t formatSize(vk::Format format) {
    switch (format)
    {
        case vk::Format::eR8Unorm:
        case vk::Format::eR8Snorm:
        case vk::Format::eR8Uint:
        case vk::Format::eR8Sint:
            return 1;

        case vk::Format::eR8G8Unorm:
        case vk::Format::eR8G8Snorm:
        case vk::Format::eR8G8Uint:
        case vk::Format::eR8G8Sint:
            return 2;

        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Snorm:
        case vk::Format::eR8G8B8A8Uint:
        case vk::Format::eR8G8B8A8Sint:
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eB8G8R8A8Srgb:
            return 4;

        case vk::Format::eR16Unorm:
        case vk::Format::eR16Sint:
        case vk::Format::eR16Uint:
        case vk::Format::eR16Sfloat:
            return 2;

        case vk::Format::eR16G16Unorm:
        case vk::Format::eR16G16Sfloat:
            return 4;

        case vk::Format::eR16G16B16A16Sfloat:
            return 8;

        case vk::Format::eR32Uint:
        case vk::Format::eR32Sfloat:
            return 4;

        case vk::Format::eR32G32Sfloat:
            return 8;

        case vk::Format::eR32G32B32Sfloat:
            return 12;

        case vk::Format::eR32G32B32A32Sfloat:
            return 16;

        case vk::Format::eD32Sfloat:
            return 4;

        case vk::Format::eD24UnormS8Uint:
            return 4;

        case vk::Format::eD32SfloatS8Uint:
            return 8;

        default:
            throw std::runtime_error("Unsupported or compressed format");
    }
}

py::object torch_module() {
    return py::module_::import("torch");
}

struct DLDataType {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
};

struct DLTensor {
    void* data;
    DLDevice device;
    std::int32_t ndim;
    DLDataType dtype;
    std::int64_t* shape;
    std::int64_t* strides;
    std::uint64_t byte_offset;
};

struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(DLManagedTensor*);
};

struct TensorOwner {
    std::shared_ptr<MemorySlice> memory;
    std::unique_ptr<std::int64_t[]> shape;
    std::unique_ptr<std::int64_t[]> strides;
};

DLDataType dlpack_dtype (ScalarType scalar) {
    switch (scalar) {
        case ScalarType::FLOAT16: return {2, 16, 1};
        case ScalarType::FLOAT32: return {2, 32, 1};
        case ScalarType::FLOAT64: return {2, 64, 1 };
        case ScalarType::INT8: return {0, 8, 1};
        case ScalarType::BOOL: return {6, 8, 1};
        case ScalarType::INT16: return {0, 16, 1};
        case ScalarType::INT32: return {0, 32, 1};
        case ScalarType::INT64: return {0, 64, 1};
        case ScalarType::UINT8: return {1, 8, 1};
        case ScalarType::UINT16: return {1, 16, 1};
        case ScalarType::UINT32: return {1, 32, 1};
        case ScalarType::UINT64: return {1, 64, 1};
        default:
            throw std::invalid_argument("Wrong scalar type");
    }
}

DLDataType dlpack_dtype(const py::object& dtype) {
    const std::string name = py::str(dtype);
    if (name.find("float16") != std::string::npos) return {2, 16, 1};
    if (name.find("bfloat16") != std::string::npos) return {4, 16, 1};
    if (name.find("float32") != std::string::npos) return {2, 32, 1};
    if (name.find("float64") != std::string::npos) return {2, 64, 1};
    if (name.find("int8") != std::string::npos) return {0, 8, 1};
    if (name.find("uint8") != std::string::npos) return {1, 8, 1};
    if (name.find("int16") != std::string::npos) return {0, 16, 1};
    if (name.find("uint16") != std::string::npos) return {1, 16, 1};
    if (name.find("int32") != std::string::npos) return {0, 32, 1};
    if (name.find("int64") != std::string::npos) return {0, 64, 1};
    if (name.find("bool") != std::string::npos) return {6, 8, 1};
    throw std::runtime_error("Unsupported dtype");
}

std::size_t dtype_itemsize(const py::object& dtype) {
    return dlpack_dtype(dtype).bits / 8;
}

void dlmanaged_tensor_deleter(DLManagedTensor* self) {
    if (!self) {
        return;
    }

    auto* owner = static_cast<TensorOwner*>(self->manager_ctx);
    delete owner;
    delete self;
}

py::object export_dltensor_py(DLManagedTensor* managed) {
    return py::capsule(
        managed,
        "dltensor", [](PyObject* cap) {
        const char* name = PyCapsule_GetName(cap);
        if (name && std::strcmp(name, "dltensor") == 0) {
            auto* dlmanaged = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
            if (dlmanaged && dlmanaged->deleter) {
                dlmanaged->deleter(dlmanaged);
            }
        }
    });
}

using ExternalImportMemoryFn = ExternalImportFn;

std::string interop_library_base_name(uint32_t vendor_id) {
    switch (vendor_id) {
        case 0x10DE: // NVIDIA
            return "vk_cuda_interop";
        default:
            return {};
    }
}

std::string interop_library_filename(std::string_view base_name) {
#if defined(_WIN32)
    return std::string(base_name) + ".dll";
#else
    return "lib" + std::string(base_name) + ".so";
#endif
}

std::filesystem::path module_install_dir() {
    py::gil_scoped_acquire acquire;
    py::module_ module = py::module_::import("vk");
    return std::filesystem::path(std::string(py::str(module.attr("__file__")))).parent_path();
}

uint32_t find_memory_type_index(
    const vk::PhysicalDeviceMemoryProperties& props,
    vk::MemoryPropertyFlags required,
    vk::MemoryPropertyFlags preferred,
    bool* found
) {
    uint32_t fallback = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        const auto flags = props.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) {
            continue;
        }
        if ((flags & preferred) == preferred) {
            *found = true;
            return i;
        }
        if (fallback == std::numeric_limits<uint32_t>::max()) {
            fallback = i;
        }
    }

    if (fallback != std::numeric_limits<uint32_t>::max()) {
        *found = true;
        return fallback;
    }

    *found = false;
    return 0;
}

vk::BufferUsageFlags full_buffer_usage_flags() {
    return vk::BufferUsageFlagBits::eTransferSrc |
           vk::BufferUsageFlagBits::eTransferDst |
           vk::BufferUsageFlagBits::eUniformTexelBuffer |
           vk::BufferUsageFlagBits::eStorageTexelBuffer |
           vk::BufferUsageFlagBits::eUniformBuffer |
           vk::BufferUsageFlagBits::eStorageBuffer |
           vk::BufferUsageFlagBits::eIndexBuffer |
           vk::BufferUsageFlagBits::eVertexBuffer |
           vk::BufferUsageFlagBits::eIndirectBuffer |
           vk::BufferUsageFlagBits::eShaderDeviceAddress;
}

} // namespace

class ExternalInteropLibraryImpl {
public:
    explicit ExternalInteropLibraryImpl(std::string library_base_name) {
        load(std::move(library_base_name));
    }

    ~ExternalInteropLibraryImpl() noexcept {
        unload();
    }

    ExternalInteropLibraryImpl(const ExternalInteropLibraryImpl&) = delete;
    ExternalInteropLibraryImpl& operator=(const ExternalInteropLibraryImpl&) = delete;
    ExternalInteropLibraryImpl(ExternalInteropLibraryImpl&&) = delete;
    ExternalInteropLibraryImpl& operator=(ExternalInteropLibraryImpl&&) = delete;

    [[nodiscard]] bool loaded() const noexcept {
        return try_import_memory_ != nullptr;
    }

    [[nodiscard]] ExternalImportMemoryFn try_import_memory_fn() const noexcept {
        return try_import_memory_;
    }

private:
#if defined(_WIN32)
    using LibraryHandle = HMODULE;

    static LibraryHandle open_library(const std::filesystem::path& path) {
        return LoadLibraryW(path.c_str());
    }

    static void close_library(LibraryHandle handle) {
        if (handle) {
            FreeLibrary(handle);
        }
    }

    static void* load_symbol(LibraryHandle handle, const char* name) {
        return handle ? reinterpret_cast<void*>(GetProcAddress(handle, name)) : nullptr;
    }
#else
    using LibraryHandle = void*;

    static LibraryHandle open_library(const std::filesystem::path& path) {
        return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    }

    static void close_library(LibraryHandle handle) {
        if (handle) {
            dlclose(handle);
        }
    }

    static void* load_symbol(LibraryHandle handle, const char* name) {
        return handle ? dlsym(handle, name) : nullptr;
    }
#endif

    void load(std::string library_base_name) {
        if (library_base_name.empty()) {
            printf("[ERROR] Library name was empty");
            return;
        }

        const std::string filename = interop_library_filename(library_base_name);
        std::vector<std::filesystem::path> candidates;
        try {
            const auto install_dir = module_install_dir();
            candidates.push_back(install_dir / filename);
            if (install_dir.has_parent_path()) {
                candidates.push_back(install_dir.parent_path() / filename);
            }
        } catch (const py::error_already_set&) {
            // If Python cannot resolve the module path, fall back to the loader search path.
            printf("[ERROR] Module Install dir could not be resolved.");
        }
        candidates.push_back(std::filesystem::current_path() / filename);
        candidates.push_back(std::filesystem::path(filename));

        for (const auto& candidate : candidates) {
            handle_ = open_library(candidate);
            if (handle_) {
                break;
            }
        }

        if (!handle_) {
            printf("[ERROR] Library could not be opened for any of the possible folders.");
            return;
        }

        auto symbol = load_symbol(handle_, "try_import_memory");
        if (!symbol) {
            printf("[ERROR] Could not be loaded symbol try_import_memory");
            unload();
            return;
        }

        try_import_memory_ = reinterpret_cast<ExternalImportMemoryFn>(symbol);
    }

    void unload() noexcept {
        if (handle_) {
            close_library(handle_);
            handle_ = nullptr;
        }
        try_import_memory_ = nullptr;
    }

    LibraryHandle handle_ = nullptr;
    ExternalImportMemoryFn try_import_memory_ = nullptr;
};

std::shared_ptr<Device> Device::create_device(uint32_t device_index, bool enable_validation_layers) {
    auto dev = std::shared_ptr<Device>(new Device(device_index, enable_validation_layers));
    dev->init();
    return dev;
}

int scalar_type_size(ScalarType type) {
    switch (type) {
        case ScalarType::FLOAT16:
        case ScalarType::INT16:
        case ScalarType::UINT16:
            return 2;
        case ScalarType::FLOAT32:
        case ScalarType::INT32:
        case ScalarType::UINT32:
            return 4;
        case ScalarType::FLOAT64:
        case ScalarType::INT64:
        case ScalarType::UINT64:
            return 8;
        case ScalarType::INT8:
        case ScalarType::UINT8:
        case ScalarType::BOOL:
            return 1;
        default:
            throw std::runtime_error("Unsupported scalar type");
    }
}


ScalarType format_scalar_type(vk::Format format)
{
    switch (format)
    {
        // =======================
        // 8-bit UNSIGNED INT
        // =======================
        case vk::Format::eR8Uint:
        case vk::Format::eR8G8Uint:
        case vk::Format::eR8G8B8Uint:
        case vk::Format::eR8G8B8A8Uint:
            return ScalarType::UINT8;

        // =======================
        // 8-bit SIGNED INT
        // =======================
        case vk::Format::eR8Sint:
        case vk::Format::eR8G8Sint:
        case vk::Format::eR8G8B8Sint:
        case vk::Format::eR8G8B8A8Sint:
            return ScalarType::INT8;

        // =======================
        // 8-bit NORMALIZED (treated as float-like)
        // =======================
        case vk::Format::eR8Unorm:
        case vk::Format::eR8Snorm:
        case vk::Format::eR8G8Unorm:
        case vk::Format::eR8G8Snorm:
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eR8G8B8A8Snorm:
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eB8G8R8A8Srgb:
            return ScalarType::FLOAT32; // normalized -> conceptual float

        // =======================
        // 16-bit UNSIGNED INT
        // =======================
        case vk::Format::eR16Uint:
        case vk::Format::eR16G16Uint:
        case vk::Format::eR16G16B16A16Uint:
            return ScalarType::UINT16;

        // =======================
        // 16-bit SIGNED INT
        // =======================
        case vk::Format::eR16Sint:
        case vk::Format::eR16G16Sint:
        case vk::Format::eR16G16B16A16Sint:
            return ScalarType::INT16;

        // =======================
        // 16-bit FLOAT
        // =======================
        case vk::Format::eR16Sfloat:
        case vk::Format::eR16G16Sfloat:
        case vk::Format::eR16G16B16A16Sfloat:
            return ScalarType::FLOAT16;

        // =======================
        // 32-bit UNSIGNED INT
        // =======================
        case vk::Format::eR32Uint:
        case vk::Format::eR32G32Uint:
        case vk::Format::eR32G32B32Uint:
        case vk::Format::eR32G32B32A32Uint:
            return ScalarType::UINT32;

        // =======================
        // 32-bit SIGNED INT
        // =======================
        case vk::Format::eR32Sint:
        case vk::Format::eR32G32Sint:
        case vk::Format::eR32G32B32Sint:
        case vk::Format::eR32G32B32A32Sint:
            return ScalarType::INT32;

        // =======================
        // 32-bit FLOAT
        // =======================
        case vk::Format::eR32Sfloat:
        case vk::Format::eR32G32Sfloat:
        case vk::Format::eR32G32B32Sfloat:
        case vk::Format::eR32G32B32A32Sfloat:
            return ScalarType::FLOAT32;

        // =======================
        // 64-bit FLOAT
        // =======================
        case vk::Format::eR64Sfloat:
        case vk::Format::eR64G64Sfloat:
        case vk::Format::eR64G64B64Sfloat:
        case vk::Format::eR64G64B64A64Sfloat:
            return ScalarType::FLOAT64;

        // =======================
        // Depth formats (special case)
        // =======================
        case vk::Format::eD16Unorm:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32Sfloat:
            return ScalarType::FLOAT32;

        // =======================
        // Stencil (special, integer)
        // =======================
        case vk::Format::eS8Uint:
            return ScalarType::UINT8;

        default:
            return ScalarType::UNDEFINED;
    }
}


ResourceData::ResourceData(
    std::shared_ptr<Device> device, std::shared_ptr<MemorySlice> memory,
    vk::Image image, vk::ImageCreateInfo image_info): device_(device), memory_(memory), image_(image), image_info_(image_info), resource_type_(image ? ResourceType::IMAGE : ResourceType::BUFFER) {
    buffer_ = memory->page_buffer();
}

ResourceData::~ResourceData() noexcept {
    dispose();
}

void ResourceData::dispose() {
    this->device_->destroy_data(shared_from_this());
    this->device_ = nullptr;
    this->memory_ = nullptr;
    this->buffer_ = nullptr;
    this->image_ = nullptr;
}

std::uint64_t ResourceData::device_ptr() const {
    if (memory_) {
        return memory_->device_ptr();
    }
    return 0;
}

std::uint64_t ResourceData::external_ptr() const {
    if (memory_) {
        return memory_->external_ptr();
    }
    return 0;
}

bool ResourceData::is_cpu() const noexcept { return memory_ && memory_->host_visible(); }

Resource::Resource(std::shared_ptr<ResourceData> resource_data, ResourceSlice view_slice) {
    data_ = resource_data;
    slice_ = view_slice;
}

Resource::~Resource() noexcept {
    dispose();
}

vk::BufferView Buffer::get_view() noexcept {
    if (!buffer_view_) {
        vk::BufferViewCreateInfo info{};
        info.buffer = data_->get_buffer();
        info.format = slice_.buffer.format;
        info.offset = slice_.buffer.offset;
        info.range = slice_.buffer.size;
        auto device = this->data_->device()->logical_device();
        buffer_view_ = device.createBufferView(info);
    }
    return buffer_view_;
}

vk::ImageView Image::get_view() noexcept {
    if (!image_view_) {
        auto image_info = data_->get_image_info();
        vk::ImageViewCreateInfo info{};
        info.image = data_->get_image();
        info.format = slice_.image.format;
        info.components = vk::ComponentMapping{
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
        };
        info.subresourceRange.aspectMask = (image_info.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) == vk::ImageUsageFlagBits::eDepthStencilAttachment ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
        info.subresourceRange.baseMipLevel = slice_.image.mip_start;
        info.subresourceRange.levelCount = slice_.image.mip_count;
        info.subresourceRange.baseArrayLayer = slice_.image.array_start;
        info.subresourceRange.layerCount = slice_.image.array_count;
        auto device = this->data_->device()->logical_device();
        image_view_ = device.createImageView(info);
    }
    return image_view_;
}

void Resource::dispose() {
    data_ = nullptr;
    slice_ = {};
}

Buffer::Buffer(std::shared_ptr<ResourceData> resource_data, ResourceSlice view_slice): Resource(resource_data, view_slice) {
}

std::uint64_t Buffer::device_ptr() const {
    return data_->device_ptr() + slice_.buffer.offset;
}

std::uint64_t Buffer::external_ptr() const {
    return data_->external_ptr() + slice_.buffer.offset;
}

Buffer::~Buffer() noexcept{
    if (buffer_view_) {
        auto device = this->data_->device()->logical_device();
        device.destroyBufferView(buffer_view_);
        buffer_view_ = nullptr;
    }
}

Image::Image(std::shared_ptr<ResourceData> resource_data, ResourceSlice view_slice):Resource(resource_data, view_slice) {
}

Image::~Image() noexcept {
    if (image_view_) {
        auto device = this->data_->device()->logical_device();
        device.destroyImageView(image_view_);
        image_view_ = nullptr;
    }
}

std::shared_ptr<Buffer> Buffer::cast_format(vk::Format new_format) const {
    if (data_->resource_type() != ResourceType::BUFFER) {
        throw std::runtime_error("buffer_cast can only be called on buffer resources");
    }
    ResourceSlice view_slice{};
    view_slice.buffer.offset = slice_.buffer.offset;
    view_slice.buffer.size = slice_.buffer.size;
    view_slice.buffer.format = new_format;
    view_slice.buffer.scalar = slice_.buffer.scalar;
    return std::make_shared<Buffer>(data_, view_slice);
}

std::shared_ptr<Buffer> Buffer::cast_scalar(ScalarType new_scalar) const {
    if (data_->resource_type() != ResourceType::BUFFER) {
        throw std::runtime_error("buffer_cast can only be called on buffer resources");
    }
    ResourceSlice view_slice{};
    view_slice.buffer.offset = slice_.buffer.offset;
    view_slice.buffer.size = slice_.buffer.size;
    view_slice.buffer.format = slice_.buffer.format;
    view_slice.buffer.scalar = new_scalar;
    return std::make_shared<Buffer>(data_, view_slice);
}

std::shared_ptr<Buffer> Buffer::slice(int offset, int size) const {
    if (data_->resource_type() != ResourceType::BUFFER) {
        throw std::runtime_error("buffer_slice can only be called on buffer resources");
    }
    if (offset < 0 || size <= 0 || offset + size > slice_.buffer.size) {
        throw std::runtime_error("Invalid buffer slice range");
    }
    ResourceSlice view_slice{};
    view_slice.buffer.offset = slice_.buffer.offset + offset;
    view_slice.buffer.size = size;
    view_slice.buffer.format = slice_.buffer.format;
    view_slice.buffer.scalar = slice_.buffer.scalar;
    return std::make_shared<Buffer>(data_, view_slice);
}

std::shared_ptr<Buffer> Buffer::slice_array(int start, int count) const {
    int scalar_size = scalar_type_size(slice_.buffer.scalar);
    return slice(start * scalar_size, count * scalar_size);
}

pybind11::object Buffer::dlpack() const {
    void* ptr = nullptr;
    auto memory = data_->get_memory();
    DLDevice device = memory->dl_device();
    if (external_ptr() == 0) {
        throw std::runtime_error("DEVICE tensor requested but external pointer is unavailable");
    }
    ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(external_ptr()));

    int element_size = scalar_type_size(slice_.buffer.scalar);
    int elements = size() / element_size;

    auto* owner = new TensorOwner();
    owner->memory = memory;
    owner->shape = std::make_unique<std::int64_t[]>(1);
    owner->strides = std::make_unique<std::int64_t[]>(1);
    owner->shape[0] = elements;
    owner->strides[0] = 1;

    auto* managed = new DLManagedTensor{};
    managed->dl_tensor.data = ptr;
    managed->dl_tensor.device = device;
    managed->dl_tensor.ndim = 1;
    managed->dl_tensor.dtype = dlpack_dtype(slice_.buffer.scalar);
    managed->dl_tensor.shape = owner->shape.get();
    managed->dl_tensor.strides = owner->strides.get();
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = owner;
    managed->deleter = dlmanaged_tensor_deleter;

    return export_dltensor_py(managed);
}

std::shared_ptr<Image> Image::cast_format(vk::Format new_format) const {
    if (data_->resource_type() != ResourceType::IMAGE) {
        throw std::runtime_error("image_cast can only be called on image resources");
    }
    ResourceSlice view_slice{};
    view_slice.image.mip_start = slice_.image.mip_start;
    view_slice.image.mip_count = slice_.image.mip_count;
    view_slice.image.array_start = slice_.image.array_start;
    view_slice.image.array_count = slice_.image.array_count;
    view_slice.image.format = new_format;
    return std::make_shared<Image>(data_, view_slice);
}

std::shared_ptr<Image> Image::slice(int mip_start, int mip_count, int array_start, int array_count) const {
    if (mip_count == -1)
        mip_count = slice_.image.mip_count - mip_start;
    if (array_count == -1)
        array_count = slice_.image.array_count - array_start;
    if (data_->resource_type() != ResourceType::IMAGE) {
        throw std::runtime_error("image_slice can only be called on image resources");
    }
    if (mip_start < 0 || mip_count <= 0 || mip_start + mip_count > slice_.image.mip_count ||
        array_start < 0 || array_count <= 0 || array_start + array_count > slice_.image.array_count) {
        throw std::runtime_error("Invalid image slice range");
    }
    ResourceSlice view_slice{};
    view_slice.image.mip_start = slice_.image.mip_start + mip_start;
    view_slice.image.mip_count = mip_count;
    view_slice.image.array_start = slice_.image.array_start + array_start;
    view_slice.image.array_count = array_count;
    view_slice.image.format = slice_.image.format;
    return std::make_shared<Image>(data_, view_slice);
}

Device::Device(uint32_t device_index, bool enable_validation_layers) {
    device_index_ = device_index;
    uint32_t loader_version = VK_API_VERSION_1_1;
    if (vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS) {
        loader_version = VK_API_VERSION_1_1;
    }

    const uint32_t api_version = std::min(loader_version, VK_API_VERSION_1_3);
    vk::ApplicationInfo app("vk", 1, "vk", 1, api_version);

    std::vector<const char*> enabled_layers;
    const std::vector<VkLayerProperties> instance_layers = enumerate_instance_layers();
    if (enable_validation_layers) {
        auto layer_it = std::find_if(instance_layers.begin(), instance_layers.end(), [](const VkLayerProperties& layer) {
            return has_validation_suffix(layer.layerName);
        });
        if (layer_it == instance_layers.end()) {
            throw std::runtime_error("Validation layers were requested, but no *_validation layer was found");
        }
        enabled_layers.push_back(layer_it->layerName);
    }

    vk::InstanceCreateInfo ci{};
    ci.pApplicationInfo = &app;
    ci.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
    ci.ppEnabledLayerNames = enabled_layers.empty() ? nullptr : enabled_layers.data();

    instance_ = vk::createInstance(ci);

    const auto gpus = instance_.enumeratePhysicalDevices();
    if (gpus.empty()) {
        throw std::runtime_error("No Vulkan GPU found");
    }
    if (device_index >= gpus.size()) {
        throw std::runtime_error("Requested Vulkan GPU index is out of range");
    }

    physical_ = gpus[device_index];
    const auto supported_extensions = physical_.enumerateDeviceExtensionProperties();

    std::vector<const char*> enabled_extensions;
    const auto add_extension_if_supported = [&](const char* extension_name) {
        if (supports_extension(supported_extensions, extension_name)) {
            enabled_extensions.push_back(extension_name);
        }
    };

    add_extension_if_supported(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    add_extension_if_supported(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    add_extension_if_supported(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    add_extension_if_supported(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    add_extension_if_supported(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    add_extension_if_supported(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    add_extension_if_supported(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    add_extension_if_supported(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#if defined(_WIN32) && defined(VK_USE_PLATFORM_WIN32_KHR)
    add_extension_if_supported("VK_KHR_external_memory_win32");
#else
    add_extension_if_supported("VK_KHR_external_memory_fd");
#endif

    vk::PhysicalDeviceProperties2 properties2{};
    vk::PhysicalDeviceVulkan12Properties vulkan12_properties{};
    vk::PhysicalDeviceVulkan13Properties vulkan13_properties{};
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties{};
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_pipeline_properties{};

    properties2.pNext = &vulkan12_properties;
    vulkan12_properties.pNext = &vulkan13_properties;
    vulkan13_properties.pNext = &acceleration_structure_properties;
    acceleration_structure_properties.pNext = &ray_tracing_pipeline_properties;
    physical_.getProperties2(&properties2);

    vk::PhysicalDeviceFeatures2 features2{};
    vk::PhysicalDeviceVulkan12Features vulkan12_features{};
    vk::PhysicalDeviceVulkan13Features vulkan13_features{};
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features{};
    vk::PhysicalDeviceRayQueryFeaturesKHR ray_query_features{};
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_pipeline_features{};
    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features{};
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};

    features2.pNext = &vulkan12_features;
    vulkan12_features.pNext = &vulkan13_features;
    vulkan13_features.pNext = &acceleration_structure_features;
    acceleration_structure_features.pNext = &ray_query_features;
    ray_query_features.pNext = &ray_tracing_pipeline_features;
    ray_tracing_pipeline_features.pNext = &cooperative_matrix_features;
    cooperative_matrix_features.pNext = &atomic_float_features;
    physical_.getFeatures2(&features2);

    constexpr float priority = 1.0f;
    const auto queue_families = physical_.getQueueFamilyProperties();
    std::vector<std::vector<float>> queue_priorities(queue_families.size());
    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    queue_create_infos.reserve(queue_families.size());

    for (uint32_t family_index = 0; family_index < queue_families.size(); ++family_index) {
        const auto& family = queue_families[family_index];
        if (family.queueCount == 0) {
            continue;
        }

        queue_priorities[family_index].assign(family.queueCount, priority);
        queue_create_infos.emplace_back(
            vk::DeviceQueueCreateFlags{},
            family_index,
            family.queueCount,
            queue_priorities[family_index].data()
        );
    }

    vk::DeviceCreateInfo dci{};
    dci.pNext = &features2;
    dci.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    dci.pQueueCreateInfos = queue_create_infos.empty() ? nullptr : queue_create_infos.data();
    dci.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    dci.ppEnabledExtensionNames = enabled_extensions.empty() ? nullptr : enabled_extensions.data();

    device_ = physical_.createDevice(dci);

    queues_.resize(queue_families.size());
    for (uint32_t family_index = 0; family_index < queue_families.size(); ++family_index) {
        const auto& family = queue_families[family_index];
        queues_[family_index].resize(family.queueCount);
        for (uint32_t queue_index = 0; queue_index < family.queueCount; ++queue_index) {
            queues_[family_index][queue_index] = device_.getQueue(family_index, queue_index);
        }
    }
}

vk::PhysicalDevice Device::physical_device() const noexcept {
    return physical_;
}

vk::Device Device::logical_device() const noexcept {
    return device_;
}

uint32_t Device::device_index() const noexcept {
    return device_index_;
}

void Device::init() {
    const auto memory_properties = physical_.getMemoryProperties();
    bool found_cpu = false;
    const uint32_t cpu_type = find_memory_type_index(
        memory_properties,
        vk::MemoryPropertyFlagBits::eHostVisible,
        vk::MemoryPropertyFlagBits::eHostCoherent,
        &found_cpu
    );
    if (!found_cpu) {
        throw std::runtime_error("Could not find host-visible Vulkan memory type");
    }
    bool found_gpu = false;
    const uint32_t gpu_type = find_memory_type_index(
        memory_properties,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::MemoryPropertyFlags{},
        &found_gpu
    );
    if (!found_gpu) {
        throw std::runtime_error("Could not find device-local Vulkan memory type");
    }
    const auto gpu_flags = memory_properties.memoryTypes[gpu_type].propertyFlags;
    const bool gpu_host_visible = (gpu_flags & vk::MemoryPropertyFlagBits::eHostVisible) == vk::MemoryPropertyFlagBits::eHostVisible;
    host_memory_manager_ = std::make_unique<MemoryManager>(shared_from_this(), cpu_type, true);
    device_memory_manager_ = std::make_unique<MemoryManager>(shared_from_this(), gpu_type, gpu_host_visible);
}

MemoryManager& Device::host_memory_manager() {
    return *host_memory_manager_;
}

MemoryManager& Device::device_memory_manager() {
    return *device_memory_manager_;
}

std::shared_ptr<Buffer> Device::create_buffer(int size, MemoryLocation location) {
    if (size <= 0) {
        throw std::runtime_error("Buffer size must be positive");
    }
    MemoryManager& manager = (location == MemoryLocation::HOST) ? host_memory_manager() : device_memory_manager();
    auto memory = manager.allocate(size, 256);
    auto data = std::make_shared<ResourceData>(shared_from_this(), memory, vk::Image{}, vk::ImageCreateInfo{});
    ResourceSlice full_slice {};
    full_slice.buffer.offset = memory->offset();
    full_slice.buffer.size = size;
    full_slice.buffer.format = vk::Format::eUndefined;
    full_slice.buffer.scalar = ScalarType::UINT8;
    return std::make_shared<Buffer>(data, full_slice);
}

std::shared_ptr<Buffer> Device::create_array(int elements, ScalarType type, MemoryLocation location) {
    if (elements <= 0) {
        throw std::runtime_error("Array elements must be positive");
    }
    const int itemsize = scalar_type_size(type);
    const int bytes = elements * itemsize;
    return create_buffer(bytes, location)->cast_scalar(type);
}

std::shared_ptr<Buffer> Device::create_texels(int elements, vk::Format format, MemoryLocation location) {
    if (elements <= 0) {
        throw std::runtime_error("Texel elements must be positive");
    }
    const int bytes = elements * formatSize(format);
    return create_buffer(bytes, location)->cast_format(format)->cast_scalar(format_scalar_type(format));
}

std::shared_ptr<Image> Device::create_image(int width, int height, int depth, int mip_levels,
    int array_layers, vk::Format format, MemoryLocation location) {
    return nullptr;
}

void Device::destroy_data(std::shared_ptr<ResourceData> data) noexcept {
}

pybind11::object Device::allocate_tensor_dlpack(const std::vector<int>& shape, ScalarType type, MemoryLocation location) {
    int dimension = shape.size();
    if (dimension <= 0) {
        throw std::runtime_error("Dimension must be positive");
    }
    int elements = 1;
    for (int i = 0; i < dimension; ++i) {
        elements *= shape[i];
    }
    int element_size = scalar_type_size(type);
    auto& manager = location == MemoryLocation::HOST ? host_memory_manager() : device_memory_manager();
    auto memory = manager.allocate(elements * element_size, 256);
    auto ptr = memory->external_ptr();
    if (ptr == 0) {
        throw std::runtime_error("DEVICE tensor requested but external pointer is unavailable");
    }
    DLDevice device = memory->dl_device();
    auto* owner = new TensorOwner();
    owner->memory = memory;
    owner->shape = std::make_unique<std::int64_t[]>(dimension);
    owner->strides = std::make_unique<std::int64_t[]>(dimension);
    for (int i=0; i<dimension; ++i) {
        owner->shape[i] = shape[i];
        elements /= shape[i];
        owner->strides[i] = elements;
    }
    auto* managed = new DLManagedTensor{};
    managed->dl_tensor.data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(ptr));
    managed->dl_tensor.device = device;
    managed->dl_tensor.ndim = dimension;
    managed->dl_tensor.dtype = dlpack_dtype(type);
    managed->dl_tensor.shape = owner->shape.get();
    managed->dl_tensor.strides = owner->strides.get();
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = owner;
    managed->deleter = dlmanaged_tensor_deleter;

    return export_dltensor_py(managed);
}


// py::object Device::tensor(int elements, const py::object& dtype, MemoryLocation location) {
//     if (elements <= 0) {
//         throw std::runtime_error("tensor elements must be positive");
//     }
//
//     const std::size_t itemsize = dtype_itemsize(dtype);
//     const int bytes = static_cast<int>(elements * itemsize);
//     MemoryManager& manager = (location == MemoryLocation::HOST) ? host_memory_manager() : device_memory_manager();
//     MemorySlice memory = manager.allocate(bytes);
//
//     void* ptr = nullptr;
//     DLDevice device;
//     if (location == MemoryLocation::HOST) {
//         device = dl_device_for_host_;
//         ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(memory.cpu_ptr()));
//     } else {
//         if (memory.external_ptr() == 0) {
//             throw std::runtime_error("DEVICE tensor requested but external pointer is unavailable");
//         }
//         device = dl_device_for_device_;
//         ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(memory.external_ptr()));
//     }
//
//     auto* owner = new TensorOwner();
//     owner->memory = std::make_shared<MemorySlice>(std::move(memory));
//     owner->shape = std::make_unique<std::int64_t[]>(1);
//     owner->strides = std::make_unique<std::int64_t[]>(1);
//     owner->shape[0] = elements;
//     owner->strides[0] = 1;
//
//     auto* managed = new DLManagedTensor{};
//     managed->dl_tensor.data = ptr;
//     managed->dl_tensor.device = device;
//     managed->dl_tensor.ndim = 1;
//     managed->dl_tensor.dtype = dlpack_dtype(dtype);
//     managed->dl_tensor.shape = owner->shape.get();
//     managed->dl_tensor.strides = owner->strides.get();
//     managed->dl_tensor.byte_offset = 0;
//     managed->manager_ctx = owner;
//     managed->deleter = dlmanaged_tensor_deleter;
//
//     return from_dlpack(managed);
// }

void Device::dispose() noexcept {
    try {
        if (device_) {
            (void)device_.waitIdle();
        }
    } catch (...) {
        // Destructors and dispose must not throw.
    }

    // Destroy now any pending resource
    for (auto r: resources_)
        destroy_data(r);

    device_memory_manager_.reset();
    host_memory_manager_.reset();
    device_index_ = 0;

    queues_.clear();

    if (device_) {
        device_.destroy();
        device_ = vk::Device{};
    }

    physical_ = vk::PhysicalDevice{};

    if (instance_) {
        instance_.destroy();
        instance_ = vk::Instance{};
    }
}

Device::~Device() noexcept {
    dispose();
}

MemorySlice::MemorySlice() = default;

MemorySlice::MemorySlice(
    std::shared_ptr<MemoryPage> page,
    int allocated_offset,
    int allocated_size,
    int offset,
    int size
) noexcept
    : page_(page),
      allocated_offset_(allocated_offset),
      allocated_size_(allocated_size),
      offset_(offset),
      size_(size) {}

MemorySlice::MemorySlice(MemorySlice&& other) noexcept
    : page_(std::move(other.page_)),
      allocated_offset_(other.allocated_offset_),
      allocated_size_(other.allocated_size_),
      offset_(other.offset_),
      size_(other.size_) {
    other.page_.reset();
    other.allocated_offset_ = 0;
    other.allocated_size_ = 0;
    other.offset_ = 0;
    other.size_ = 0;
}

MemorySlice& MemorySlice::operator=(MemorySlice&& other) noexcept {
    if (this != &other) {
        release();
        page_ = other.page_;
        allocated_offset_ = other.allocated_offset_;
        allocated_size_ = other.allocated_size_;
        offset_ = other.offset_;
        size_ = other.size_;

        other.page_.reset();
        other.allocated_offset_ = 0;
        other.allocated_size_ = 0;
        other.offset_ = 0;
        other.size_ = 0;
    }
    return *this;
}

vk::Buffer MemorySlice::page_buffer() const noexcept {
    if (page_) {
        return page_->buffer();
    }
    return vk::Buffer{};
}

std::uint64_t MemorySlice::device_ptr() const noexcept {
    if (!page_) {
        return 0;
    }
    return page_->device_ptr() + static_cast<std::uint64_t>(offset_);
}

std::uint64_t MemorySlice::external_ptr() const noexcept {
    if (!page_) {
        return 0;
    }
    return page_->external_ptr() + static_cast<std::uint64_t>(offset_);
}

bool MemorySlice::host_visible() const noexcept { return page_ && page_->host_visible(); }

DLDevice MemorySlice::dl_device() const noexcept { return page_ ? page_->dl_device() : DLDevice{0, 0}; }

void MemorySlice::release() noexcept {
    //printf("MemorySlice::release() called for offset %d, size %d\n", allocated_offset_, allocated_size_);
    if (page_) {
        page_->free_memory(allocated_offset_);
    }
    page_.reset();
    offset_ = 0;
    size_ = 0;
}

MemorySlice::~MemorySlice() noexcept {
    release();
}

MemoryManager::MemoryManager(std::shared_ptr<Device> device, uint32_t memory_type_index, bool host_visible):
      device_(device),
      memory_type_index_(memory_type_index),
      host_visible_(host_visible) {
    const auto vendor_id = device_->physical_device().getProperties().vendorID;
    printf("[INFO] Loading interop library for vendor ID: 0x%04X\n", vendor_id);
    interop_library_ = std::make_shared<ExternalInteropLibraryImpl>(interop_library_base_name(vendor_id));
    if (!interop_library_) {
        printf("[WARNING] Library not found.");
    }
    try_import_memory_ = interop_library_ && interop_library_->loaded()
        ? interop_library_->try_import_memory_fn()
        : nullptr;
    if (!try_import_memory_) {
        printf("[WARNING] try_import_memory could not be loaded.");
    }
}

MemoryManager::~MemoryManager() noexcept = default;

std::shared_ptr<MemorySlice> MemoryManager::allocate(int size, int alignment) {
    if (size <= 0) {
        throw std::runtime_error("Allocation size must be positive");
    }

    for (auto& page : pages_) {
        if (page->can_allocate(size, alignment)) {
            return page->allocate(size, alignment);
        }
    }

    int page_capacity = std::max(size, next_page_capacity_); // big data create their own page.

    pages_.push_back(std::make_shared<MemoryPage>(device_, memory_type_index_, host_visible_, page_capacity, try_import_memory_));
    if (next_page_capacity_ <= std::numeric_limits<int>::max() / 2) {
        next_page_capacity_ *= 2;
    }

    return pages_.back()->allocate(size, alignment);
}

std::uint64_t MemoryManager::external_to_device(std::uint64_t external_ptr) const noexcept {
    for (auto& page : pages_) {
        auto device_ptr = page->external_to_device(external_ptr);
        if (device_ptr != 0) {
            return device_ptr;
        }
    }
    return 0;
}

std::uint64_t MemoryManager::device_to_external(std::uint64_t device_ptr) const noexcept {
    for (auto& page : pages_) {
        auto external_ptr = page->device_to_external(device_ptr);
        if (external_ptr != 0) {
            return external_ptr;
        }
    }
    return 0;
}

MemoryPage::MemoryPage(std::shared_ptr<Device> device, uint32_t memory_type_index, bool host_visible, int capacity, ExternalImportFn try_import_memory)
    : device_(device),
      memory_type_index_(memory_type_index),
      host_visible_(host_visible),
      capacity_(capacity),
      allocator_(std::make_unique<MemoryAllocator>(capacity)),
      try_import_memory_(try_import_memory) {
    if (capacity <= 0) {
        throw std::runtime_error("Page capacity must be positive");
    }

    auto dev = device->vk_device();

    vk::ExternalMemoryBufferCreateInfo external_buffer_info{};
#if defined(_WIN32)
    external_buffer_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
#else
    external_buffer_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
#endif

    vk::BufferCreateInfo buffer_info(
        vk::BufferCreateFlags{},
        static_cast<vk::DeviceSize>(capacity_),
        full_buffer_usage_flags(),
        vk::SharingMode::eExclusive
    );
    if (!host_visible_) {
        buffer_info.pNext = &external_buffer_info;
    }
    buffer_ = dev.createBuffer(buffer_info);

    const vk::MemoryRequirements requirements = dev.getBufferMemoryRequirements(buffer_);
    vk::ExportMemoryAllocateInfo export_alloc_info{};
#if defined(_WIN32)
    export_alloc_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
#else
    export_alloc_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
#endif

    vk::MemoryAllocateFlagsInfo alloc_flags_info{};
    alloc_flags_info.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
    vk::MemoryAllocateInfo alloc_info(requirements.size, memory_type_index_);
    if (!host_visible_) {
        alloc_flags_info.pNext = &export_alloc_info;
    }
    alloc_info.pNext = &alloc_flags_info;
    memory_ = dev.allocateMemory(alloc_info);
    dev.bindBufferMemory(buffer_, memory_, 0);

    try {
        const vk::BufferDeviceAddressInfo address_info(buffer_);
        device_ptr_ = static_cast<std::uint64_t>(dev.getBufferAddress(address_info));
    } catch (...) {
        device_ptr_ = 0;
    }

    if (host_visible_) {
        external_ptr_ = reinterpret_cast<std::uint64_t>(dev.mapMemory(memory_, 0, VK_WHOLE_SIZE));
    } else if (try_import_memory_) {
        external_ptr_ = try_import_memory_(dev, device->device_index(), static_cast<VkDeviceMemory>(memory_), capacity_);
    }

    const int device_id = device->device_index();
    if (host_visible_) {
        dl_device_ = {1, device_id};
    } else if (external_ptr_) {
        dl_device_ = {2, device_id};
    } else {
        dl_device_ = {0, device_id};
    }
}

MemoryPage::~MemoryPage() noexcept {
    if (!device_) {
        return;
    }

    auto dev = device_->vk_device();
    if (host_visible_ && memory_) {
        dev.unmapMemory(memory_);
    }
    if (buffer_) {
        dev.destroyBuffer(buffer_);
        buffer_ = vk::Buffer{};
    }
    if (memory_) {
        dev.freeMemory(memory_);
        memory_ = vk::DeviceMemory{};
    }
    device_ptr_ = 0;
    external_ptr_ = 0;
}

bool MemoryPage::can_allocate(int size, int alignment) const {
    return allocator_->can_allocate(size + alignment - 1);
}

std::shared_ptr<MemorySlice> MemoryPage::allocate(int size, int alignment) {
    const int offset = allocator_->allocate(size + alignment - 1);
    int aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
    return std::make_shared<MemorySlice>(shared_from_this(), offset, size + alignment - 1, aligned_offset, size);
}

void MemoryPage::free_memory(int allocated_offset) noexcept {
    allocator_->free_memory(allocated_offset);
}

int MemoryPage::capacity() const noexcept { return capacity_; }
vk::Buffer MemoryPage::buffer() const noexcept { return buffer_; }
vk::DeviceMemory MemoryPage::memory() const noexcept { return memory_; }
std::uint64_t MemoryPage::device_ptr() const noexcept { return device_ptr_; }
std::uint64_t MemoryPage::external_ptr() const noexcept { return external_ptr_; }

std::uint64_t MemoryPage::external_to_device(std::uint64_t external_ptr) const noexcept {
    if (external_ptr_ == 0) {
        return 0;
    }
    if (external_ptr < external_ptr_ || external_ptr >= external_ptr_ + capacity_) {
        return 0;
    }
    return device_ptr_ + (external_ptr - external_ptr_);
}

std::uint64_t MemoryPage::device_to_external(std::uint64_t device_ptr) const noexcept {
    if (external_ptr_ == 0) {
        return 0;
    }
    if (device_ptr < device_ptr_ || device_ptr >= device_ptr_ + capacity_) {
        return 0;
    }
    return external_ptr_ + (device_ptr - device_ptr_);
}

MemoryAllocator::MemoryAllocator(int capacity) : capacity_(capacity), bins_(32) {
    if (capacity_ <= 0) {
        throw std::runtime_error("Allocator capacity must be positive");
    }

    auto node = std::make_unique<Range>();
    node->offset = 0;
    node->size = capacity_;
    head_ = node.get();
    nodes_.push_back(std::move(node));
    rebuild_bins();
}

MemoryAllocator::~MemoryAllocator() noexcept = default;

int MemoryAllocator::bin_for_size(int size) const {
    int value = 1;
    int bin = 0;
    while (value < size && bin < static_cast<int>(bins_.size()) - 1) {
        value <<= 1;
        ++bin;
    }
    return bin;
}

void MemoryAllocator::clear_bins() {
    for (auto& bucket : bins_) {
        bucket.clear();
    }
}

void MemoryAllocator::rebuild_bins() {
    clear_bins();
    for (Range* node = head_; node != nullptr; node = node->next) {
        node->bin = bin_for_size(node->size);
        bins_[node->bin].push_back(node);
    }
}

void MemoryAllocator::remove_node(Range* node) noexcept {
    if (!node) {
        return;
    }

    if (node->prev) {
        node->prev->next = node->next;
    } else {
        head_ = node->next;
    }
    if (node->next) {
        node->next->prev = node->prev;
    }

    const auto it = std::find_if(nodes_.begin(), nodes_.end(), [node](const std::unique_ptr<Range>& p) {
        return p.get() == node;
    });
    if (it != nodes_.end()) {
        nodes_.erase(it);
    }
}

bool MemoryAllocator::can_allocate(int size) const {
    if (size <= 0) {
        return false;
    }
    const int start_bin = bin_for_size(size);
    for (int bin = start_bin; bin < static_cast<int>(bins_.size()); ++bin) {
        for (const Range* node : bins_[bin]) {
            if (node->size >= size) {
                return true;
            }
        }
    }
    return false;
}

int MemoryAllocator::allocate(int size) {
    if (size <= 0) {
        throw std::runtime_error("Allocation size must be positive");
    }

    const int start_bin = bin_for_size(size);
    Range* selected = nullptr;
    for (int bin = start_bin; bin < static_cast<int>(bins_.size()) && !selected; ++bin) {
        for (Range* node : bins_[bin]) {
            if (node->size >= size) {
                selected = node;
                break;
            }
        }
    }

    if (!selected) {
        throw std::runtime_error("Out of memory in page allocator");
    }

    const int offset = selected->offset;
    if (selected->size == size) {
        remove_node(selected);
    } else {
        selected->offset += size;
        selected->size -= size;
    }

    allocations_[offset] = size;
    rebuild_bins();
    return offset;
}

void MemoryAllocator::free_memory(int offset) noexcept {
    const auto allocation = allocations_.find(offset);
    if (allocation == allocations_.end()) {
        return;
    }

    const int size = allocation->second;
    allocations_.erase(allocation);

    Range* prev = nullptr;
    Range* curr = head_;
    while (curr && curr->offset < offset) {
        prev = curr;
        curr = curr->next;
    }

    auto node = std::make_unique<Range>();
    node->offset = offset;
    node->size = size;
    node->prev = prev;
    node->next = curr;
    Range* inserted = node.get();
    if (prev) {
        prev->next = inserted;
    } else {
        head_ = inserted;
    }
    if (curr) {
        curr->prev = inserted;
    }
    nodes_.push_back(std::move(node));

    if (inserted->prev && inserted->prev->offset + inserted->prev->size == inserted->offset) {
        Range* merged = inserted->prev;
        merged->size += inserted->size;
        remove_node(inserted);
        inserted = merged;
    }

    if (inserted && inserted->next && inserted->offset + inserted->size == inserted->next->offset) {
        inserted->size += inserted->next->size;
        remove_node(inserted->next);
    }

    rebuild_bins();
}
