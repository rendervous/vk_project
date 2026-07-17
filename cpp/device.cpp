#include "device.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#if defined(_WIN32) && defined(VK_USE_PLATFORM_WIN32_KHR)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

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

uint32_t formatSize(Format format) {
    vk::Format vk_format = (vk::Format) format;
    switch (vk_format)
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

// Maps a scalar component type/count (as found in a vertex-layout struct
// field) to the named Format entry Vulkan expects for a vertex attribute.
Format vertex_attribute_format(Type component_type, int components) {
    switch (component_type) {
        case Type::FLOAT32:
            switch (components) {
                case 1: return Format::R32_Float;
                case 2: return Format::RG32_Float;
                case 3: return Format::RGB32_Float;
                case 4: return Format::RGBA32_Float;
                default: break;
            }
            break;
        case Type::INT32:
            switch (components) {
                case 1: return Format::R32_SInt;
                case 2: return Format::RG32_SInt;
                case 3: return Format::RGB32_SInt;
                case 4: return Format::RGBA32_SInt;
                default: break;
            }
            break;
        case Type::UINT32:
            switch (components) {
                case 1: return Format::R32_UInt;
                case 2: return Format::RG32_UInt;
                case 3: return Format::RGB32_UInt;
                case 4: return Format::RGBA32_UInt;
                default: break;
            }
            break;
        case Type::FLOAT64:
            switch (components) {
                case 1: return Format::R64_Float;
                case 2: return Format::RG64_Float;
                case 3: return Format::RGB64_Float;
                case 4: return Format::RGBA64_Float;
                default: break;
            }
            break;
        case Type::INT64:
            switch (components) {
                case 1: return Format::R64_SInt;
                case 2: return Format::RG64_SInt;
                case 3: return Format::RGB64_SInt;
                case 4: return Format::RGBA64_SInt;
                default: break;
            }
            break;
        case Type::UINT64:
            switch (components) {
                case 1: return Format::R64_UInt;
                case 2: return Format::RG64_UInt;
                case 3: return Format::RGB64_UInt;
                case 4: return Format::RGBA64_UInt;
                default: break;
            }
            break;
        case Type::FLOAT16:
            switch (components) {
                case 1: return Format::R16_Float;
                case 2: return Format::RG16_Float;
                case 3: return Format::RGB16_Float;
                case 4: return Format::RGBA16_Float;
                default: break;
            }
            break;
        case Type::INT16:
            switch (components) {
                case 1: return Format::R16_SInt;
                case 2: return Format::RG16_SInt;
                case 3: return Format::RGB16_SInt;
                case 4: return Format::RGBA16_SInt;
                default: break;
            }
            break;
        case Type::UINT16:
            switch (components) {
                case 1: return Format::R16_UInt;
                case 2: return Format::RG16_UInt;
                case 3: return Format::RGB16_UInt;
                case 4: return Format::RGBA16_UInt;
                default: break;
            }
            break;
        case Type::INT8:
            switch (components) {
                case 1: return Format::R8_SInt;
                case 2: return Format::RG8_SInt;
                case 3: return Format::RGB8_SInt;
                case 4: return Format::RGBA8_SInt;
                default: break;
            }
            break;
        case Type::UINT8:
        case Type::BOOL:
            switch (components) {
                case 1: return Format::R8_UInt;
                case 2: return Format::RG8_UInt;
                case 3: return Format::RGB8_UInt;
                case 4: return Format::RGBA8_UInt;
                default: break;
            }
            break;
        default:
            break;
    }
    throw std::runtime_error("vertex_layout: unsupported scalar/vector component type or count for a vertex attribute");
}

vk::DescriptorType to_vk_descriptor_type(DescriptorType type) {
    switch (type) {
        case DescriptorType::STORAGE_BUFFER: return vk::DescriptorType::eStorageBuffer;
        case DescriptorType::UNIFORM_BUFFER: return vk::DescriptorType::eUniformBuffer;
        case DescriptorType::STORAGE_IMAGE: return vk::DescriptorType::eStorageImage;
        case DescriptorType::SAMPLED_IMAGE: return vk::DescriptorType::eSampledImage;
        case DescriptorType::SAMPLER: return vk::DescriptorType::eSampler;
        case DescriptorType::COMBINED_IMAGE_SAMPLER: return vk::DescriptorType::eCombinedImageSampler;
        case DescriptorType::ACCELERATION_STRUCTURE: return vk::DescriptorType::eAccelerationStructureKHR;
        default: throw std::runtime_error("Unsupported descriptor type");
    }
}

vk::ShaderStageFlagBits to_vk_shader_stage(ShaderStageType type) {
    switch (type) {
        case ShaderStageType::VERTEX: return vk::ShaderStageFlagBits::eVertex;
        case ShaderStageType::FRAGMENT: return vk::ShaderStageFlagBits::eFragment;
        case ShaderStageType::GEOMETRY: return vk::ShaderStageFlagBits::eGeometry;
        case ShaderStageType::TESS_CONTROL: return vk::ShaderStageFlagBits::eTessellationControl;
        case ShaderStageType::TESS_EVAL: return vk::ShaderStageFlagBits::eTessellationEvaluation;
        case ShaderStageType::COMPUTE: return vk::ShaderStageFlagBits::eCompute;
        default: throw std::runtime_error("Unsupported shader stage type");
    }
}

// Stage name understood by glslangValidator's `-S` flag.
const char* to_glslang_stage_name(ShaderStageType type) {
    switch (type) {
        case ShaderStageType::VERTEX: return "vert";
        case ShaderStageType::FRAGMENT: return "frag";
        case ShaderStageType::GEOMETRY: return "geom";
        case ShaderStageType::TESS_CONTROL: return "tesc";
        case ShaderStageType::TESS_EVAL: return "tese";
        case ShaderStageType::COMPUTE: return "comp";
        default: throw std::runtime_error("Unsupported shader stage type");
    }
}

// Locates the glslangValidator executable: relies on it being on PATH by
// default (this project has no build- or link-time dependency on the Vulkan
// SDK for shader compilation), falling back to $VULKAN_SDK/Bin as a
// convenience if that environment variable happens to be set and PATH
// wasn't configured.
std::string glslang_validator_path() {
#if defined(_WIN32)
    constexpr const char* exe_name = "glslangValidator.exe";
#else
    constexpr const char* exe_name = "glslangValidator";
#endif
    if (const char* sdk = std::getenv("VULKAN_SDK")) {
        std::filesystem::path candidate = std::filesystem::path(sdk) / "Bin" / exe_name;
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
            return candidate.string();
        }
    }
    return exe_name;
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

DLDataType dlpack_dtype (Type scalar) {
    switch (scalar) {
        case Type::FLOAT16: return {2, 16, 1};
        case Type::FLOAT32: return {2, 32, 1};
        case Type::FLOAT64: return {2, 64, 1 };
        case Type::INT8: return {0, 8, 1};
        case Type::BOOL: return {6, 8, 1};
        case Type::INT16: return {0, 16, 1};
        case Type::INT32: return {0, 32, 1};
        case Type::INT64: return {0, 64, 1};
        case Type::UINT8: return {1, 8, 1};
        case Type::UINT16: return {1, 16, 1};
        case Type::UINT32: return {1, 32, 1};
        case Type::UINT64: return {1, 64, 1};
        default:
            throw std::invalid_argument("Wrong scalar type");
    }
}

// Reverse of dlpack_dtype(): the Type matching an incoming DLPack
// tensor's dtype, for Device::wrap().
Type scalar_type_from_dlpack_dtype(const DLDataType& dtype) {
    switch (dtype.code) {
        case 0: // signed int
            switch (dtype.bits) {
                case 8: return Type::INT8;
                case 16: return Type::INT16;
                case 32: return Type::INT32;
                case 64: return Type::INT64;
                default: break;
            }
            break;
        case 1: // unsigned int
            switch (dtype.bits) {
                case 8: return Type::UINT8;
                case 16: return Type::UINT16;
                case 32: return Type::UINT32;
                case 64: return Type::UINT64;
                default: break;
            }
            break;
        case 2: // float
            switch (dtype.bits) {
                case 16: return Type::FLOAT16;
                case 32: return Type::FLOAT32;
                case 64: return Type::FLOAT64;
                default: break;
            }
            break;
        case 6: // bool
            return Type::BOOL;
        default:
            break;
    }
    throw std::runtime_error("Device::wrap: unsupported DLPack dtype");
}

// Best-effort Type from a Python buffer-protocol format string
// (struct module conventions), for Device::wrap().
Type scalar_type_from_buffer_format(const std::string& format, std::uint64_t itemsize) {
    if (format == "f") return Type::FLOAT32;
    if (format == "d") return Type::FLOAT64;
    if (format == "e") return Type::FLOAT16;
    if (format == "b" || format == "c") return Type::INT8;
    if (format == "B") return Type::UINT8;
    if (format == "?") return Type::BOOL;
    if (format == "h") return Type::INT16;
    if (format == "H") return Type::UINT16;
    if (format == "i" || format == "l") return itemsize == 8 ? Type::INT64 : Type::INT32;
    if (format == "I" || format == "L") return itemsize == 8 ? Type::UINT64 : Type::UINT32;
    if (format == "q") return Type::INT64;
    if (format == "Q") return Type::UINT64;
    throw std::runtime_error("Device::wrap: unsupported buffer format '" + format + "'");
}

// True if `shape`/`strides` (DLPack convention: strides in units of
// elements) describe a C-contiguous (row-major, tightly packed) tensor.
// `strides == nullptr` is the DLPack convention for "contiguous, compute it
// yourself".
bool dltensor_is_contiguous(const std::int64_t* shape, const std::int64_t* strides, int ndim) {
    if (!strides) return true;
    std::int64_t expected = 1;
    for (int d = ndim - 1; d >= 0; --d) {
        if (shape[d] > 1 && strides[d] != expected) return false;
        expected *= shape[d];
    }
    return true;
}

// Host-only analogue of vk_cuda_interop.cpp's copy_strided: copies between
// a flat contiguous buffer and a strided view, for the case where both
// sides are guaranteed CPU-dereferenceable (no CUDA device memory
// involved). See Device::vk_copy_in/vk_copy_out.
void copy_strided_host(
    void* contiguous, void* strided, const std::int64_t* shape, const std::int64_t* strides,
    int ndim, std::uint64_t itemsize, bool contiguous_is_dst) {
    if (ndim == 0) {
        std::memcpy(contiguous_is_dst ? contiguous : strided, contiguous_is_dst ? strided : contiguous, itemsize);
        return;
    }
    std::uint64_t row_elements = 1;
    for (int d = 1; d < ndim; ++d) row_elements *= static_cast<std::uint64_t>(shape[d]);
    bool inner_packed = true;
    std::int64_t expected = 1;
    for (int d = ndim - 1; d >= 1; --d) {
        if (strides[d] != expected) { inner_packed = false; break; }
        expected *= shape[d];
    }
    if (ndim == 1 || inner_packed) {
        const std::uint64_t row_bytes = row_elements * itemsize;
        for (std::int64_t i = 0; i < shape[0]; ++i) {
            char* s = reinterpret_cast<char*>(strided) + static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(strides[0]) * itemsize;
            char* c = reinterpret_cast<char*>(contiguous) + static_cast<std::uint64_t>(i) * row_bytes;
            std::memcpy(contiguous_is_dst ? c : s, contiguous_is_dst ? s : c, row_bytes);
        }
        return;
    }
    for (std::int64_t i = 0; i < shape[0]; ++i) {
        char* next_strided = reinterpret_cast<char*>(strided) + static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(strides[0]) * itemsize;
        char* next_contiguous = reinterpret_cast<char*>(contiguous) + static_cast<std::uint64_t>(i) * row_elements * itemsize;
        copy_strided_host(next_contiguous, next_strided, shape + 1, strides + 1, ndim - 1, itemsize, contiguous_is_dst);
    }
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
// Mirrors vk_cuda_interop.cpp's try_import_semaphore: imports a Vulkan
// timeline semaphore (Device::vk_interop_semaphore()) into CUDA, returning
// an opaque cudaExternalSemaphore_t handle (0 on failure/unsupported).
using ExternalImportSemaphoreFn = std::uint64_t (*)(vk::Device device, int device_index, vk::Semaphore semaphore);
// Mirrors vk_cuda_interop.cpp's copy_from_dlpack_to_ptr/copy_ptr_to_dlpack
// signatures exactly (shape/strides: `ndim` entries, in units of `itemsize`
// elements, per the DLPack convention). `wait_semaphore`/`wait_value`: see
// try_import_semaphore's comment -- a GPU-side wait for a specific Vulkan
// submission, enqueued before the copy itself. Both return 0 on success.
using ExternalCopyFromDlpackFn = int (*)(
    void* tensor_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
    std::uint64_t itemsize, std::uint64_t dst_ptr, std::uint64_t total_bytes,
    std::uint64_t wait_semaphore, std::uint64_t wait_value);
using ExternalCopyToDlpackFn = int (*)(
    std::uint64_t src_ptr, std::uint64_t total_bytes,
    void* tensor_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
    std::uint64_t itemsize, std::uint64_t wait_semaphore, std::uint64_t wait_value);

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

// `acceleration_structure_supported` gates
// VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR (so
// pooled buffers -- e.g. ADSTriangles::vertices -- can be used directly as
// acceleration structure build input): it comes from
// VK_KHR_acceleration_structure, so setting it when that extension isn't
// enabled would be a validation error (even on a buffer never actually used
// for one) -- see Device::vk_acceleration_structure_supported(). An
// acceleration structure's own backing/storage buffer is allocated
// separately (see vk_AccelerationStructure), with its own dedicated
// VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR usage, so that bit
// has no place here.
vk::BufferUsageFlags full_buffer_usage_flags(bool acceleration_structure_supported) {
    vk::BufferUsageFlags flags = vk::BufferUsageFlagBits::eTransferSrc |
           vk::BufferUsageFlagBits::eTransferDst |
           vk::BufferUsageFlagBits::eUniformTexelBuffer |
           vk::BufferUsageFlagBits::eStorageTexelBuffer |
           vk::BufferUsageFlagBits::eUniformBuffer |
           vk::BufferUsageFlagBits::eStorageBuffer |
           vk::BufferUsageFlagBits::eIndexBuffer |
           vk::BufferUsageFlagBits::eVertexBuffer |
           vk::BufferUsageFlagBits::eIndirectBuffer |
           vk::BufferUsageFlagBits::eShaderDeviceAddress;
    if (acceleration_structure_supported) {
        flags |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    }
    return flags;
}

vk::ImageUsageFlags full_image_usage_flags() {
    return vk::ImageUsageFlagBits::eTransferSrc |
           vk::ImageUsageFlagBits::eTransferDst |
           vk::ImageUsageFlagBits::eSampled |
           vk::ImageUsageFlagBits::eStorage |
           vk::ImageUsageFlagBits::eColorAttachment;
}

// Defined further below, alongside the rest of the Layout-inspection
// helpers; forward-declared here so CommandBuffer::bind_indices (which
// comes first in the file) can use it.
Type resolve_component_type(const Layout& layout);

// Copies `total_bytes` from a DLPack-compatible or Python buffer-protocol
// object `source` into `dst_external_ptr` (`dst_location`). Same
// DLPack/buffer-protocol handling and vk_copy_in dispatch as
// Device::wrap()'s copy-in path; used by Buffer::load()/Tensor::load().
void copy_pyobject_into(Device& device, std::uint64_t dst_external_ptr, MemoryLocation dst_location,
    std::uint64_t total_bytes, const pybind11::object& source) {
    if (pybind11::hasattr(source, "__dlpack__")) {
        pybind11::object capsule = source.attr("__dlpack__")();
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
        if (!managed) {
            throw std::runtime_error("load(): invalid DLPack capsule (expected a \"dltensor\" capsule)");
        }
        const DLTensor& t = managed->dl_tensor;
        char* data_ptr = reinterpret_cast<char*>(t.data) + t.byte_offset;
        const bool is_cuda_device = (t.device.device_type == 2);
        const std::uint64_t itemsize = (static_cast<std::uint64_t>(t.dtype.bits) / 8) * static_cast<std::uint64_t>(t.dtype.lanes);
        std::uint64_t src_bytes = itemsize;
        for (int i = 0; i < t.ndim; ++i) src_bytes *= static_cast<std::uint64_t>(t.shape[i]);
        if (src_bytes != total_bytes) {
            throw std::runtime_error("load(): source size does not match the destination's size");
        }
        device.vk_copy_in(dst_external_ptr, dst_location, data_ptr, t.shape, t.strides, t.ndim, itemsize, is_cuda_device);
        return;
    }
    if (PyObject_CheckBuffer(source.ptr())) {
        pybind11::buffer buf = pybind11::reinterpret_borrow<pybind11::buffer>(source);
        pybind11::buffer_info info = buf.request();
        const std::uint64_t src_bytes = static_cast<std::uint64_t>(info.size) * static_cast<std::uint64_t>(info.itemsize);
        if (src_bytes != total_bytes) {
            throw std::runtime_error("load(): source size does not match the destination's size");
        }
        const std::int64_t total = static_cast<std::int64_t>(info.size);
        const std::int64_t stride1 = 1;
        device.vk_copy_in(dst_external_ptr, dst_location, info.ptr, &total, &stride1, 1, static_cast<std::uint64_t>(info.itemsize), false);
        return;
    }
    throw std::runtime_error("load(): source must be a DLPack-compatible object or a Python buffer object");
}

// Symmetric to copy_pyobject_into: copies `total_bytes` from
// `src_external_ptr` (`src_location`) into `target` (a DLPack-compatible
// or Python buffer-protocol object, already sized to receive them). Used
// by Buffer::save()/Tensor::save().
void copy_pyobject_from(Device& device, std::uint64_t src_external_ptr, MemoryLocation src_location,
    std::uint64_t total_bytes, const pybind11::object& target) {
    if (pybind11::hasattr(target, "__dlpack__")) {
        pybind11::object capsule = target.attr("__dlpack__")();
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
        if (!managed) {
            throw std::runtime_error("save(): invalid DLPack capsule (expected a \"dltensor\" capsule)");
        }
        const DLTensor& t = managed->dl_tensor;
        char* data_ptr = reinterpret_cast<char*>(t.data) + t.byte_offset;
        const bool is_cuda_device = (t.device.device_type == 2);
        const std::uint64_t itemsize = (static_cast<std::uint64_t>(t.dtype.bits) / 8) * static_cast<std::uint64_t>(t.dtype.lanes);
        std::uint64_t dst_bytes = itemsize;
        for (int i = 0; i < t.ndim; ++i) dst_bytes *= static_cast<std::uint64_t>(t.shape[i]);
        if (dst_bytes != total_bytes) {
            throw std::runtime_error("save(): target size does not match the source's size");
        }
        device.vk_copy_out(src_external_ptr, src_location, data_ptr, t.shape, t.strides, t.ndim, itemsize, is_cuda_device);
        return;
    }
    if (PyObject_CheckBuffer(target.ptr())) {
        pybind11::buffer buf = pybind11::reinterpret_borrow<pybind11::buffer>(target);
        pybind11::buffer_info info = buf.request();
        if (info.readonly) {
            throw std::runtime_error("save(): target buffer is read-only");
        }
        const std::uint64_t dst_bytes = static_cast<std::uint64_t>(info.size) * static_cast<std::uint64_t>(info.itemsize);
        if (dst_bytes != total_bytes) {
            throw std::runtime_error("save(): target size does not match the source's size");
        }
        const std::int64_t total = static_cast<std::int64_t>(info.size);
        const std::int64_t stride1 = 1;
        device.vk_copy_out(src_external_ptr, src_location, info.ptr, &total, &stride1, 1, static_cast<std::uint64_t>(info.itemsize), false);
        return;
    }
    throw std::runtime_error("save(): target must be a DLPack-compatible object or a writable Python buffer object");
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

    // Optional: only present in interop plugins that implement strided
    // DLPack<->contiguous-pointer copies (see Device::wrap). Null (not an
    // error) if the loaded plugin predates these, or has neither.
    [[nodiscard]] ExternalCopyFromDlpackFn copy_from_dlpack_fn() const noexcept {
        return copy_from_dlpack_;
    }
    [[nodiscard]] ExternalCopyToDlpackFn copy_to_dlpack_fn() const noexcept {
        return copy_to_dlpack_;
    }
    // Optional: only present in interop plugins that implement cross-API
    // semaphore import (see Device::vk_interop_semaphore). Null (not an
    // error) if the loaded plugin predates it.
    [[nodiscard]] ExternalImportSemaphoreFn try_import_semaphore_fn() const noexcept {
        return try_import_semaphore_;
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
            throw std::runtime_error("Failed to resolve module install directory for interop library");
            // If Python cannot resolve the module path, fall back to the loader search path.
            // printf("[ERROR] Module Install dir could not be resolved.");
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
            throw std::runtime_error("Library could not be opened for any of the possible folders.");
            // printf("[ERROR] Library could not be opened for any of the possible folders.");
            return;
        }

        auto symbol = load_symbol(handle_, "try_import_memory");
        if (!symbol) {
            throw std::runtime_error("Could not be loaded symbol try_import_memory");
            // printf("[ERROR] Could not be loaded symbol try_import_memory");
            unload();
            return;
        }

        try_import_memory_ = reinterpret_cast<ExternalImportMemoryFn>(symbol);

        // Optional symbols: absent in older/other plugins, not an error.
        if (auto copy_from = load_symbol(handle_, "copy_from_dlpack_to_ptr")) {
            copy_from_dlpack_ = reinterpret_cast<ExternalCopyFromDlpackFn>(copy_from);
        }
        if (auto copy_to = load_symbol(handle_, "copy_ptr_to_dlpack")) {
            copy_to_dlpack_ = reinterpret_cast<ExternalCopyToDlpackFn>(copy_to);
        }
        if (auto import_semaphore = load_symbol(handle_, "try_import_semaphore")) {
            try_import_semaphore_ = reinterpret_cast<ExternalImportSemaphoreFn>(import_semaphore);
        }
    }

    void unload() noexcept {
        if (handle_) {
            close_library(handle_);
            handle_ = nullptr;
        }
        try_import_memory_ = nullptr;
        copy_from_dlpack_ = nullptr;
        copy_to_dlpack_ = nullptr;
        try_import_semaphore_ = nullptr;
    }

    LibraryHandle handle_ = nullptr;
    ExternalImportMemoryFn try_import_memory_ = nullptr;
    ExternalCopyFromDlpackFn copy_from_dlpack_ = nullptr;
    ExternalCopyToDlpackFn copy_to_dlpack_ = nullptr;
    ExternalImportSemaphoreFn try_import_semaphore_ = nullptr;
};

std::shared_ptr<Device> Device::create_device(uint32_t device_index, bool enable_validation_layers) {
    auto dev = std::shared_ptr<Device>(new Device(device_index, enable_validation_layers));
    dev->vk_init(); // initializes all components of the device that requires a pointer to the Device object.
    return dev;
}

int scalar_type_size(Type type) {
    switch (type) {
        case Type::FLOAT16:
        case Type::INT16:
        case Type::UINT16:
            return 2;
        case Type::FLOAT32:
        case Type::INT32:
        case Type::UINT32:
            return 4;
        case Type::FLOAT64:
        case Type::INT64:
        case Type::UINT64:
            return 8;
        case Type::INT8:
        case Type::UINT8:
        case Type::BOOL:
            return 1;
        default:
            throw std::runtime_error("Unsupported scalar type");
    }
}

TypeTraits type_traits(Type type) {
    switch (type) {
        case Type::VEC2: return { Type::FLOAT32, 2, 1 };
        case Type::VEC3: return { Type::FLOAT32, 3, 1 };
        case Type::VEC4: return { Type::FLOAT32, 4, 1 };
        case Type::IVEC2: return { Type::INT32, 2, 1 };
        case Type::IVEC3: return { Type::INT32, 3, 1 };
        case Type::IVEC4: return { Type::INT32, 4, 1 };
        case Type::UVEC2: return { Type::UINT32, 2, 1 };
        case Type::UVEC3: return { Type::UINT32, 3, 1 };
        case Type::UVEC4: return { Type::UINT32, 4, 1 };
        case Type::BVEC2: return { Type::BOOL, 2, 1 };
        case Type::BVEC3: return { Type::BOOL, 3, 1 };
        case Type::BVEC4: return { Type::BOOL, 4, 1 };
        case Type::MAT2: return { Type::FLOAT32, 2, 2 };
        case Type::MAT2x3: return { Type::FLOAT32, 3, 2 };
        case Type::MAT2x4: return { Type::FLOAT32, 4, 2 };
        case Type::MAT3x2: return { Type::FLOAT32, 2, 3 };
        case Type::MAT3: return { Type::FLOAT32, 3, 3 };
        case Type::MAT3x4: return { Type::FLOAT32, 4, 3 };
        case Type::MAT4x2: return { Type::FLOAT32, 2, 4 };
        case Type::MAT4x3: return { Type::FLOAT32, 3, 4 };
        case Type::MAT4: return { Type::FLOAT32, 4, 4 };
        default: return { type, 1, 1 }; // plain scalar
    }
}

TypeDescriptor TypeDescriptor::single(Type type) {
    TypeDescriptor td;
    td.payload_ = SingleDesc{ type };
    return td;
}

TypeDescriptor TypeDescriptor::array_of(std::shared_ptr<TypeDescriptor> element_type, std::uint64_t count) {
    TypeDescriptor td;
    td.payload_ = ArrayDesc{ std::move(element_type), count };
    return td;
}

TypeDescriptor TypeDescriptor::struct_of(std::vector<StructField> fields) {
    TypeDescriptor td;
    td.payload_ = StructDesc{ std::move(fields) };
    return td;
}

TypeKind TypeDescriptor::kind() const noexcept {
    // Payload variant alternatives are declared in the same order as TypeKind's values.
    return static_cast<TypeKind>(payload_.index());
}

namespace {
    std::uint64_t round_up(std::uint64_t value, std::uint64_t align) {
        return align == 0 ? value : ((value + align - 1) / align) * align;
    }

    // Alignment of an ARRAY/MATRIX built from elements/columns with this
    // layout, under `rule` (a matrix is laid out as an array of columns).
    std::uint64_t compute_array_alignment(const std::shared_ptr<Layout>& element_layout, LayoutRule rule) {
        return rule == LayoutRule::Std140 ? round_up(element_layout->alignment, 16) : element_layout->alignment;
    }

    // Propagates `root` (the outermost Layout compute_layout() was called for)
    // into every LayoutField reachable from `layout`, however deeply nested,
    // so any field can resolve `array_index * root.aligned_size + offset` on
    // its own (see LayoutField::root). Safe to call redundantly on nested
    // layouts during construction: the outermost call always runs last and
    // overwrites everything to point at the true root.
    void assign_root(const std::shared_ptr<Layout>& layout, const std::weak_ptr<Layout>& root) {
        if (layout->kind == TypeKind::STRUCT) {
            for (auto& field : layout->fields) {
                field.root = root;
                assign_root(field.layout, root);
            }
        } else if (layout->element_layout) {
            assign_root(layout->element_layout, root);
        }
    }
}

std::shared_ptr<Layout> compute_layout(const TypeDescriptor& type, LayoutRule rule) {
    auto layout = std::make_shared<Layout>();
    std::visit([&](auto&& desc) {
        using T = std::decay_t<decltype(desc)>;
        if constexpr (std::is_same_v<T, SingleDesc>) {
            const TypeTraits traits = type_traits(desc.type);
            layout->kind = TypeKind::SINGLE;
            layout->type = desc.type;
            layout->component_type = traits.component_type;
            const std::uint64_t base = static_cast<std::uint64_t>(scalar_type_size(traits.component_type));
            if (traits.columns == 1 && traits.rows == 1) {
                // Plain scalar.
                layout->size = base;
                layout->alignment = base;
            } else if (traits.columns == 1) {
                // Vector: `rows` components of `component_type`.
                layout->size = static_cast<std::uint64_t>(traits.rows) * base;
                if (rule == LayoutRule::Scalar)
                    layout->alignment = base;
                else
                    layout->alignment = (traits.rows == 2) ? 2 * base : 4 * base;
            } else {
                // Matrix: `columns` columns, each a vector of `rows` components.
                TypeDescriptor column = TypeDescriptor::single(
                    traits.rows == 2 ? Type::VEC2 : traits.rows == 3 ? Type::VEC3 : Type::VEC4);
                auto column_layout = compute_layout(column, rule);
                layout->element_layout = column_layout;
                layout->stride = column_layout->aligned_size;
                layout->count = static_cast<std::uint64_t>(traits.columns);
                layout->size = layout->stride * layout->count;
                layout->alignment = compute_array_alignment(column_layout, rule);
            }
        } else if constexpr (std::is_same_v<T, ArrayDesc>) {
            auto element_layout = compute_layout(*desc.element_type, rule);
            layout->kind = TypeKind::ARRAY;
            layout->element_layout = element_layout;
            layout->stride = element_layout->aligned_size;
            layout->count = desc.count;
            layout->size = layout->stride * layout->count;
            layout->alignment = compute_array_alignment(element_layout, rule);
        } else if constexpr (std::is_same_v<T, StructDesc>) {
            std::uint64_t running_offset = 0;
            std::uint64_t struct_align = 1;
            std::vector<LayoutField> fields;
            fields.reserve(desc.fields.size());
            for (const auto& field : desc.fields) {
                auto field_layout = compute_layout(*field.type, rule);
                running_offset = round_up(running_offset, field_layout->alignment);
                fields.push_back(LayoutField{ field.name, running_offset, field_layout });
                running_offset += field_layout->size;
                struct_align = std::max(struct_align, field_layout->alignment);
            }
            const std::uint64_t align = (rule == LayoutRule::Std140) ? round_up(struct_align, 16) : struct_align;
            layout->kind = TypeKind::STRUCT;
            layout->fields = std::move(fields);
            layout->alignment = align;
            layout->size = round_up(running_offset, align);
        }
    }, type.payload());
    // Size this layout would occupy as one element of an array of itself
    // (see Layout::aligned_size).
    layout->aligned_size = (rule == LayoutRule::Std140)
        ? round_up(std::max(layout->size, layout->alignment), 16)
        : round_up(layout->size, layout->alignment);
    assign_root(layout, layout);
    return layout;
}

const LayoutField& Layout::field(const std::string& name) const {
    if (kind != TypeKind::STRUCT) {
        throw std::runtime_error("Layout::field: this layout is not a struct");
    }
    for (const auto& f : fields) {
        if (f.name == name) return f;
    }
    throw std::runtime_error("Layout::field: no field named '" + name + "'");
}

const std::shared_ptr<Layout>& Layouts::index16() {
    static const std::shared_ptr<Layout> layout = compute_layout(TypeDescriptor::single(Type::UINT16), LayoutRule::Scalar);
    return layout;
}

const std::shared_ptr<Layout>& Layouts::index32() {
    static const std::shared_ptr<Layout> layout = compute_layout(TypeDescriptor::single(Type::UINT32), LayoutRule::Scalar);
    return layout;
}

const std::shared_ptr<Layout>& Layouts::position() {
    static const std::shared_ptr<Layout> layout = compute_layout(TypeDescriptor::single(Type::VEC3), LayoutRule::Scalar);
    return layout;
}

const std::shared_ptr<Layout>& Layouts::aabb() {
    // Matches VkAabbPositionsKHR (6 tightly-packed floats) byte-for-byte
    // under LayoutRule::Scalar.
    static const std::shared_ptr<Layout> layout = compute_layout(TypeDescriptor::struct_of({
        { "min_x", std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::FLOAT32)) },
        { "min_y", std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::FLOAT32)) },
        { "min_z", std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::FLOAT32)) },
        { "max_x", std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::FLOAT32)) },
        { "max_y", std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::FLOAT32)) },
        { "max_z", std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::FLOAT32)) },
    }), LayoutRule::Scalar);
    return layout;
}

const std::shared_ptr<Layout>& Layouts::instance() {
    // Matches VkAccelerationStructureInstanceKHR byte-for-byte under
    // LayoutRule::Scalar: "transform" is VkTransformMatrixKHR (a plain,
    // tightly-packed float[3][4], i.e. 3 row-major vec4 rows -- NOT one of
    // this project's own column-major matrix Types, which wouldn't match
    // this layout); the two bitfield-packed uint32 words and the trailing
    // uint64 BLAS reference follow with no padding (56 already being
    // 8-aligned), for a total of exactly 64 bytes, matching
    // sizeof(VkAccelerationStructureInstanceKHR).
    static const std::shared_ptr<Layout> layout = compute_layout(TypeDescriptor::struct_of({
        { "transform", std::make_shared<TypeDescriptor>(TypeDescriptor::array_of(
            std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::VEC4)), 3)) },
        { "instance_custom_index_and_mask", std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::UINT32)) },
        { "instance_shader_binding_table_record_offset_and_flags", std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::UINT32)) },
        { "acceleration_structure_reference", std::make_shared<TypeDescriptor>(TypeDescriptor::single(Type::UINT64)) },
    }), LayoutRule::Scalar);
    return layout;
}

Type format_scalar_type(Format format)
{
    vk::Format vk_format = (vk::Format) format;
    switch (vk_format)
    {
        // =======================
        // 8-bit UNSIGNED INT
        // =======================
        case vk::Format::eR8Uint:
        case vk::Format::eR8G8Uint:
        case vk::Format::eR8G8B8Uint:
        case vk::Format::eR8G8B8A8Uint:
            return Type::UINT8;

        // =======================
        // 8-bit SIGNED INT
        // =======================
        case vk::Format::eR8Sint:
        case vk::Format::eR8G8Sint:
        case vk::Format::eR8G8B8Sint:
        case vk::Format::eR8G8B8A8Sint:
            return Type::INT8;

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
            return Type::FLOAT32; // normalized -> conceptual float

        // =======================
        // 16-bit UNSIGNED INT
        // =======================
        case vk::Format::eR16Uint:
        case vk::Format::eR16G16Uint:
        case vk::Format::eR16G16B16A16Uint:
            return Type::UINT16;

        // =======================
        // 16-bit SIGNED INT
        // =======================
        case vk::Format::eR16Sint:
        case vk::Format::eR16G16Sint:
        case vk::Format::eR16G16B16A16Sint:
            return Type::INT16;

        // =======================
        // 16-bit FLOAT
        // =======================
        case vk::Format::eR16Sfloat:
        case vk::Format::eR16G16Sfloat:
        case vk::Format::eR16G16B16A16Sfloat:
            return Type::FLOAT16;

        // =======================
        // 32-bit UNSIGNED INT
        // =======================
        case vk::Format::eR32Uint:
        case vk::Format::eR32G32Uint:
        case vk::Format::eR32G32B32Uint:
        case vk::Format::eR32G32B32A32Uint:
            return Type::UINT32;

        // =======================
        // 32-bit SIGNED INT
        // =======================
        case vk::Format::eR32Sint:
        case vk::Format::eR32G32Sint:
        case vk::Format::eR32G32B32Sint:
        case vk::Format::eR32G32B32A32Sint:
            return Type::INT32;

        // =======================
        // 32-bit FLOAT
        // =======================
        case vk::Format::eR32Sfloat:
        case vk::Format::eR32G32Sfloat:
        case vk::Format::eR32G32B32Sfloat:
        case vk::Format::eR32G32B32A32Sfloat:
            return Type::FLOAT32;

        // =======================
        // 64-bit FLOAT
        // =======================
        case vk::Format::eR64Sfloat:
        case vk::Format::eR64G64Sfloat:
        case vk::Format::eR64G64B64Sfloat:
        case vk::Format::eR64G64B64A64Sfloat:
            return Type::FLOAT64;

        // =======================
        // Depth formats (special case)
        // =======================
        case vk::Format::eD16Unorm:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32Sfloat:
            return Type::FLOAT32;

        // =======================
        // Stencil (special, integer)
        // =======================
        case vk::Format::eS8Uint:
            return Type::UINT8;

        default:
            return Type::UNDEFINED;
    }
}

vk_ResourceData::vk_ResourceData(
    std::shared_ptr<Device> device, const std::shared_ptr<MemorySlice>& memory,
    vk::Image image, vk::ImageCreateInfo image_info, vk::DeviceMemory dedicated_image_memory,
    bool owns_image)
    : device_(std::move(device)), memory_(memory), image_(image), image_info_(image_info),
      resource_type_(image ? ResourceType::IMAGE : ResourceType::BUFFER),
      dedicated_image_memory_(dedicated_image_memory), owns_image_(owns_image) {
    buffer_ = memory->page_buffer();
}

vk_ResourceData::~vk_ResourceData() noexcept {
    dispose();
}

void vk_ResourceData::dispose() {
    if (!this->device_)
        return;
    auto device = this->device_->logical_device();
    if (this->image_ && this->owns_image_) {
        device.destroyImage(this->image_);
    }
    if (this->dedicated_image_memory_) {
        device.freeMemory(this->dedicated_image_memory_);
    }
    auto dev = this->device_;
    this->device_ = nullptr;
    this->memory_ = nullptr;
    this->buffer_ = nullptr;
    this->image_ = nullptr;
    this->dedicated_image_memory_ = nullptr;
}

std::uint64_t vk_ResourceData::device_ptr() const {
    if (memory_) {
        return memory_->device_ptr();
    }
    return 0;
}

std::uint64_t vk_ResourceData::external_ptr() const {
    if (memory_) {
        return memory_->external_ptr();
    }
    return 0;
}

bool vk_ResourceData::is_cpu() const noexcept { return memory_ && memory_->host_visible(); }

Resource::Resource(std::shared_ptr<vk_ResourceData> resource_data, ResourceSlice view_slice) {
    data_ = std::move(resource_data);
    slice_ = view_slice;
}

Resource::~Resource() noexcept {
    dispose();
}

std::uint32_t Resource::device_index() const noexcept {
    return data_->device()->device_index();
}

vk::BufferView Buffer::get_view() {
    if (!buffer_view_) {
        vk::BufferViewCreateInfo info{};
        info.buffer = data_->get_buffer();
        info.format = (vk::Format)format_;
        info.offset = data_->get_memory()->offset() + slice_.buffer.offset;
        info.range = slice_.buffer.size;
        auto device = this->data_->device()->logical_device();
        buffer_view_ = device.createBufferView(info);
    }
    return buffer_view_;
}

vk::ImageView Image::get_view() {
    if (!image_view_) {
        auto image_info = data_->get_image_info();
        const bool arrayed = slice_.image.array_count > 1;
        vk::ImageViewType view_type;
        switch (image_info.imageType) {
            case vk::ImageType::e1D: view_type = arrayed ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D; break;
            case vk::ImageType::e3D: view_type = vk::ImageViewType::e3D; break;
            default: view_type = arrayed ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D; break;
        }
        vk::ImageViewCreateInfo info{};
        info.image = data_->get_image();
        info.viewType = view_type;
        info.format = (vk::Format)slice_.image.format;
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

SubmittedTask::SubmittedTask(std::weak_ptr<vk_Engine> engine, std::uint64_t submission_id) {
    submission_id_ = submission_id;
    engine_ = std::move(engine);
}

void SubmittedTask::wait() {
    auto e = engine_.lock();
    if (!e) return;
    e->vk_wait_for(submission_id_);
}

bool SubmittedTask::is_complete() {
    auto e = engine_.lock();
    if (!e) return true;
    return submission_id_ <= e->vk_check_completion();
}

void SubmittedTask::vk_notify_completion() {
    for (auto& cb : submitted_) {
        cb->vk_command_buffer()->notify_executed();
    }
    submitted_.clear();
}

vk_CommandBuffer::vk_CommandBuffer(vk::CommandBuffer command_buffer) noexcept {
    this->command_buffer = command_buffer;
    this->state = vk_CommandBufferInternalState::CREATED;
}

void vk_CommandBuffer::begin() {
    assert(state == vk_CommandBufferInternalState::CREATED || state == vk_CommandBufferInternalState::EXECUTABLE);
    if (state == vk_CommandBufferInternalState::EXECUTABLE) {
        command_buffer.reset({});
    }
    command_buffer.begin(vk::CommandBufferBeginInfo{});
    state = vk_CommandBufferInternalState::RECORDING;
}

void vk_CommandBuffer::end() {
    assert(state == vk_CommandBufferInternalState::RECORDING);
    command_buffer.end();
    state = vk_CommandBufferInternalState::EXECUTABLE;
}

void vk_CommandBuffer::notify_submitted() {
    assert(state == vk_CommandBufferInternalState::EXECUTABLE);
    state = vk_CommandBufferInternalState::SUBMITTED;
}

void vk_CommandBuffer::notify_executed() {
    assert(state == vk_CommandBufferInternalState::SUBMITTED);
    state = vk_CommandBufferInternalState::EXECUTABLE;
}

void CommandBuffer::close() {
    if (render_pass_active_) {
        command_buffer_->command_buffer.endRenderPass();
        render_pass_active_ = false;
    }
    command_buffer_->end();
}

void CommandBuffer::transfer(const std::shared_ptr<Buffer>& source, const std::shared_ptr<Buffer>& destination) {
    if (is_closed()) {
        throw std::runtime_error("Cannot record a transfer into a closed command buffer");
    }
    std::uint64_t bytes = source->size();
    assert(destination->size() == bytes);
    vk::BufferCopy region(source->vk_buffer_offset(), destination->vk_buffer_offset(), static_cast<vk::DeviceSize>(bytes));
    command_buffer_->command_buffer.copyBuffer(source->vk_buffer(), destination->vk_buffer(), region);
}

namespace {

// Builds one vk::BufferImageCopy per mip level `image` covers (tightly
// packed: each mip's region immediately follows the previous one's bytes
// in the buffer, no row/slice padding), covering every array layer `image`
// covers in a single region per mip. Shared by
// Device::create_staging(image) (sizing only, via the returned total) and
// CommandBuffer::transfer(Image<->Buffer) (the actual copy regions).
struct ImageCopyPlan {
    std::vector<vk::BufferImageCopy> regions;
    std::uint64_t total_bytes = 0;
};

ImageCopyPlan plan_image_copy(const Image& image, std::uint64_t buffer_offset) {
    const vk::ImageCreateInfo info = image.vk_image_info();
    const vk::ImageAspectFlags aspect = is_depth_format(image.format())
        ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
    const std::uint64_t texel_size = static_cast<std::uint64_t>(formatSize(image.format()));
    const std::uint64_t layer_count = static_cast<std::uint64_t>(image.array_count());

    ImageCopyPlan plan;
    plan.regions.reserve(static_cast<std::size_t>(image.mip_count()));
    std::uint64_t offset = buffer_offset;
    for (int m = 0; m < image.mip_count(); ++m) {
        const int mip = image.vk_mip_start() + m;
        const std::uint32_t w = std::max<std::uint32_t>(1, info.extent.width >> mip);
        const std::uint32_t h = std::max<std::uint32_t>(1, info.extent.height >> mip);
        const std::uint32_t d = std::max<std::uint32_t>(1, info.extent.depth >> mip);

        vk::BufferImageCopy region{};
        region.bufferOffset = static_cast<vk::DeviceSize>(offset);
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = aspect;
        region.imageSubresource.mipLevel = static_cast<std::uint32_t>(mip);
        region.imageSubresource.baseArrayLayer = static_cast<std::uint32_t>(image.vk_array_start());
        region.imageSubresource.layerCount = static_cast<std::uint32_t>(image.array_count());
        region.imageOffset = vk::Offset3D{0, 0, 0};
        region.imageExtent = vk::Extent3D{w, h, d};
        plan.regions.push_back(region);

        offset += static_cast<std::uint64_t>(w) * h * d * texel_size * layer_count;
    }
    plan.total_bytes = offset - buffer_offset;
    return plan;
}

} // namespace

// Total byte size of `image` (every mip level/array layer it covers,
// tightly packed with no row/slice padding) -- the size
// Device::create_staging(image) allocates and CommandBuffer::transfer
// (Image<->Buffer) reads/writes.
std::uint64_t vk_image_byte_size(const Image& image) {
    return plan_image_copy(image, 0).total_bytes;
}

void CommandBuffer::transfer(const std::shared_ptr<Image>& source, const std::shared_ptr<Buffer>& destination) {
    if (is_closed()) {
        throw std::runtime_error("Cannot record a transfer into a closed command buffer");
    }
    const ImageCopyPlan plan = plan_image_copy(*source, destination->vk_buffer_offset());
    if (destination->size() != plan.total_bytes) {
        throw std::runtime_error("CommandBuffer::transfer: destination buffer size does not match the source image's byte size");
    }
    command_buffer_->command_buffer.copyImageToBuffer(
        source->vk_image(), vk::ImageLayout::eGeneral, destination->vk_buffer(), plan.regions);
    bound_images_.push_back(source);
    bound_vertex_index_buffers_.push_back(destination);
}

void CommandBuffer::transfer(const std::shared_ptr<Buffer>& source, const std::shared_ptr<Image>& destination) {
    if (is_closed()) {
        throw std::runtime_error("Cannot record a transfer into a closed command buffer");
    }
    const ImageCopyPlan plan = plan_image_copy(*destination, source->vk_buffer_offset());
    if (source->size() != plan.total_bytes) {
        throw std::runtime_error("CommandBuffer::transfer: source buffer size does not match the destination image's byte size");
    }
    command_buffer_->command_buffer.copyBufferToImage(
        source->vk_buffer(), destination->vk_image(), vk::ImageLayout::eGeneral, plan.regions);
    bound_images_.push_back(destination);
    bound_vertex_index_buffers_.push_back(source);
}

void CommandBuffer::set_pipeline(const std::shared_ptr<Pipeline>& pipeline) {
    if (is_closed()) {
        throw std::runtime_error("Cannot set a pipeline on a closed command buffer");
    }
    auto vk_pipeline = pipeline->vk_pipeline();
    if (!vk_pipeline->is_closed()) {
        throw std::runtime_error("CommandBuffer::set_pipeline: pipeline must be closed first");
    }
    if (vk_pipeline->type() == PipelineType::RASTERIZATION && !render_pass_active_) {
        throw std::runtime_error("CommandBuffer::set_pipeline: a RASTERIZATION pipeline requires an active render pass (call set_framebuffer first)");
    }
    const vk::PipelineBindPoint bind_point = vk_pipeline->type() == PipelineType::COMPUTE
        ? vk::PipelineBindPoint::eCompute
        : vk::PipelineBindPoint::eGraphics;
    command_buffer_->command_buffer.bindPipeline(bind_point, vk_pipeline->vk_handle());
    // Appended, not overwritten: a command buffer may bind several
    // pipelines over the course of one recording, and every one of them
    // must stay alive (its raw vk::Pipeline handle is already baked into
    // this command buffer) for as long as this command buffer is.
    bound_pipelines_.push_back(pipeline);
}

void CommandBuffer::bind(int initial_set, const std::vector<std::shared_ptr<DescriptorSet>>& descriptor_sets) {
    if (is_closed()) {
        throw std::runtime_error("Cannot bind descriptor sets on a closed command buffer");
    }
    if (bound_pipelines_.empty()) {
        throw std::runtime_error("CommandBuffer::bind: no pipeline is currently bound (call set_pipeline first)");
    }
    if (descriptor_sets.empty()) return;
    auto vk_pipeline = bound_pipelines_.back()->vk_pipeline();
    const vk::PipelineBindPoint bind_point = vk_pipeline->type() == PipelineType::COMPUTE
        ? vk::PipelineBindPoint::eCompute
        : vk::PipelineBindPoint::eGraphics;
    std::vector<vk::DescriptorSet> sets;
    sets.reserve(descriptor_sets.size());
    for (const auto& ds : descriptor_sets) {
        sets.push_back(ds->vk_descriptor_set()->vk_handle());
    }
    command_buffer_->command_buffer.bindDescriptorSets(
        bind_point, vk_pipeline->vk_pipeline_layout(), static_cast<std::uint32_t>(initial_set), sets, {});
    // Appended, not overwritten: see bound_pipelines_ above -- the same
    // applies to every descriptor set ever bound during this recording.
    bound_descriptor_sets_.insert(bound_descriptor_sets_.end(), descriptor_sets.begin(), descriptor_sets.end());
}

void CommandBuffer::dispatch_threads(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    if (is_closed()) {
        throw std::runtime_error("Cannot dispatch on a closed command buffer");
    }
    if (bound_pipelines_.empty()) {
        throw std::runtime_error("CommandBuffer::dispatch_threads: no pipeline is currently bound (call set_pipeline first)");
    }
    auto vk_pipeline = bound_pipelines_.back()->vk_pipeline();
    if (vk_pipeline->type() != PipelineType::COMPUTE) {
        throw std::runtime_error("CommandBuffer::dispatch_threads: bound pipeline is not a COMPUTE pipeline");
    }
    auto group_count = [](std::uint32_t threads, std::uint32_t local_size) {
        return (threads + local_size - 1) / local_size;
    };
    command_buffer_->command_buffer.dispatch(
        group_count(x, vk_pipeline->vk_local_size_x()),
        group_count(y, vk_pipeline->vk_local_size_y()),
        group_count(z, vk_pipeline->vk_local_size_z()));
}

void CommandBuffer::set_framebuffer(const std::shared_ptr<Framebuffer>& framebuffer) {
    if (is_closed()) {
        throw std::runtime_error("Cannot set a framebuffer on a closed command buffer");
    }
    if (render_pass_active_) {
        command_buffer_->command_buffer.endRenderPass();
        render_pass_active_ = false;
    }

    // Every attachment's render pass loadOp is eClear (see vk_Pipeline::vk_close),
    // so a clear value must be provided for each; a plain opaque black is
    // as reasonable a default as any, since there's no per-attachment
    // clear-color API yet. The depth attachment, if any, is always last
    // (see vk_Pipeline::vk_close()'s comment) and cleared to 1.0 (the
    // "farthest" depth) instead.
    std::vector<vk::ClearValue> clear_values(framebuffer->vk_attachment_count());
    for (auto& cv : clear_values) {
        cv.color.float32[0] = 0.0f;
        cv.color.float32[1] = 0.0f;
        cv.color.float32[2] = 0.0f;
        cv.color.float32[3] = 1.0f;
    }
    if (framebuffer->vk_has_depth()) {
        clear_values.back().depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };
    }

    vk::RenderPassBeginInfo info{};
    info.renderPass = framebuffer->vk_render_pass();
    info.framebuffer = framebuffer->vk_framebuffer();
    info.renderArea.offset = vk::Offset2D{ 0, 0 };
    info.renderArea.extent = vk::Extent2D{ framebuffer->width(), framebuffer->height() };
    info.clearValueCount = static_cast<std::uint32_t>(clear_values.size());
    info.pClearValues = clear_values.data();

    command_buffer_->command_buffer.beginRenderPass(info, vk::SubpassContents::eInline);
    render_pass_active_ = true;
    active_framebuffer_has_depth_ = framebuffer->vk_has_depth();
    // Appended, not overwritten: see bound_pipelines_ above -- the same
    // applies to every framebuffer ever set during this recording.
    bound_framebuffers_.push_back(framebuffer);
}

void CommandBuffer::set_viewport(float x, float y, float width, float height) {
    if (is_closed()) {
        throw std::runtime_error("Cannot set a viewport on a closed command buffer");
    }
    vk::Viewport viewport{ x, y, width, height, 0.0f, 1.0f };
    vk::Rect2D scissor{
        vk::Offset2D{ static_cast<std::int32_t>(x), static_cast<std::int32_t>(y) },
        vk::Extent2D{ static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height) }
    };
    command_buffer_->command_buffer.setViewport(0, viewport);
    command_buffer_->command_buffer.setScissor(0, scissor);
}

namespace {
vk::CullModeFlags to_vk_cull_mode(CullMode mode) {
    switch (mode) {
        case CullMode::NONE: return vk::CullModeFlagBits::eNone;
        case CullMode::FRONT: return vk::CullModeFlagBits::eFront;
        case CullMode::BACK: return vk::CullModeFlagBits::eBack;
        case CullMode::FRONT_AND_BACK: return vk::CullModeFlagBits::eFrontAndBack;
    }
    throw std::runtime_error("Invalid CullMode");
}
} // namespace

void CommandBuffer::set_cull_mode(CullMode mode) {
    if (is_closed()) {
        throw std::runtime_error("Cannot set cull mode on a closed command buffer");
    }
    if (!device_->vk_extended_dynamic_state_supported()) {
        throw std::runtime_error("CommandBuffer::set_cull_mode: requires Vulkan 1.3 (extended dynamic state)");
    }
    command_buffer_->command_buffer.setCullMode(to_vk_cull_mode(mode));
}

void CommandBuffer::set_front_face(FrontFace front_face) {
    if (is_closed()) {
        throw std::runtime_error("Cannot set front face on a closed command buffer");
    }
    if (!device_->vk_extended_dynamic_state_supported()) {
        throw std::runtime_error("CommandBuffer::set_front_face: requires Vulkan 1.3 (extended dynamic state)");
    }
    command_buffer_->command_buffer.setFrontFace(
        front_face == FrontFace::CLOCKWISE ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);
}

void CommandBuffer::set_depth_test(bool enable, bool write_enable, CompareOp compare_op) {
    if (is_closed()) {
        throw std::runtime_error("Cannot set depth test state on a closed command buffer");
    }
    if (!device_->vk_extended_dynamic_state_supported()) {
        throw std::runtime_error("CommandBuffer::set_depth_test: requires Vulkan 1.3 (extended dynamic state)");
    }
    if (!render_pass_active_) {
        throw std::runtime_error("CommandBuffer::set_depth_test: no active render pass (call set_framebuffer first)");
    }
    if (enable && !active_framebuffer_has_depth_) {
        throw std::runtime_error("CommandBuffer::set_depth_test: the active framebuffer has no depth attachment (see Pipeline::attach_depth)");
    }
    command_buffer_->command_buffer.setDepthTestEnable(enable ? VK_TRUE : VK_FALSE);
    command_buffer_->command_buffer.setDepthWriteEnable(write_enable ? VK_TRUE : VK_FALSE);
    command_buffer_->command_buffer.setDepthCompareOp(static_cast<vk::CompareOp>(compare_op));
}

void CommandBuffer::blit_image(const std::shared_ptr<Image>& src, const std::shared_ptr<Image>& dst, Filter filter) {
    if (is_closed()) {
        throw std::runtime_error("Cannot blit on a closed command buffer");
    }
    if (render_pass_active_) {
        throw std::runtime_error("CommandBuffer::blit_image: cannot blit while a render pass is active (set_framebuffer with a different framebuffer, or close(), ends it)");
    }
    const vk::Extent3D src_extent = src->vk_image_info().extent;
    const vk::Extent3D dst_extent = dst->vk_image_info().extent;

    vk::ImageBlit region{};
    region.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.srcSubresource.mipLevel = static_cast<std::uint32_t>(src->vk_mip_start());
    region.srcSubresource.baseArrayLayer = static_cast<std::uint32_t>(src->vk_array_start());
    region.srcSubresource.layerCount = static_cast<std::uint32_t>(src->array_count());
    region.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
    region.srcOffsets[1] = vk::Offset3D{
        static_cast<std::int32_t>(src_extent.width), static_cast<std::int32_t>(src_extent.height), static_cast<std::int32_t>(src_extent.depth) };

    region.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.dstSubresource.mipLevel = static_cast<std::uint32_t>(dst->vk_mip_start());
    region.dstSubresource.baseArrayLayer = static_cast<std::uint32_t>(dst->vk_array_start());
    region.dstSubresource.layerCount = static_cast<std::uint32_t>(dst->array_count());
    region.dstOffsets[0] = vk::Offset3D{ 0, 0, 0 };
    region.dstOffsets[1] = vk::Offset3D{
        static_cast<std::int32_t>(dst_extent.width), static_cast<std::int32_t>(dst_extent.height), static_cast<std::int32_t>(dst_extent.depth) };

    command_buffer_->command_buffer.blitImage(
        src->vk_image(), vk::ImageLayout::eGeneral,
        dst->vk_image(), vk::ImageLayout::eGeneral,
        region, static_cast<vk::Filter>(filter));

    bound_images_.push_back(src);
    bound_images_.push_back(dst);
}

void CommandBuffer::bind_vertices(int binding, const std::shared_ptr<Buffer>& vertex_buffer) {
    if (is_closed()) {
        throw std::runtime_error("Cannot bind vertices on a closed command buffer");
    }
    vk::Buffer buf = vertex_buffer->vk_buffer();
    vk::DeviceSize offset = vertex_buffer->vk_buffer_offset();
    command_buffer_->command_buffer.bindVertexBuffers(static_cast<std::uint32_t>(binding), 1, &buf, &offset);
    bound_vertex_index_buffers_.push_back(vertex_buffer);
}

void CommandBuffer::bind_indices(const std::shared_ptr<Buffer>& index_buffer) {
    if (is_closed()) {
        throw std::runtime_error("Cannot bind indices on a closed command buffer");
    }
    const Layout& layout = *index_buffer->element_layout();
    vk::IndexType index_type;
    if (layout.kind != TypeKind::STRUCT && resolve_component_type(layout) == Type::UINT16) {
        index_type = vk::IndexType::eUint16;
    } else if (layout.kind != TypeKind::STRUCT && resolve_component_type(layout) == Type::UINT32) {
        index_type = vk::IndexType::eUint32;
    } else {
        throw std::runtime_error("CommandBuffer::bind_indices: index buffer's element_layout must be a scalar UINT16 or UINT32");
    }
    command_buffer_->command_buffer.bindIndexBuffer(index_buffer->vk_buffer(), index_buffer->vk_buffer_offset(), index_type);
    bound_vertex_index_buffers_.push_back(index_buffer);
}

void CommandBuffer::dispatch_primitives(std::uint32_t vertices, std::uint32_t vertex_start) {
    if (is_closed()) {
        throw std::runtime_error("Cannot dispatch on a closed command buffer");
    }
    if (bound_pipelines_.empty()) {
        throw std::runtime_error("CommandBuffer::dispatch_primitives: no pipeline is currently bound (call set_pipeline first)");
    }
    if (bound_pipelines_.back()->vk_pipeline()->type() != PipelineType::RASTERIZATION) {
        throw std::runtime_error("CommandBuffer::dispatch_primitives: bound pipeline is not a RASTERIZATION pipeline");
    }
    if (!render_pass_active_) {
        throw std::runtime_error("CommandBuffer::dispatch_primitives: no active render pass (call set_framebuffer first)");
    }
    command_buffer_->command_buffer.draw(vertices, 1, vertex_start, 0);
}

void CommandBuffer::dispatch_indexed_primitives(std::uint32_t indices, std::uint32_t index_start, std::int32_t vertex_offset) {
    if (is_closed()) {
        throw std::runtime_error("Cannot dispatch on a closed command buffer");
    }
    if (bound_pipelines_.empty()) {
        throw std::runtime_error("CommandBuffer::dispatch_indexed_primitives: no pipeline is currently bound (call set_pipeline first)");
    }
    if (bound_pipelines_.back()->vk_pipeline()->type() != PipelineType::RASTERIZATION) {
        throw std::runtime_error("CommandBuffer::dispatch_indexed_primitives: bound pipeline is not a RASTERIZATION pipeline");
    }
    if (!render_pass_active_) {
        throw std::runtime_error("CommandBuffer::dispatch_indexed_primitives: no active render pass (call set_framebuffer first)");
    }
    command_buffer_->command_buffer.drawIndexed(indices, 1, index_start, vertex_offset, 0);
}

void CommandBuffer::release() {
    if (device_ == nullptr)
        return;
    if (command_buffer_->state == vk_CommandBufferInternalState::SUBMITTED) {
        throw std::runtime_error("Command buffer can not be released while submitted. Explicitly use wait before releasing.");
    }
    engine_->vk_release_command_buffer(command_buffer_);
    command_buffer_.reset();
    device_.reset();
    engine_.reset();
    bound_pipelines_.clear();
    bound_descriptor_sets_.clear();
    bound_framebuffers_.clear();
    bound_vertex_index_buffers_.clear();
    bound_images_.clear();
    render_pass_active_ = false;
}

std::uint32_t CommandBuffer::device_index() const {
    if (is_released()) throw std::runtime_error("CommandBuffer: already released");
    return device_->device_index();
}

EngineType CommandBuffer::engine_type() const {
    if (is_released()) throw std::runtime_error("CommandBuffer: already released");
    return engine_->engine_type();
}

std::uint32_t CommandBuffer::engine_index() const {
    if (is_released()) throw std::runtime_error("CommandBuffer: already released");
    return engine_->engine_index();
}

void vk_Engine::dispose() noexcept {
    auto d = device_.lock();
    vk::Device dev = (d && !d->is_disposed()) ? d->logical_device() : vk::Device{};
    vk_dispose_with(dev);
}

void vk_Engine::vk_dispose_with(vk::Device dev) noexcept {
    if (!queue_) return; // already disposed.
    wait();
    if (dev) {
        dev.destroySemaphore(timeline_semaphore_);
    }
    reusable_command_buffers_.clear();
    queue_ = nullptr;
    command_pool_ = nullptr; // not owned here
    device_.reset();
    timeline_semaphore_ = nullptr;
}

std::shared_ptr<vk_CommandBuffer> vk_Engine::create_command_buffer() {
    std::shared_ptr<vk_CommandBuffer> command_buffer;
    if (!reusable_command_buffers_.empty()) {
        command_buffer = reusable_command_buffers_.back();
        reusable_command_buffers_.pop_back();
    }
    else {
        auto dev = device_.lock();
        if (dev && !dev->is_disposed()) {
            vk::CommandBufferAllocateInfo alloc_info{};
            alloc_info.commandPool = command_pool_;
            alloc_info.level = vk::CommandBufferLevel::ePrimary;
            alloc_info.commandBufferCount = 1;
            auto command_buffers = dev->logical_device().allocateCommandBuffers(alloc_info);
            command_buffers_.push_back(command_buffers[0]);
            command_buffer = std::make_shared<vk_CommandBuffer>(command_buffers[0]);
        }
        else {
            throw std::runtime_error("Device is disposed.");
        }
    }
    command_buffer->begin();
    return command_buffer;
}

void vk_Engine::release_command_buffer(std::shared_ptr<vk_CommandBuffer> command_buffer) {
    if (command_buffer->state == vk_CommandBufferInternalState::RECORDING)
        command_buffer->end(); // close unfinished command buffer first before reset
    assert(command_buffer->state == vk_CommandBufferInternalState::EXECUTABLE);
    reusable_command_buffers_.push_back(command_buffer);
}

std::shared_ptr<SubmittedTask> vk_Engine::submit(std::uint32_t count, vk::CommandBuffer* command_buffers) {
    // submit the command buffers in command_buffers to the queue
    // map command_buffers to a vec of vk::CommandBuffer
    if (count == 0) { // important silent behaviour for dynamic submission cases. Empty submission will preserve submission id from previous task.
        return std::make_shared<SubmittedTask>(shared_from_this(), current_submission_id_);
    }
    current_submission_id_ ++; // if there is something to send to the GPU, there is something to wait for
    auto task = std::make_shared<SubmittedTask>(shared_from_this(), current_submission_id_);

    // In addition to this engine's own private timeline semaphore, also
    // signal the device-wide, exported interop_semaphore_ (if available):
    // this is what lets a later CUDA-side memcpy (see
    // MemoryManager::vk_copy_to_dlpack/vk_copy_from_dlpack) issue a real
    // GPU-side wait for this exact submission, rather than relying solely
    // on a host-side wait for cross-API memory visibility.
    std::array<vk::Semaphore, 2> signal_semaphores{ timeline_semaphore_, nullptr };
    std::array<std::uint64_t, 2> signal_values{ current_submission_id_, 0 };
    std::uint32_t signal_count = 1;
    if (auto device = device_.lock()) {
        if (vk::Semaphore interop_semaphore = device->vk_interop_semaphore()) {
            signal_semaphores[1] = interop_semaphore;
            signal_values[1] = device->vk_bump_interop_semaphore_value();
            signal_count = 2;
        }
    }

    vk::TimelineSemaphoreSubmitInfo timeline_submit_info{
        0,
        nullptr,
        signal_count,
        signal_values.data(),
    };
    vk::SubmitInfo submit_info{
        0,
        nullptr,
        nullptr,
        count,
        command_buffers,
        signal_count,
        signal_semaphores.data(),
    };
    submit_info.setPNext(&timeline_submit_info);
    queue_.submit(submit_info);
    pending_submitted_tasks_.push_back(task);
    return task;
}

void vk_Engine::wait() {
    queue_.waitIdle(); // waits for all submitted task to be completed
    vk_collect_all_completed(std::numeric_limits<std::uint64_t>::max()); // clear all pending tasks
}

void vk_Engine::vk_wait_for(std::uint64_t submission_id) {
    // gpu waits for signal submission_id
    if (auto d = device_.lock()) {
        if (d->logical_device().waitSemaphores(vk::SemaphoreWaitInfo{
            vk::SemaphoreWaitFlagBits::eAny,
            1,
            &timeline_semaphore_,
            &submission_id
        }, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess)
            throw std::runtime_error("vk_wait_for failed while waitSemaphores.");
    }
    // all tasks with submission_id <= submission_id will be notified and removed from pending
    vk_collect_all_completed(submission_id);
}

std::uint64_t vk_Engine::vk_check_completion() {
    std::uint64_t gpu_timeline_value = std::numeric_limits<std::uint64_t>::max();
    if (auto d = device_.lock()) {
        if (d->logical_device().getSemaphoreCounterValue(timeline_semaphore_, &gpu_timeline_value) != vk::Result::eSuccess)
            throw std::runtime_error("Failed to get semaphore counter value");
    }
    // all tasks with submission_id <= submission_id will be notified and removed from pending
    vk_collect_all_completed(gpu_timeline_value);
    return gpu_timeline_value;
}

void vk_Engine::vk_collect_all_completed(std::uint64_t submission_id) {
    // all tasks with submission_id <= submission_id will be notified and removed from pending
    while (!pending_submitted_tasks_.empty()) {
        if (pending_submitted_tasks_.front()->vk_submission_id() <= submission_id) {
            pending_submitted_tasks_.front()->vk_notify_completion();
            pending_submitted_tasks_.pop_front();
        }
        else {
            break;
        }
    }
}

Engine::Engine(std::shared_ptr<Device> device, std::shared_ptr<vk_Engine> engine, uint32_t index) noexcept {
    device_ = std::move(device);
    engine_ = std::move(engine);
    engine_index_ = index;
}

std::uint32_t Engine::device_index() const noexcept {
    return device_->device_index();
}

void Resource::dispose() {
    data_ = nullptr;
    slice_ = {};
}

Buffer::Buffer(std::shared_ptr<vk_ResourceData> resource_data, ResourceSlice view_slice, std::shared_ptr<Layout> layout, Format format)
    : Resource(resource_data, view_slice), layout_(std::move(layout)), format_(format) {
    assert (view_slice.type == ResourceType::BUFFER);
}

std::uint64_t Buffer::device_ptr() const {
    return data_->device_ptr() + slice_.buffer.offset;
}

std::uint64_t Buffer::external_ptr() const {
    return data_->external_ptr() + slice_.buffer.offset;
}

void Buffer::load(const pybind11::object& source) const {
    copy_pyobject_into(*data_->device(), external_ptr(), data_->is_cpu() ? MemoryLocation::HOST : MemoryLocation::DEVICE, size(), source);
}

void Buffer::save(const pybind11::object& target) const {
    copy_pyobject_from(*data_->device(), external_ptr(), data_->is_cpu() ? MemoryLocation::HOST : MemoryLocation::DEVICE, size(), target);
}

std::uint64_t Buffer::vk_buffer_offset() const noexcept {
    return data_->get_memory()->offset() + slice_.buffer.offset;
}

std::uint64_t Buffer::vk_buffer_size() const noexcept {
    return slice_.buffer.size;
}

Buffer::~Buffer() noexcept{
    if (buffer_view_) {
        auto device = this->data_->device();
        if (device && !device->is_disposed()) {
            device->logical_device().destroyBufferView(buffer_view_);
        }
        buffer_view_ = nullptr;
    }
}

Image::Image(std::shared_ptr<vk_ResourceData> resource_data, ResourceSlice view_slice):Resource(resource_data, view_slice) {
}

Image::~Image() noexcept {
    if (image_view_) {
        auto device = this->data_->device();
        if (device && !device->is_disposed()) {
            device->logical_device().destroyImageView(image_view_);
        }
        image_view_ = nullptr;
    }
}

std::shared_ptr<Buffer> Buffer::cast(Type scalar) const {
    if (data_->resource_type() != ResourceType::BUFFER) {
        throw std::runtime_error("Buffer::cast can only be called on buffer resources");
    }
    auto layout = compute_layout(TypeDescriptor::single(scalar), LayoutRule::Scalar);
    ResourceSlice view_slice{};
    view_slice.type = ResourceType::BUFFER;
    view_slice.buffer.offset = slice_.buffer.offset;
    view_slice.buffer.size = slice_.buffer.size;
    return std::make_shared<Buffer>(data_, view_slice, std::move(layout));
}

std::shared_ptr<Buffer> Buffer::cast(Format format) const {
    if (data_->resource_type() != ResourceType::BUFFER) {
        throw std::runtime_error("Buffer::cast can only be called on buffer resources");
    }
    const Type component_type = format_scalar_type(format);
    const int components = static_cast<int>(formatSize(format)) / scalar_type_size(component_type);
    auto element_type = std::make_shared<TypeDescriptor>(TypeDescriptor::single(component_type));
    auto layout = compute_layout(TypeDescriptor::array_of(std::move(element_type), components), LayoutRule::Scalar);
    ResourceSlice view_slice{};
    view_slice.type = ResourceType::BUFFER;
    view_slice.buffer.offset = slice_.buffer.offset;
    view_slice.buffer.size = slice_.buffer.size;
    return std::make_shared<Buffer>(data_, view_slice, std::move(layout), format);
}

std::shared_ptr<Buffer> Buffer::cast(const std::shared_ptr<Layout>& layout) const {
    if (data_->resource_type() != ResourceType::BUFFER) {
        throw std::runtime_error("Buffer::cast can only be called on buffer resources");
    }
    ResourceSlice view_slice{};
    view_slice.type = ResourceType::BUFFER;
    view_slice.buffer.offset = slice_.buffer.offset;
    view_slice.buffer.size = slice_.buffer.size;
    return std::make_shared<Buffer>(data_, view_slice, layout);
}

std::shared_ptr<Buffer> Buffer::slice(std::uint64_t start_element, std::uint64_t count) const {
    if (data_->resource_type() != ResourceType::BUFFER) {
        throw std::runtime_error("Buffer::slice can only be called on buffer resources");
    }
    const std::uint64_t element_size = layout_->aligned_size;
    const std::uint64_t byte_offset = start_element * element_size;
    const std::uint64_t byte_size = count * element_size;
    if (count == 0 || byte_offset + byte_size > slice_.buffer.size) {
        throw std::runtime_error("Invalid buffer slice range");
    }
    ResourceSlice view_slice{};
    view_slice.type = ResourceType::BUFFER;
    view_slice.buffer.offset = slice_.buffer.offset + byte_offset;
    view_slice.buffer.size = byte_size;
    return std::make_shared<Buffer>(data_, view_slice, layout_, format_);
}

std::shared_ptr<Buffer> Buffer::element(std::uint64_t index) const {
    return slice(index, 1);
}

DLDevice Buffer::vk_dlpack_device() const noexcept {
    return data_->get_memory()->dl_device();
}

namespace {
    // Leaf scalar component type of `layout`: itself for a SINGLE (scalar,
    // vector or matrix), or recursively its element's for ARRAY. Throws for
    // STRUCT (and any layout that somehow has neither a component_type nor
    // an element_layout), since a struct isn't a single numeric type.
    Type resolve_component_type(const Layout& layout) {
        if (layout.component_type != Type::UNDEFINED) {
            return layout.component_type;
        }
        if (layout.element_layout) {
            return resolve_component_type(*layout.element_layout);
        }
        throw std::runtime_error("Cannot resolve a scalar component type for this layout");
    }

    // Appends the tensor dimensions (shape + strides, in elements of
    // `scalar_size`-byte components) contributed by `layout` itself: none
    // for a scalar SINGLE, one for a vector SINGLE, two for a matrix SINGLE
    // (columns, rows), and one plus whatever its element contributes for
    // ARRAY. Throws for STRUCT.
    void append_tensor_dims(const Layout& layout, std::uint64_t scalar_size,
        std::vector<std::int64_t>& shape, std::vector<std::int64_t>& strides) {
        switch (layout.kind) {
            case TypeKind::SINGLE:
                if (layout.element_layout) {
                    // Matrix.
                    shape.push_back(static_cast<std::int64_t>(layout.count));
                    shape.push_back(static_cast<std::int64_t>(layout.element_layout->size / scalar_size));
                    strides.push_back(static_cast<std::int64_t>(layout.stride / scalar_size));
                    strides.push_back(1);
                } else if (type_traits(layout.type).rows > 1) {
                    // Vector.
                    shape.push_back(static_cast<std::int64_t>(layout.size / scalar_size));
                    strides.push_back(1);
                }
                // Else: plain scalar, no dims contributed.
                break;
            case TypeKind::ARRAY:
                shape.push_back(static_cast<std::int64_t>(layout.count));
                strides.push_back(static_cast<std::int64_t>(layout.stride / scalar_size));
                append_tensor_dims(*layout.element_layout, scalar_size, shape, strides);
                break;
            case TypeKind::STRUCT:
                throw std::runtime_error("Cannot create a dlpack tensor view over a struct field; "
                    "select a scalar/vector/matrix/array field instead");
        }
    }

    // Copies every element described by `shape`/`strides` (as produced by
    // append_tensor_dims: at most 2 dims for the vector/matrix/array-of-
    // scalars fields Buffer::read()/write() support) between `strided`
    // (this field's own, possibly padded, location inside the buffer) and
    // `tight` (a contiguous scratch/user buffer of exactly
    // element_count * scalar_size bytes). `shape`/`strides` are in units of
    // `scalar_size`-byte elements, matching append_tensor_dims's convention.
    // Recursive step for copy_field_elements(): walks dimension `dim` of
    // `shape`/`strides`. The innermost dimension is copied as a single
    // contiguous run whenever it actually is contiguous (strides[dim] == 1,
    // true for a VECTOR's own dimension and a MATRIX's row dimension, and
    // hence for anything nested under one of those, e.g. an array of
    // vectors/matrices); the one case where even the innermost dimension
    // isn't contiguous is a padded ARRAY-of-SCALAR (e.g. std140), copied one
    // scalar at a time instead.
    void copy_field_elements_dim(
        char* strided, char* tight, std::uint64_t scalar_size,
        const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& strides,
        std::size_t dim, bool strided_to_tight) {
        const bool last_dim = (dim + 1 == shape.size());
        if (last_dim && strides[dim] == 1) {
            const std::uint64_t row_bytes = static_cast<std::uint64_t>(shape[dim]) * scalar_size;
            if (strided_to_tight) std::memcpy(tight, strided, row_bytes);
            else std::memcpy(strided, tight, row_bytes);
            return;
        }
        std::uint64_t tight_row_bytes = scalar_size;
        for (std::size_t d = dim + 1; d < shape.size(); ++d) {
            tight_row_bytes *= static_cast<std::uint64_t>(shape[d]);
        }
        for (std::int64_t i = 0; i < shape[dim]; ++i) {
            char* s = strided + static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(strides[dim]) * scalar_size;
            char* t = tight + static_cast<std::uint64_t>(i) * tight_row_bytes;
            if (last_dim) {
                if (strided_to_tight) std::memcpy(t, s, scalar_size);
                else std::memcpy(s, t, scalar_size);
            } else {
                copy_field_elements_dim(s, t, scalar_size, shape, strides, dim + 1, strided_to_tight);
            }
        }
    }

    // Copies every element described by `shape`/`strides` (as produced by
    // append_tensor_dims, at any nesting depth: vectors, matrices, and
    // arrays of either) between `strided` (this field's own, possibly
    // padded, location inside the buffer) and `tight` (a contiguous
    // scratch/user buffer of exactly element_count * scalar_size bytes).
    // `shape`/`strides` are in units of `scalar_size`-byte elements,
    // matching append_tensor_dims's convention.
    void copy_field_elements(
        char* strided, char* tight, std::uint64_t scalar_size,
        const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& strides,
        bool strided_to_tight) {
        if (shape.empty()) {
            if (strided_to_tight) std::memcpy(tight, strided, scalar_size);
            else std::memcpy(strided, tight, scalar_size);
            return;
        }
        copy_field_elements_dim(strided, tight, scalar_size, shape, strides, 0, strided_to_tight);
    }

    // Reads one scalar of `type` from `ptr`, returned as a plain Python number.
    pybind11::object read_scalar(const void* ptr, Type type) {
        switch (type) {
            case Type::BOOL: { std::uint8_t v; std::memcpy(&v, ptr, 1); return pybind11::cast(v != 0); }
            case Type::INT8: { std::int8_t v; std::memcpy(&v, ptr, 1); return pybind11::cast(v); }
            case Type::UINT8: { std::uint8_t v; std::memcpy(&v, ptr, 1); return pybind11::cast(v); }
            case Type::INT16: { std::int16_t v; std::memcpy(&v, ptr, 2); return pybind11::cast(v); }
            case Type::UINT16: { std::uint16_t v; std::memcpy(&v, ptr, 2); return pybind11::cast(v); }
            case Type::FLOAT16: { std::uint16_t v; std::memcpy(&v, ptr, 2); return pybind11::cast(v); } // raw bit pattern
            case Type::INT32: { std::int32_t v; std::memcpy(&v, ptr, 4); return pybind11::cast(v); }
            case Type::UINT32: { std::uint32_t v; std::memcpy(&v, ptr, 4); return pybind11::cast(v); }
            case Type::FLOAT32: { float v; std::memcpy(&v, ptr, 4); return pybind11::cast(v); }
            case Type::INT64: { std::int64_t v; std::memcpy(&v, ptr, 8); return pybind11::cast(v); }
            case Type::UINT64: { std::uint64_t v; std::memcpy(&v, ptr, 8); return pybind11::cast(v); }
            case Type::FLOAT64: { double v; std::memcpy(&v, ptr, 8); return pybind11::cast(v); }
            default: throw std::runtime_error("Unsupported scalar type");
        }
    }

    // Writes one scalar of `type`, given as a plain Python number, to `ptr`.
    void write_scalar(void* ptr, Type type, const pybind11::object& value) {
        switch (type) {
            case Type::BOOL: { std::uint8_t v = value.cast<bool>() ? 1 : 0; std::memcpy(ptr, &v, 1); return; }
            case Type::INT8: { auto v = value.cast<std::int8_t>(); std::memcpy(ptr, &v, 1); return; }
            case Type::UINT8: { auto v = value.cast<std::uint8_t>(); std::memcpy(ptr, &v, 1); return; }
            case Type::INT16: { auto v = value.cast<std::int16_t>(); std::memcpy(ptr, &v, 2); return; }
            case Type::UINT16: { auto v = value.cast<std::uint16_t>(); std::memcpy(ptr, &v, 2); return; }
            case Type::FLOAT16: { auto v = value.cast<std::uint16_t>(); std::memcpy(ptr, &v, 2); return; } // raw bit pattern
            case Type::INT32: { auto v = value.cast<std::int32_t>(); std::memcpy(ptr, &v, 4); return; }
            case Type::UINT32: { auto v = value.cast<std::uint32_t>(); std::memcpy(ptr, &v, 4); return; }
            case Type::FLOAT32: { auto v = value.cast<float>(); std::memcpy(ptr, &v, 4); return; }
            case Type::INT64: { auto v = value.cast<std::int64_t>(); std::memcpy(ptr, &v, 8); return; }
            case Type::UINT64: { auto v = value.cast<std::uint64_t>(); std::memcpy(ptr, &v, 8); return; }
            case Type::FLOAT64: { auto v = value.cast<double>(); std::memcpy(ptr, &v, 8); return; }
            default: throw std::runtime_error("Unsupported scalar type");
        }
    }

    // Imports the `vk.math3d` module (mirrors module_install_dir()'s use of
    // py::module_::import for a different purpose). Not cached in a static:
    // a static py::object's destructor runs at process-exit time, which can
    // fire after Py_Finalize and crash trying to Py_DECREF with no GIL/
    // interpreter left. Cheap to call repeatedly regardless -- Python's own
    // sys.modules cache makes a re-import of an already-imported module a
    // simple dict lookup. Returns an empty (falsy) object if the import
    // fails for any reason; callers fall back to raw bytes in that case.
    pybind11::object vk_math3d_module() {
        try {
            return pybind11::module_::import("vk.math3d");
        } catch (const pybind11::error_already_set&) {
            return pybind11::object();
        }
    }

    // Name of the vk.math3d class matching a VECTOR field (e.g. FLOAT32/3
    // -> "vec3", INT32/2 -> "ivec2"), or "" if there's no equivalent
    // math3d type (math3d only covers the GLSL-like float/int/uint/bool
    // scalar types, 2 to 4 components).
    std::string vk_math3d_vector_name(Type component_type, int components) {
        if (components < 2 || components > 4) return {};
        const char* prefix;
        switch (component_type) {
            case Type::FLOAT32: prefix = "vec"; break;
            case Type::INT32: prefix = "ivec"; break;
            case Type::UINT32: prefix = "uvec"; break;
            case Type::BOOL: prefix = "bvec"; break;
            default: return {};
        }
        return prefix + std::to_string(components);
    }

    // Name of the vk.math3d class matching a MATRIX field, or "" if there's
    // no equivalent (math3d, like GLSL, only has float matrices).
    std::string vk_math3d_matrix_name(Type component_type, int columns, int rows) {
        if (component_type != Type::FLOAT32) return {};
        if (columns < 2 || columns > 4 || rows < 2 || rows > 4) return {};
        return "mat" + std::to_string(columns) + (columns == rows ? "" : "x" + std::to_string(rows));
    }

    // Calls a Python callable with a dynamically-sized tuple of positional
    // arguments -- pybind11 has no C++ spelling for Python's `f(*args)`.
    pybind11::object vk_call_with_args(const pybind11::object& callable, const pybind11::tuple& args) {
        pybind11::object result = pybind11::reinterpret_steal<pybind11::object>(PyObject_CallObject(callable.ptr(), args.ptr()));
        if (!result) throw pybind11::error_already_set();
        return result;
    }

    // Recursively flattens `value` -- expected to be nested `shape.size()`
    // levels deep, each level of length `shape[dim]` (matching
    // append_tensor_dims's shape convention: one dim of components for a
    // vector, [columns, rows] for a matrix) -- into `out`, `scalar_size`
    // bytes at a time via write_scalar. Accepts any object supporting the
    // Python sequence protocol at each level: a math3d vec*/mat* (whose
    // `__getitem__` yields components, or columns for a matrix), or a
    // plain nested Python list/tuple.
    void vk_flatten_sequence_into(
        const pybind11::object& value, Type component_type, std::uint64_t scalar_size,
        const std::vector<std::int64_t>& shape, std::size_t dim, char*& out) {
        if (dim == shape.size()) {
            write_scalar(out, component_type, value);
            out += scalar_size;
            return;
        }
        pybind11::sequence seq = value.cast<pybind11::sequence>();
        if (static_cast<std::int64_t>(seq.size()) != shape[dim]) {
            throw std::runtime_error("Buffer::write: source sequence size does not match the field's shape");
        }
        for (auto item : seq) {
            vk_flatten_sequence_into(pybind11::reinterpret_borrow<pybind11::object>(item), component_type, scalar_size, shape, dim + 1, out);
        }
    }
}

pybind11::object Buffer::vk_dlpack() const {
    void* ptr = nullptr;
    auto memory = data_->get_memory();
    DLDevice device = memory->dl_device();
    if (external_ptr() == 0) {
        throw std::runtime_error("A DEVICE tensor was requested but the external pointer is unavailable");
    }
    ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(external_ptr()));

    // Shape/dtype are derived from element_layout(): a struct has no single
    // numeric type, so it falls back to raw bytes; everything else (scalar,
    // vector, matrix, array) uses the same shape/stride machinery as
    // field(), just for the buffer's own top-level layout instead of one of
    // its fields (with instance_count = size() / element_layout()->aligned_size
    // as the outer dimension).
    Type dtype_scalar;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    if (layout_->kind == TypeKind::STRUCT) {
        dtype_scalar = Type::UINT8;
        shape = { static_cast<std::int64_t>(size()) };
        strides = { 1 };
    } else {
        dtype_scalar = resolve_component_type(*layout_);
        const std::uint64_t scalar_size = static_cast<std::uint64_t>(scalar_type_size(dtype_scalar));
        const std::uint64_t instance_count = layout_->aligned_size > 0 ? size() / layout_->aligned_size : 1;
        shape.push_back(static_cast<std::int64_t>(instance_count));
        strides.push_back(static_cast<std::int64_t>(layout_->aligned_size / scalar_size));
        append_tensor_dims(*layout_, scalar_size, shape, strides);
    }
    const int dimension = static_cast<int>(shape.size());

    auto* owner = new TensorOwner();
    owner->memory = memory;
    owner->shape = std::make_unique<std::int64_t[]>(dimension);
    owner->strides = std::make_unique<std::int64_t[]>(dimension);
    for (int i = 0; i < dimension; ++i) {
        owner->shape[i] = shape[i];
        owner->strides[i] = strides[i];
    }

    auto* managed = new DLManagedTensor{};
    managed->dl_tensor.data = ptr;
    managed->dl_tensor.device = device;
    managed->dl_tensor.ndim = dimension;
    managed->dl_tensor.dtype = dlpack_dtype(dtype_scalar);
    managed->dl_tensor.shape = owner->shape.get();
    managed->dl_tensor.strides = owner->strides.get();
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = owner;
    managed->deleter = dlmanaged_tensor_deleter;

    return export_dltensor_py(managed);
}

void Buffer::vk_require_struct_layout(const char* method_name) const {
    if (layout_->kind != TypeKind::STRUCT) {
        throw std::runtime_error(std::string("Buffer::") + method_name + ": only valid on a buffer whose element_layout() is a struct");
    }
}

pybind11::object Buffer::field(const LayoutField& field) const {
    vk_require_struct_layout("field");
    auto root = field.root.lock();
    if (!root) {
        throw std::runtime_error("LayoutField has no root layout (was it obtained from compute_layout()?)");
    }
    if (field.layout->kind == TypeKind::STRUCT) {
        throw std::runtime_error("Cannot create a dlpack tensor view over a struct field; "
            "select a scalar/vector/matrix/array field instead");
    }
    if (external_ptr() == 0) {
        throw std::runtime_error("A DEVICE tensor was requested but the external pointer is unavailable");
    }

    const Type component_type = resolve_component_type(*field.layout);
    const std::uint64_t scalar_size = static_cast<std::uint64_t>(scalar_type_size(component_type));

    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    // Outer dimension: how many instances of `root` this buffer holds.
    const std::uint64_t instance_count = root->aligned_size > 0 ? size() / root->aligned_size : 1;
    shape.push_back(static_cast<std::int64_t>(instance_count));
    strides.push_back(static_cast<std::int64_t>(root->aligned_size / scalar_size));
    append_tensor_dims(*field.layout, scalar_size, shape, strides);

    auto memory = data_->get_memory();
    const int dimension = static_cast<int>(shape.size());
    auto* owner = new TensorOwner();
    owner->memory = memory;
    owner->shape = std::make_unique<std::int64_t[]>(dimension);
    owner->strides = std::make_unique<std::int64_t[]>(dimension);
    for (int i = 0; i < dimension; ++i) {
        owner->shape[i] = shape[i];
        owner->strides[i] = strides[i];
    }

    auto* managed = new DLManagedTensor{};
    managed->dl_tensor.data = reinterpret_cast<char*>(static_cast<std::uintptr_t>(external_ptr())) + field.offset;
    managed->dl_tensor.device = memory->dl_device();
    managed->dl_tensor.ndim = dimension;
    managed->dl_tensor.dtype = dlpack_dtype(component_type);
    managed->dl_tensor.shape = owner->shape.get();
    managed->dl_tensor.strides = owner->strides.get();
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = owner;
    managed->deleter = dlmanaged_tensor_deleter;

    return export_dltensor_py(managed);
}

char* Buffer::vk_resolve_field_ptr(const LayoutField& field) const {
    vk_require_struct_layout("read/write");
    auto root = field.root.lock();
    if (!root) {
        throw std::runtime_error("LayoutField has no root layout (was it obtained from compute_layout()?)");
    }
    if (!data_->get_memory()->host_visible()) {
        throw std::runtime_error("Buffer::read/write: only supported on host-visible buffers");
    }
    if (size() != root->aligned_size) {
        throw std::runtime_error("Buffer::read/write: buffer must hold exactly one instance of the field's root layout");
    }
    if (external_ptr() == 0) {
        throw std::runtime_error("Buffer::read/write: no host-accessible pointer available for this buffer");
    }
    return reinterpret_cast<char*>(static_cast<std::uintptr_t>(external_ptr())) + field.offset;
}

pybind11::object Buffer::read(const LayoutField& field) const {
    char* ptr = vk_resolve_field_ptr(field);
    const Layout& layout = *field.layout;

    const bool is_matrix = layout.kind == TypeKind::SINGLE && layout.element_layout != nullptr;
    const bool is_vector = layout.kind == TypeKind::SINGLE && !is_matrix && type_traits(layout.type).rows > 1;
    if (layout.kind == TypeKind::SINGLE && !is_matrix && !is_vector) {
        return read_scalar(ptr, layout.component_type);
    }

    const Type component_type = resolve_component_type(layout);
    const std::uint64_t scalar_size = static_cast<std::uint64_t>(scalar_type_size(component_type));
    std::vector<std::int64_t> shape, strides;
    append_tensor_dims(layout, scalar_size, shape, strides);
    std::uint64_t total_elements = 1;
    for (auto d : shape) total_elements *= static_cast<std::uint64_t>(d);

    std::string packed(static_cast<std::size_t>(total_elements * scalar_size), '\0');
    copy_field_elements(ptr, packed.data(), scalar_size, shape, strides, /*strided_to_tight=*/true);

    // A vector or matrix field is returned as the matching vk.math3d
    // vec*/mat* object (e.g. FLOAT32 vec3 -> vk.math3d.vec3, INT32 vec2 ->
    // ivec2) when one exists, instead of raw bytes.
    std::string type_name;
    if (is_vector) {
        type_name = vk_math3d_vector_name(component_type, static_cast<int>(total_elements));
    } else if (is_matrix) {
        type_name = vk_math3d_matrix_name(
            component_type, static_cast<int>(layout.count),
            static_cast<int>(layout.element_layout->size / scalar_size));
    }
    if (!type_name.empty()) {
        pybind11::object module = vk_math3d_module();
        if (module && pybind11::hasattr(module, type_name.c_str())) {
            pybind11::object cls = module.attr(type_name.c_str());
            pybind11::tuple args(static_cast<std::size_t>(total_elements));
            const char* p = packed.data();
            for (std::uint64_t i = 0; i < total_elements; ++i) {
                args[static_cast<std::size_t>(i)] = read_scalar(p, component_type);
                p += scalar_size;
            }
            return vk_call_with_args(cls, args);
        }
    }
    return pybind11::bytes(packed);
}

void Buffer::write(const LayoutField& field, const pybind11::object& value) const {
    char* ptr = vk_resolve_field_ptr(field);
    const Layout& layout = *field.layout;

    const bool is_matrix = layout.kind == TypeKind::SINGLE && layout.element_layout != nullptr;
    const bool is_vector = layout.kind == TypeKind::SINGLE && !is_matrix && type_traits(layout.type).rows > 1;
    if (layout.kind == TypeKind::SINGLE && !is_matrix && !is_vector) {
        write_scalar(ptr, layout.component_type, value);
        return;
    }

    const Type component_type = resolve_component_type(layout);
    const std::uint64_t scalar_size = static_cast<std::uint64_t>(scalar_type_size(component_type));
    std::vector<std::int64_t> shape, strides;
    append_tensor_dims(layout, scalar_size, shape, strides);
    std::uint64_t total_elements = 1;
    for (auto d : shape) total_elements *= static_cast<std::uint64_t>(d);
    const std::uint64_t total_bytes = total_elements * scalar_size;

    // A plain Python buffer object (e.g. a numpy array), assumed C-contiguous.
    if (PyObject_CheckBuffer(value.ptr())) {
        pybind11::buffer buf = pybind11::reinterpret_borrow<pybind11::buffer>(value);
        pybind11::buffer_info info = buf.request();
        const std::uint64_t src_bytes = static_cast<std::uint64_t>(info.size) * static_cast<std::uint64_t>(info.itemsize);
        if (src_bytes != total_bytes) {
            throw std::runtime_error("Buffer::write: source buffer size does not match the field's size");
        }
        copy_field_elements(ptr, reinterpret_cast<char*>(info.ptr), scalar_size, shape, strides, /*strided_to_tight=*/false);
        return;
    }

    // A DLPack-compatible object (e.g. a CPU torch tensor): pull its
    // "dltensor" capsule directly rather than going through from_dlpack(),
    // which is the expensive path this method exists to avoid.
    if (pybind11::hasattr(value, "__dlpack__")) {
        pybind11::object capsule = value.attr("__dlpack__")();
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
        if (!managed) {
            throw std::runtime_error("Buffer::write: invalid DLPack capsule (expected a \"dltensor\" capsule)");
        }
        const DLTensor& src = managed->dl_tensor;
        // device_type 1 == CPU, 3 == CUDA host/pinned memory: both are
        // directly dereferenceable from the CPU. Anything else (e.g. a
        // plain CUDA device tensor) can't be safely memcpy'd from here.
        if (src.device.device_type != 1 && src.device.device_type != 3) {
            throw std::runtime_error("Buffer::write: DLPack source tensor must be CPU-accessible");
        }
        const std::uint64_t src_itemsize = (static_cast<std::uint64_t>(src.dtype.bits) / 8) * static_cast<std::uint64_t>(src.dtype.lanes);
        std::uint64_t src_bytes = src_itemsize;
        for (std::int32_t i = 0; i < src.ndim; ++i) {
            src_bytes *= static_cast<std::uint64_t>(src.shape[i]);
        }
        if (src_bytes != total_bytes) {
            throw std::runtime_error("Buffer::write: DLPack source tensor size does not match the field's size");
        }
        char* src_ptr = reinterpret_cast<char*>(src.data) + src.byte_offset;
        copy_field_elements(ptr, src_ptr, scalar_size, shape, strides, /*strided_to_tight=*/false);
        return;
    }

    // A math3d-style vec*/mat* object, or any plain (possibly nested, for
    // a matrix) Python sequence of numbers: written element-by-element via
    // write_scalar, so int/uint/bool components convert correctly
    // regardless of the source's own representation. Only reached for
    // vector/matrix fields (an ARRAY-of-scalars field still requires a
    // buffer-protocol/DLPack source, above).
    if ((is_vector || is_matrix)
        && pybind11::hasattr(value, "__len__") && pybind11::hasattr(value, "__getitem__")) {
        std::string packed(static_cast<std::size_t>(total_bytes), '\0');
        char* out = packed.data();
        vk_flatten_sequence_into(value, component_type, scalar_size, shape, 0, out);
        copy_field_elements(ptr, packed.data(), scalar_size, shape, strides, /*strided_to_tight=*/false);
        return;
    }

    throw std::runtime_error(
        "Buffer::write: value must be a plain number for a scalar field, a vk.math3d vec*/mat* "
        "object or nested Python sequence for a vector/matrix field, or a Python buffer object / "
        "DLPack-compatible object for a vector, matrix or array-of-scalars field");
}

std::shared_ptr<Image> Image::cast_format(Format new_format) const {
    if (data_->resource_type() != ResourceType::IMAGE) {
        throw std::runtime_error("image_cast can only be called on image resources");
    }
    ResourceSlice view_slice{};
    view_slice.type = ResourceType::IMAGE;
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
    view_slice.type = ResourceType::IMAGE;
    view_slice.image.mip_start = slice_.image.mip_start + mip_start;
    view_slice.image.mip_count = mip_count;
    view_slice.image.array_start = slice_.image.array_start + array_start;
    view_slice.image.array_count = array_count;
    view_slice.image.format = slice_.image.format;
    return std::make_shared<Image>(data_, view_slice);
}

namespace {

// Best-effort PCI vendor ID -> friendly name, for the vendors this project
// is realistically tested against; falls back to a hex string for
// anything else (still useful, just not a friendly name).
std::string vk_vendor_name(std::uint32_t vendor_id) {
    switch (vendor_id) {
        case 0x1002: return "AMD";
        case 0x10DE: return "NVIDIA";
        case 0x8086: return "Intel";
        case 0x13B5: return "ARM";
        case 0x5143: return "Qualcomm";
        case 0x1010: return "ImgTec";
        case 0x106B: return "Apple";
        default: {
            std::ostringstream oss;
            oss << "0x" << std::hex << vendor_id;
            return oss.str();
        }
    }
}

std::string vk_device_type_name(vk::PhysicalDeviceType type) {
    switch (type) {
        case vk::PhysicalDeviceType::eIntegratedGpu: return "integrated_gpu";
        case vk::PhysicalDeviceType::eDiscreteGpu: return "discrete_gpu";
        case vk::PhysicalDeviceType::eVirtualGpu: return "virtual_gpu";
        case vk::PhysicalDeviceType::eCpu: return "cpu";
        default: return "other";
    }
}

std::string vk_format_api_version(std::uint32_t version) {
    std::ostringstream oss;
    oss << VK_API_VERSION_MAJOR(version) << "." << VK_API_VERSION_MINOR(version) << "." << VK_API_VERSION_PATCH(version);
    return oss.str();
}

} // namespace

std::vector<DeviceInfo> query_device_infos() {
    // Same instance-creation recipe as Device::Device(), minus the
    // layers/surface extensions (irrelevant for a query-only instance
    // that's destroyed before this function returns).
    uint32_t loader_version = VK_API_VERSION_1_1;
    if (vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS) {
        loader_version = VK_API_VERSION_1_1;
    }
    const uint32_t api_version = std::min(loader_version, VK_API_VERSION_1_3);
    vk::ApplicationInfo app("vk", 1, "vk", 1, api_version);
    vk::InstanceCreateInfo ci{};
    ci.pApplicationInfo = &app;

    vk::Instance instance = vk::createInstance(ci);
    std::vector<DeviceInfo> result;
    try {
        const auto gpus = instance.enumeratePhysicalDevices();
        result.reserve(gpus.size());
        for (std::uint32_t i = 0; i < gpus.size(); ++i) {
            const vk::PhysicalDeviceProperties props = gpus[i].getProperties();
            const vk::PhysicalDeviceMemoryProperties mem_props = gpus[i].getMemoryProperties();
            std::uint64_t vram_bytes = 0;
            for (std::uint32_t h = 0; h < mem_props.memoryHeapCount; ++h) {
                if (mem_props.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                    vram_bytes += mem_props.memoryHeaps[h].size;
                }
            }
            DeviceInfo info{};
            info.index = i;
            info.name = std::string(props.deviceName.data());
            info.vendor = vk_vendor_name(props.vendorID);
            info.vendor_id = props.vendorID;
            info.device_id = props.deviceID;
            info.device_type = vk_device_type_name(props.deviceType);
            info.api_version = vk_format_api_version(props.apiVersion);
            info.driver_version = props.driverVersion;
            info.vram_bytes = vram_bytes;
            result.push_back(std::move(info));
        }
    } catch (...) {
        instance.destroy();
        throw;
    }
    instance.destroy();
    return result;
}

Device::Device(uint32_t device_index, bool enable_validation_layers) {
    device_index_ = device_index;
    uint32_t loader_version = VK_API_VERSION_1_1;
    if (vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS) {
        loader_version = VK_API_VERSION_1_1;
    }

    const uint32_t api_version = std::min(loader_version, VK_API_VERSION_1_3);
    vk::ApplicationInfo app("vk", 1, "vk", 1, api_version);
    // Extended dynamic state (dynamic cull mode/front face/depth test/depth
    // write/depth compare op -- see CommandBuffer::set_cull_mode/
    // set_front_face/set_depth_test) is core, unconditional functionality
    // from Vulkan 1.3 onwards; no separate extension/feature check needed.
    extended_dynamic_state_supported_ = api_version >= VK_API_VERSION_1_3;

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

    // Surface extensions, needed for Window/swapchain support. Enabled
    // defensively (whichever the platform actually supports) rather than
    // deferred until the first Window is created, since VkInstance
    // extensions can only be chosen once, up front, and glfwGetRequiredInstanceExtensions
    // needs glfwInit() to already have been called (a chicken-and-egg
    // problem avoided by just enabling every surface extension this
    // platform supports up front instead).
    const std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    std::vector<const char*> enabled_instance_extensions;
    const auto add_instance_extension_if_supported = [&](const char* name) {
        if (supports_extension(instance_extensions, name)) {
            enabled_instance_extensions.push_back(name);
        }
    };
    add_instance_extension_if_supported(VK_KHR_SURFACE_EXTENSION_NAME);
#if defined(_WIN32)
    add_instance_extension_if_supported(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
    // Literal names instead of VK_KHR_XLIB_SURFACE_EXTENSION_NAME/
    // VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME: those macros only exist once
    // vulkan_xlib.h/vulkan_wayland.h are pulled in (behind
    // VK_USE_PLATFORM_XLIB_KHR/_WAYLAND_KHR), which isn't guaranteed here
    // (e.g. Wayland dev headers aren't installed in the manylinux build image).
    add_instance_extension_if_supported("VK_KHR_xlib_surface");
    add_instance_extension_if_supported("VK_KHR_wayland_surface");
#endif

    vk::InstanceCreateInfo ci{};
    ci.pApplicationInfo = &app;
    ci.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
    ci.ppEnabledLayerNames = enabled_layers.empty() ? nullptr : enabled_layers.data();
    ci.enabledExtensionCount = static_cast<uint32_t>(enabled_instance_extensions.size());
    ci.ppEnabledExtensionNames = enabled_instance_extensions.empty() ? nullptr : enabled_instance_extensions.data();

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
    add_extension_if_supported(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    add_extension_if_supported(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#if defined(_WIN32) && defined(VK_USE_PLATFORM_WIN32_KHR)
    add_extension_if_supported("VK_KHR_external_memory_win32");
    const char* external_semaphore_platform_extension = "VK_KHR_external_semaphore_win32";
#else
    add_extension_if_supported("VK_KHR_external_memory_fd");
    const char* external_semaphore_platform_extension = "VK_KHR_external_semaphore_fd";
#endif
    add_extension_if_supported(external_semaphore_platform_extension);
    // Only meaningful if every extension needed to export/import a
    // cross-API semaphore (see vk_init()'s interop_semaphore_ setup) is
    // actually present -- interop-semaphore-based synchronization is
    // skipped entirely otherwise, falling back to a plain host-side wait.
    external_semaphore_supported_ =
        supports_extension(supported_extensions, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME) &&
        supports_extension(supported_extensions, external_semaphore_platform_extension);
    // Gates Device::create_window: without VK_KHR_swapchain there is no
    // way to present anything.
    swapchain_supported_ = supports_extension(supported_extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    // Gates Device::create_ads() and the acceleration-structure-related
    // buffer usage flags (see full_buffer_usage_flags()).
    // VK_KHR_deferred_host_operations is VK_KHR_acceleration_structure's own
    // hard dependency.
    acceleration_structure_supported_ =
        supports_extension(supported_extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
        supports_extension(supported_extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

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

    // See Device::vk_dynamic_dispatch()'s comment: acceleration-structure
    // entry points aren't guaranteed to be resolvable as directly-linked
    // symbols, so load them dynamically here instead.
    dynamic_dispatch_.init(static_cast<VkInstance>(instance_), vkGetInstanceProcAddr,
        static_cast<VkDevice>(device_), vkGetDeviceProcAddr);
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

void Device::vk_init() {
    engines_.resize(8); // one slot per EngineType bitmask combination (1..7)

    if (external_semaphore_supported_) {
        vk::SemaphoreTypeCreateInfo semaphore_type_create_info(vk::SemaphoreType::eTimeline, 0);
#if defined(_WIN32) && defined(VK_USE_PLATFORM_WIN32_KHR)
        vk::ExportSemaphoreCreateInfo export_info(vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32);
#else
        vk::ExportSemaphoreCreateInfo export_info(vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd);
#endif
        semaphore_type_create_info.pNext = &export_info;
        interop_semaphore_ = device_.createSemaphore(vk::SemaphoreCreateInfo{
            vk::SemaphoreCreateFlags{},
            &semaphore_type_create_info
        });
    }

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
    host_memory_manager_ = std::make_shared<MemoryManager>(shared_from_this(), cpu_type, true);
    device_memory_manager_ = std::make_shared<MemoryManager>(shared_from_this(), gpu_type, gpu_host_visible);

    const auto queue_families = physical_.getQueueFamilyProperties();

    command_pools_.resize(queue_families.size()); // holds one command pool for each family if necessary

    for (uint32_t family_index = 0; family_index < queue_families.size(); ++family_index) {
        const auto& family = queue_families[family_index];
        for (uint32_t engine_combination_index = 1; engine_combination_index < 8; ++engine_combination_index) {
            std::uint32_t fflags = (std::uint32_t)family.queueFlags;
            if ((fflags & engine_combination_index) == engine_combination_index) {
                if (engine_type_index_[engine_combination_index] == -1 || fflags < engine_queue_flags_[engine_combination_index]) {
                    engine_type_index_[engine_combination_index] = family_index;
                    engine_queue_flags_[engine_combination_index] = fflags;
                    engine_queue_count_[engine_combination_index] = family.queueCount;
                }
            }
        }
    }
}

std::shared_ptr<Engine> Device::create_engine(EngineType type, uint32_t index) {
    int engine_index = (int)type;
    assert (engine_index > 0 && engine_index < 8);

    int family_index = engine_type_index_[engine_index];
    if (family_index == -1) {
        throw std::runtime_error("No suitable queue family found for requested engine type");
    }

    if (engines_[engine_index].size() == 0) {
        engines_[engine_index].resize(engine_queue_count_[engine_index]);
    }

    uint32_t queue_index = index % engine_queue_count_[engine_index];

    if (engines_[engine_index][queue_index] == nullptr) {
        auto dev = logical_device();
        vk::Queue queue = dev.getQueue(family_index, queue_index);
        if (!command_pools_[family_index]) {
            vk::CommandPoolCreateInfo pool_info{};
            pool_info.queueFamilyIndex = family_index;
            pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
            command_pools_[family_index] = dev.createCommandPool(pool_info);
        }
        engines_[engine_index][queue_index] = std::make_shared<vk_Engine>(shared_from_this(), queue, command_pools_[family_index], type);
    }

    return std::make_shared<Engine>(shared_from_this(), engines_[engine_index][queue_index], index);
}

std::shared_ptr<Pipeline> Device::create_pipeline(PipelineType type) {
    auto pipeline = std::make_shared<vk_Pipeline>(shared_from_this(), type);
    return std::make_shared<Pipeline>(shared_from_this(), pipeline);
}

ShaderSource ShaderSource::from_spirv(std::vector<std::uint32_t> code, const std::string& entry_point) {
    ShaderSource source;
    source.code_ = std::move(code);
    source.entry_point_ = entry_point;
    return source;
}

ShaderSource ShaderSource::from_glsl(
    const std::string& source_text,
    ShaderStageType stage,
    const std::string& entry_point,
    const std::vector<std::string>& include_dirs) {
    namespace fs = std::filesystem;

    // Shells out to glslangValidator rather than linking shaderc, so this
    // project has no build- or link-time dependency on the Vulkan SDK --
    // only a runtime dependency on glslangValidator being available. The
    // source is piped through the child's stdin (no temp input file needed);
    // the compiled SPIR-V and any diagnostics always land in the same two
    // fixed-name temp files (calls are expected to happen one at a time, not
    // concurrently from multiple threads).
    const fs::path dir = fs::temp_directory_path();
    const fs::path output_path = dir / "temp_shader_compilation_output.temp.spv";
    const fs::path log_path = dir / "temp_shader_compilation_output.temp.log";

    std::string command =
        "\"" + glslang_validator_path() + "\" --stdin -V -S " + to_glslang_stage_name(stage) +
        " --source-entrypoint " + entry_point + " -e " + entry_point;
    for (const auto& include_dir : include_dirs) {
        command += " -I\"" + include_dir + "\"";
    }
    command += " -o \"" + output_path.string() + "\" > \"" + log_path.string() + "\" 2>&1";
#if defined(_WIN32)
    // Both system() and _popen() on Windows run the command through
    // `cmd.exe /c`, which has a well known quirk: a command line whose
    // first token is itself quoted (as ours is, to handle spaces in the
    // executable path) gets mis-parsed ("The filename, directory name, or
    // volume label syntax is incorrect.") unless the entire command is
    // wrapped in one more, outer pair of quotes.
    command = "\"" + command + "\"";
    FILE* pipe = _popen(command.c_str(), "wb");
#else
    FILE* pipe = popen(command.c_str(), "w");
#endif
    if (!pipe) {
        throw std::runtime_error("ShaderSource::from_glsl: failed to launch glslangValidator");
    }
    std::fwrite(source_text.data(), 1, source_text.size(), pipe);
#if defined(_WIN32)
    const int exit_code = _pclose(pipe);
#else
    const int exit_code = pclose(pipe);
#endif

    std::string log;
    {
        std::ifstream log_file(log_path, std::ios::binary);
        if (log_file) {
            std::ostringstream oss;
            oss << log_file.rdbuf();
            log = oss.str();
        }
    }
    std::error_code ec;
    fs::remove(log_path, ec);

    if (exit_code != 0) {
        fs::remove(output_path, ec);
        throw std::runtime_error(
            "ShaderSource::from_glsl: glslangValidator failed (is it installed and on PATH?):\n" + log);
    }

    std::vector<std::uint32_t> code;
    {
        std::ifstream output_file(output_path, std::ios::binary | std::ios::ate);
        if (!output_file) {
            throw std::runtime_error("ShaderSource::from_glsl: glslangValidator did not produce an output file");
        }
        const std::streamsize length = output_file.tellg();
        output_file.seekg(0);
        code.resize(static_cast<std::size_t>(length) / 4);
        output_file.read(reinterpret_cast<char*>(code.data()), length);
    }
    fs::remove(output_path, ec);

    return from_spirv(std::move(code), entry_point);
}

ShaderSource ShaderSource::from_file(
    const std::string& path,
    ShaderStageType stage,
    const std::string& entry_point,
    const std::vector<std::string>& include_dirs) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("ShaderSource::from_file: cannot open '" + path + "'");
    }
    const std::streamsize length = file.tellg();
    file.seekg(0);

    const bool is_spirv_binary = path.size() >= 4 && path.compare(path.size() - 4, 4, ".spv") == 0;
    if (is_spirv_binary) {
        if (length % 4 != 0) {
            throw std::runtime_error("ShaderSource::from_file: '" + path + "' is not a valid SPIR-V binary (size not a multiple of 4)");
        }
        std::vector<std::uint32_t> code(static_cast<std::size_t>(length) / 4);
        file.read(reinterpret_cast<char*>(code.data()), length);
        return from_spirv(std::move(code), entry_point);
    }

    std::string text(static_cast<std::size_t>(length), '\0');
    file.read(text.data(), length);
    return from_glsl(text, stage, entry_point, include_dirs);
}

vk_Pipeline::vk_Pipeline(std::weak_ptr<Device> device, PipelineType type)
    : device_(std::move(device)), type_(type) {}

vk_Pipeline::~vk_Pipeline() noexcept {
    auto device = device_.lock();
    if (!device || device->is_disposed()) return;
    vk::Device dev = device->logical_device();
    for (const auto& stage : stages_) {
        dev.destroyShaderModule(stage.module);
    }
    if (pipeline_) dev.destroyPipeline(pipeline_);
    if (pipeline_layout_) dev.destroyPipelineLayout(pipeline_layout_);
    if (render_pass_) dev.destroyRenderPass(render_pass_);
    if (descriptor_pool_) dev.destroyDescriptorPool(descriptor_pool_);
    for (auto& set_layout : descriptor_set_layouts_) {
        if (set_layout) dev.destroyDescriptorSetLayout(set_layout);
    }
}

void vk_Pipeline::vk_stage(ShaderStageType type, const ShaderSource& source) {
    if (closed_) throw std::runtime_error("Pipeline::stage: pipeline is already closed");
    auto device = device_.lock();
    if (!device) throw std::runtime_error("Pipeline::stage: device has been disposed");
    vk::ShaderModuleCreateInfo info{};
    info.codeSize = source.vk_code().size() * sizeof(std::uint32_t);
    info.pCode = source.vk_code().data();
    vk::ShaderModule module = device->logical_device().createShaderModule(info);
    stages_.push_back({ type, module, source.vk_entry_point() });
}

int vk_Pipeline::vk_layout(int set, int binding, DescriptorType description, int count) {
    if (closed_) throw std::runtime_error("Pipeline::layout: pipeline is already closed");
    if (set < 0 || binding < 0 || count <= 0) throw std::runtime_error("Pipeline::layout: invalid set/binding/count");
    bindings_.push_back({ set, binding, description, count });
    return static_cast<int>(bindings_.size()) - 1;
}

void vk_Pipeline::vk_vertex_layout(int start_location, const Layout& layout) {
    if (closed_) throw std::runtime_error("Pipeline::vertex_layout: pipeline is already closed");
    if (type_ != PipelineType::RASTERIZATION) throw std::runtime_error("Pipeline::vertex_layout: only valid for RASTERIZATION pipelines");
    if (layout.kind != TypeKind::STRUCT) throw std::runtime_error("Pipeline::vertex_layout: layout must describe a struct");

    VertexBinding binding{};
    binding.stride = layout.aligned_size;
    int location = start_location;
    for (const auto& field : layout.fields) {
        const Layout& field_layout = *field.layout;
        if (field_layout.kind != TypeKind::SINGLE) {
            throw std::runtime_error("Pipeline::vertex_layout: fields must be scalar, vector or matrix");
        }
        if (field_layout.element_layout) {
            // Matrix: one vertex attribute location per column.
            const int column_components = static_cast<int>(field_layout.element_layout->size / scalar_type_size(field_layout.component_type));
            const Format fmt = vertex_attribute_format(field_layout.component_type, column_components);
            for (std::uint64_t c = 0; c < field_layout.count; ++c) {
                binding.fields.push_back({ location++, (vk::Format)fmt, field.offset + c * field_layout.stride });
            }
        } else {
            // Scalar or vector.
            const int components = type_traits(field_layout.type).rows;
            const Format fmt = vertex_attribute_format(field_layout.component_type, components);
            binding.fields.push_back({ location++, (vk::Format)fmt, field.offset });
        }
    }
    vertex_bindings_.push_back(std::move(binding));
}

int vk_Pipeline::vk_attach(int slot, Format format) {
    if (closed_) throw std::runtime_error("Pipeline::attach: pipeline is already closed");
    if (type_ != PipelineType::RASTERIZATION) throw std::runtime_error("Pipeline::attach: only valid for RASTERIZATION pipelines");
    for (const auto& attachment : attachments_) {
        if (attachment.slot == slot) throw std::runtime_error("Pipeline::attach: slot already attached");
    }
    attachments_.push_back({ slot, format });
    return slot;
}

void vk_Pipeline::vk_attach_depth(Format format) {
    if (closed_) throw std::runtime_error("Pipeline::attach_depth: pipeline is already closed");
    if (type_ != PipelineType::RASTERIZATION) throw std::runtime_error("Pipeline::attach_depth: only valid for RASTERIZATION pipelines");
    if (!is_depth_format(format)) throw std::runtime_error("Pipeline::attach_depth: format must be one of the Format::Depth* values");
    if (depth_attachment_format_.has_value()) throw std::runtime_error("Pipeline::attach_depth: a depth attachment is already declared");
    auto device = device_.lock();
    if (!device) throw std::runtime_error("Pipeline::attach_depth: device has been disposed");
    if (!device->vk_extended_dynamic_state_supported()) {
        throw std::runtime_error("Pipeline::attach_depth: requires Vulkan 1.3 (extended dynamic state)");
    }
    depth_attachment_format_ = format;
}

void vk_Pipeline::vk_set_local_size(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    if (closed_) throw std::runtime_error("Pipeline::local_size: pipeline is already closed");
    if (type_ != PipelineType::COMPUTE) throw std::runtime_error("Pipeline::local_size: only valid for COMPUTE pipelines");
    if (x == 0 || y == 0 || z == 0) throw std::runtime_error("Pipeline::local_size: dimensions must be positive");
    local_size_x_ = x;
    local_size_y_ = y;
    local_size_z_ = z;
}

void vk_Pipeline::vk_close() {
    if (closed_) throw std::runtime_error("Pipeline::close: already closed");
    auto device = device_.lock();
    if (!device) throw std::runtime_error("Pipeline::close: device has been disposed");
    vk::Device dev = device->logical_device();

    int max_set = -1;
    for (const auto& binding : bindings_) max_set = std::max(max_set, binding.set);
    descriptor_set_layouts_.assign(static_cast<std::size_t>(max_set + 1), vk::DescriptorSetLayout{});
    for (int set = 0; set <= max_set; ++set) {
        std::vector<vk::DescriptorSetLayoutBinding> vk_bindings;
        for (const auto& binding : bindings_) {
            if (binding.set != set) continue;
            vk::DescriptorSetLayoutBinding vk_binding{};
            vk_binding.binding = static_cast<std::uint32_t>(binding.binding);
            vk_binding.descriptorType = to_vk_descriptor_type(binding.type);
            vk_binding.descriptorCount = static_cast<std::uint32_t>(binding.count);
            vk_binding.stageFlags = vk::ShaderStageFlagBits::eAll;
            vk_bindings.push_back(vk_binding);
        }
        vk::DescriptorSetLayoutCreateInfo layout_info{};
        layout_info.bindingCount = static_cast<std::uint32_t>(vk_bindings.size());
        layout_info.pBindings = vk_bindings.data();
        descriptor_set_layouts_[static_cast<std::size_t>(set)] = dev.createDescriptorSetLayout(layout_info);
    }

    vk::PipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.setLayoutCount = static_cast<std::uint32_t>(descriptor_set_layouts_.size());
    pipeline_layout_info.pSetLayouts = descriptor_set_layouts_.data();
    pipeline_layout_ = dev.createPipelineLayout(pipeline_layout_info);

    if (max_set >= 0) {
        // Sized to allow a handful of independently-allocated descriptor sets
        // per declared set index (e.g. for multi-buffering); not a hard
        // architectural limit, just a pragmatic default pool size.
        constexpr std::uint32_t kMaxSetInstances = 16;
        std::unordered_map<VkDescriptorType, std::uint32_t> counts;
        for (const auto& binding : bindings_) {
            counts[static_cast<VkDescriptorType>(to_vk_descriptor_type(binding.type))] += static_cast<std::uint32_t>(binding.count) * kMaxSetInstances;
        }
        std::vector<vk::DescriptorPoolSize> pool_sizes;
        pool_sizes.reserve(counts.size());
        for (auto& [vk_type, count] : counts) {
            pool_sizes.push_back({ static_cast<vk::DescriptorType>(vk_type), count });
        }
        vk::DescriptorPoolCreateInfo pool_info{};
        pool_info.maxSets = static_cast<std::uint32_t>(descriptor_set_layouts_.size()) * kMaxSetInstances;
        pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        descriptor_pool_ = dev.createDescriptorPool(pool_info);
    }

    std::vector<vk::PipelineShaderStageCreateInfo> stage_infos;
    stage_infos.reserve(stages_.size());
    for (const auto& stage : stages_) {
        vk::PipelineShaderStageCreateInfo info{};
        info.stage = to_vk_shader_stage(stage.type);
        info.module = stage.module;
        info.pName = stage.entry_point.c_str();
        stage_infos.push_back(info);
    }

    if (type_ == PipelineType::COMPUTE) {
        if (stage_infos.size() != 1 || stages_[0].type != ShaderStageType::COMPUTE) {
            throw std::runtime_error("Pipeline::close: a COMPUTE pipeline requires exactly one COMPUTE stage");
        }
        vk::ComputePipelineCreateInfo info{};
        info.stage = stage_infos[0];
        info.layout = pipeline_layout_;
        auto result = dev.createComputePipeline(nullptr, info);
        if (result.result != vk::Result::eSuccess) throw std::runtime_error("Pipeline::close: failed to create compute pipeline");
        pipeline_ = result.value;
    } else if (type_ == PipelineType::RASTERIZATION) {
        std::vector<vk::AttachmentDescription> attachment_descs;
        std::vector<vk::AttachmentReference> attachment_refs;
        attachment_descs.reserve(attachments_.size());
        attachment_refs.reserve(attachments_.size());
        for (std::size_t i = 0; i < attachments_.size(); ++i) {
            vk::AttachmentDescription desc{};
            desc.format = (vk::Format)attachments_[i].format;
            desc.samples = vk::SampleCountFlagBits::e1;
            desc.loadOp = vk::AttachmentLoadOp::eClear;
            desc.storeOp = vk::AttachmentStoreOp::eStore;
            desc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            desc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            desc.initialLayout = vk::ImageLayout::eUndefined;
            // The subpass itself still writes through eColorAttachmentOptimal
            // (attachment_refs, below), but the render pass's automatic
            // final transition targets eGeneral -- the one layout every
            // other use site of an Image assumes uniformly (descriptor
            // binds, blit_image; see Image's docstring), so a rendered-to
            // image stays consistently usable afterwards without any
            // per-image layout tracking.
            desc.finalLayout = vk::ImageLayout::eGeneral;
            attachment_descs.push_back(desc);
            attachment_refs.push_back({ static_cast<std::uint32_t>(i), vk::ImageLayout::eColorAttachmentOptimal });
        }

        // Depth/stencil attachment, if attach_depth() was called: appended
        // after every color attachment, so its render-pass attachment index
        // is always attachment_descs.size() at this point -- Pipeline::
        // create_framebuffer() and CommandBuffer::set_framebuffer() rely on
        // this same "depth is last" ordering.
        vk::AttachmentReference depth_ref{};
        if (depth_attachment_format_) {
            vk::AttachmentDescription desc{};
            desc.format = (vk::Format)*depth_attachment_format_;
            desc.samples = vk::SampleCountFlagBits::e1;
            desc.loadOp = vk::AttachmentLoadOp::eClear;
            desc.storeOp = vk::AttachmentStoreOp::eStore;
            desc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            desc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            desc.initialLayout = vk::ImageLayout::eUndefined;
            desc.finalLayout = vk::ImageLayout::eGeneral;
            depth_ref = { static_cast<std::uint32_t>(attachment_descs.size()), vk::ImageLayout::eDepthStencilAttachmentOptimal };
            attachment_descs.push_back(desc);
        }

        vk::SubpassDescription subpass{};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = static_cast<std::uint32_t>(attachment_refs.size());
        subpass.pColorAttachments = attachment_refs.data();
        if (depth_attachment_format_) subpass.pDepthStencilAttachment = &depth_ref;

        vk::RenderPassCreateInfo render_pass_info{};
        render_pass_info.attachmentCount = static_cast<std::uint32_t>(attachment_descs.size());
        render_pass_info.pAttachments = attachment_descs.data();
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;
        render_pass_ = dev.createRenderPass(render_pass_info);

        std::vector<vk::VertexInputBindingDescription> binding_descs;
        std::vector<vk::VertexInputAttributeDescription> attribute_descs;
        for (std::size_t b = 0; b < vertex_bindings_.size(); ++b) {
            binding_descs.push_back({
                static_cast<std::uint32_t>(b),
                static_cast<std::uint32_t>(vertex_bindings_[b].stride),
                vk::VertexInputRate::eVertex
            });
            for (const auto& field : vertex_bindings_[b].fields) {
                attribute_descs.push_back({
                    static_cast<std::uint32_t>(field.location),
                    static_cast<std::uint32_t>(b),
                    field.format,
                    static_cast<std::uint32_t>(field.offset)
                });
            }
        }
        vk::PipelineVertexInputStateCreateInfo vertex_input_info{};
        vertex_input_info.vertexBindingDescriptionCount = static_cast<std::uint32_t>(binding_descs.size());
        vertex_input_info.pVertexBindingDescriptions = binding_descs.data();
        vertex_input_info.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attribute_descs.size());
        vertex_input_info.pVertexAttributeDescriptions = attribute_descs.data();

        vk::PipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.topology = vk::PrimitiveTopology::eTriangleList;

        vk::PipelineViewportStateCreateInfo viewport_state{};
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        vk::PipelineRasterizationStateCreateInfo rasterization{};
        rasterization.polygonMode = vk::PolygonMode::eFill;
        rasterization.cullMode = vk::CullModeFlagBits::eNone;
        rasterization.frontFace = vk::FrontFace::eCounterClockwise;
        rasterization.lineWidth = 1.0f;

        vk::PipelineMultisampleStateCreateInfo multisample{};
        multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

        std::vector<vk::PipelineColorBlendAttachmentState> blend_attachments(attachments_.size());
        for (auto& blend : blend_attachments) {
            blend.blendEnable = VK_FALSE;
            blend.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        }
        vk::PipelineColorBlendStateCreateInfo color_blend{};
        color_blend.attachmentCount = static_cast<std::uint32_t>(blend_attachments.size());
        color_blend.pAttachments = blend_attachments.data();

        // Depth test enable/write/compare-op are never baked into the
        // pipeline -- only ever set dynamically, via
        // CommandBuffer::set_depth_test() -- so the values here are just
        // placeholders, overridden at draw time. depthBoundsTestEnable/
        // stencilTestEnable stay off: this project doesn't implement depth
        // bounds or stencil testing.
        vk::PipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.depthTestEnable = VK_TRUE;
        depth_stencil.depthWriteEnable = VK_TRUE;
        depth_stencil.depthCompareOp = vk::CompareOp::eLess;
        depth_stencil.minDepthBounds = 0.0f;
        depth_stencil.maxDepthBounds = 1.0f;

        // Extended dynamic state (cull mode/front face always, when
        // supported; depth test/write/compare op additionally when a depth
        // attachment was declared) -- see
        // Device::vk_extended_dynamic_state_supported()'s comment.
        const bool extended_dynamic_state = device->vk_extended_dynamic_state_supported();
        std::vector<vk::DynamicState> dynamic_states{ vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        if (extended_dynamic_state) {
            dynamic_states.push_back(vk::DynamicState::eCullMode);
            dynamic_states.push_back(vk::DynamicState::eFrontFace);
            if (depth_attachment_format_) {
                dynamic_states.push_back(vk::DynamicState::eDepthTestEnable);
                dynamic_states.push_back(vk::DynamicState::eDepthWriteEnable);
                dynamic_states.push_back(vk::DynamicState::eDepthCompareOp);
            }
        }
        vk::PipelineDynamicStateCreateInfo dynamic_state{};
        dynamic_state.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
        dynamic_state.pDynamicStates = dynamic_states.data();

        vk::GraphicsPipelineCreateInfo info{};
        info.stageCount = static_cast<std::uint32_t>(stage_infos.size());
        info.pStages = stage_infos.data();
        info.pVertexInputState = &vertex_input_info;
        info.pInputAssemblyState = &input_assembly;
        info.pViewportState = &viewport_state;
        info.pRasterizationState = &rasterization;
        info.pMultisampleState = &multisample;
        info.pColorBlendState = &color_blend;
        info.pDepthStencilState = depth_attachment_format_ ? &depth_stencil : nullptr;
        info.pDynamicState = &dynamic_state;
        info.layout = pipeline_layout_;
        info.renderPass = render_pass_;
        info.subpass = 0;

        auto result = dev.createGraphicsPipeline(nullptr, info);
        if (result.result != vk::Result::eSuccess) throw std::runtime_error("Pipeline::close: failed to create graphics pipeline");
        pipeline_ = result.value;
    } else {
        throw std::runtime_error("Pipeline::close: RAYTRACING pipelines are not yet supported");
    }

    for (const auto& stage : stages_) dev.destroyShaderModule(stage.module);
    stages_.clear();

    closed_ = true;
}

vk::DescriptorSet vk_Pipeline::vk_allocate_descriptor_set(int set) {
    if (!closed_) throw std::runtime_error("Pipeline::create_descriptor_set: pipeline must be closed first");
    if (set < 0 || set >= static_cast<int>(descriptor_set_layouts_.size())) {
        throw std::runtime_error("Pipeline::create_descriptor_set: invalid set index");
    }
    auto device = device_.lock();
    if (!device) throw std::runtime_error("Pipeline::create_descriptor_set: device has been disposed");
    vk::DescriptorSetAllocateInfo alloc_info{};
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layouts_[static_cast<std::size_t>(set)];
    return device->logical_device().allocateDescriptorSets(alloc_info)[0];
}

const vk_DescriptorBinding& vk_Pipeline::vk_binding(int layout_id) const {
    if (layout_id < 0 || layout_id >= static_cast<int>(bindings_.size())) {
        throw std::runtime_error("Invalid layout_id");
    }
    return bindings_[static_cast<std::size_t>(layout_id)];
}

std::shared_ptr<Framebuffer> Pipeline::create_framebuffer(
    std::vector<std::pair<AttachHandle, std::shared_ptr<Image>>> attachments, std::shared_ptr<Image> depth_image) {
    if (!pipeline_->is_closed()) throw std::runtime_error("Pipeline::create_framebuffer: pipeline must be closed first");
    if (pipeline_->type() != PipelineType::RASTERIZATION) throw std::runtime_error("Pipeline::create_framebuffer: only valid for RASTERIZATION pipelines");
    if (pipeline_->vk_has_depth_attachment() != (depth_image != nullptr)) {
        throw std::runtime_error(
            pipeline_->vk_has_depth_attachment()
                ? "Pipeline::create_framebuffer: this pipeline declared a depth attachment (attach_depth()); depth_image is required"
                : "Pipeline::create_framebuffer: this pipeline declared no depth attachment; depth_image must not be provided");
    }

    const auto& attachment_descs = pipeline_->vk_attachments();
    const bool has_depth = depth_image != nullptr;
    std::vector<vk::ImageView> views(attachment_descs.size() + (has_depth ? 1 : 0));
    std::vector<bool> filled(attachment_descs.size(), false);
    std::uint32_t width = 0, height = 0;

    for (auto& [slot, image] : attachments) {
        int index = -1;
        for (std::size_t i = 0; i < attachment_descs.size(); ++i) {
            if (attachment_descs[i].slot == slot.vk_slot()) { index = static_cast<int>(i); break; }
        }
        if (index < 0) throw std::runtime_error("Pipeline::create_framebuffer: no attachment declared for this slot");
        if (image->format() != attachment_descs[static_cast<std::size_t>(index)].format) {
            throw std::runtime_error("Pipeline::create_framebuffer: image format does not match the attachment's declared format");
        }
        const vk::Extent3D extent = image->vk_image_info().extent;
        if (width == 0) {
            width = extent.width;
            height = extent.height;
        } else if (width != extent.width || height != extent.height) {
            throw std::runtime_error("Pipeline::create_framebuffer: attachment images have mismatched dimensions");
        }
        views[static_cast<std::size_t>(index)] = image->get_view();
        filled[static_cast<std::size_t>(index)] = true;
    }
    for (bool f : filled) {
        if (!f) throw std::runtime_error("Pipeline::create_framebuffer: missing an image for one of the declared attachments");
    }

    if (has_depth) {
        if (depth_image->format() != pipeline_->vk_depth_attachment_format()) {
            throw std::runtime_error("Pipeline::create_framebuffer: depth_image format does not match attach_depth()'s declared format");
        }
        const vk::Extent3D extent = depth_image->vk_image_info().extent;
        if (width != 0 && (width != extent.width || height != extent.height)) {
            throw std::runtime_error("Pipeline::create_framebuffer: depth_image dimensions do not match the color attachments");
        }
        width = extent.width;
        height = extent.height;
        // Always last -- see vk_Pipeline::vk_close()'s comment on depth
        // attachment ordering.
        views.back() = depth_image->get_view();
    }

    vk::FramebufferCreateInfo info{};
    info.renderPass = pipeline_->vk_render_pass();
    info.attachmentCount = static_cast<std::uint32_t>(views.size());
    info.pAttachments = views.data();
    info.width = width;
    info.height = height;
    info.layers = 1;

    vk::Framebuffer framebuffer = device_->logical_device().createFramebuffer(info);
    auto vk_fb = std::make_shared<vk_Framebuffer>(
        device_, framebuffer, pipeline_->vk_render_pass(), static_cast<std::uint32_t>(views.size()), width, height, has_depth);
    device_->vk_register_framebuffer(vk_fb);
    return std::make_shared<Framebuffer>(std::move(vk_fb));
}

std::shared_ptr<DescriptorSet> Pipeline::create_descriptor_set(int set) {
    vk::DescriptorSet ds = pipeline_->vk_allocate_descriptor_set(set);
    auto vk_ds = std::make_shared<vk_DescriptorSet>(device_, pipeline_, set, ds);
    return std::make_shared<DescriptorSet>(std::move(vk_ds));
}

std::uint32_t Pipeline::device_index() const noexcept {
    return device_->device_index();
}

vk_Framebuffer::vk_Framebuffer(std::weak_ptr<Device> device, vk::Framebuffer framebuffer, vk::RenderPass render_pass,
    std::uint32_t attachment_count, std::uint32_t width, std::uint32_t height, bool has_depth) noexcept
    : device_(std::move(device)), framebuffer_(framebuffer), render_pass_(render_pass),
      attachment_count_(attachment_count), width_(width), height_(height), has_depth_(has_depth) {}

std::uint32_t vk_Framebuffer::device_index() const {
    auto device = device_.lock();
    if (!device) throw std::runtime_error("Framebuffer: device is disposed");
    return device->device_index();
}

void vk_Framebuffer::dispose() noexcept {
    if (!framebuffer_) return;
    auto device = device_.lock();
    if (device && !device->is_disposed()) {
        device->logical_device().destroyFramebuffer(framebuffer_);
    }
    framebuffer_ = nullptr;
}

vk_Sampler::vk_Sampler(std::weak_ptr<Device> device, vk::Sampler sampler) noexcept
    : device_(std::move(device)), sampler_(sampler) {}

std::uint32_t vk_Sampler::device_index() const {
    auto device = device_.lock();
    if (!device) throw std::runtime_error("Sampler: device is disposed");
    return device->device_index();
}

void vk_Sampler::dispose() noexcept {
    if (!sampler_) return;
    auto device = device_.lock();
    if (device && !device->is_disposed()) {
        device->logical_device().destroySampler(sampler_);
    }
    sampler_ = nullptr;
}

vk_AccelerationStructure::vk_AccelerationStructure(std::weak_ptr<Device> device, vk::AccelerationStructureKHR handle,
    vk::Buffer storage_buffer, vk::DeviceMemory storage_memory,
    vk::AccelerationStructureTypeKHR type, std::uint64_t build_scratch_size) noexcept
    : device_(std::move(device)), handle_(handle), storage_buffer_(storage_buffer), storage_memory_(storage_memory),
      type_(type), build_scratch_size_(build_scratch_size) {}

void vk_AccelerationStructure::dispose() noexcept {
    if (!handle_) return;
    auto device = device_.lock();
    if (device && !device->is_disposed()) {
        vk::Device dev = device->logical_device();
        dev.destroyAccelerationStructureKHR(handle_, nullptr, device->vk_dynamic_dispatch());
        if (storage_buffer_) dev.destroyBuffer(storage_buffer_);
        if (storage_memory_) dev.freeMemory(storage_memory_);
    }
    handle_ = nullptr;
    storage_buffer_ = nullptr;
    storage_memory_ = nullptr;
}

std::uint64_t vk_AccelerationStructure::vk_device_address() const {
    if (device_address_ != 0) return device_address_;
    auto device = device_.lock();
    if (!device || device->is_disposed() || !handle_) {
        throw std::runtime_error("AccelerationStructure::device_address: acceleration structure is disposed");
    }
    vk::AccelerationStructureDeviceAddressInfoKHR info{};
    info.accelerationStructure = handle_;
    device_address_ = static_cast<std::uint64_t>(
        device->logical_device().getAccelerationStructureAddressKHR(info, device->vk_dynamic_dispatch()));
    return device_address_;
}

std::uint32_t vk_AccelerationStructure::device_index() const {
    auto device = device_.lock();
    if (!device) throw std::runtime_error("AccelerationStructure: device is disposed");
    return device->device_index();
}

namespace {

// Finds a queue family supporting both graphics (needed for the blit/copy/
// barrier commands recorded below) and presentation to `surface`. Throws
// if none exists (exceedingly rare in practice: every desktop driver
// exposes at least one such family).
std::uint32_t find_present_queue_family(vk::PhysicalDevice physical, vk::SurfaceKHR surface) {
    const auto families = physical.getQueueFamilyProperties();
    for (std::uint32_t i = 0; i < families.size(); ++i) {
        const bool graphics = (families[i].queueFlags & vk::QueueFlagBits::eGraphics) == vk::QueueFlagBits::eGraphics;
        if (!graphics) continue;
        if (physical.getSurfaceSupportKHR(i, surface)) {
            return i;
        }
    }
    throw std::runtime_error("Window: no queue family supports both graphics and presentation to this surface");
}

// Records a full image layout transition (both access masks and pipeline
// stages broad enough to be correct, if not maximally tight, for the
// handful of operations Window's pre-recorded command buffers perform).
void record_image_barrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout) {
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
        vk::DependencyFlags{}, {}, {}, barrier);
}

// Reference-counts glfwInit()/glfwTerminate() across however many Windows
// are alive at once, since each successful glfwInit() must be matched by
// exactly one glfwTerminate() (not one per Window).
int g_glfw_ref_count = 0;

void glfw_acquire() {
    if (g_glfw_ref_count == 0) {
        glfwInit();
    }
    ++g_glfw_ref_count;
}

void glfw_release() {
    if (--g_glfw_ref_count == 0) {
        glfwTerminate();
    }
}

} // namespace

vk_Window::vk_Window(std::shared_ptr<Device> device, std::uint32_t width, std::uint32_t height,
    const std::string& title, Format format, std::uint32_t frames_on_the_fly, bool vsync)
    : device_(device), format_(format), frames_on_the_fly_(std::max<std::uint32_t>(frames_on_the_fly, 1)), vsync_(vsync) {
    if (!device->vk_swapchain_supported()) {
        throw std::runtime_error("Window: VK_KHR_swapchain is not supported/enabled on this device");
    }

    glfw_acquire();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow* glfw_window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title.c_str(), nullptr, nullptr);
    if (!glfw_window) {
        glfw_release();
        throw std::runtime_error("Window: glfwCreateWindow failed");
    }
    glfw_window_ = glfw_window;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(static_cast<VkInstance>(device->vk_instance()), glfw_window, nullptr, &surface) != VK_SUCCESS) {
        glfwDestroyWindow(glfw_window);
        glfw_release();
        throw std::runtime_error("Window: glfwCreateWindowSurface failed");
    }
    surface_ = surface;

    present_queue_family_ = find_present_queue_family(device->physical_device(), surface_);
    present_queue_ = device->logical_device().getQueue(present_queue_family_, 0);

    // Must happen before create_swapchain(): it creates imgui_render_pass_,
    // which the first create_swapchain() call needs to build imgui_framebuffers_.
    vk_imgui_init(device);

    create_swapchain();
}

void vk_Window::vk_imgui_init(const std::shared_ptr<Device>& device) {
    vk::Device dev = device->logical_device();

    IMGUI_CHECKVERSION();
    ImGuiContext* ctx = ImGui::CreateContext();
    imgui_context_ = ctx;
    ImGui::SetCurrentContext(ctx);
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // a library shouldn't write files behind the caller's back

    ImGui_ImplGlfw_InitForVulkan(static_cast<GLFWwindow*>(glfw_window_), true);

    // Sized generously for ImGui's own internal descriptor sets (font
    // texture, plus any user textures added later).
    vk::DescriptorPoolSize pool_size{ vk::DescriptorType::eCombinedImageSampler, 64 };
    vk::DescriptorPoolCreateInfo pool_info{};
    pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    pool_info.maxSets = 64;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    imgui_descriptor_pool_ = dev.createDescriptorPool(pool_info);

    // Drawn as a final overlay pass onto the swapchain image: eLoad (not
    // eClear) to preserve whatever the caller already rendered/blitted
    // there; initial/final layout ePresentSrcKHR to match the layout the
    // pre-recorded target-to-render_target command buffers always leave
    // render_target in by the time this pass runs (see vk_present_frame).
    vk::AttachmentDescription attachment{};
    attachment.format = vk::Format::eB8G8R8A8Unorm;
    attachment.samples = vk::SampleCountFlagBits::e1;
    attachment.loadOp = vk::AttachmentLoadOp::eLoad;
    attachment.storeOp = vk::AttachmentStoreOp::eStore;
    attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    attachment.initialLayout = vk::ImageLayout::ePresentSrcKHR;
    attachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference color_ref{ 0, vk::ImageLayout::eColorAttachmentOptimal };
    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;

    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = vk::AccessFlags{};
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo render_pass_info{};
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;
    imgui_render_pass_ = dev.createRenderPass(render_pass_info);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = static_cast<VkInstance>(device->vk_instance());
    init_info.PhysicalDevice = static_cast<VkPhysicalDevice>(device->physical_device());
    init_info.Device = static_cast<VkDevice>(dev);
    init_info.QueueFamily = present_queue_family_;
    init_info.Queue = static_cast<VkQueue>(present_queue_);
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = static_cast<VkDescriptorPool>(imgui_descriptor_pool_);
    init_info.RenderPass = static_cast<VkRenderPass>(imgui_render_pass_);
    init_info.Subpass = 0;
    init_info.MinImageCount = std::max<std::uint32_t>(frames_on_the_fly_, 2);
    init_info.ImageCount = init_info.MinImageCount;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&init_info);

    last_frame_time_point_ = std::chrono::steady_clock::now();
}

void vk_Window::vk_imgui_shutdown(vk::Device dev) noexcept {
    if (!imgui_context_) return;
    ImGui::SetCurrentContext(static_cast<ImGuiContext*>(imgui_context_));
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(static_cast<ImGuiContext*>(imgui_context_));
    imgui_context_ = nullptr;
    if (dev) {
        if (imgui_render_pass_) dev.destroyRenderPass(imgui_render_pass_);
        if (imgui_descriptor_pool_) dev.destroyDescriptorPool(imgui_descriptor_pool_);
    }
    imgui_render_pass_ = nullptr;
    imgui_descriptor_pool_ = nullptr;
}

void vk_Window::vk_imgui_ensure_current() const noexcept {
    if (imgui_context_) ImGui::SetCurrentContext(static_cast<ImGuiContext*>(imgui_context_));
}

void vk_Window::vk_imgui_text(const std::string& text) {
    vk_imgui_ensure_current();
    ImGui::Text("%s", text.c_str());
}

bool vk_Window::vk_imgui_button(const std::string& text) {
    vk_imgui_ensure_current();
    return ImGui::Button(text.c_str());
}

bool vk_Window::vk_imgui_checkbox(const std::string& label, bool& value) {
    vk_imgui_ensure_current();
    return ImGui::Checkbox(label.c_str(), &value);
}

bool vk_Window::vk_imgui_slider_float(const std::string& label, float& value, float min, float max) {
    vk_imgui_ensure_current();
    return ImGui::SliderFloat(label.c_str(), &value, min, max);
}

bool vk_Window::vk_imgui_slider_int(const std::string& label, int& value, int min, int max) {
    vk_imgui_ensure_current();
    return ImGui::SliderInt(label.c_str(), &value, min, max);
}

bool vk_Window::vk_imgui_combobox(const std::string& label, const std::vector<std::string>& items, int& selected_index) {
    vk_imgui_ensure_current();
    bool changed = false;
    const std::string preview = (selected_index >= 0 && static_cast<std::size_t>(selected_index) < items.size())
        ? items[static_cast<std::size_t>(selected_index)] : std::string{};
    if (ImGui::BeginCombo(label.c_str(), preview.c_str())) {
        for (int i = 0; i < static_cast<int>(items.size()); ++i) {
            const bool is_selected = (i == selected_index);
            if (ImGui::Selectable(items[static_cast<std::size_t>(i)].c_str(), is_selected)) {
                selected_index = i;
                changed = true;
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}

std::uint64_t vk_next_widget_id(const std::weak_ptr<vk_Window>& window) {
    auto w = window.lock();
    return w ? w->vk_next_widget_id() : 0;
}

double Stats::fps() const {
    auto w = window_.lock();
    return w ? w->vk_fps() : 0.0;
}

bool Checkbox::draw() {
    auto w = window_.lock();
    if (!w) return false;
    const std::string id_label = label_ + "##cb" + std::to_string(id_);
    return w->vk_imgui_checkbox(id_label, value_);
}

bool SliderFloat::draw() {
    auto w = window_.lock();
    if (!w) return false;
    const std::string id_label = label_ + "##sf" + std::to_string(id_);
    return w->vk_imgui_slider_float(id_label, value_, min_, max_);
}

bool SliderInt::draw() {
    auto w = window_.lock();
    if (!w) return false;
    const std::string id_label = label_ + "##si" + std::to_string(id_);
    return w->vk_imgui_slider_int(id_label, value_, min_, max_);
}

bool Combobox::draw() {
    auto w = window_.lock();
    if (!w) return false;
    const std::string id_label = label_ + "##cx" + std::to_string(id_);
    return w->vk_imgui_combobox(id_label, items_, selected_index_);
}

vk_Window::~vk_Window() noexcept {
    dispose();
}

void vk_Window::dispose() noexcept {
    auto device = device_.lock();
    vk::Device dev = (device && !device->is_disposed()) ? device->logical_device() : vk::Device{};
    vk::Instance instance = (device && !device->is_disposed()) ? device->vk_instance() : vk::Instance{};
    vk_dispose_with(dev, instance);
}

void vk_Window::vk_dispose_with(vk::Device dev, vk::Instance instance) noexcept {
    if (dev) {
        try {
            dev.waitIdle();
        } catch (...) {
            // dispose must not throw.
        }
    }
    destroy_swapchain_resources_with(dev);
    vk_imgui_shutdown(dev);
    if (dev && instance && surface_) {
        instance.destroySurfaceKHR(surface_);
    }
    surface_ = nullptr;
    if (glfw_window_) {
        glfwDestroyWindow(static_cast<GLFWwindow*>(glfw_window_));
        glfw_window_ = nullptr;
        glfw_release();
    }
}

void vk_Window::destroy_swapchain_resources_with(vk::Device dev) noexcept {
    // Must go before slots_.clear() below: each framebuffer wraps a
    // slot's render_target image view, which slots_.clear() may destroy
    // (if that was the view's last reference).
    if (dev) {
        for (auto& fb : imgui_framebuffers_) {
            if (fb) dev.destroyFramebuffer(fb);
        }
    }
    imgui_framebuffers_.clear();
    imgui_command_buffers_.clear(); // freed below, along with every other command buffer in command_pool_

    if (dev) {
        for (auto& sync : sync_groups_) {
            if (sync.image_available_semaphore) dev.destroySemaphore(sync.image_available_semaphore);
            if (sync.content_ready_semaphore) dev.destroySemaphore(sync.content_ready_semaphore);
            if (sync.render_finished_semaphore) dev.destroySemaphore(sync.render_finished_semaphore);
            if (sync.fence) dev.destroyFence(sync.fence);
        }
        if (command_pool_) {
            dev.destroyCommandPool(command_pool_); // also frees every command buffer allocated from it
        }
        if (swapchain_) {
            dev.destroySwapchainKHR(swapchain_);
        }
    }
    sync_groups_.clear();
    slots_.clear();
    swapchain_images_.clear();
    frame_to_image_index_.clear();
    images_in_flight_.clear();
    command_pool_ = nullptr;
    swapchain_ = nullptr;
}

namespace {
// Physical, byte-exact per-channel scalar type and channel count for a
// Format -- unlike format_scalar_type(), which reports the shader-
// conceptual read type (collapsing every 8-bit normalized format to a
// single "float" component). buffer_target/tensor_target need the
// former: their raw bytes are copied directly into image_target via
// vkCmdCopyBufferToImage, and tensor_target's dtype/shape are exposed
// as-is to the caller (e.g. via DLPack).
std::pair<Type, std::uint64_t> physical_pixel_layout(Format format) {
    switch (static_cast<vk::Format>(format)) {
        case vk::Format::eR8Unorm:
        case vk::Format::eR8G8Unorm:
        case vk::Format::eR8G8B8Unorm:
        case vk::Format::eR8G8B8A8Unorm:
        case vk::Format::eB8G8R8A8Unorm:
            return { Type::UINT8, formatSize(format) };
        case vk::Format::eR8Snorm:
        case vk::Format::eR8G8Snorm:
        case vk::Format::eR8G8B8Snorm:
        case vk::Format::eR8G8B8A8Snorm:
            return { Type::INT8, formatSize(format) };
        default: {
            const Type scalar = format_scalar_type(format);
            return { scalar, formatSize(format) / scalar_type_size(scalar) };
        }
    }
}
} // namespace

void vk_Window::create_swapchain() {
    auto device = device_.lock();
    if (!device) throw std::runtime_error("Window: device has been disposed");
    vk::PhysicalDevice physical = device->physical_device();
    vk::Device dev = device->logical_device();

    dev.waitIdle();
    vk::SwapchainKHR old_swapchain = swapchain_;
    // Destroy everything except the swapchain itself (needed as
    // oldSwapchain below, for a resize to hand over ownership cleanly).
    // Framebuffers must go before slots_.clear(), same reasoning as
    // destroy_swapchain_resources_with.
    for (auto& fb : imgui_framebuffers_) {
        if (fb) dev.destroyFramebuffer(fb);
    }
    imgui_framebuffers_.clear();
    imgui_command_buffers_.clear(); // freed below, along with every other command buffer in command_pool_
    for (auto& sync : sync_groups_) {
        if (sync.image_available_semaphore) dev.destroySemaphore(sync.image_available_semaphore);
        if (sync.content_ready_semaphore) dev.destroySemaphore(sync.content_ready_semaphore);
        if (sync.render_finished_semaphore) dev.destroySemaphore(sync.render_finished_semaphore);
        if (sync.fence) dev.destroyFence(sync.fence);
    }
    sync_groups_.clear();
    slots_.clear();
    swapchain_images_.clear();
    images_in_flight_.clear();
    if (command_pool_) {
        dev.destroyCommandPool(command_pool_);
        command_pool_ = nullptr;
    }

    const vk::SurfaceCapabilitiesKHR capabilities = physical.getSurfaceCapabilitiesKHR(surface_);
    const auto surface_formats = physical.getSurfaceFormatsKHR(surface_);
    const auto present_modes = physical.getSurfacePresentModesKHR(surface_);

    // The swapchain itself is always BGRA8_UNorm (the native format on
    // most Windows/Linux drivers); `format_` is only for image_target/
    // buffer_target/tensor_target, which may be any format the caller
    // finds convenient -- blit_image (used to composite them into
    // render_target at present time) converts between formats natively.
    const vk::Format swapchain_format = vk::Format::eB8G8R8A8Unorm;
    bool format_supported = false;
    for (const auto& f : surface_formats) {
        if (f.format == swapchain_format && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            format_supported = true;
            break;
        }
    }
    if (!format_supported) {
        throw std::runtime_error("Window: this surface does not support VK_FORMAT_B8G8R8A8_UNORM (the format every swapchain is created with)");
    }
    vk_format_ = swapchain_format;

    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo; // always supported, vsync'd
    if (!vsync_) {
        // Prefer Mailbox (uncapped, triple-buffered, doesn't tear) over
        // Immediate (uncapped, can tear); silently falls back to the
        // vsync'd Fifo above if this surface supports neither.
        for (const auto& mode : present_modes) {
            if (mode == vk::PresentModeKHR::eMailbox) { present_mode = mode; break; }
        }
        if (present_mode == vk::PresentModeKHR::eFifo) {
            for (const auto& mode : present_modes) {
                if (mode == vk::PresentModeKHR::eImmediate) { present_mode = mode; break; }
            }
        }
    }

    vk::Extent2D extent;
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        extent = capabilities.currentExtent;
    } else {
        int actual_width = 0, actual_height = 0;
        glfwGetFramebufferSize(static_cast<GLFWwindow*>(glfw_window_), &actual_width, &actual_height);
        extent.width = std::clamp(static_cast<std::uint32_t>(actual_width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(static_cast<std::uint32_t>(actual_height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    extent_ = extent;

    std::uint32_t image_count = std::max(frames_on_the_fly_, capabilities.minImageCount);
    if (capabilities.maxImageCount > 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    // eTransferSrc is not strictly needed to render+present, but it's
    // virtually universally supported and lets render_target be read back
    // (e.g. vkCmdCopyImageToBuffer, for a screenshot) or used as a blit
    // source -- cheap to request defensively.
    vk::ImageUsageFlags image_usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
    if (capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferSrc) {
        image_usage |= vk::ImageUsageFlagBits::eTransferSrc;
    }

    vk::SwapchainCreateInfoKHR swapchain_info{};
    swapchain_info.surface = surface_;
    swapchain_info.minImageCount = image_count;
    swapchain_info.imageFormat = vk_format_;
    swapchain_info.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    swapchain_info.imageExtent = extent_;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = image_usage;
    swapchain_info.imageSharingMode = vk::SharingMode::eExclusive;
    swapchain_info.preTransform = capabilities.currentTransform;
    swapchain_info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapchain_info.presentMode = present_mode;
    swapchain_info.clipped = VK_TRUE;
    swapchain_info.oldSwapchain = old_swapchain;

    swapchain_ = dev.createSwapchainKHR(swapchain_info);
    if (old_swapchain) {
        dev.destroySwapchainKHR(old_swapchain);
    }

    swapchain_images_ = dev.getSwapchainImagesKHR(swapchain_);
    const std::uint32_t actual_image_count = static_cast<std::uint32_t>(swapchain_images_.size());
    frames_on_the_fly_ = actual_image_count;

    vk::CommandPoolCreateInfo pool_info{};
    pool_info.queueFamilyIndex = present_queue_family_;
    pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    command_pool_ = dev.createCommandPool(pool_info);

    vk::ImageCreateInfo swapchain_image_info{};
    swapchain_image_info.imageType = vk::ImageType::e2D;
    swapchain_image_info.format = vk_format_;
    swapchain_image_info.extent = vk::Extent3D{ extent_.width, extent_.height, 1 };
    swapchain_image_info.mipLevels = 1;
    swapchain_image_info.arrayLayers = 1;
    swapchain_image_info.samples = vk::SampleCountFlagBits::e1;
    swapchain_image_info.tiling = vk::ImageTiling::eOptimal;
    swapchain_image_info.usage = swapchain_info.imageUsage;
    swapchain_image_info.sharingMode = vk::SharingMode::eExclusive;
    swapchain_image_info.initialLayout = vk::ImageLayout::eUndefined;

    slots_.resize(actual_image_count);
    images_in_flight_.assign(actual_image_count, vk::Fence{});
    for (std::uint32_t i = 0; i < actual_image_count; ++i) {
        auto memory_slice = std::make_shared<MemorySlice>(device, nullptr, 0, 0, 0, 0);
        auto data = std::make_shared<vk_ResourceData>(device, memory_slice, swapchain_images_[i], swapchain_image_info,
            /*dedicated_image_memory=*/nullptr, /*owns_image=*/false);
        ResourceSlice slice{};
        slice.type = ResourceType::IMAGE;
        slice.image.format = Format::BGRA8_UNorm; // the swapchain's real, fixed format
        slice.image.mip_start = 0;
        slice.image.mip_count = 1;
        slice.image.array_start = 0;
        slice.image.array_count = 1;
        slots_[i].render_target = std::make_shared<Image>(data, slice);
        slots_[i].image_target = device->create_image(
            static_cast<int>(extent_.width), static_cast<int>(extent_.height), 1, 1, 1, format_, MemoryLocation::DEVICE);
        const std::uint64_t pixel_count = static_cast<std::uint64_t>(extent_.width) * extent_.height;
        slots_[i].buffer_target = device->create_buffer(pixel_count, format_, MemoryLocation::DEVICE);
        const auto [target_scalar_type, target_channels] = physical_pixel_layout(format_);
        slots_[i].tensor_target = device->create_tensor(
            { extent_.height, extent_.width, target_channels }, target_scalar_type, MemoryLocation::DEVICE);

        vk::CommandBufferAllocateInfo alloc_info{};
        alloc_info.commandPool = command_pool_;
        alloc_info.level = vk::CommandBufferLevel::ePrimary;
        alloc_info.commandBufferCount = 4;
        auto cmds = dev.allocateCommandBuffers(alloc_info);
        slots_[i].cmd_from_render_target = cmds[0];
        slots_[i].cmd_from_image_target = cmds[1];
        slots_[i].cmd_from_buffer_target = cmds[2];
        slots_[i].cmd_from_tensor_target = cmds[3];

        record_command_buffers(i);
    }

    // ImGui overlay framebuffer: one per slot, wrapping the same
    // render_target view every other command buffer targets, so the
    // overlay pass draws on top of whatever content is already there.
    imgui_framebuffers_.resize(actual_image_count);
    for (std::uint32_t i = 0; i < actual_image_count; ++i) {
        vk::ImageView view = slots_[i].render_target->get_view();
        vk::FramebufferCreateInfo fb_info{};
        fb_info.renderPass = imgui_render_pass_;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments = &view;
        fb_info.width = extent_.width;
        fb_info.height = extent_.height;
        fb_info.layers = 1;
        imgui_framebuffers_[i] = dev.createFramebuffer(fb_info);
    }
    ImGui_ImplVulkan_SetMinImageCount(std::max<std::uint32_t>(frames_on_the_fly_, 2));

    sync_groups_.resize(frames_on_the_fly_);
    for (auto& sync : sync_groups_) {
        sync.image_available_semaphore = dev.createSemaphore(vk::SemaphoreCreateInfo{});
        sync.content_ready_semaphore = dev.createSemaphore(vk::SemaphoreCreateInfo{});
        sync.render_finished_semaphore = dev.createSemaphore(vk::SemaphoreCreateInfo{});
        sync.fence = dev.createFence(vk::FenceCreateInfo{});
        sync.fence_submitted = false;
    }
    imgui_command_buffers_.resize(frames_on_the_fly_);
    {
        vk::CommandBufferAllocateInfo alloc_info{};
        alloc_info.commandPool = command_pool_;
        alloc_info.level = vk::CommandBufferLevel::ePrimary;
        alloc_info.commandBufferCount = frames_on_the_fly_;
        auto cmds = dev.allocateCommandBuffers(alloc_info);
        for (std::uint32_t i = 0; i < frames_on_the_fly_; ++i) {
            imgui_command_buffers_[i] = cmds[i];
        }
    }
    frame_to_image_index_.assign(frames_on_the_fly_, 0);
}

void vk_Window::record_command_buffers(std::size_t slot_index) {
    auto& slot = slots_[slot_index];
    const vk::Image render_target_image = slot.render_target->vk_image();
    const vk::Image image_target_image = slot.image_target->vk_image();

    // render_target used directly: the caller's own render pass already
    // left it in eGeneral (initialLayout=eUndefined, finalLayout=eGeneral --
    // see vk_Pipeline::vk_close), regardless of the image's true state
    // beforehand, so only the final present transition is needed here.
    slot.cmd_from_render_target.begin(vk::CommandBufferBeginInfo{});
    record_image_barrier(slot.cmd_from_render_target, render_target_image, vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);
    slot.cmd_from_render_target.end();

    // image_target: blit it into render_target, then present. render_target
    // is transitioned from eUndefined (a discard -- valid regardless of its
    // true prior layout, e.g. ePresentSrcKHR from the last time this slot
    // was presented) since the caller never wrote to it directly.
    slot.cmd_from_image_target.begin(vk::CommandBufferBeginInfo{});
    record_image_barrier(slot.cmd_from_image_target, render_target_image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    {
        vk::ImageBlit region{};
        region.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.srcSubresource.layerCount = 1;
        region.srcOffsets[1] = vk::Offset3D{ static_cast<std::int32_t>(extent_.width), static_cast<std::int32_t>(extent_.height), 1 };
        region.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.dstSubresource.layerCount = 1;
        region.dstOffsets[1] = vk::Offset3D{ static_cast<std::int32_t>(extent_.width), static_cast<std::int32_t>(extent_.height), 1 };
        slot.cmd_from_image_target.blitImage(
            image_target_image, vk::ImageLayout::eGeneral,
            render_target_image, vk::ImageLayout::eGeneral,
            region, vk::Filter::eNearest);
    }
    record_image_barrier(slot.cmd_from_image_target, render_target_image, vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);
    slot.cmd_from_image_target.end();

    // buffer_target/tensor_target: upload into image_target (already
    // eGeneral since Device::create_image leaves every image there), then
    // exactly the image_target sequence above.
    auto record_from_linear_source = [&](vk::CommandBuffer cmd, vk::Buffer src_buffer, std::uint64_t src_offset) {
        cmd.begin(vk::CommandBufferBeginInfo{});
        record_image_barrier(cmd, render_target_image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        vk::BufferImageCopy copy{};
        copy.bufferOffset = src_offset;
        copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = vk::Extent3D{ extent_.width, extent_.height, 1 };
        cmd.copyBufferToImage(src_buffer, image_target_image, vk::ImageLayout::eGeneral, copy);
        vk::ImageBlit region{};
        region.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.srcSubresource.layerCount = 1;
        region.srcOffsets[1] = vk::Offset3D{ static_cast<std::int32_t>(extent_.width), static_cast<std::int32_t>(extent_.height), 1 };
        region.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.dstSubresource.layerCount = 1;
        region.dstOffsets[1] = vk::Offset3D{ static_cast<std::int32_t>(extent_.width), static_cast<std::int32_t>(extent_.height), 1 };
        cmd.blitImage(
            image_target_image, vk::ImageLayout::eGeneral,
            render_target_image, vk::ImageLayout::eGeneral,
            region, vk::Filter::eNearest);
        record_image_barrier(cmd, render_target_image, vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);
        cmd.end();
    };
    record_from_linear_source(slot.cmd_from_buffer_target, slot.buffer_target->vk_buffer(), slot.buffer_target->vk_buffer_offset());
    record_from_linear_source(slot.cmd_from_tensor_target, slot.tensor_target->vk_buffer(), slot.tensor_target->vk_buffer_offset());
}

bool vk_Window::check_alive() {
    glfwPollEvents();
    if (glfwWindowShouldClose(static_cast<GLFWwindow*>(glfw_window_))) {
        alive_ = false;
    }
    return alive_;
}

std::uint32_t vk_Window::device_index() const {
    auto device = device_.lock();
    if (!device) throw std::runtime_error("Window: device is disposed");
    return device->device_index();
}

void vk_Window::set_title(const std::string& title) {
    glfwSetWindowTitle(static_cast<GLFWwindow*>(glfw_window_), title.c_str());
}

void vk_Window::set_size(std::uint32_t width, std::uint32_t height) {
    glfwSetWindowSize(static_cast<GLFWwindow*>(glfw_window_), static_cast<int>(width), static_cast<int>(height));
    create_swapchain();
}

vk_Window::FrameResources vk_Window::vk_begin_frame() {
    auto device = device_.lock();
    if (!device) throw std::runtime_error("Window::begin_frame: device has been disposed");
    vk::Device dev = device->logical_device();

    const std::uint32_t sync_index = static_cast<std::uint32_t>(current_frame_index_ % frames_on_the_fly_);
    auto& sync = sync_groups_[sync_index];
    if (sync.fence_submitted) {
        auto wait_result = dev.waitForFences(sync.fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
        (void)wait_result;
        dev.resetFences(sync.fence);
    }

    std::uint32_t image_index = 0;
    VkResult result = vkAcquireNextImageKHR(
        static_cast<VkDevice>(dev), static_cast<VkSwapchainKHR>(swapchain_),
        std::numeric_limits<std::uint64_t>::max(),
        static_cast<VkSemaphore>(sync.image_available_semaphore), VK_NULL_HANDLE, &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        create_swapchain();
        auto& sync2 = sync_groups_[sync_index];
        result = vkAcquireNextImageKHR(
            static_cast<VkDevice>(dev), static_cast<VkSwapchainKHR>(swapchain_),
            std::numeric_limits<std::uint64_t>::max(),
            static_cast<VkSemaphore>(sync2.image_available_semaphore), VK_NULL_HANDLE, &image_index);
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Window::begin_frame: failed to acquire a swapchain image");
    }

    frame_to_image_index_[sync_index] = image_index;

    // vkAcquireNextImageKHR does not guarantee round-robin order, so this
    // image may have last been used under a *different* sync group --
    // whose fence (not this one) is what actually guards its command
    // buffers/ImGui framebuffer. Wait for that specific fence (unless
    // it's already this sync group's, e.g. on a stable round-robin
    // driver, in which case it was already waited on above) before
    // reusing anything belonging to this image. See images_in_flight_'s
    // docstring.
    if (images_in_flight_[image_index] && images_in_flight_[image_index] != sync.fence) {
        auto wait_result = dev.waitForFences(images_in_flight_[image_index], VK_TRUE, std::numeric_limits<std::uint64_t>::max());
        (void)wait_result;
    }
    images_in_flight_[image_index] = sync.fence;

    const auto& slot = slots_[image_index];

    // Starts this frame's ImGui immediate-mode block: label()/button()/
    // checkbox()/slider_*()/combobox() calls made between now and the
    // matching present() land in this frame's draw data (see
    // vk_present_frame, which calls ImGui::Render() and submits it).
    vk_imgui_ensure_current();
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last_frame_time_point_).count();
    last_frame_time_point_ = now;
    if (dt > 0.0) {
        const double instantaneous_fps = 1.0 / dt;
        // Exponential moving average: a single frame's instantaneous fps
        // is too noisy to display directly.
        fps_ = (fps_ == 0.0) ? instantaneous_fps : (fps_ * 0.9 + instantaneous_fps * 0.1);
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    FrameResources resources{};
    resources.render_target = slot.render_target;
    resources.image_target = slot.image_target;
    resources.buffer_target = slot.buffer_target;
    resources.tensor_target = slot.tensor_target;
    resources.index = current_frame_index_;
    ++current_frame_index_;
    return resources;
}

void vk_Window::vk_present_frame(std::int64_t index,
    const std::shared_ptr<Image>& render_target, const std::shared_ptr<Image>& image_target,
    const std::shared_ptr<Buffer>& buffer_target, const std::shared_ptr<Tensor>& tensor_target) {
    if (index != last_enqueue_frame_index_ + 1) {
        throw std::runtime_error("Window::present: frames must be presented in order");
    }
    last_enqueue_frame_index_ = index;

    auto device = device_.lock();
    if (!device) throw std::runtime_error("Window::present: device has been disposed");
    vk::Device dev = device->logical_device();

    const std::uint32_t sync_index = static_cast<std::uint32_t>(index % frames_on_the_fly_);
    const std::uint32_t image_index = frame_to_image_index_[sync_index];
    auto& sync = sync_groups_[sync_index];
    const auto& slot = slots_[image_index];

    vk::CommandBuffer cmd;
    if (buffer_target) {
        cmd = slot.cmd_from_buffer_target;
    } else if (tensor_target) {
        cmd = slot.cmd_from_tensor_target;
    } else if (image_target) {
        cmd = slot.cmd_from_image_target;
    } else {
        cmd = slot.cmd_from_render_target;
    }

    // Wait, GPU-side, for any Vulkan work submitted since begin_frame()
    // (the caller's own rendering into whichever target they picked) --
    // see Device::vk_interop_semaphore's docstring: this is a real
    // cross-submission wait, not just "the host thinks it's done".
    const bool has_interop_wait = device->vk_interop_semaphore() != nullptr;
    const std::uint64_t interop_wait_value = device->vk_interop_semaphore_value();

    std::array<vk::Semaphore, 2> wait_semaphores{ sync.image_available_semaphore, nullptr };
    std::array<std::uint64_t, 2> wait_values{ 0, 0 };
    std::array<vk::PipelineStageFlags, 2> wait_stages{ vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eAllCommands };
    std::uint32_t wait_count = 1;
    if (has_interop_wait) {
        wait_semaphores[1] = device->vk_interop_semaphore();
        wait_values[1] = interop_wait_value;
        wait_count = 2;
    }

    vk::TimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.waitSemaphoreValueCount = wait_count;
    timeline_info.pWaitSemaphoreValues = wait_values.data();

    // Signals content_ready_semaphore (not render_finished_semaphore
    // directly): the ImGui overlay submission below runs after this one,
    // drawing on top of whatever this command buffer just produced, and
    // is what actually signals render_finished_semaphore/the fence --
    // see this function's docstring.
    vk::SubmitInfo submit_info{};
    submit_info.waitSemaphoreCount = wait_count;
    submit_info.pWaitSemaphores = wait_semaphores.data();
    submit_info.pWaitDstStageMask = wait_stages.data();
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &sync.content_ready_semaphore;
    submit_info.setPNext(&timeline_info);

    present_queue_.submit(submit_info, nullptr);

    // ImGui's draw data is inherently per-frame, so unlike `cmd` above
    // (pre-recorded once at swapchain-creation time), this command buffer
    // is recorded fresh every present() call.
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    vk::CommandBuffer imgui_cmd = imgui_command_buffers_[sync_index];
    imgui_cmd.reset();
    imgui_cmd.begin(vk::CommandBufferBeginInfo{});
    vk::RenderPassBeginInfo rp_begin{};
    rp_begin.renderPass = imgui_render_pass_;
    rp_begin.framebuffer = imgui_framebuffers_[image_index];
    rp_begin.renderArea = vk::Rect2D{ { 0, 0 }, extent_ };
    imgui_cmd.beginRenderPass(rp_begin, vk::SubpassContents::eInline);
    ImGui_ImplVulkan_RenderDrawData(draw_data, static_cast<VkCommandBuffer>(imgui_cmd));
    imgui_cmd.endRenderPass();
    imgui_cmd.end();

    vk::PipelineStageFlags imgui_wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo imgui_submit_info{};
    imgui_submit_info.waitSemaphoreCount = 1;
    imgui_submit_info.pWaitSemaphores = &sync.content_ready_semaphore;
    imgui_submit_info.pWaitDstStageMask = &imgui_wait_stage;
    imgui_submit_info.commandBufferCount = 1;
    imgui_submit_info.pCommandBuffers = &imgui_cmd;
    imgui_submit_info.signalSemaphoreCount = 1;
    imgui_submit_info.pSignalSemaphores = &sync.render_finished_semaphore;

    present_queue_.submit(imgui_submit_info, sync.fence);
    sync.fence_submitted = true;

    vk::PresentInfoKHR present_info{};
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &sync.render_finished_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_;
    present_info.pImageIndices = &image_index;

    const VkResult present_result = vkQueuePresentKHR(static_cast<VkQueue>(present_queue_), reinterpret_cast<const VkPresentInfoKHR*>(&present_info));
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
        create_swapchain();
    } else if (present_result != VK_SUCCESS) {
        throw std::runtime_error("Window::present: vkQueuePresentKHR failed");
    }
}

Frame::Frame(std::shared_ptr<Window> window, std::shared_ptr<Image> render_target, std::shared_ptr<Image> image_target,
    std::shared_ptr<Buffer> buffer_target, std::shared_ptr<Tensor> tensor_target, std::int64_t index) noexcept :
window_(std::move(window)),
render_target_(std::move(render_target)),
image_target_(std::move(image_target)),
buffer_target_(std::move(buffer_target)),
tensor_target_(std::move(tensor_target)),
index_(index)
{
}

void Frame::present() {
    if (presented())
        return;
    if (!is_target_acquired()) {
        throw std::runtime_error("Frame::present: cannot present a frame that has not acquired any resource.");
    }
    window_->vk_present_frame(index_, render_target_, image_target_, buffer_target_, tensor_target_);
    window_.reset();
    render_target_.reset();
    image_target_.reset();
    buffer_target_.reset();
    tensor_target_.reset();
}

Window::Window(std::shared_ptr<Device> device, std::uint32_t width, std::uint32_t height, const std::string& title,
    Format format, std::uint32_t frames_on_the_fly, bool vsync)
    : window_(std::make_shared<vk_Window>(std::move(device), width, height, title, format, frames_on_the_fly, vsync)) {
}

std::shared_ptr<Frame> Window::begin_frame() {
    auto resources = window_->vk_begin_frame();
    return std::make_shared<Frame>(shared_from_this(), resources.render_target, resources.image_target,
        resources.buffer_target, resources.tensor_target, resources.index);
}

std::shared_ptr<Window> Device::create_window(std::uint32_t width, std::uint32_t height, const std::string& title,
    Format format, std::uint32_t frames_on_the_fly, bool vsync) {
    auto window = std::make_shared<Window>(shared_from_this(), width, height, title, format, frames_on_the_fly, vsync);
    vk_register_window(window->vk_window());
    return window;
}

vk_DescriptorSet::vk_DescriptorSet(std::weak_ptr<Device> device, std::shared_ptr<vk_Pipeline> pipeline, int set, vk::DescriptorSet descriptor_set) noexcept
    : device_(std::move(device)), pipeline_(std::move(pipeline)), set_(set), descriptor_set_(descriptor_set) {}

std::uint32_t vk_DescriptorSet::device_index() const {
    auto device = device_.lock();
    if (!device) throw std::runtime_error("DescriptorSet: device is disposed");
    return device->device_index();
}

void vk_DescriptorSet::vk_bind_buffer(LayoutHandle layout_id, const std::shared_ptr<Buffer>& buffer) {
    const auto& binding = pipeline_->vk_binding(layout_id.vk_id());
    if (binding.set != set_) throw std::runtime_error("DescriptorSet::bind: layout_id does not belong to this descriptor set");
    if (binding.type != DescriptorType::STORAGE_BUFFER && binding.type != DescriptorType::UNIFORM_BUFFER) {
        throw std::runtime_error("DescriptorSet::bind: this binding does not expect a buffer");
    }
    auto device = device_.lock();
    if (!device) throw std::runtime_error("DescriptorSet::bind: device has been disposed");

    vk::DescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffer->vk_buffer();
    buffer_info.offset = buffer->vk_buffer_offset();
    buffer_info.range = buffer->vk_buffer_size();

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = static_cast<std::uint32_t>(binding.binding);
    write.descriptorCount = 1;
    write.descriptorType = to_vk_descriptor_type(binding.type);
    write.pBufferInfo = &buffer_info;

    device->logical_device().updateDescriptorSets(1, &write, 0, nullptr);
}

void vk_DescriptorSet::vk_bind_image(LayoutHandle layout_id, const std::shared_ptr<Image>& image) {
    const auto& binding = pipeline_->vk_binding(layout_id.vk_id());
    if (binding.set != set_) throw std::runtime_error("DescriptorSet::bind: layout_id does not belong to this descriptor set");
    if (binding.type != DescriptorType::STORAGE_IMAGE && binding.type != DescriptorType::SAMPLED_IMAGE) {
        throw std::runtime_error(
            "DescriptorSet::bind: this binding's declared type requires a sampler; "
            "call bind(layout_id, sampler) for a SAMPLER binding, or bind(layout_id, image, sampler) for COMBINED_IMAGE_SAMPLER");
    }
    auto device = device_.lock();
    if (!device) throw std::runtime_error("DescriptorSet::bind: device has been disposed");

    vk::DescriptorImageInfo image_info{};
    image_info.imageView = image->get_view();
    // No layout-tracking exists yet for images (Image has no method to record
    // or transition its current vk::ImageLayout), so eGeneral is used here as
    // the one layout valid for both storage-image and sampled-image access
    // without requiring a prior explicit transition.
    image_info.imageLayout = vk::ImageLayout::eGeneral;

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = static_cast<std::uint32_t>(binding.binding);
    write.descriptorCount = 1;
    write.descriptorType = to_vk_descriptor_type(binding.type);
    write.pImageInfo = &image_info;

    device->logical_device().updateDescriptorSets(1, &write, 0, nullptr);
}

void vk_DescriptorSet::vk_bind_sampler(LayoutHandle layout_id, const std::shared_ptr<Sampler>& sampler) {
    const auto& binding = pipeline_->vk_binding(layout_id.vk_id());
    if (binding.set != set_) throw std::runtime_error("DescriptorSet::bind: layout_id does not belong to this descriptor set");
    if (binding.type != DescriptorType::SAMPLER) {
        throw std::runtime_error("DescriptorSet::bind: this binding does not expect a standalone sampler");
    }
    auto device = device_.lock();
    if (!device) throw std::runtime_error("DescriptorSet::bind: device has been disposed");

    vk::DescriptorImageInfo image_info{};
    image_info.sampler = sampler->vk_sampler();

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = static_cast<std::uint32_t>(binding.binding);
    write.descriptorCount = 1;
    write.descriptorType = to_vk_descriptor_type(binding.type);
    write.pImageInfo = &image_info;

    device->logical_device().updateDescriptorSets(1, &write, 0, nullptr);
}

void vk_DescriptorSet::vk_bind_combined(LayoutHandle layout_id, const std::shared_ptr<Image>& image, const std::shared_ptr<Sampler>& sampler) {
    const auto& binding = pipeline_->vk_binding(layout_id.vk_id());
    if (binding.set != set_) throw std::runtime_error("DescriptorSet::bind: layout_id does not belong to this descriptor set");
    if (binding.type != DescriptorType::COMBINED_IMAGE_SAMPLER) {
        throw std::runtime_error("DescriptorSet::bind: this binding is not a COMBINED_IMAGE_SAMPLER");
    }
    auto device = device_.lock();
    if (!device) throw std::runtime_error("DescriptorSet::bind: device has been disposed");

    vk::DescriptorImageInfo image_info{};
    image_info.sampler = sampler->vk_sampler();
    image_info.imageView = image->get_view();
    image_info.imageLayout = vk::ImageLayout::eGeneral; // see vk_bind_image's comment

    vk::WriteDescriptorSet write{};
    write.dstSet = descriptor_set_;
    write.dstBinding = static_cast<std::uint32_t>(binding.binding);
    write.descriptorCount = 1;
    write.descriptorType = to_vk_descriptor_type(binding.type);
    write.pImageInfo = &image_info;

    device->logical_device().updateDescriptorSets(1, &write, 0, nullptr);
}

std::shared_ptr<vk_ResourceData> Device::vk_allocate_buffer_data(std::uint64_t size, MemoryLocation location) {
    auto& manager = location == MemoryLocation::HOST ? host_memory_manager_ : device_memory_manager_;
    auto memory = manager->allocate(size, 256);
    auto data = std::make_shared<vk_ResourceData>(shared_from_this(), memory, vk::Image{}, vk::ImageCreateInfo{});
    resources_.push_back(data); // save a weak reference to force destroy of hanging resources.
    return data;
}

std::shared_ptr<Buffer> Device::create_buffer(std::uint64_t elements, Type type, MemoryLocation location) {
    auto layout = compute_layout(TypeDescriptor::single(type), LayoutRule::Scalar);
    const std::uint64_t bytes = elements * layout->aligned_size;
    auto data = vk_allocate_buffer_data(bytes, location);
    ResourceSlice full_slice {};
    // Relative to this resource's own memory slice (see Buffer::device_ptr/external_ptr
    // and Buffer::offset()), not the absolute offset within the underlying page.
    full_slice.type = ResourceType::BUFFER;
    full_slice.buffer.offset = 0;
    full_slice.buffer.size = bytes;
    return std::make_shared<Buffer>(data, full_slice, std::move(layout));
}

std::shared_ptr<Buffer> Device::create_buffer(std::uint64_t elements, Format format, MemoryLocation location) {
    const Type component_type = format_scalar_type(format);
    const int components = static_cast<int>(formatSize(format)) / scalar_type_size(component_type);
    auto element_type = std::make_shared<TypeDescriptor>(TypeDescriptor::single(component_type));
    auto layout = compute_layout(TypeDescriptor::array_of(std::move(element_type), components), LayoutRule::Scalar);
    const std::uint64_t bytes = elements * layout->aligned_size;
    auto data = vk_allocate_buffer_data(bytes, location);
    ResourceSlice full_slice {};
    full_slice.type = ResourceType::BUFFER;
    full_slice.buffer.offset = 0;
    full_slice.buffer.size = bytes;
    return std::make_shared<Buffer>(data, full_slice, std::move(layout), format);
}

std::shared_ptr<Buffer> Device::create_buffer(std::uint64_t elements, const std::shared_ptr<Layout>& layout, MemoryLocation location) {
    if (elements == 0)
        throw std::runtime_error("create_buffer: elements must be positive");
    // Always uses aligned_size (not the possibly-tighter size), even for elements == 1,
    // so that buffer_size / layout.aligned_size reliably recovers `elements` later
    // (e.g. in Buffer::field(field)).
    const std::uint64_t size = layout->aligned_size * elements;
    if (size == 0)
        throw std::runtime_error("create_buffer: computed size is 0");
    auto data = vk_allocate_buffer_data(size, location);
    ResourceSlice full_slice {};
    full_slice.type = ResourceType::BUFFER;
    full_slice.buffer.offset = 0;
    full_slice.buffer.size = size;
    return std::make_shared<Buffer>(data, full_slice, layout);
}

void Device::vk_transition_image_layout(vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
    std::uint32_t mip_levels, std::uint32_t array_layers, vk::ImageAspectFlags aspect) {
    std::shared_ptr<Engine> engine;
    for (EngineType type : { EngineType::COMPUTE, EngineType::GRAPHICS, EngineType::TRANSFER }) {
        try {
            engine = create_engine(type, 0);
            break;
        } catch (const std::runtime_error&) {
            continue;
        }
    }
    if (!engine) {
        throw std::runtime_error("Device::create_image: no queue available to perform the initial image layout transition");
    }

    auto cmd = engine->create_command_buffer(); // already begin()-ed
    auto vk_cmd = cmd->vk_command_buffer();

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mip_levels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = array_layers;
    barrier.srcAccessMask = vk::AccessFlags{};
    // Over-broad on purpose (both color- and depth/stencil-attachment write
    // access, regardless of `aspect`): harmless, and keeps this one helper
    // shared by both create_image() and create_depth_buffer_image().
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
        | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite
        | vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

    vk_cmd->command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eAllCommands,
        vk::DependencyFlags{}, {}, {}, barrier);

    cmd->close();
    engine->submit({ cmd })->wait();
}

std::shared_ptr<Image> Device::create_image(int width, int height, int depth, int mip_levels,
    int array_layers, Format format, MemoryLocation location) {
    if (width <= 0 || height <= 0 || depth <= 0 || mip_levels <= 0 || array_layers <= 0) {
        throw std::runtime_error("Device::create_image: width, height, depth, mip_levels and array_layers must all be positive");
    }
    if (depth > 1 && array_layers > 1) {
        throw std::runtime_error("Device::create_image: a 3D image (depth > 1) cannot have more than one array layer");
    }

    const vk::ImageType image_type = depth > 1 ? vk::ImageType::e3D
        : (height > 1 ? vk::ImageType::e2D : vk::ImageType::e1D);

    vk::ImageCreateInfo image_info{};
    image_info.imageType = image_type;
    image_info.format = static_cast<vk::Format>(format);
    image_info.extent = vk::Extent3D{
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), static_cast<std::uint32_t>(depth) };
    image_info.mipLevels = static_cast<std::uint32_t>(mip_levels);
    image_info.arrayLayers = static_cast<std::uint32_t>(array_layers);
    image_info.samples = vk::SampleCountFlagBits::e1;
    image_info.tiling = vk::ImageTiling::eOptimal;
    image_info.usage = full_image_usage_flags();
    image_info.sharingMode = vk::SharingMode::eExclusive;
    image_info.initialLayout = vk::ImageLayout::eUndefined;

    vk::Device dev = logical_device();
    vk::Image image = dev.createImage(image_info);

    const vk::MemoryRequirements requirements = dev.getImageMemoryRequirements(image);
    auto& manager = location == MemoryLocation::HOST ? host_memory_manager_ : device_memory_manager_;
    vk::MemoryAllocateInfo alloc_info(requirements.size, manager->vk_memory_type_index());
    vk::DeviceMemory memory;
    try {
        memory = dev.allocateMemory(alloc_info);
        dev.bindImageMemory(image, memory, 0);
    } catch (...) {
        dev.destroyImage(image);
        throw;
    }

    // Freshly created images start in eUndefined, which isn't valid for
    // sampling/storage/blit access -- transition once here so every other
    // use site can keep assuming eGeneral uniformly (see
    // vk_transition_image_layout's comment).
    try {
        vk_transition_image_layout(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            static_cast<std::uint32_t>(mip_levels), static_cast<std::uint32_t>(array_layers));
    } catch (...) {
        dev.destroyImage(image);
        dev.freeMemory(memory);
        throw;
    }

    // Images don't sub-allocate from a shared MemoryPage the way buffers do
    // (see vk_ResourceData's dedicated_image_memory_ comment), so this
    // MemorySlice is just an inert placeholder to satisfy vk_ResourceData's
    // constructor -- its null `page` makes every one of its accessors
    // (page_buffer/device_ptr/external_ptr/host_visible/release) a no-op.
    auto memory_slice = std::make_shared<MemorySlice>(shared_from_this(), nullptr, 0, 0, 0, 0);
    auto data = std::make_shared<vk_ResourceData>(shared_from_this(), memory_slice, image, image_info, memory);
    resources_.push_back(data); // save a weak reference to force destroy of hanging resources.

    ResourceSlice slice{};
    slice.type = ResourceType::IMAGE;
    slice.image.format = format;
    slice.image.mip_start = 0;
    slice.image.mip_count = mip_levels;
    slice.image.array_start = 0;
    slice.image.array_count = array_layers;

    return std::make_shared<Image>(data, slice);
}

std::shared_ptr<Image> Device::create_depth_buffer_image(int width, int height, Format format, MemoryLocation location) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Device::create_depth_buffer_image: width and height must be positive");
    }
    if (!is_depth_format(format)) {
        throw std::runtime_error("Device::create_depth_buffer_image: format must be one of the Format::Depth* values");
    }

    vk::ImageCreateInfo image_info{};
    image_info.imageType = vk::ImageType::e2D;
    image_info.format = static_cast<vk::Format>(format);
    image_info.extent = vk::Extent3D{ static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1 };
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = vk::SampleCountFlagBits::e1;
    image_info.tiling = vk::ImageTiling::eOptimal;
    // Not full_image_usage_flags(): eColorAttachment/eStorage are invalid
    // usages for a depth/stencil format. eSampled is kept so a depth buffer
    // can still be read from a shader later (e.g. for shadow mapping).
    image_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment
        | vk::ImageUsageFlagBits::eSampled
        | vk::ImageUsageFlagBits::eTransferSrc
        | vk::ImageUsageFlagBits::eTransferDst;
    image_info.sharingMode = vk::SharingMode::eExclusive;
    image_info.initialLayout = vk::ImageLayout::eUndefined;

    vk::Device dev = logical_device();
    vk::Image image = dev.createImage(image_info);

    const vk::MemoryRequirements requirements = dev.getImageMemoryRequirements(image);
    auto& manager = location == MemoryLocation::HOST ? host_memory_manager_ : device_memory_manager_;
    vk::MemoryAllocateInfo alloc_info(requirements.size, manager->vk_memory_type_index());
    vk::DeviceMemory memory;
    try {
        memory = dev.allocateMemory(alloc_info);
        dev.bindImageMemory(image, memory, 0);
    } catch (...) {
        dev.destroyImage(image);
        throw;
    }

    // Same "always eGeneral" convention as create_image() -- see
    // vk_transition_image_layout's comment -- just with the depth aspect
    // instead of color.
    try {
        vk_transition_image_layout(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, 1, 1, vk::ImageAspectFlagBits::eDepth);
    } catch (...) {
        dev.destroyImage(image);
        dev.freeMemory(memory);
        throw;
    }

    auto memory_slice = std::make_shared<MemorySlice>(shared_from_this(), nullptr, 0, 0, 0, 0);
    auto data = std::make_shared<vk_ResourceData>(shared_from_this(), memory_slice, image, image_info, memory);
    resources_.push_back(data);

    ResourceSlice slice{};
    slice.type = ResourceType::IMAGE;
    slice.image.format = format;
    slice.image.mip_start = 0;
    slice.image.mip_count = 1;
    slice.image.array_start = 0;
    slice.image.array_count = 1;

    return std::make_shared<Image>(data, slice);
}

std::shared_ptr<Sampler> Device::create_sampler(
    Filter mag_filter, Filter min_filter, MipmapMode mipmap_mode,
    WrapMode wrap_u, WrapMode wrap_v, WrapMode wrap_w) {
    vk::SamplerCreateInfo info{};
    info.magFilter = static_cast<vk::Filter>(mag_filter);
    info.minFilter = static_cast<vk::Filter>(min_filter);
    info.mipmapMode = static_cast<vk::SamplerMipmapMode>(mipmap_mode);
    info.addressModeU = static_cast<vk::SamplerAddressMode>(wrap_u);
    info.addressModeV = static_cast<vk::SamplerAddressMode>(wrap_v);
    info.addressModeW = static_cast<vk::SamplerAddressMode>(wrap_w);
    info.borderColor = vk::BorderColor::eFloatOpaqueBlack;
    info.anisotropyEnable = VK_FALSE;
    info.compareEnable = VK_FALSE;
    info.minLod = 0.0f;
    info.maxLod = 1000.0f; // effectively unbounded: every real image has far fewer mip levels.
    info.unnormalizedCoordinates = VK_FALSE;

    vk::Sampler sampler = logical_device().createSampler(info);
    auto data = std::make_shared<vk_Sampler>(weak_from_this(), sampler);
    vk_register_sampler(data);
    return std::make_shared<Sampler>(std::move(data));
}

namespace {

// Vulkan geometry description (plus primitive/instance count and implied AS
// type) for `declaration`, shared by Device::create_ads() (sizing only) and
// CommandBuffer::build_ads() (the actual build). Throws if a required
// buffer is missing, or an index buffer's element type isn't UINT16/UINT32.
struct AdsGeometryInfo {
    vk::AccelerationStructureGeometryKHR geometry;
    std::uint32_t primitive_count = 0;
    vk::AccelerationStructureTypeKHR type = vk::AccelerationStructureTypeKHR::eBottomLevel;
};

AdsGeometryInfo vk_build_ads_geometry_info(const ADSDeclaration& declaration) {
    AdsGeometryInfo result{};
    std::visit([&](auto&& decl) {
        using T = std::decay_t<decltype(decl)>;
        if constexpr (std::is_same_v<T, ADSTriangles>) {
            const vk::GeometryFlagsKHR flags = decl.opaque ? vk::GeometryFlagsKHR(vk::GeometryFlagBitsKHR::eOpaque) : vk::GeometryFlagsKHR{};
            if (!decl.vertices) throw std::runtime_error("ADSTriangles: vertices buffer is required");
            vk::AccelerationStructureGeometryTrianglesDataKHR triangles{};
            triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
            triangles.vertexData.deviceAddress = decl.vertices->device_ptr();
            triangles.vertexStride = decl.vertices->element_layout()->aligned_size;
            triangles.maxVertex = decl.vertex_count > 0 ? decl.vertex_count - 1 : 0;
            if (decl.indices) {
                const Type idx_scalar = decl.indices->element_layout()->component_type;
                if (idx_scalar == Type::UINT16) triangles.indexType = vk::IndexType::eUint16;
                else if (idx_scalar == Type::UINT32) triangles.indexType = vk::IndexType::eUint32;
                else throw std::runtime_error("ADSTriangles: indices must be a UINT16 or UINT32 buffer");
                triangles.indexData.deviceAddress = decl.indices->device_ptr();
            } else {
                triangles.indexType = vk::IndexType::eNoneKHR;
            }
            result.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
            result.primitive_count = decl.primitive_count;
            result.geometry = vk::AccelerationStructureGeometryKHR(vk::GeometryTypeKHR::eTriangles, triangles, flags);
        } else if constexpr (std::is_same_v<T, ADSAABB>) {
            const vk::GeometryFlagsKHR flags = decl.opaque ? vk::GeometryFlagsKHR(vk::GeometryFlagBitsKHR::eOpaque) : vk::GeometryFlagsKHR{};
            if (!decl.aabbs) throw std::runtime_error("ADSAABB: aabbs buffer is required");
            vk::AccelerationStructureGeometryAabbsDataKHR aabbs{};
            aabbs.data.deviceAddress = decl.aabbs->device_ptr();
            aabbs.stride = decl.aabbs->element_layout()->aligned_size;
            result.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
            result.primitive_count = decl.count;
            result.geometry = vk::AccelerationStructureGeometryKHR(vk::GeometryTypeKHR::eAabbs, aabbs, flags);
        } else if constexpr (std::is_same_v<T, ADSInstances>) {
            if (!decl.instances) throw std::runtime_error("ADSInstances: instances buffer is required");
            vk::AccelerationStructureGeometryInstancesDataKHR instances{};
            instances.arrayOfPointers = VK_FALSE;
            instances.data.deviceAddress = decl.instances->device_ptr();
            result.type = vk::AccelerationStructureTypeKHR::eTopLevel;
            result.primitive_count = decl.count;
            result.geometry = vk::AccelerationStructureGeometryKHR(vk::GeometryTypeKHR::eInstances, instances, vk::GeometryFlagsKHR{});
        }
    }, declaration);
    return result;
}

} // namespace

std::shared_ptr<AccelerationStructure> Device::create_ads(const ADSDeclaration& declaration) {
    if (!acceleration_structure_supported_) {
        throw std::runtime_error("Device::create_ads: VK_KHR_acceleration_structure is not supported/enabled on this device");
    }

    AdsGeometryInfo info = vk_build_ads_geometry_info(declaration);

    vk::AccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.type = info.type;
    build_info.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    build_info.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    build_info.geometryCount = 1;
    build_info.pGeometries = &info.geometry;

    vk::Device dev = logical_device();
    vk::AccelerationStructureBuildSizesInfoKHR size_info = dev.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, build_info, info.primitive_count, dynamic_dispatch_);

    // The acceleration structure's backing storage buffer is allocated
    // separately from every other Buffer (which sub-allocate from a shared,
    // pooled MemoryPage -- see MemoryPage's constructor): it needs its own
    // dedicated VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
    // usage, which full_buffer_usage_flags() deliberately does not grant to
    // the shared pool (see its own comment).
    vk::BufferCreateInfo storage_info{};
    storage_info.size = size_info.accelerationStructureSize;
    storage_info.usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    storage_info.sharingMode = vk::SharingMode::eExclusive;
    vk::Buffer storage_buffer = dev.createBuffer(storage_info);

    const vk::MemoryRequirements requirements = dev.getBufferMemoryRequirements(storage_buffer);
    vk::MemoryAllocateFlagsInfo alloc_flags_info{};
    alloc_flags_info.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
    vk::MemoryAllocateInfo alloc_info(requirements.size, device_memory_manager_->vk_memory_type_index());
    alloc_info.pNext = &alloc_flags_info;
    vk::DeviceMemory storage_memory;
    vk::AccelerationStructureKHR handle;
    try {
        storage_memory = dev.allocateMemory(alloc_info);
        dev.bindBufferMemory(storage_buffer, storage_memory, 0);

        vk::AccelerationStructureCreateInfoKHR create_info{};
        create_info.buffer = storage_buffer;
        create_info.offset = 0;
        create_info.size = size_info.accelerationStructureSize;
        create_info.type = info.type;
        handle = dev.createAccelerationStructureKHR(create_info, nullptr, dynamic_dispatch_);
    } catch (...) {
        dev.destroyBuffer(storage_buffer);
        if (storage_memory) dev.freeMemory(storage_memory);
        throw;
    }

    auto vk_ads = std::make_shared<vk_AccelerationStructure>(
        weak_from_this(), handle, storage_buffer, storage_memory, info.type, size_info.buildScratchSize);
    vk_register_acceleration_structure(vk_ads);
    return std::make_shared<AccelerationStructure>(std::move(vk_ads));
}

void CommandBuffer::build_ads(const std::shared_ptr<AccelerationStructure>& ads, const ADSDeclaration& declaration) {
    if (is_closed()) {
        throw std::runtime_error("Cannot build an acceleration structure on a closed command buffer");
    }
    if (!ads) {
        throw std::runtime_error("CommandBuffer::build_ads: ads must not be null");
    }

    AdsGeometryInfo info = vk_build_ads_geometry_info(declaration);
    const auto& vk_ads = ads->vk_ads();
    if (vk_ads->vk_type() != info.type) {
        throw std::runtime_error("CommandBuffer::build_ads: declaration's acceleration structure type does not match ads's own type");
    }

    // A fresh, temporary buffer every build -- see bound_scratch_buffers_'s
    // comment: it only needs to stay alive/valid while the GPU executes
    // this command buffer, no different from a vertex/index buffer.
    auto scratch_buffer = device_->create_buffer(
        std::max<std::uint64_t>(vk_ads->vk_build_scratch_size(), 1), Type::UINT8, MemoryLocation::DEVICE);

    vk::AccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.type = info.type;
    build_info.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    build_info.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    build_info.dstAccelerationStructure = vk_ads->vk_handle();
    build_info.geometryCount = 1;
    build_info.pGeometries = &info.geometry;
    build_info.scratchData.deviceAddress = scratch_buffer->device_ptr();

    vk::AccelerationStructureBuildRangeInfoKHR range_info{};
    range_info.primitiveCount = info.primitive_count;
    const vk::AccelerationStructureBuildRangeInfoKHR* p_range_info = &range_info;

    command_buffer_->command_buffer.buildAccelerationStructuresKHR(build_info, p_range_info, device_->vk_dynamic_dispatch());

    bound_acceleration_structures_.push_back(ads);
    bound_scratch_buffers_.push_back(scratch_buffer);
    std::visit([&](auto&& decl) {
        using T = std::decay_t<decltype(decl)>;
        if constexpr (std::is_same_v<T, ADSTriangles>) {
            bound_ads_input_buffers_.push_back(decl.vertices);
            if (decl.indices) bound_ads_input_buffers_.push_back(decl.indices);
        } else if constexpr (std::is_same_v<T, ADSAABB>) {
            bound_ads_input_buffers_.push_back(decl.aabbs);
        } else if constexpr (std::is_same_v<T, ADSInstances>) {
            bound_ads_input_buffers_.push_back(decl.instances);
        }
    }, declaration);
}

std::shared_ptr<Buffer> Device::create_staging(const std::shared_ptr<Buffer>& buffer, MemoryLocation location) {
    return create_buffer(buffer->size(), Type::UINT8, location);
}

std::shared_ptr<Buffer> Device::create_staging(const std::shared_ptr<Image>& image, MemoryLocation location) {
    // A contiguous buffer of texels in `image`'s own format: each element
    // is [components, component_type] (same convention as Buffer::cast(Format)/
    // create_buffer(elements, Format, ...)), and the element count is the
    // product of every extent dimension (width * height * depth).
    const vk::ImageCreateInfo info = image->vk_image_info();
    const Type component_type = format_scalar_type(image->format());
    const int components = static_cast<int>(formatSize(image->format())) / scalar_type_size(component_type);
    const std::uint64_t texel_count =
        static_cast<std::uint64_t>(info.extent.width) * info.extent.height * info.extent.depth;
    auto component_desc = std::make_shared<TypeDescriptor>(TypeDescriptor::single(component_type));
    auto texel_layout = compute_layout(TypeDescriptor::array_of(std::move(component_desc), components), LayoutRule::Scalar);
    return create_buffer(texel_count, texel_layout, location);
}

Tensor::Tensor(std::shared_ptr<vk_ResourceData> resource_data, ResourceSlice view_slice,
    std::vector<std::uint64_t> shape, Type scalar_type)
    : Resource(std::move(resource_data), view_slice), shape_(std::move(shape)), scalar_type_(scalar_type) {
    assert(view_slice.type == ResourceType::BUFFER);
}

DLDevice Tensor::vk_dlpack_device() const noexcept {
    return data_->get_memory()->dl_device();
}

std::uint64_t Tensor::vk_buffer_offset() const noexcept {
    return data_->get_memory()->offset() + slice_.buffer.offset;
}

pybind11::object Tensor::vk_dlpack() const {
    if (external_ptr() == 0) {
        throw std::runtime_error("A DEVICE tensor was requested but the external pointer is unavailable");
    }
    void* ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(external_ptr()));
    DLDevice device = data_->get_memory()->dl_device();

    const int dimension = static_cast<int>(shape_.size());
    auto* owner = new TensorOwner();
    owner->memory = data_->get_memory();
    owner->shape = std::make_unique<std::int64_t[]>(dimension);
    owner->strides = std::make_unique<std::int64_t[]>(dimension);
    std::int64_t stride = 1;
    for (int i = dimension - 1; i >= 0; --i) {
        owner->shape[i] = static_cast<std::int64_t>(shape_[static_cast<std::size_t>(i)]);
        owner->strides[i] = stride;
        stride *= static_cast<std::int64_t>(shape_[static_cast<std::size_t>(i)]);
    }

    auto* managed = new DLManagedTensor{};
    managed->dl_tensor.data = ptr;
    managed->dl_tensor.device = device;
    managed->dl_tensor.ndim = dimension;
    managed->dl_tensor.dtype = dlpack_dtype(scalar_type_);
    managed->dl_tensor.shape = owner->shape.get();
    managed->dl_tensor.strides = owner->strides.get();
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = owner;
    managed->deleter = dlmanaged_tensor_deleter;

    return export_dltensor_py(managed);
}

void Tensor::load(const pybind11::object& source) const {
    copy_pyobject_into(*data_->device(), external_ptr(), data_->is_cpu() ? MemoryLocation::HOST : MemoryLocation::DEVICE, size(), source);
}

void Tensor::save(const pybind11::object& target) const {
    copy_pyobject_from(*data_->device(), external_ptr(), data_->is_cpu() ? MemoryLocation::HOST : MemoryLocation::DEVICE, size(), target);
}

std::shared_ptr<Tensor> Device::create_tensor(const std::vector<std::uint64_t>& shape, Type type, MemoryLocation location) {
    if (shape.empty()) {
        throw std::runtime_error("Device::create_tensor: shape must have at least one dimension");
    }
    std::uint64_t elements = 1;
    for (auto d : shape) elements *= d;
    const std::uint64_t size = elements * static_cast<std::uint64_t>(scalar_type_size(type));
    if (size == 0) {
        throw std::runtime_error("Device::create_tensor: computed size is 0");
    }
    auto data = vk_allocate_buffer_data(size, location);
    ResourceSlice full_slice{};
    full_slice.type = ResourceType::BUFFER;
    full_slice.buffer.offset = 0;
    full_slice.buffer.size = size;
    return std::make_shared<Tensor>(data, full_slice, shape, type);
}

void Device::vk_copy_in(
    std::uint64_t dst_external_ptr, MemoryLocation dst_location,
    void* src_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
    std::uint64_t itemsize, bool source_is_cuda) {
    std::uint64_t total_bytes = itemsize;
    for (int d = 0; d < ndim; ++d) total_bytes *= static_cast<std::uint64_t>(shape[d]);

    if (dst_location == MemoryLocation::DEVICE || source_is_cuda) {
        if (!device_memory_manager_->vk_copy_from_dlpack(src_data, shape, strides, ndim, itemsize, dst_external_ptr, total_bytes)) {
            throw std::runtime_error("Device::wrap: no interop library available to copy CUDA-resident/DEVICE data");
        }
        return;
    }
    copy_strided_host(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst_external_ptr)), src_data, shape, strides, ndim, itemsize, /*contiguous_is_dst=*/true);
}

void Device::vk_copy_out(
    std::uint64_t src_external_ptr, MemoryLocation src_location,
    void* dst_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
    std::uint64_t itemsize, bool dst_is_cuda) {
    std::uint64_t total_bytes = itemsize;
    for (int d = 0; d < ndim; ++d) total_bytes *= static_cast<std::uint64_t>(shape[d]);

    if (src_location == MemoryLocation::DEVICE || dst_is_cuda) {
        if (!device_memory_manager_->vk_copy_to_dlpack(src_external_ptr, total_bytes, dst_data, shape, strides, ndim, itemsize)) {
            throw std::runtime_error("Device::wrap: no interop library available to copy back to CUDA-resident/DEVICE data");
        }
        return;
    }
    copy_strided_host(reinterpret_cast<void*>(static_cast<std::uintptr_t>(src_external_ptr)), dst_data, shape, strides, ndim, itemsize, /*contiguous_is_dst=*/false);
}

std::shared_ptr<WrappedMemory> Device::wrap(pybind11::object obj, MemoryLocation location) {
    // Case 1: one of our own Buffers -- always a direct, contiguous mapping
    // already in the right place; never needs a copy.
    if (py::isinstance<Buffer>(obj)) {
        auto buffer = obj.cast<std::shared_ptr<Buffer>>();
        const Layout& layout = *buffer->element_layout();
        const Type scalar = layout.kind == TypeKind::STRUCT ? Type::UINT8 : resolve_component_type(layout);
        const std::uint64_t itemsize = static_cast<std::uint64_t>(scalar_type_size(scalar));
        std::vector<std::uint64_t> shape{ buffer->size() / itemsize };
        return std::make_shared<WrappedMemory>(
            weak_from_this(), buffer->device_ptr(), std::move(shape), scalar,
            nullptr, location, WrappedMemory::SourceKind::NONE, pybind11::object());
    }

    // Case 1b: one of our own Tensors -- same as a Buffer, always a direct
    // mapping, never needs a copy (checked before the generic DLPack case
    // below, since Tensor also implements __dlpack__).
    if (py::isinstance<Tensor>(obj)) {
        auto tensor = obj.cast<std::shared_ptr<Tensor>>();
        return std::make_shared<WrappedMemory>(
            weak_from_this(), tensor->device_ptr(), tensor->shape(), tensor->scalar_type(),
            nullptr, location, WrappedMemory::SourceKind::NONE, pybind11::object());
    }

    // Case 2: a DLPack-compatible object (e.g. a torch tensor).
    if (pybind11::hasattr(obj, "__dlpack__")) {
        pybind11::object capsule = obj.attr("__dlpack__")();
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
        if (!managed) {
            throw std::runtime_error("Device::wrap: invalid DLPack capsule (expected a \"dltensor\" capsule)");
        }
        const DLTensor& t = managed->dl_tensor;
        char* data_ptr = reinterpret_cast<char*>(t.data) + t.byte_offset;
        const bool is_cuda_device = (t.device.device_type == 2);
        const bool contiguous = dltensor_is_contiguous(t.shape, t.strides, t.ndim);

        std::vector<std::uint64_t> shape(static_cast<std::size_t>(t.ndim));
        for (int i = 0; i < t.ndim; ++i) shape[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(t.shape[i]);
        const Type scalar = scalar_type_from_dlpack_dtype(t.dtype);
        std::uint64_t elements = 1;
        for (auto d : shape) elements *= d;

        if (is_cuda_device && contiguous) {
            const auto external_ptr = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data_ptr));
            // Already one of our own allocations (e.g. it came from
            // torch.from_dlpack() on one of our own Buffers)? Reuse its
            // corresponding Vulkan device address directly -- no copy
            // needed either way.
            std::uint64_t own_device_ptr = host_memory_manager_->external_to_device(external_ptr);
            if (own_device_ptr == 0) {
                own_device_ptr = device_memory_manager_->external_to_device(external_ptr);
            }
            if (own_device_ptr != 0) {
                return std::make_shared<WrappedMemory>(
                    weak_from_this(), own_device_ptr, std::move(shape), scalar,
                    nullptr, location, WrappedMemory::SourceKind::NONE, pybind11::object());
            }
        }

        // No copy happens here: the returned WrappedMemory starts out
        // CPU-fresh, and update_gpu() lazily performs this copy on demand.
        auto temp = create_buffer(elements, scalar, location);
        return std::make_shared<WrappedMemory>(
            weak_from_this(), temp->device_ptr(), std::move(shape), scalar,
            temp, location, WrappedMemory::SourceKind::DLPACK, std::move(obj));
    }

    // Case 3: a plain Python buffer-protocol object (e.g. a numpy array).
    if (PyObject_CheckBuffer(obj.ptr())) {
        pybind11::buffer buf = pybind11::reinterpret_borrow<pybind11::buffer>(obj);
        pybind11::buffer_info info = buf.request();
        const Type scalar = scalar_type_from_buffer_format(info.format, static_cast<std::uint64_t>(info.itemsize));
        const std::uint64_t elements = static_cast<std::uint64_t>(info.size);

        auto temp = create_buffer(elements, scalar, location);
        std::vector<std::uint64_t> shape{ elements };
        return std::make_shared<WrappedMemory>(
            weak_from_this(), temp->device_ptr(), std::move(shape), scalar,
            temp, location, WrappedMemory::SourceKind::BUFFER_PROTOCOL, std::move(obj));
    }

    throw std::runtime_error("Device::wrap: value must be a Buffer, a DLPack-compatible object, or a Python buffer object");
}

WrappedMemory::WrappedMemory(
    std::weak_ptr<Device> device,
    std::uint64_t device_ptr,
    std::vector<std::uint64_t> shape,
    Type scalar_type,
    std::shared_ptr<Buffer> owned_buffer,
    MemoryLocation owned_location,
    SourceKind source_kind,
    pybind11::object source) noexcept
    : device_(std::move(device)), device_ptr_(device_ptr), shape_(std::move(shape)), scalar_type_(scalar_type),
      owned_buffer_(std::move(owned_buffer)), owned_location_(owned_location), source_kind_(source_kind),
      source_(source.is_none() ? nullptr : std::make_unique<pybind11::object>(std::move(source))),
      // A freshly wrapped copy starts out CPU-fresh: the wrapped object's
      // data hasn't been pushed into owned_buffer_ yet, so the first
      // update_gpu() call performs that copy. Irrelevant when is_direct().
      cpu_version_(1), gpu_version_(0) {
}

void WrappedMemory::make_cpu_dirty() noexcept {
    if (is_direct()) return;
    cpu_version_ = gpu_version_ + 1;
}

void WrappedMemory::make_gpu_dirty() noexcept {
    if (is_direct()) return;
    gpu_version_ = cpu_version_ + 1;
}

void WrappedMemory::update_gpu() {
    if (is_direct()) return;
    if (gpu_version_ >= cpu_version_) return;
    auto device = device_.lock();
    if (!device || device->is_disposed()) return;

    const std::uint64_t itemsize = static_cast<std::uint64_t>(scalar_type_size(scalar_type_));
    if (source_kind_ == SourceKind::DLPACK) {
        pybind11::object capsule = source_->attr("__dlpack__")();
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
        if (managed) {
            const DLTensor& t = managed->dl_tensor;
            char* data_ptr = reinterpret_cast<char*>(t.data) + t.byte_offset;
            const bool is_cuda_device = (t.device.device_type == 2);
            device->vk_copy_in(owned_buffer_->external_ptr(), owned_location_, data_ptr, t.shape, t.strides, t.ndim, itemsize, is_cuda_device);
        }
    } else if (source_kind_ == SourceKind::BUFFER_PROTOCOL) {
        pybind11::buffer buf = pybind11::reinterpret_borrow<pybind11::buffer>(*source_);
        pybind11::buffer_info info = buf.request();
        const std::int64_t total = static_cast<std::int64_t>(info.size);
        const std::int64_t stride1 = 1;
        device->vk_copy_in(owned_buffer_->external_ptr(), owned_location_, info.ptr, &total, &stride1, 1, itemsize, /*source_is_cuda=*/false);
    }
    gpu_version_ = cpu_version_;
}

void WrappedMemory::update_cpu() {
    if (is_direct()) return;
    if (cpu_version_ >= gpu_version_) return;
    auto device = device_.lock();
    if (!device || device->is_disposed()) return;

    const std::uint64_t itemsize = static_cast<std::uint64_t>(scalar_type_size(scalar_type_));
    if (source_kind_ == SourceKind::DLPACK) {
        pybind11::object capsule = source_->attr("__dlpack__")();
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
        if (managed) {
            const DLTensor& t = managed->dl_tensor;
            char* data_ptr = reinterpret_cast<char*>(t.data) + t.byte_offset;
            const bool is_cuda_device = (t.device.device_type == 2);
            device->vk_copy_out(owned_buffer_->external_ptr(), owned_location_, data_ptr, t.shape, t.strides, t.ndim, itemsize, is_cuda_device);
        }
    } else if (source_kind_ == SourceKind::BUFFER_PROTOCOL) {
        pybind11::buffer buf = pybind11::reinterpret_borrow<pybind11::buffer>(*source_);
        pybind11::buffer_info info = buf.request();
        const std::int64_t total = static_cast<std::int64_t>(info.size);
        const std::int64_t stride1 = 1;
        device->vk_copy_out(owned_buffer_->external_ptr(), owned_location_, info.ptr, &total, &stride1, 1, itemsize, /*dst_is_cuda=*/false);
    }
    cpu_version_ = gpu_version_;
}

WrappedMemory::~WrappedMemory() noexcept = default;

void Device::dispose() noexcept {
    try {
        if (device_) {
            (void)device_.waitIdle();
        }
    } catch (...) {
        // Destructors and dispose must not throw.
    }

    // dispose all created engines
    for (auto& engine_family : engines_) {
        for (auto& engine : engine_family) {
            if (engine) {
                engine->vk_dispose_with(device_);
            }
        }
        engine_family.clear();
    }
    engines_.clear();

    // destroy all command pools
    for (auto& pool : command_pools_) {
        if (pool) {
            auto dev = logical_device();
            dev.destroyCommandPool(pool);
        }
    }
    command_pools_.clear();

    // Destroy any still-alive framebuffers first: each references a render
    // pass and image views, so they must go before the images/pipelines
    // those views/render passes point into -- and well before the device
    // and instance themselves, since leaving one around across
    // vkDestroyDevice/vkDestroyInstance has been observed to crash on at
    // least one driver.
    for (const auto& fb : framebuffers_) {
        auto framebuffer = fb.lock();
        if (framebuffer)
            framebuffer->dispose();
    }
    framebuffers_.clear();

    // Destroy any still-alive windows next: each owns a swapchain/surface,
    // semaphores/fences and per-slot render/image/buffer/tensor targets.
    // vk_Window only holds a weak_ptr<Device> (to avoid a reference cycle
    // back to Device), so without this, whether its Vulkan objects get
    // destroyed before or after vkDestroyDevice would depend entirely on
    // the unspecified order in which Python drops its last references to
    // the Device and the Window (e.g. plain module-level globals, never
    // explicitly disposed) -- tripping vkDestroyDevice's child-object
    // validation checks whenever the Window loses that race.
    for (const auto& w : windows_) {
        auto window = w.lock();
        if (window)
            window->vk_dispose_with(device_, instance_);
    }
    windows_.clear();

    // Destroy any still-alive samplers before the device/instance (same
    // rationale as windows_/framebuffers_ above).
    for (const auto& s : samplers_) {
        auto sampler = s.lock();
        if (sampler)
            sampler->dispose();
    }
    samplers_.clear();

    // Destroy any still-alive acceleration structures before the
    // device/instance (same rationale as windows_/samplers_ above).
    for (const auto& a : acceleration_structures_) {
        auto ads = a.lock();
        if (ads)
            ads->dispose();
    }
    acceleration_structures_.clear();

    // Destroy now any pending resource
    for (const auto& r: resources_) {
        auto res = r.lock();
        if (res)
            res->dispose();
    }
    resources_.clear();

    // Destroy every memory page's underlying vk::Buffer/vk::DeviceMemory
    // using device_ directly (see vk_dispose_with's docstring) before
    // resetting the managers -- otherwise ~MemoryPage would try
    // device_.lock() on its own weak_ptr<Device>, which is unconditionally
    // expired by the time Device::dispose() runs via ~Device(), leaking
    // every page's backing buffer/memory.
    if (device_memory_manager_) device_memory_manager_->vk_dispose_with(device_);
    if (host_memory_manager_) host_memory_manager_->vk_dispose_with(device_);
    device_memory_manager_.reset();
    host_memory_manager_.reset();
    device_index_ = 0;

    if (interop_semaphore_ && device_) {
        device_.destroySemaphore(interop_semaphore_);
        interop_semaphore_ = nullptr;
    }

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

vk_Engine::vk_Engine(std::weak_ptr<Device> device, vk::Queue queue, vk::CommandPool pool, EngineType type)
    : device_(std::move(device)), queue_(queue), command_pool_(pool), engine_type_(type) {
    auto d = device_.lock();
    if (d && !d->is_disposed()) {
        auto semaphore_type_create_info = vk::SemaphoreTypeCreateInfo(
                vk::SemaphoreType::eTimeline,
                0,
                nullptr
        );
        timeline_semaphore_ = d->logical_device().createSemaphore(vk::SemaphoreCreateInfo{
            vk::SemaphoreCreateFlags{},
            &semaphore_type_create_info
        });
    }
    current_submission_id_ = 0;
}

MemorySlice::MemorySlice(
    std::shared_ptr<Device> device,
    std::shared_ptr<MemoryPage> page,
    std::uint64_t allocated_offset,
    std::uint64_t allocated_size,
    std::uint64_t offset,
    std::uint64_t size
) noexcept
    : device_(device),
      page_(page),
      allocated_offset_(allocated_offset),
      allocated_size_(allocated_size),
      offset_(offset),
      size_(size) {}

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
    device_.reset();
    offset_ = 0;
    size_ = 0;
}

MemorySlice::~MemorySlice() noexcept {
    release();
}

MemoryManager::MemoryManager(std::weak_ptr<Device> device, uint32_t memory_type_index, bool host_visible):
      device_(device),
      memory_type_index_(memory_type_index),
      host_visible_(host_visible) {
    auto dev = device_.lock();
    const auto vendor_id = dev->physical_device().getProperties().vendorID;
    //std::cout << "[INFO] Loading interop library for vendor ID" << vendor_id << std::endl;
    // printf("[INFO] Loading interop library for vendor ID: 0x%04X\n", vendor_id);
    interop_library_ = std::make_shared<ExternalInteropLibraryImpl>(interop_library_base_name(vendor_id));
    if (!interop_library_) {
        throw std::runtime_error("Unable to load interop library");
        // std::cout << "[WARNING] Library not found." << std::endl;
    }
    try_import_memory_ = interop_library_ && interop_library_->loaded()
        ? interop_library_->try_import_memory_fn()
        : nullptr;
    if (!try_import_memory_) {
        throw std::runtime_error("Unable to load interop library");
        // std::cout << "[WARNING] Library not loaded." << std::endl;
    }
}

MemoryManager::~MemoryManager() noexcept = default;

void MemoryManager::vk_dispose_with(vk::Device dev) noexcept {
    for (auto& page : pages_) {
        if (page) page->vk_dispose_with(dev);
    }
    pages_.clear();
}

std::shared_ptr<MemorySlice> MemoryManager::allocate(std::uint64_t size, int alignment) {
    auto d = device_.lock();
    if (!d) {
        throw std::runtime_error("Unable to allocate new memory in a disposed device");
    }
    if (size <= 0) {
        throw std::runtime_error("Allocation size must be positive");
    }
    for (auto& page : pages_) {
        if (page->can_allocate(size, alignment)) {
            return page->allocate(size, alignment);
        }
    }

    std::uint64_t page_capacity = std::max(size, next_page_capacity_); // big data create their own page.

    pages_.push_back(std::make_shared<MemoryPage>(d, memory_type_index_, host_visible_, page_capacity, try_import_memory_));
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

std::uint64_t MemoryManager::vk_interop_semaphore_handle() const {
    if (interop_semaphore_import_attempted_) return interop_semaphore_handle_;
    interop_semaphore_import_attempted_ = true;
    if (!interop_library_) return 0;
    auto import_fn = interop_library_->try_import_semaphore_fn();
    if (!import_fn) return 0;
    auto device = device_.lock();
    if (!device) return 0;
    vk::Semaphore semaphore = device->vk_interop_semaphore();
    if (!semaphore) return 0;
    interop_semaphore_handle_ = import_fn(device->vk_device(), static_cast<int>(device->device_index()), semaphore);
    return interop_semaphore_handle_;
}

bool MemoryManager::vk_copy_from_dlpack(
    void* tensor_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
    std::uint64_t itemsize, std::uint64_t dst_ptr, std::uint64_t total_bytes) const {
    if (!interop_library_) return false;
    auto fn = interop_library_->copy_from_dlpack_fn();
    if (!fn) return false;
    const std::uint64_t wait_semaphore = vk_interop_semaphore_handle();
    std::uint64_t wait_value = 0;
    if (wait_semaphore) {
        if (auto device = device_.lock()) wait_value = device->vk_interop_semaphore_value();
    }
    return fn(tensor_data, shape, strides, ndim, itemsize, dst_ptr, total_bytes, wait_semaphore, wait_value) == 0;
}

bool MemoryManager::vk_copy_to_dlpack(
    std::uint64_t src_ptr, std::uint64_t total_bytes,
    void* tensor_data, const std::int64_t* shape, const std::int64_t* strides, int ndim,
    std::uint64_t itemsize) const {
    if (!interop_library_) return false;
    auto fn = interop_library_->copy_to_dlpack_fn();
    if (!fn) return false;
    const std::uint64_t wait_semaphore = vk_interop_semaphore_handle();
    std::uint64_t wait_value = 0;
    if (wait_semaphore) {
        if (auto device = device_.lock()) wait_value = device->vk_interop_semaphore_value();
    }
    return fn(src_ptr, total_bytes, tensor_data, shape, strides, ndim, itemsize, wait_semaphore, wait_value) == 0;
}

MemoryPage::MemoryPage(std::weak_ptr<Device> device, uint32_t memory_type_index, bool host_visible, std::uint64_t capacity, ExternalImportFn try_import_memory)
    : device_(device),
      memory_type_index_(memory_type_index),
      host_visible_(host_visible),
      capacity_(capacity),
      allocator_(std::make_unique<MemoryAllocator>(capacity)),
      try_import_memory_(try_import_memory) {
    if (capacity <= 0) {
        throw std::runtime_error("Page capacity must be positive");
    }

    auto device_ptr = device_.lock();

    auto dev = device_ptr->vk_device();

    vk::ExternalMemoryBufferCreateInfo external_buffer_info{};
#if defined(_WIN32)
    external_buffer_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
#else
    external_buffer_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
#endif

    vk::BufferCreateInfo buffer_info(
        vk::BufferCreateFlags{},
        static_cast<vk::DeviceSize>(capacity_),
        full_buffer_usage_flags(device_ptr->vk_acceleration_structure_supported()),
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
    }
    // Non-host_visible_ (DEVICE) pages: external_ptr_ is left at 0 here.
    // The CUDA import (try_import_memory_) is deliberately deferred to
    // ensure_external_ptr_imported(), the first time external_ptr()/
    // dl_device() is actually called -- see that method's comment.
}

MemoryPage::~MemoryPage() noexcept {
    auto device_ptr = device_.lock();
    vk::Device dev = (device_ptr && !device_ptr->is_disposed()) ? device_ptr->vk_device() : vk::Device{};
    vk_dispose_with(dev);
}

void MemoryPage::vk_dispose_with(vk::Device dev) noexcept {
    if (dev) {
        if (host_visible_ && memory_) {
            dev.unmapMemory(memory_);
        }
        if (buffer_) {
            dev.destroyBuffer(buffer_);
        }
        if (memory_) {
            dev.freeMemory(memory_);
        }
    }
    buffer_ = vk::Buffer{};
    memory_ = vk::DeviceMemory{};
    device_ptr_ = 0;
    external_ptr_ = 0;
}

bool MemoryPage::can_allocate(std::uint64_t size, std::uint64_t alignment) const {
    return allocator_->can_allocate(size + alignment - 1);
}

std::shared_ptr<MemorySlice> MemoryPage::allocate(std::uint64_t size, std::uint64_t alignment) {
    auto device = device_.lock(); // get the device
    if (!device) {
        throw std::runtime_error("Page can not allocate memory in a disposed device");
    }
    const int offset = allocator_->allocate(size + alignment - 1);
    int aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
    return std::make_shared<MemorySlice>(device, shared_from_this(), offset, size + alignment - 1, aligned_offset, size);
}

void MemoryPage::free_memory(std::uint64_t allocated_offset) noexcept {
    allocator_->free_memory(allocated_offset);
}

std::uint64_t MemoryPage::capacity() const noexcept { return capacity_; }
vk::Buffer MemoryPage::buffer() const noexcept { return buffer_; }
vk::DeviceMemory MemoryPage::memory() const noexcept { return memory_; }
std::uint64_t MemoryPage::device_ptr() const noexcept { return device_ptr_; }

void MemoryPage::ensure_external_ptr_imported() const noexcept {
    if (host_visible_ || external_ptr_import_attempted_) return;
    external_ptr_import_attempted_ = true;
    if (!try_import_memory_) return;
    auto device = device_.lock();
    if (!device) return;
    external_ptr_ = try_import_memory_(device->vk_device(), device->device_index(), static_cast<VkDeviceMemory>(memory_), capacity_);
}

std::uint64_t MemoryPage::external_ptr() const noexcept {
    ensure_external_ptr_imported();
    return external_ptr_;
}

DLDevice MemoryPage::dl_device() const noexcept {
    ensure_external_ptr_imported();
    auto device = device_.lock();
    const int device_id = device ? device->device_index() : 0;
    if (host_visible_) return {1, device_id};
    if (external_ptr_) return {2, device_id};
    return {0, device_id};
}

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

MemoryAllocator::MemoryAllocator(std::uint64_t capacity) : capacity_(capacity), bins_(32) {
    auto node = std::make_unique<Range>();
    node->offset = 0;
    node->size = capacity_;
    head_ = node.get();
    nodes_.push_back(std::move(node));
    rebuild_bins();
}

MemoryAllocator::~MemoryAllocator() noexcept = default;

int MemoryAllocator::bin_for_size(std::uint64_t size) const {
    std::uint64_t value = 1;
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

bool MemoryAllocator::can_allocate(std::uint64_t size) const {
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

std::uint64_t MemoryAllocator::allocate(std::uint64_t size) {
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

    const std::uint64_t offset = selected->offset;
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

void MemoryAllocator::free_memory(std::uint64_t offset) noexcept {
    const auto allocation = allocations_.find(offset);
    if (allocation == allocations_.end()) {
        return;
    }

    const std::uint64_t size = allocation->second;
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
