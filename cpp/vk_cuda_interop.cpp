#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>

#if defined(_WIN32)
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

extern "C" {

// The plugin exposes a C symbol with this signature.
// It receives the Vulkan device handle and the device memory and should
// return an external pointer (e.g., CUDA device pointer) or 0 on failure.
#ifdef _WIN32
#  define PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
#  define PLUGIN_EXPORT extern "C"
#endif

PLUGIN_EXPORT uint64_t try_import_memory(VkDevice device, int device_index, VkDeviceMemory memory, unsigned long long size) {
    std::cout << "try_import_memory" << std::endl;
#if defined(_WIN32)
    // Try to obtain the vkGetDeviceProcAddr function and then obtain
    // vkGetMemoryWin32HandleKHR dynamically. If anything fails, return 0.
    PFN_vkGetDeviceProcAddr pfnGetDeviceProcAddr = &vkGetDeviceProcAddr;
    if (!pfnGetDeviceProcAddr) {
        std::cout << "Failed to get vkGetDeviceProcAddr" << std::endl;
        return 0;
    }

    auto pfnGetMemoryWin32Handle = (PFN_vkGetMemoryWin32HandleKHR)pfnGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    if (!pfnGetMemoryWin32Handle) {
        std::cout << "Failed to get vkGetMemoryWin32HandleKHR" << std::endl;
        return 0;
    }

    VkMemoryGetWin32HandleInfoKHR handle_info{};
    handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    handle_info.memory = memory;

    HANDLE h = NULL;
    if (pfnGetMemoryWin32Handle(device, &handle_info, &h) != VK_SUCCESS) {
        std::cout << "Failed to get Win32 handle for Vulkan memory" << std::endl;
        return 0;
    }

    if (!h) {
        std::cout << "Win32 handle is NULL" << std::endl;
        return 0;
    }

    // Dynamically load CUDA runtime and import external memory.
    HMODULE cudart = LoadLibraryA("cudart64_12.dll");
    if (!cudart) cudart = LoadLibraryA("cudart64_110.dll");
    if (!cudart) {
        CloseHandle(h);
        std::cout << "Failed to load CUDA runtime library" << std::endl;
        return 0;
    }

    using ImportFn = int (*)(void**, const void*);
    using MapFn = int (*)(void**, void*, const void*);
    using DestroyFn = int (*)(void*);

    auto import_fn = (ImportFn)GetProcAddress(cudart, "cudaImportExternalMemory");
    auto map_fn = (MapFn)GetProcAddress(cudart, "cudaExternalMemoryGetMappedBuffer");
    auto destroy_fn = (DestroyFn)GetProcAddress(cudart, "cudaDestroyExternalMemory");

    if (!import_fn || !map_fn || !destroy_fn) {
        FreeLibrary(cudart);
        CloseHandle(h);
        std::cout << "Failed to get CUDA import/map/destroy functions" << std::endl;
        return 0;
    }

    // Minimal opaque structs (we avoid including CUDA headers to keep build simple).
    struct CUDAExternalMemoryHandleDesc {
        int type;
        union {
            int fd;
            struct {
                void *handle;
                const void *name;
            } win32;
            const void *nvSciBufObject;
        } handle;
        unsigned long long size;
        unsigned int flags;
    } desc{};
    // Opaque: choose a sensible constant (PLATFORM DEPENDENT). If import fails, we return 0.
    desc.type = 2; // opaque win32
    desc.handle.win32.handle = h;
    desc.size = size; // unknown; driver may accept zero or ignore
    desc.flags = 0;

    void* imported = nullptr;
    if (import_fn(&imported, &desc) != 0) {
        FreeLibrary(cudart);
        CloseHandle(h);
        std::cout << "Failed to import external memory into CUDA" << std::endl;
        return 0;
    }

    struct CUDAExternalMemoryBufferDesc {
        unsigned long long offset;
        unsigned long long size;
        unsigned int flags;
    } bufdesc{};
    bufdesc.offset = 0;
    bufdesc.size = size;
    bufdesc.flags = 0;

    void* mapped_buffer = nullptr;
    if (map_fn(&mapped_buffer, imported, &bufdesc) != 0) {
        destroy_fn(imported);
        FreeLibrary(cudart);
        CloseHandle(h);
        std::cout << "Failed to map external memory buffer into CUDA" << std::endl;
        return 0;
    }

    uint64_t ptr = reinterpret_cast<uint64_t>(mapped_buffer);
    // Keep cudart loaded; we don't manage imported lifetime across process exit.
    CloseHandle(h);
    return ptr;
#else
    // On POSIX systems, attempt to get an fd and import via libcudart (if available).
    auto pfnGetDeviceProcAddr = &vkGetDeviceProcAddr;
    if (!pfnGetDeviceProcAddr) {
        throw std::runtime_error("Failed to get vkGetDeviceProcAddr");
        // std::cout << "Failed to get vkGetDeviceProcAddr" << std::endl;
        return 0;
    }

    auto pfnGetMemoryFd = (PFN_vkGetMemoryFdKHR)pfnGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!pfnGetMemoryFd) {
        throw std::runtime_error("Failed to get vkGetMemoryFdKHR");
        // std::cout << "Failed to get vkGetMemoryFdKHR" << std::endl;
        return 0;
    }

    VkMemoryGetFdInfoKHR fd_info{};
    fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.memory = memory;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    if (pfnGetMemoryFd(device, &fd_info, &fd) != VK_SUCCESS) return 0;
    if (fd < 0) {
        throw std::runtime_error("Failed to get vkGetMemoryFdKHR");
        // std::cout << "Failed to get valid file descriptor for Vulkan memory" << std::endl;
        return 0;
    }

    void* libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) libcudart = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) libcudart = dlopen("libcudart.so.12.0", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) { close(fd);
        throw std::runtime_error("Failed to load CUDA library from libcudart");
        // std::cout << "Failed to load CUDA library" << std::endl;
        return 0;
    }

    using ImportFn = int (*)(void**, const void*);
    using MapFn = int (*)(void**, void*, const void*);
    using DestroyFn = int (*)(void*);

    auto import_fn = (ImportFn)dlsym(libcudart, "cudaImportExternalMemory");
    auto map_fn = (MapFn)dlsym(libcudart, "cudaExternalMemoryGetMappedBuffer");
    auto destroy_fn = (DestroyFn)dlsym(libcudart, "cudaDestroyExternalMemory");

    if (!import_fn || !map_fn || !destroy_fn) { dlclose(libcudart); close(fd);
        throw std::runtime_error("Failed to get CUDA import/map/destroy functions");
        // std::cout << "Failed to get CUDA import/map/destroy functions" << std::endl;
        return 0;
    }

    struct CUDAExternalMemoryHandleDesc {
        int type;
        union {
            int fd;
            struct {
                void *handle;
                const void *name;
            } win32;
            const void *nvSciBufObject;
        } handle;
        unsigned long long size;
        unsigned int flags;
    } desc{};
    desc.type = 1; // opaque fd
    desc.handle.fd = fd;
    desc.size = size;
    desc.flags = 0;

    void* imported = nullptr;
    if (import_fn(&imported, &desc) != 0) { dlclose(libcudart); close(fd);
        throw std::runtime_error("Failed to import external memory into CUDA");
        // std::cout << "Failed to import external memory into CUDA" << std::endl;
        return 0; }

    struct CUDAExternalMemoryBufferDesc {
        unsigned long long offset;
        unsigned long long size;
        unsigned int flags;
    } bufdesc{};
    bufdesc.offset = 0;
    bufdesc.size = size;
    bufdesc.flags = 0;

    void* mapped_buffer = nullptr;
    if (map_fn(&mapped_buffer, imported, &bufdesc) != 0) { destroy_fn(imported); dlclose(libcudart); close(fd);
        throw std::runtime_error("Failed to map external memory buffer into CUDA");
        // std::cout << "Failed to map external memory buffer into CUDA" << std::endl;
        return 0; }

    uint64_t ptr = reinterpret_cast<uint64_t>(mapped_buffer);
    // keep libcudart loaded
    close(fd);
    return ptr;
#endif
}

// Imports `semaphore` (a Vulkan VK_SEMAPHORE_TYPE_TIMELINE semaphore,
// exported by Device::vk_init() as VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32/FD_BIT)
// into CUDA as a cudaExternalSemaphore_t, returned as an opaque handle (0
// on failure). Mirrors try_import_memory's shape, but for a semaphore
// instead of a memory allocation.
//
// This is what lets copy_from_dlpack_to_ptr/copy_ptr_to_dlpack, below,
// issue a real GPU-side wait (cudaWaitExternalSemaphoresAsync) for a
// specific Vulkan submission before touching shared memory -- unlike a
// purely host-side Vulkan wait (vkWaitSemaphores/queue-idle), a GPU-side
// wait is what actually guarantees a Vulkan write is visible to a later
// CUDA-side read on at least some drivers (observed: a plain host wait,
// with no GPU-side cross-API semaphore wait, can leave the CUDA side
// reading stale data even though the Vulkan work has genuinely finished).
PLUGIN_EXPORT uint64_t try_import_semaphore(VkDevice device, int device_index, VkSemaphore semaphore) {
    (void)device_index;
    if (!semaphore) return 0;
#if defined(_WIN32)
    PFN_vkGetDeviceProcAddr pfnGetDeviceProcAddr = &vkGetDeviceProcAddr;
    auto pfnGetSemaphoreWin32Handle = (PFN_vkGetSemaphoreWin32HandleKHR)pfnGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
    if (!pfnGetSemaphoreWin32Handle) return 0;

    VkSemaphoreGetWin32HandleInfoKHR handle_info{};
    handle_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handle_info.semaphore = semaphore;
    handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE h = NULL;
    if (pfnGetSemaphoreWin32Handle(device, &handle_info, &h) != VK_SUCCESS || !h) return 0;

    HMODULE cudart = LoadLibraryA("cudart64_12.dll");
    if (!cudart) cudart = LoadLibraryA("cudart64_110.dll");
    if (!cudart) { CloseHandle(h); return 0; }

    using ImportSemaphoreFn = int (*)(void**, const void*);
    auto import_fn = (ImportSemaphoreFn)GetProcAddress(cudart, "cudaImportExternalSemaphore");
    if (!import_fn) { CloseHandle(h); return 0; }

    struct CudaExternalSemaphoreHandleDesc {
        int type;
        union {
            int fd;
            struct { void* handle; const void* name; } win32;
            const void* nvSciSyncObj;
        } handle;
        unsigned int flags;
    } desc{};
    constexpr int kCudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10;
    desc.type = kCudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    desc.handle.win32.handle = h;
    desc.flags = 0;

    void* imported = nullptr;
    if (import_fn(&imported, &desc) != 0) { CloseHandle(h); return 0; }
    CloseHandle(h);
    return reinterpret_cast<uint64_t>(imported);
#else
    PFN_vkGetDeviceProcAddr pfnGetDeviceProcAddr = &vkGetDeviceProcAddr;
    auto pfnGetSemaphoreFd = (PFN_vkGetSemaphoreFdKHR)pfnGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    if (!pfnGetSemaphoreFd) return 0;

    VkSemaphoreGetFdInfoKHR fd_info{};
    fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fd_info.semaphore = semaphore;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    if (pfnGetSemaphoreFd(device, &fd_info, &fd) != VK_SUCCESS || fd < 0) return 0;

    void* libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) libcudart = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) libcudart = dlopen("libcudart.so.12.0", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) { close(fd); return 0; }

    using ImportSemaphoreFn = int (*)(void**, const void*);
    auto import_fn = (ImportSemaphoreFn)dlsym(libcudart, "cudaImportExternalSemaphore");
    if (!import_fn) { close(fd); return 0; }

    struct CudaExternalSemaphoreHandleDesc {
        int type;
        union {
            int fd;
            struct { void* handle; const void* name; } win32;
            const void* nvSciSyncObj;
        } handle;
        unsigned int flags;
    } desc{};
    constexpr int kCudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9;
    desc.type = kCudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    desc.handle.fd = fd;
    desc.flags = 0;

    void* imported = nullptr;
    if (import_fn(&imported, &desc) != 0) { close(fd); return 0; }
    return reinterpret_cast<uint64_t>(imported);
#endif
}

// ---- Strided <-> contiguous copy helpers ----
//
// Used by Device::wrap()/copy() to move data between a DLPack tensor
// (possibly strided, possibly host- or CUDA-resident) and a plain
// contiguous CUDA device pointer (always how this project's own DEVICE
// buffers are laid out). `shape`/`strides` follow the DLPack convention:
// `ndim` entries, strides in units of `itemsize`-byte elements. Returns 0
// on success, nonzero on failure.

struct CudaRuntimeApi {
    using MemcpyAsyncFn = int (*)(void*, const void*, size_t, int, void*);
    using Memcpy2DAsyncFn = int (*)(void*, size_t, const void*, size_t, size_t, size_t, int, void*);
    using StreamSynchronizeFn = int (*)(void*);
    using WaitExternalSemaphoresAsyncFn = int (*)(const void*, const void*, unsigned int, void*);
    MemcpyAsyncFn memcpy_async_fn = nullptr;
    Memcpy2DAsyncFn memcpy2d_async_fn = nullptr;
    StreamSynchronizeFn stream_synchronize_fn = nullptr;
    WaitExternalSemaphoresAsyncFn wait_external_semaphores_async_fn = nullptr;
    bool ok() const { return memcpy_async_fn && memcpy2d_async_fn && stream_synchronize_fn; }
};

// cudaMemcpyKind::cudaMemcpyDefault -- let the driver infer the direction
// from the pointers themselves (unified addressing), so this works
// regardless of whether the strided side is host or CUDA device memory.
constexpr int kCudaMemcpyDefault = 4;
// Default (legacy) CUDA stream: used so the semaphore wait (if any) and
// every copy issued after it are strictly ordered relative to each other,
// without needing to create/manage a dedicated stream.
constexpr void* kCudaDefaultStream = nullptr;

static CudaRuntimeApi load_cuda_runtime_api() {
    CudaRuntimeApi api{};
#if defined(_WIN32)
    HMODULE cudart = LoadLibraryA("cudart64_12.dll");
    if (!cudart) cudart = LoadLibraryA("cudart64_110.dll");
    if (!cudart) return api;
    api.memcpy_async_fn = (CudaRuntimeApi::MemcpyAsyncFn)GetProcAddress(cudart, "cudaMemcpyAsync");
    api.memcpy2d_async_fn = (CudaRuntimeApi::Memcpy2DAsyncFn)GetProcAddress(cudart, "cudaMemcpy2DAsync");
    api.stream_synchronize_fn = (CudaRuntimeApi::StreamSynchronizeFn)GetProcAddress(cudart, "cudaStreamSynchronize");
    api.wait_external_semaphores_async_fn = (CudaRuntimeApi::WaitExternalSemaphoresAsyncFn)GetProcAddress(cudart, "cudaWaitExternalSemaphoresAsync");
#else
    void* libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) libcudart = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) libcudart = dlopen("libcudart.so.12.0", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) return api;
    api.memcpy_async_fn = (CudaRuntimeApi::MemcpyAsyncFn)dlsym(libcudart, "cudaMemcpyAsync");
    api.memcpy2d_async_fn = (CudaRuntimeApi::Memcpy2DAsyncFn)dlsym(libcudart, "cudaMemcpy2DAsync");
    api.stream_synchronize_fn = (CudaRuntimeApi::StreamSynchronizeFn)dlsym(libcudart, "cudaStreamSynchronize");
    api.wait_external_semaphores_async_fn = (CudaRuntimeApi::WaitExternalSemaphoresAsyncFn)dlsym(libcudart, "cudaWaitExternalSemaphoresAsync");
#endif
    return api;
}

// Oversized, zeroed stand-in for CUDA's cudaExternalSemaphoreWaitParams:
// only `params.fence.value` (unambiguously the first 8 bytes of the real
// struct) is ever set, for a timeline semaphore wait. The remaining bytes
// are padding, generously larger than the real struct so the driver never
// reads past what's allocated here, regardless of exact reserved-field
// sizes in a given CUDA Toolkit version.
struct CudaExternalSemaphoreWaitParamsPadded {
    unsigned long long fence_value = 0;
    unsigned char padding[248] = {};
};

// Enqueues (on the default stream) a GPU-side wait for `semaphore_handle`
// (an imported cudaExternalSemaphore_t, from try_import_semaphore) to
// reach `wait_value`. A no-op returning true if `semaphore_handle` is 0
// (unsupported/unavailable) or the driver doesn't expose
// cudaWaitExternalSemaphoresAsync.
static bool wait_for_semaphore(const CudaRuntimeApi& api, unsigned long long semaphore_handle, unsigned long long wait_value) {
    if (!semaphore_handle || !api.wait_external_semaphores_async_fn) return true;
    void* sem = reinterpret_cast<void*>(semaphore_handle);
    CudaExternalSemaphoreWaitParamsPadded params{};
    params.fence_value = wait_value;
    return api.wait_external_semaphores_async_fn(&sem, &params, 1, kCudaDefaultStream) == 0;
}

// Copies between `contiguous` (a flat buffer of product(shape)*itemsize
// bytes) and a strided view `data`/`shape`/`strides` (`ndim` dims).
// `contiguous_is_dst` selects the direction. All copies are issued on the
// default stream (async), so they're properly ordered after the semaphore
// wait enqueued by the caller; the caller synchronizes the stream once,
// after every copy below has been issued.
//
// Fast paths (a single cudaMemcpyAsync or cudaMemcpy2DAsync call):
// everything but the outermost dimension is already tightly packed --
// true for every shape this project's own dlpack()/field() ever produce
// (at most one strided "instance" dimension over an otherwise-contiguous
// element), and for the common case of a plain, unpermuted
// array-of-structs view from numpy/torch. General fallback (arbitrarily
// permuted/transposed views): recurse one dimension at a time, which
// still hits the fast path one level down as soon as the remaining
// dimensions happen to be packed.
static int copy_strided(
    const CudaRuntimeApi& api, void* contiguous, void* data,
    const long long* shape, const long long* strides, int ndim,
    unsigned long long itemsize, bool contiguous_is_dst)
{
    if (ndim == 0) {
        void* dst = contiguous_is_dst ? contiguous : data;
        void* src = contiguous_is_dst ? data : contiguous;
        return api.memcpy_async_fn(dst, src, (size_t)itemsize, kCudaMemcpyDefault, kCudaDefaultStream);
    }

    unsigned long long row_elements = 1;
    for (int d = 1; d < ndim; ++d) row_elements *= (unsigned long long)shape[d];
    bool inner_packed = true;
    unsigned long long expected_stride = 1;
    for (int d = ndim - 1; d >= 1; --d) {
        if (strides[d] != (long long)expected_stride) { inner_packed = false; break; }
        expected_stride *= (unsigned long long)shape[d];
    }

    if (ndim == 1 || inner_packed) {
        const unsigned long long row_bytes = row_elements * itemsize;
        const unsigned long long outer_pitch_bytes = (unsigned long long)strides[0] * itemsize;
        if (outer_pitch_bytes == row_bytes) {
            // Fully contiguous end to end: one flat copy.
            return api.memcpy_async_fn(contiguous_is_dst ? contiguous : data,
                                        contiguous_is_dst ? data : contiguous,
                                        (size_t)(row_bytes * (unsigned long long)shape[0]), kCudaMemcpyDefault, kCudaDefaultStream);
        }
        void* dst = contiguous_is_dst ? contiguous : data;
        void* src = contiguous_is_dst ? data : contiguous;
        size_t dpitch = contiguous_is_dst ? (size_t)row_bytes : (size_t)outer_pitch_bytes;
        size_t spitch = contiguous_is_dst ? (size_t)outer_pitch_bytes : (size_t)row_bytes;
        return api.memcpy2d_async_fn(dst, dpitch, src, spitch, (size_t)row_bytes, (size_t)shape[0], kCudaMemcpyDefault, kCudaDefaultStream);
    }

    // Outer dimension only, one recursive call per index.
    for (long long i = 0; i < shape[0]; ++i) {
        char* next_data = (char*)data + (unsigned long long)i * (unsigned long long)strides[0] * itemsize;
        char* next_contiguous = (char*)contiguous + (unsigned long long)i * row_elements * itemsize;
        int rc = copy_strided(api, next_contiguous, next_data, shape + 1, strides + 1, ndim - 1, itemsize, contiguous_is_dst);
        if (rc != 0) return rc;
    }
    return 0;
}

// Copies a DLPack tensor (`tensor_data`/`shape`/`strides`/`ndim`, possibly
// strided, possibly host- or CUDA-resident) into `dst_ptr`, a contiguous
// CUDA device pointer of `total_bytes` bytes. Returns 0 on success.
//
// `wait_semaphore`/`wait_value`: if `wait_semaphore` is nonzero (an
// imported cudaExternalSemaphore_t from try_import_semaphore), a GPU-side
// wait for `wait_value` is enqueued before the copy, ensuring any prior
// Vulkan write up to that submission is actually visible to CUDA (see
// try_import_semaphore's comment) -- not just "the host knows it's done".
PLUGIN_EXPORT int copy_from_dlpack_to_ptr(
    void* tensor_data, const long long* shape, const long long* strides, int ndim,
    unsigned long long itemsize, unsigned long long dst_ptr, unsigned long long total_bytes,
    unsigned long long wait_semaphore, unsigned long long wait_value)
{
    CudaRuntimeApi api = load_cuda_runtime_api();
    if (!api.ok()) return -1;
    if (!wait_for_semaphore(api, wait_semaphore, wait_value)) return -1;
    int rc;
    if (ndim == 0) {
        rc = api.memcpy_async_fn((void*)dst_ptr, tensor_data, (size_t)total_bytes, kCudaMemcpyDefault, kCudaDefaultStream);
    } else {
        rc = copy_strided(api, (void*)dst_ptr, tensor_data, shape, strides, ndim, itemsize, /*contiguous_is_dst=*/true);
    }
    if (rc != 0) return rc;
    return api.stream_synchronize_fn(kCudaDefaultStream);
}

// Copies `total_bytes` contiguous bytes from `src_ptr` (a CUDA device
// pointer) into a DLPack tensor (`tensor_data`/`shape`/`strides`/`ndim`,
// possibly strided, possibly host- or CUDA-resident). Returns 0 on
// success. `wait_semaphore`/`wait_value`: see copy_from_dlpack_to_ptr.
PLUGIN_EXPORT int copy_ptr_to_dlpack(
    unsigned long long src_ptr, unsigned long long total_bytes,
    void* tensor_data, const long long* shape, const long long* strides, int ndim,
    unsigned long long itemsize, unsigned long long wait_semaphore, unsigned long long wait_value)
{
    CudaRuntimeApi api = load_cuda_runtime_api();
    if (!api.ok()) return -1;
    if (!wait_for_semaphore(api, wait_semaphore, wait_value)) return -1;
    int rc;
    if (ndim == 0) {
        rc = api.memcpy_async_fn(tensor_data, (void*)src_ptr, (size_t)total_bytes, kCudaMemcpyDefault, kCudaDefaultStream);
    } else {
        rc = copy_strided(api, (void*)src_ptr, tensor_data, shape, strides, ndim, itemsize, /*contiguous_is_dst=*/false);
    }
    if (rc != 0) return rc;
    return api.stream_synchronize_fn(kCudaDefaultStream);
}

} // extern "C"
