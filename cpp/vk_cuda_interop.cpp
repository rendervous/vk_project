#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>

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
    printf("Importing memory...");
#if defined(_WIN32)
    // Try to obtain the vkGetDeviceProcAddr function and then obtain
    // vkGetMemoryWin32HandleKHR dynamically. If anything fails, return 0.
    PFN_vkGetDeviceProcAddr pfnGetDeviceProcAddr = &vkGetDeviceProcAddr;
    if (!pfnGetDeviceProcAddr) {
        printf("Failed to get vkGetDeviceProcAddr\n");
        return 0;
    }

    auto pfnGetMemoryWin32Handle = (PFN_vkGetMemoryWin32HandleKHR)pfnGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    if (!pfnGetMemoryWin32Handle) {
        printf("Failed to get vkGetMemoryWin32HandleKHR\n");
        return 0;
    }

    VkMemoryGetWin32HandleInfoKHR handle_info{};
    handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    handle_info.memory = memory;

    HANDLE h = NULL;
    if (pfnGetMemoryWin32Handle(device, &handle_info, &h) != VK_SUCCESS) {
        printf("Failed to get Win32 handle for Vulkan memory\n");
        return 0;
    }

    if (!h) {
        printf("Win32 handle is NULL\n");
        return 0;
    }

    // Dynamically load CUDA runtime and import external memory.
    HMODULE cudart = LoadLibraryA("cudart64_12.dll");
    if (!cudart) cudart = LoadLibraryA("cudart64_110.dll");
    if (!cudart) {
        CloseHandle(h);
        printf("Failed to load CUDA runtime library\n");
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
        printf("Failed to get CUDA import/map/destroy functions\n");
        return 0;
    }

    // Minimal opaque structs (we avoid including CUDA headers to keep build simple).
    struct CUDAExternalMemoryHandleDesc_Win32 {
        int type;
        void* handle;
        const void *name;
        unsigned long long size;
        unsigned int flags;
    } desc{};
    // Opaque: choose a sensible constant (PLATFORM DEPENDENT). If import fails, we return 0.
    desc.type = 2; // opaque win32
    desc.handle = h;
    desc.size = size; // unknown; driver may accept zero or ignore

    void* imported = nullptr;
    if (import_fn(&imported, &desc) != 0) {
        FreeLibrary(cudart);
        CloseHandle(h);
        printf("Failed to import external memory into CUDA\n");
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
        printf("Failed to map external memory buffer into CUDA\n");
        return 0;
    }

    uint64_t ptr = reinterpret_cast<uint64_t>(mapped_buffer);
    // Keep cudart loaded; we don't manage imported lifetime across process exit.
    CloseHandle(h);
    return ptr;
#else
    // On POSIX systems, attempt to get an fd and import via libcudart (if available).
    auto pfnGetDeviceProcAddr = &vkGetDeviceProcAddr;
    if (!pfnGetDeviceProcAddr) return 0;

    auto pfnGetMemoryFd = (PFN_vkGetMemoryFdKHR)pfnGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!pfnGetMemoryFd) return 0;

    VkMemoryGetFdInfoKHR fd_info{};
    fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.memory = memory;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    if (pfnGetMemoryFd(device, &fd_info, &fd) != VK_SUCCESS) return 0;
    if (fd < 0) return 0;

    void* libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) libcudart = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_LOCAL);
    if (!libcudart) { close(fd); return 0; }

    using ImportFn = int (*)(void**, const void*);
    using MapFn = int (*)(void**, void*, const void*);
    using DestroyFn = int (*)(void*);

    auto import_fn = (ImportFn)dlsym(libcudart, "cudaImportExternalMemory");
    auto map_fn = (MapFn)dlsym(libcudart, "cudaExternalMemoryGetMappedBuffer");
    auto destroy_fn = (DestroyFn)dlsym(libcudart, "cudaDestroyExternalMemory");

    if (!import_fn || !map_fn || !destroy_fn) { dlclose(libcudart); close(fd); return 0; }

    struct CUDAExternalMemoryHandleDesc {
        int type;
        int fd;
        unsigned long long size;
    } desc{};
    desc.type = 1; // opaque fd
    desc.fd = fd;
    desc.size = size;

    void* imported = nullptr;
    if (import_fn(&imported, &desc) != 0) { dlclose(libcudart); close(fd); return 0; }

    struct CUDAExternalMemoryBufferDesc {
        unsigned long long offset;
        unsigned long long size;
        unsigned int flags;
    } bufdesc{};
    bufdesc.offset = 0;
    bufdesc.size = size;
    bufdesc.flags = 0;

    void* mapped_buffer = nullptr;
    if (map_fn(&mapped_buffer, imported, &bufdesc) != 0) { destroy_fn(imported); dlclose(libcudart); close(fd); return 0; }

    uint64_t ptr = reinterpret_cast<uint64_t>(mapped_buffer);
    // keep libcudart loaded
    close(fd);
    return ptr;
#endif
}

} // extern "C"
