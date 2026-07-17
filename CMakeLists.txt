cmake_minimum_required(VERSION 3.18)

project(vk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# -----------------------------------------------------------------------------
# pybind11 (installed in the active Python environment)
# -----------------------------------------------------------------------------

if(NOT pybind11_DIR)
    execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    if(_pybind11_cmake_dir)
        list(PREPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

# -----------------------------------------------------------------------------
# Vulkan
# -----------------------------------------------------------------------------

# On Windows, the Vulkan SDK normally sets VULKAN_SDK.
# Help CMake locate the SDK if it is present.
if(WIN32 AND DEFINED ENV{VULKAN_SDK})
    set(Vulkan_ROOT "$ENV{VULKAN_SDK}" CACHE PATH "Vulkan SDK")
elseif(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_sdk/Include/vulkan/vulkan.h")
    set(Vulkan_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_sdk" CACHE PATH "Vulkan SDK")
endif()

find_package(Vulkan REQUIRED)

# -----------------------------------------------------------------------------
# GLFW (fetched and built from source, statically linked -- no system-
# installed GLFW required on either Windows or Linux, and no extra runtime
# library to ship alongside vk's own extension module)
# -----------------------------------------------------------------------------

include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

# -----------------------------------------------------------------------------
# Dear ImGui (fetched and built from source; ships as loose sources with no
# CMakeLists.txt of its own, so its files are added directly to the vk
# target below instead of via add_subdirectory/FetchContent_MakeAvailable)
# -----------------------------------------------------------------------------

FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.9b
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

# -----------------------------------------------------------------------------
# tinyobjloader (fetched and built from source; used by loaders.cpp to load
# .obj scenes)
# -----------------------------------------------------------------------------

FetchContent_Declare(
        tinyobjloader
        GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
        GIT_TAG v2.0.0rc13
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(tinyobjloader)

set(IMGUI_SOURCES
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

# -----------------------------------------------------------------------------
# CUDA Interop Library (shared library)
# -----------------------------------------------------------------------------

add_library(vk_cuda_interop SHARED
        cpp/vk_cuda_interop.cpp
)

target_link_libraries(vk_cuda_interop
        PRIVATE
        Vulkan::Vulkan
)

target_include_directories(vk_cuda_interop
        PRIVATE
        ${Vulkan_INCLUDE_DIRS}
)

if(UNIX AND NOT APPLE)
    target_link_libraries(vk_cuda_interop PRIVATE ${CMAKE_DL_LIBS})
endif()

if(WIN32)
    target_compile_definitions(vk_cuda_interop
            PRIVATE
            VK_USE_PLATFORM_WIN32_KHR
            NOMINMAX
    )
endif()

# Set output directory for the library to be discoverable
set_target_properties(vk_cuda_interop PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/vk_cuda_interop_lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/vk_cuda_interop_lib"
)

# Platform-specific naming
if(WIN32)
    set_target_properties(vk_cuda_interop PROPERTIES PREFIX "")
    set_target_properties(vk_cuda_interop PROPERTIES SUFFIX ".dll")
elseif(APPLE)
    set_target_properties(vk_cuda_interop PROPERTIES PREFIX "lib")
    set_target_properties(vk_cuda_interop PROPERTIES SUFFIX ".dylib")
else()
    set_target_properties(vk_cuda_interop PROPERTIES PREFIX "lib")
    set_target_properties(vk_cuda_interop PROPERTIES SUFFIX ".so")
endif()

# Install the CUDA interop library to Vulkan bin folder
if(WIN32)
    install(TARGETS vk_cuda_interop
            LIBRARY DESTINATION "${Vulkan_ROOT}/Bin"
            RUNTIME DESTINATION "${Vulkan_ROOT}/Bin"
    )
else()
    install(TARGETS vk_cuda_interop
            LIBRARY DESTINATION "${Vulkan_ROOT}/lib"
    )
endif()

# Also install to package for Python wheel
install(TARGETS vk_cuda_interop
        LIBRARY DESTINATION vk
        RUNTIME DESTINATION vk
)

# -----------------------------------------------------------------------------
# Python extension
# -----------------------------------------------------------------------------

pybind11_add_module(vk
        cpp/bindings.cpp
        cpp/device.cpp
        cpp/loaders.cpp
        ${IMGUI_SOURCES}
)

target_link_libraries(vk
        PRIVATE
        Vulkan::Vulkan
        glfw
        tinyobjloader
)

target_include_directories(vk
        PRIVATE
        ${Vulkan_INCLUDE_DIRS}
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
)

if(UNIX AND NOT APPLE)
    target_link_libraries(vk PRIVATE ${CMAKE_DL_LIBS})
endif()

if(WIN32)
    target_compile_definitions(vk
            PRIVATE
            VK_USE_PLATFORM_WIN32_KHR
            NOMINMAX
    )
endif()

# Add dependency so CUDA interop library is built before vk Python extension
add_dependencies(vk vk_cuda_interop)

# Note: vk_cuda_interop is a plugin that is dynamically loaded at runtime,
# not linked at compile time. It will be discovered via the Vulkan SDK path or wheel package.

# -----------------------------------------------------------------------------
# Install into the Python package
# -----------------------------------------------------------------------------

install(TARGETS vk
        LIBRARY DESTINATION vk
        RUNTIME DESTINATION vk
)