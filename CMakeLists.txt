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
endif()

find_package(Vulkan REQUIRED)

# -----------------------------------------------------------------------------
# Python extension
# -----------------------------------------------------------------------------

pybind11_add_module(vk
        cpp/bindings.cpp
        cpp/device.cpp
)

target_link_libraries(vk
        PRIVATE
        Vulkan::Vulkan
)

target_include_directories(vk
        PRIVATE
        ${Vulkan_INCLUDE_DIRS}
)

if(WIN32)
    target_compile_definitions(vk
            PRIVATE
            VK_USE_PLATFORM_WIN32_KHR
            NOMINMAX
    )
endif()

# -----------------------------------------------------------------------------
# Install into the Python package
# -----------------------------------------------------------------------------

install(TARGETS vk
        LIBRARY DESTINATION vk
        RUNTIME DESTINATION vk
)