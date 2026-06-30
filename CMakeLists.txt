cmake_minimum_required(VERSION 3.18)

project(vk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# ---------------------------------------------------------------------------
# pybind11 (pip environment)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Vulkan (required for build, provided by CI)
# ---------------------------------------------------------------------------

if(DEFINED ENV{VULKAN_SDK})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
endif()

find_package(Vulkan REQUIRED CONFIG QUIET)

if(NOT Vulkan_FOUND)
    find_package(Vulkan REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Python extension module (IMPORT NAME = vk)
# ---------------------------------------------------------------------------

pybind11_add_module(vk
        cpp/bindings.cpp
        cpp/device.cpp
)

target_link_libraries(vk
        PRIVATE
        Vulkan::Vulkan
)

if(WIN32)
    target_compile_definitions(vk
            PRIVATE
            VK_USE_PLATFORM_WIN32_KHR
            NOMINMAX
    )
endif()

# ---------------------------------------------------------------------------
# Install into Python package (CRITICAL FOR WHEELS)
# ---------------------------------------------------------------------------

install(TARGETS vk
        LIBRARY DESTINATION vk
        RUNTIME DESTINATION vk
)