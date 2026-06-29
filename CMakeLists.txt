cmake_minimum_required(VERSION 3.18)
project(vk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Resolve pybind11 from the active Python environment (e.g. venv/site-packages).
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
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

# CMake's FindVulkan honors Vulkan_ROOT/VULKAN_SDK; set a sensible default.
if(DEFINED ENV{VULKAN_SDK} AND NOT Vulkan_ROOT)
    set(Vulkan_ROOT "$ENV{VULKAN_SDK}" CACHE PATH "Path to Vulkan SDK")
endif()
find_package(Vulkan REQUIRED)

pybind11_add_module(_vk
    cpp/bindings.cpp
    cpp/device.cpp
)

foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(_vk PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${cfg} "${CMAKE_SOURCE_DIR}/src/vk"
            LIBRARY_OUTPUT_DIRECTORY_${cfg} "${CMAKE_SOURCE_DIR}/src/vk"
    )
endforeach()

target_include_directories(_vk PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(_vk PRIVATE Vulkan::Vulkan)

if(WIN32)
    target_compile_definitions(_vk PRIVATE VK_USE_PLATFORM_WIN32_KHR NOMINMAX)
endif()

install(TARGETS _vk
    LIBRARY DESTINATION vk
    RUNTIME DESTINATION vk
    ARCHIVE DESTINATION vk
)
