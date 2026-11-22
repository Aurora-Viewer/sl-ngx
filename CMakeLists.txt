# Build: cmake -B build -DDLSS_SDK=/path/to/the/NVIDIA/DLSS/SDK && cmake --build build --config Release
# Linux is tested; the Windows build (MSVC) is written but untested.
cmake_minimum_required(VERSION 3.16)
project(sl_ngx CXX)
set(CMAKE_CXX_STANDARD 17)
set(DLSS_SDK "" CACHE PATH "folder holding NVIDIA's DLSS SDK (include/, lib/)")
if(NOT DLSS_SDK)
    message(FATAL_ERROR "set -DDLSS_SDK to the folder holding NVIDIA's DLSS SDK")
endif()
find_package(Vulkan REQUIRED)

add_library(sl_ngx SHARED src/sl_ngx.cpp)
target_include_directories(sl_ngx PRIVATE include ${DLSS_SDK}/include)
target_link_libraries(sl_ngx PRIVATE Vulkan::Vulkan)
if(WIN32)
    # NVIDIA's static NGX core for the MSVC runtime the build uses
    if(CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL" OR NOT CMAKE_MSVC_RUNTIME_LIBRARY)
        set(NGX_LIB ${DLSS_SDK}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib)
    else()
        set(NGX_LIB ${DLSS_SDK}/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib)
    endif()
    target_link_libraries(sl_ngx PRIVATE ${NGX_LIB})
else()
    target_compile_options(sl_ngx PRIVATE -fvisibility=hidden -Wall -Wno-deprecated-declarations)
    target_link_libraries(sl_ngx PRIVATE ${DLSS_SDK}/lib/Linux_x86_64/libnvsdk_ngx.a dl)
    target_link_options(sl_ngx PRIVATE -Wl,--exclude-libs,ALL
        -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map -Wl,--no-undefined)
endif()
