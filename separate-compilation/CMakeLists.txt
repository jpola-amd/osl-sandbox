cmake_minimum_required(VERSION 3.0)

project(separate-compilation)

set(ENV{HIP_PATH} "/opt/rocm/")
#set(ROCM_HOME /opt/rocm/)
list(APPEND CMAKE_MODULE_PATH /opt/rocm/lib/cmake/hip/)
find_package(HIP REQUIRED)

HIP_ADD_EXECUTABLE(separate-compilation main.cpp)
target_include_directories(separate-compilation PRIVATE /opt/rocm/include)
target_compile_definitions(separate-compilation PRIVATE __HIP_PLATFORM_AMD__)
set_target_properties(separate-compilation PROPERTIES HIP_ARCHITECTURES "gfx1030")
