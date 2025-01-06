cmake_minimum_required(VERSION 3.0)

project(load_shade_ops)

set(ENV{HIP_PATH} "/opt/rocm/")
#set(ROCM_HOME /opt/rocm/)
list(APPEND CMAKE_MODULE_PATH /opt/rocm/lib/cmake/hip/)
find_package(HIP REQUIRED)

HIP_ADD_EXECUTABLE(load_shade_ops main.cpp)
target_include_directories(load_shade_ops PRIVATE /opt/rocm/include)
target_compile_definitions(load_shade_ops PRIVATE __HIP_PLATFORM_AMD__)
set_target_properties(load_shade_ops PROPERTIES HIP_ARCHITECTURES "gfx1030")

# copy the shader file to the build directory
add_custom_command(TARGET load_shade_ops POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/shadeops_hip.hsaco ${CMAKE_CURRENT_BINARY_DIR}/shadeops_hip.hsaco
    COMMENT "Copying shade.ops to the build directory"
)
