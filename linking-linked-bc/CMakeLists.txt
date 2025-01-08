cmake_minimum_required(VERSION 3.0)

project(linking_linked_bc)

set(ENV{HIP_PATH} "/opt/rocm/")

list(APPEND CMAKE_MODULE_PATH /opt/rocm/lib/cmake/hip/)
find_package(HIP REQUIRED)

HIP_ADD_EXECUTABLE(linking_linked_bc main.cpp)
target_include_directories(linking_linked_bc PRIVATE /opt/rocm/include)
target_compile_definitions(linking_linked_bc PRIVATE __HIP_PLATFORM_AMD__)
set_target_properties(linking_linked_bc PROPERTIES HIP_ARCHITECTURES "gfx1030")

# run the prepare_ABops.sh script to generate the shader file
add_custom_command(TARGET linking_linked_bc PRE_BUILD
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/prepare_ABops.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/prepare_ABops.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/opsA.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/opsB.cpp
    COMMENT "Generating opsAB.bc"
) 
# copy the shader file to the build directory
add_custom_command(TARGET linking_linked_bc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/opsAB.bc ${CMAKE_CURRENT_BINARY_DIR}/opsAB.bc
    COMMENT "Copying shade.ops to the build directory"
)

# copy all the *.bc files to the build directory
add_custom_command(TARGET linking_linked_bc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/linked_shadeops_hip.bc ${CMAKE_CURRENT_BINARY_DIR}/linked_shadeops_hip.bc
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/shadeops_hip.gcn.bc ${CMAKE_CURRENT_BINARY_DIR}/shadeops_hip.gcn.bc
    COMMENT "Copying *.bc files to the build directory"
)
