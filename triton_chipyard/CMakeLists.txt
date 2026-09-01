option(TRITON_BUILD_CHIPYARD_BACKEND "Build triton-chipyard backend" ON)

set(TRITON_CHIPYARD_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}") 
set(TRITON_CHIPYARD_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

function(add_symlink source_file link_name)
    if (EXISTS ${source_file})
        file(CREATE_LINK ${source_file} ${link_name} SYMBOLIC)
    else()
        message(WARNING "${source_file} not found. Skipping symlink creation for ${link_name}.")
    endif()
endfunction()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/triton-chipyard-opt)

if (TRITON_BUILD_CHIPYARD_BACKEND)
    add_triton_plugin(TritonChipyard ${CMAKE_CURRENT_SOURCE_DIR}/triton_chipyard.cc) # LINK_LIBS ...)
    target_link_libraries(TritonChipyard PRIVATE Python3::Module pybind11::headers)
endif()
