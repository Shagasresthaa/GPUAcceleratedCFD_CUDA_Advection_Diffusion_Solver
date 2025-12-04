cmake_minimum_required(VERSION 3.20)

# Force CUDA to use gcc as host compiler before project()
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc)

project(GPUAcceleratedCFD_CUDA_Advection_Diffusion_Solver C CXX CUDA)

# Prefering C23 standard since im running Gentoo with gcc 14.3.1
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output to build dir
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

# Find packages
find_package(LibXml2 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CPU executable creation
add_executable(cfd_sim src/main.c src/grid.c src/parameters.c src/solver.c src/io.c)

target_include_directories(cfd_sim PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LIBXML2_INCLUDE_DIR}
)

target_link_libraries(cfd_sim
    m
    ${LIBXML2_LIBRARIES}
)

# CUDA executable creation
add_executable(cfd_sim_cuda src/main_cuda.cu src/solver_cuda.cu src/grid.c src/parameters.c src/io.c)

target_include_directories(cfd_sim_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LIBXML2_INCLUDE_DIR}
)

target_link_libraries(cfd_sim_cuda
    m
    ${LIBXML2_LIBRARIES}
    CUDA::cudart
)

# Set CUDA architectures set for my 3070ti mobile gpu which is v8.6
set_property(TARGET cfd_sim_cuda PROPERTY CUDA_ARCHITECTURES 86)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Property flags
target_compile_options(cfd_sim PRIVATE -O3)
set_target_properties(cfd_sim_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
