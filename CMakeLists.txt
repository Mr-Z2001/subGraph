cmake_minimum_required(VERSION 3.22)
project(SubGraphIsomorphism LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12/bin/nvcc")
#set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda-12/include")
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(test)

add_executable(SubGraph src/main.cu)

target_link_libraries(SubGraph PUBLIC GRAPH JOIN FILTER ORDER)

target_compile_features(SubGraph PUBLIC cxx_std_17)

set_target_properties(SubGraph PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
