cmake_minimum_required(VERSION 3.9)
project(SubGraphIsomorphism LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(test)

add_executable(SubGraph src/main.cu)

target_link_libraries(SubGraph PUBLIC LEVELCSR)

set_target_properties(SubGraph PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
