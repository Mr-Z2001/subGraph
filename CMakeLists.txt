cmake_minimum_required(VERSION 3.25)
project(SubGraphIsomorphism CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(SubGraphIsomorphism main.cu)

set_target_properties(SubGraphIsomorphism PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
