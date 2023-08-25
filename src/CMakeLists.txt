include_directories(${PROJECT_SOURCE_DIR}/include)

#find_package(CUDA REQUIRED)

add_library(GRAPH Graph.cpp)
add_library(COORDINATE Coordinate.cpp)
add_library(JOIN Join.cu)
add_library(FILTER Filter.cu)
add_library(ORDER Ordering.cu)

#target_link_libraries(LEVELCSR PUBLIC COORDINATE)
target_link_libraries(FILTER PUBLIC GRAPH)
target_link_libraries(JOIN PUBLIC GRAPH)
target_link_libraries(ORDER PUBLIC GRAPH)