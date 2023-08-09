include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(LEVELCSR LevelCSR.cpp)
add_library(COORDINATE Coordinate.cpp)

add_library(FILTER filter.cu)

target_link_libraries(LEVELCSR PUBLIC COORDINATE)
target_link_libraries(FILTER PUBLIC LEVELCSR)