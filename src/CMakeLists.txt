include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(LEVELCSR LevelCSR.cpp)
add_library(COORDINATE Coordinate.cpp)

target_link_libraries(LEVELCSR PUBLIC COORDINATE)