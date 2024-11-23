cmake_minimum_required(VERSION 3.10)
project(yolo)

# Set the required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "/home/rahim-khan/libtorch")

# Find required packages
find_package(Torch REQUIRED)


# Add executable and libraries
add_executable(main main.cpp)
add_library(config src/config.cpp)
add_library(yolov11 src/yolov11.cpp)
add_library(conv src/conv.cpp)
add_library(block src/block.cpp)

# Link libraries
target_link_libraries(main 
    PRIVATE 
    "${TORCH_LIBRARIES}"
    config
    yolov11
    conv
    block
)


# Ensure CMake can find the Torch headers
include_directories(${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIRS})

# Set C++ standard for the target
set_property(TARGET main PROPERTY CXX_STANDARD 17)

