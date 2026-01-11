cmake_minimum_required(VERSION 3.10)
project(VisionC)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src/test_wrapper)

# Create the wrapper library
add_library(opencv_wrapper STATIC
    src/test_wrapper/opencv_wrapper.cpp
    src/test_wrapper/opencv_wrapper.h
)
target_link_libraries(opencv_wrapper ${OpenCV_LIBS})

# Create the grayscale converter executable
add_executable(grayscale_converter
    src/test_wrapper/grayscale_converter.c
)
target_link_libraries(grayscale_converter opencv_wrapper ${OpenCV_LIBS})
