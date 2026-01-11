cmake_minimum_required(VERSION 3.10)
project(VisionC)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# Global include directory for public headers
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${LIBCAMERA_INCLUDE_DIRS})

# ===== WRAPPER LIBRARIES =====

# OpenCV wrapper library
add_library(opencv_wrapper STATIC
    src/wrappers/opencv_wrapper.cpp
    include/opencv_wrapper.h
)
target_link_libraries(opencv_wrapper ${OpenCV_LIBS})

# libcamera wrapper library (placeholder)
add_library(libcamera_wrapper STATIC
    src/wrappers/libcamera_wrapper.cpp
    include/libcamera_wrapper.h
)
target_link_libraries(libcamera_wrapper ${LIBCAMERA_LIBRARIES})

# ===== MAIN APPLICATION =====

add_executable(vision_main
    src/main.c
)
target_link_libraries(vision_main 
    opencv_wrapper
    libcamera_wrapper
)

# ===== UTILITY APPS =====
add_subdirectory(apps)

# ===== TESTS =====
add_subdirectory(tests)
