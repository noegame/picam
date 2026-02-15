# rod_cv - OpenCV wrapper library
# Provides C interface to OpenCV for ArUco detection and image processing

# Create opencv_wrapper library
add_library(opencv_wrapper SHARED
    opencv_wrapper.cpp
)

# Link OpenCV libraries
target_link_libraries(opencv_wrapper
    ${OpenCV_LIBS}
)

# Set library properties
set_target_properties(opencv_wrapper PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "opencv_wrapper.h"
)

# Include directories for this target
target_include_directories(opencv_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Install rules
install(TARGETS opencv_wrapper
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/rod_cv
)
