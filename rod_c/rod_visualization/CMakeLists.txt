# ROD Visualization Library
# Functions for annotating images and creating debug outputs

add_library(rod_visualization STATIC
    rod_visualization.c
    rod_visualization.h
)

target_include_directories(rod_visualization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link with required libraries
target_link_libraries(rod_visualization PUBLIC
    opencv_wrapper
    rod_cv
)
