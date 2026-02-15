# ROD Config Library
# Centralized configuration for marker IDs and detector parameters

add_library(rod_config STATIC
    rod_config.c
    rod_config.h
)

target_include_directories(rod_config PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link with OpenCV wrapper for detector parameter types
target_link_libraries(rod_config PUBLIC
    opencv_wrapper
)
