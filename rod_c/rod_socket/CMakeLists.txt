# ROD Socket Library
# Unix domain socket communication for detection data transmission

add_library(rod_socket STATIC
    rod_socket.c
    rod_socket.h
)

target_include_directories(rod_socket PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link with required libraries
target_link_libraries(rod_socket PUBLIC
    rod_cv
    rod_config
)
