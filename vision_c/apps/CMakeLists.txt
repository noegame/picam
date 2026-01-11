# Utility applications

# Grayscale converter
add_executable(grayscale_converter
    grayscale_converter.c
)
target_link_libraries(grayscale_converter opencv_wrapper)

# Camera capture application (not yet implemented)
# add_executable(camera_capture camera_capture.c)
# target_link_libraries(camera_capture opencv_wrapper libcamera_wrapper)
