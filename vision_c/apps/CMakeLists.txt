# Utility applications

# Grayscale converter
add_executable(grayscale_converter
    grayscale_converter.c
)
target_link_libraries(grayscale_converter opencv_wrapper)

# Grayscale converter
add_executable(detect_aruco
    detect_aruco.c
)
target_link_libraries(detect_aruco opencv_wrapper)

# ArUco detection pipeline test
add_executable(test_aruco_detection_pipeline
    test_aruco_detection_pipeline.c
)
target_link_libraries(test_aruco_detection_pipeline opencv_wrapper)

# Camera capture application (not yet implemented)
# add_executable(camera_capture camera_capture.c)
# target_link_libraries(camera_capture opencv_wrapper libcamera_wrapper)
