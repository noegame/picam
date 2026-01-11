# Test OpenCV wrapper
add_executable(test_opencv_wrapper
    test_opencv_wrapper.c
)
target_link_libraries(test_opencv_wrapper opencv_wrapper)

# Test libcamera wrapper
# add_executable(test_libcamera_wrapper 
#     test_libcamera_wrapper.c
# )
# target_link_libraries(test_libcamera_wrapper libcamera_wrapper)

# Test openCV
add_executable(test_opencv
    test_opencv.cpp
)
target_link_libraries(test_opencv opencv_wrapper)

# Test libcamera
# add_executable(test_libcamera
#     test_libcamera.cpp
# )
# target_link_libraries(test_libcamera libcamera_wrapper)

