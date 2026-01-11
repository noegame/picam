#ifndef LIBCAMERA_WRAPPER_H
#define LIBCAMERA_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

struct LibCameraContext {
    std::unique_ptr<CameraManager> camera_manager;
    std::shared_ptr<Camera> camera;
    std::unique_ptr<CameraConfiguration> config;
    FrameBufferAllocator *allocator;
    std::vector<std::unique_ptr<Request>> requests;
};

LibCameraContext* libcamera_init();
int libcamera_open_camera(LibCameraContext* ctx, int camera_index);
int libcamera_configure(LibCameraContext* ctx, int width, int height);
int libcamera_start(LibCameraContext* ctx);
int libcamera_stop(LibCameraContext* ctx);
int libcamera_capture_frame(LibCameraContext* ctx, uint8_t** out_buffer,
                            size_t* out_size, int timeout_ms);
void libcamera_cleanup(LibCameraContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // OPENCV_WRAPPER_H