#include "include/libcamera_wrapper.h"
#include <stdio.h>

int main() {
    LibCameraContext* ctx; 
    int index = 0;
    int width = 4000;
    int height = 4000;

    ctx = libcamera_init();
    libcamera_open_camera(ctx, 0);
    libcamera_configure(ctx, width, height);
    libcamera_start(ctx);
    libcamera_capture_frame(ctx, NULL, NULL, 3000);
    libcamera_stop(ctx);
    libcamera_cleanup(ctx);
    
    return 0;
}
