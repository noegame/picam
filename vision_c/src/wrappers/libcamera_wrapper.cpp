#include "libcamera_wrapper.h"
#include <libcamera/libcamera.h>
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>
#include <sys/mman.h>

using namespace libcamera;

extern "C" {

LibCameraContext* libcamera_init() {
    LibCameraContext* ctx = new LibCameraContext();
    ctx->camera_manager = std::make_unique<CameraManager>();
    ctx->allocator = nullptr;
    
    int ret = ctx->camera_manager->start();
    if (ret) {
        delete ctx;
        return nullptr;
    }
    
    return ctx;
}

int libcamera_open_camera(LibCameraContext* ctx, int camera_index) {
    if (!ctx || !ctx->camera_manager)
        return -1;
    
    auto cameras = ctx->camera_manager->cameras();
    if (cameras.empty() || camera_index >= (int)cameras.size())
        return -1;
    
    ctx->camera = cameras[camera_index];
    if (ctx->camera->acquire()) {
        return -1;
    }
    
    return 0;
}

int libcamera_configure(LibCameraContext* ctx, int width, int height) {
    if (!ctx || !ctx->camera)
        return -1;
    
    ctx->config = ctx->camera->generateConfiguration({StreamRole::VideoRecording});
    if (!ctx->config)
        return -1;
    
    StreamConfiguration &streamConfig = ctx->config->at(0);
    streamConfig.size.width = width;
    streamConfig.size.height = height;
    streamConfig.pixelFormat = PixelFormat::fromString("YUV420");
    
    if (ctx->config->validate() == CameraConfiguration::Invalid)
        return -1;
    
    if (ctx->camera->configure(ctx->config.get()) < 0)
        return -1;
    
    return 0;
}

int libcamera_start(LibCameraContext* ctx) {
    if (!ctx || !ctx->camera)
        return -1;
    
    ctx->allocator = new FrameBufferAllocator(ctx->camera);
    
    Stream *stream = ctx->config->at(0).stream();
    if (ctx->allocator->allocate(stream) < 0)
        return -1;
    
    return ctx->camera->start();
}

int libcamera_stop(LibCameraContext* ctx) {
    if (!ctx || !ctx->camera)
        return -1;
    
    return ctx->camera->stop();
}

/**
 * Capture a single frame and return its buffer and size.
 * The caller is responsible for unmapping the buffer.
 * Returns 0 on success, -1 on failure.
 */
int libcamera_capture_frame(LibCameraContext* ctx, uint8_t** out_buffer,
                            size_t* out_size, int timeout_ms) {
    if (!ctx || !ctx->camera || !ctx->allocator)
        return -1;
    
    std::unique_ptr<Request> request = ctx->camera->createRequest();
    if (!request)
        return -1;
    
    Stream *stream = ctx->config->at(0).stream();
    const auto &buffers = ctx->allocator->buffers(stream);
    if (buffers.empty())
        return -1;
    
    request->addBuffer(stream, buffers[0].get());
    
    int ret = ctx->camera->queueRequest(request.get());
    if (ret < 0)
        return -1;
    
    // Wait for request completion (simplified for example purposes)
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
    
    FrameBuffer *buffer = request->findBuffer(stream);
    if (!buffer)
        return -1;
    
    // Map buffer and return data
    const FrameBuffer::Plane &plane = buffer->planes().front();
    void *mem = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
    if (mem == MAP_FAILED)
        return -1;
    *out_buffer = static_cast<uint8_t*>(mem);
    *out_size = plane.length;
    
    return 0;
}

void libcamera_cleanup(LibCameraContext* ctx) {
    if (!ctx)
        return;
    
    if (ctx->camera) {
        ctx->camera->stop();
        ctx->camera->release();
    }
    
    if (ctx->allocator)
        delete ctx->allocator;
    
    if (ctx->camera_manager)
        ctx->camera_manager->stop();
    
    delete ctx;
}



}