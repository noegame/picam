#include <libcamera/libcamera.h>
#include <iostream>
#include <memory>
#include <iomanip>
#include <fstream>
#include <thread>
#include <chrono>

using namespace libcamera;

int main() {
    // Initialize camera manager
    std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
    cm->start();

    // Get available cameras
    if (cm->cameras().empty()) {
        std::cerr << "No cameras available" << std::endl;
        return -1;
    }

    // Get the first camera
    std::shared_ptr<Camera> camera = cm->cameras()[0];
    std::cout << "Using camera: " << camera->id() << std::endl;

    // Acquire the camera
    if (camera->acquire()) {
        std::cerr << "Failed to acquire camera" << std::endl;
        return -1;
    }

    // Generate camera configuration
    std::unique_ptr<CameraConfiguration> config = 
        camera->generateConfiguration({StreamRole::StillCapture});
    
    // Validate and apply configuration
    config->validate();
    camera->configure(config.get());

    // Allocate buffers
    FrameBufferAllocator allocator(camera);
    for (StreamConfiguration &cfg : *config) {
        allocator.allocate(cfg.stream());
    }

    // Create requests
    std::vector<std::unique_ptr<Request>> requests;
    for (StreamConfiguration &cfg : *config) {
        Stream *stream = cfg.stream();
        const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator.buffers(stream);
        std::unique_ptr<Request> request = camera->createRequest();
        request->addBuffer(stream, buffers[0].get());
        requests.push_back(std::move(request));
    }

    // Start camera and queue request
    camera->start();
    camera->queueRequest(requests[0].get());

    // Wait for completion (simplified - you'd normally use event loop)
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Stop and cleanup
    camera->stop();
    allocator.free(config->at(0).stream());
    camera->release();
    cm->stop();

    std::cout << "Picture captured successfully" << std::endl;
    return 0;
}