#include <opencv2/opencv.hpp>
#include "opencv_wrapper.h"

extern "C" {

ImageHandle* load_image(const char* path) {
    cv::Mat* image = new cv::Mat();
    *image = cv::imread(path, cv::IMREAD_COLOR);
    
    if (image->empty()) {
        delete image;
        return nullptr;
    }
    
    return reinterpret_cast<ImageHandle*>(image);
}

void release_image(ImageHandle* handle) {
    if (handle != nullptr) {
        cv::Mat* image = reinterpret_cast<cv::Mat*>(handle);
        delete image;
    }
}

int get_image_width(ImageHandle* handle) {
    if (handle == nullptr) return 0;
    cv::Mat* image = reinterpret_cast<cv::Mat*>(handle);
    return image->cols;
}

int get_image_height(ImageHandle* handle) {
    if (handle == nullptr) return 0;
    cv::Mat* image = reinterpret_cast<cv::Mat*>(handle);
    return image->rows;
}

ImageHandle* convert_to_grayscale(ImageHandle* handle) {
    if (handle == nullptr) return nullptr;
    
    cv::Mat* src = reinterpret_cast<cv::Mat*>(handle);
    cv::Mat* gray = new cv::Mat();
    
    cv::cvtColor(*src, *gray, cv::COLOR_BGR2GRAY);
    
    return reinterpret_cast<ImageHandle*>(gray);
}

int save_image(const char* path, ImageHandle* handle) {
    if (handle == nullptr) return 0;
    
    cv::Mat* image = reinterpret_cast<cv::Mat*>(handle);
    return cv::imwrite(path, *image) ? 1 : 0;
}

}