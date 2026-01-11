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

ArucoDictionaryHandle* getPredefinedDictionary(int dict_id) {
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dict_id);
    cv::aruco::Dictionary* dict_ptr = new cv::aruco::Dictionary(dictionary);
    return reinterpret_cast<ArucoDictionaryHandle*>(dict_ptr);
}

DetectorParametersHandle* createDetectorParameters() {
    cv::aruco::DetectorParameters* params = new cv::aruco::DetectorParameters();
    return reinterpret_cast<DetectorParametersHandle*>(params);
}

ArucoDetectorHandle* createArucoDetector(ArucoDictionaryHandle* dict, DetectorParametersHandle* params) {
    if (dict == nullptr || params == nullptr) return nullptr;
    
    cv::aruco::Dictionary* dictionary = reinterpret_cast<cv::aruco::Dictionary*>(dict);
    cv::aruco::DetectorParameters* detectorParams = reinterpret_cast<cv::aruco::DetectorParameters*>(params);
    
    cv::aruco::ArucoDetector* detector = new cv::aruco::ArucoDetector(*dictionary, *detectorParams);
    return reinterpret_cast<ArucoDetectorHandle*>(detector);
}

void releaseArucoDetector(ArucoDetectorHandle* detector) {
    if (detector != nullptr) {
        cv::aruco::ArucoDetector* det = reinterpret_cast<cv::aruco::ArucoDetector*>(detector);
        delete det;
    }
}

void releaseArucoDictionary(ArucoDictionaryHandle* dict) {
    if (dict != nullptr) {
        cv::aruco::Dictionary* dictionary = reinterpret_cast<cv::aruco::Dictionary*>(dict);
        delete dictionary;
    }
}

void releaseDetectorParameters(DetectorParametersHandle* params) {
    if (params != nullptr) {
        cv::aruco::DetectorParameters* detectorParams = reinterpret_cast<cv::aruco::DetectorParameters*>(params);
        delete detectorParams;
    }
}

DetectionResult* detectMarkersWithConfidence(ArucoDetectorHandle* detector, ImageHandle* image) {
    if (detector == nullptr || image == nullptr) return nullptr;
    
    cv::aruco::ArucoDetector* det = reinterpret_cast<cv::aruco::ArucoDetector*>(detector);
    cv::Mat* img = reinterpret_cast<cv::Mat*>(image);
    
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    
    // Detect markers
    det->detectMarkers(*img, markerCorners, markerIds, rejectedCandidates);
    
    // Create result structure
    DetectionResult* result = new DetectionResult();
    result->count = markerIds.size();
    
    if (result->count > 0) {
        result->markers = new DetectedMarker[result->count];
        
        for (int i = 0; i < result->count; i++) {
            result->markers[i].id = markerIds[i];
            
            // Copy corner coordinates
            for (int j = 0; j < 4; j++) {
                result->markers[i].corners[j][0] = markerCorners[i][j].x;
                result->markers[i].corners[j][1] = markerCorners[i][j].y;
            }
            
            // Calculate confidence score based on corner quality
            // For now, we use a simple metric: marker perimeter vs expected perimeter
            // A better metric would consider image quality, corner sharpness, etc.
            float perimeter = 0.0f;
            for (int j = 0; j < 4; j++) {
                int next = (j + 1) % 4;
                float dx = markerCorners[i][next].x - markerCorners[i][j].x;
                float dy = markerCorners[i][next].y - markerCorners[i][j].y;
                perimeter += std::sqrt(dx * dx + dy * dy);
            }
            
            // Normalize confidence (larger markers = higher confidence, up to 1.0)
            // This is a simplified metric; adjust based on your needs
            result->markers[i].confidence = std::min(1.0f, perimeter / 400.0f);
        }
    } else {
        result->markers = nullptr;
    }
    
    return result;
}

void releaseDetectionResult(DetectionResult* result) {
    if (result != nullptr) {
        if (result->markers != nullptr) {
            delete[] result->markers;
        }
        delete result;
    }
}

ImageHandle* drawDetectedMarkers(ImageHandle* image, DetectionResult* result) {
    if (image == nullptr || result == nullptr) return nullptr;
    
    cv::Mat* src = reinterpret_cast<cv::Mat*>(image);
    cv::Mat* output = new cv::Mat();
    src->copyTo(*output);
    
    // Draw each detected marker
    for (int i = 0; i < result->count; i++) {
        DetectedMarker* marker = &result->markers[i];
        
        // Convert corners to cv::Point format
        std::vector<cv::Point2f> corners;
        for (int j = 0; j < 4; j++) {
            corners.push_back(cv::Point2f(marker->corners[j][0], marker->corners[j][1]));
        }
        
        // Draw marker outline
        for (int j = 0; j < 4; j++) {
            cv::line(*output, corners[j], corners[(j + 1) % 4], 
                    cv::Scalar(0, 255, 0), 2);
        }
        
        // Draw corner circles
        for (int j = 0; j < 4; j++) {
            cv::circle(*output, corners[j], 3, cv::Scalar(255, 0, 0), -1);
        }
        
        // Calculate center point
        cv::Point2f center(0, 0);
        for (int j = 0; j < 4; j++) {
            center.x += corners[j].x;
            center.y += corners[j].y;
        }
        center.x /= 4.0f;
        center.y /= 4.0f;
        
        // Draw marker ID
        char text[32];
        snprintf(text, sizeof(text), "ID:%d", marker->id);
        cv::putText(*output, text, cv::Point(center.x - 20, center.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        
        // Draw confidence score
        snprintf(text, sizeof(text), "%.2f", marker->confidence);
        cv::putText(*output, text, cv::Point(center.x - 20, center.y + 15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    }
    
    return reinterpret_cast<ImageHandle*>(output);
}


} // extern "C"

