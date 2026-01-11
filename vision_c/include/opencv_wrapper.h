#ifndef OPENCV_WRAPPER_H
#define OPENCV_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for image data
typedef struct ImageHandle ImageHandle;
typedef struct ArucoDictionaryHandle ArucoDictionaryHandle;
typedef struct DetectorParametersHandle DetectorParametersHandle;
typedef struct ArucoDetectorHandle ArucoDetectorHandle;
typedef struct MarkerHandle MarkerHandle; 

// ArUco dictionary constants (matching OpenCV's aruco::PredefinedDictionaryType)
#define DICT_4X4_50 0
#define DICT_4X4_100 1
#define DICT_4X4_250 2
#define DICT_4X4_1000 3
#define DICT_5X5_50 4
#define DICT_5X5_100 5
#define DICT_5X5_250 6
#define DICT_5X5_1000 7
#define DICT_6X6_50 8
#define DICT_6X6_100 9
#define DICT_6X6_250 10
#define DICT_6X6_1000 11
#define DICT_7X7_50 12
#define DICT_7X7_100 13
#define DICT_7X7_250 14
#define DICT_7X7_1000 15 


// Load an image from file
ImageHandle* load_image(const char* path);

// Release image memory
void release_image(ImageHandle* handle);

// Get image dimensions
int get_image_width(ImageHandle* handle);
int get_image_height(ImageHandle* handle);

// Convert image to grayscale
ImageHandle* convert_to_grayscale(ImageHandle* handle);

// Save image to file
int save_image(const char* path, ImageHandle* handle);

// ArUco marker detection structures
typedef struct {
    int id;
    float corners[4][2];  // 4 corners, each with x,y coordinates
    float confidence;
} DetectedMarker;

typedef struct {
    DetectedMarker* markers;
    int count;
} DetectionResult;

// ArUco functions
ArucoDictionaryHandle* getPredefinedDictionary(int dict_id);
DetectorParametersHandle* createDetectorParameters();
ArucoDetectorHandle* createArucoDetector(ArucoDictionaryHandle* dict, DetectorParametersHandle* params);
void releaseArucoDetector(ArucoDetectorHandle* detector);
void releaseArucoDictionary(ArucoDictionaryHandle* dict);
void releaseDetectorParameters(DetectorParametersHandle* params);

// Detect markers with confidence scores
DetectionResult* detectMarkersWithConfidence(ArucoDetectorHandle* detector, ImageHandle* image);
void releaseDetectionResult(DetectionResult* result);

// Draw detected markers on an image
// Returns a new image with markers drawn (original image is not modified)
ImageHandle* drawDetectedMarkers(ImageHandle* image, DetectionResult* result);

#ifdef __cplusplus
}
#endif

#endif // OPENCV_WRAPPER_H