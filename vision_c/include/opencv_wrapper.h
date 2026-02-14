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

// ===== Image Processing Functions =====

// Convert BGRA to BGR
ImageHandle* convert_bgra_to_bgr(ImageHandle* handle);

// Apply mask to image using bitwise AND
ImageHandle* bitwise_and_mask(ImageHandle* image, ImageHandle* mask);

// Apply sharpening filter to image
ImageHandle* sharpen_image(ImageHandle* image);

// Resize image
ImageHandle* resize_image(ImageHandle* image, int new_width, int new_height);

// ===== Drawing Functions =====

// Color structure for drawing
typedef struct {
    unsigned char b;
    unsigned char g;
    unsigned char r;
} Color;

// Add text to image
void put_text(ImageHandle* image, const char* text, int x, int y, 
              double font_scale, Color color, int thickness);

// Fill polygon on image (for mask creation)
ImageHandle* fill_poly(ImageHandle* image, float* points, int num_points, Color color);

// Create empty image (for mask)
ImageHandle* create_empty_image(int width, int height, int channels);

// ===== Geometric Transformation Functions =====

// Opaque handle for matrices
typedef struct MatrixHandle MatrixHandle;

// Point structure
typedef struct {
    float x;
    float y;
} Point2f;

typedef struct {
    float x;
    float y;
    float z;
} Point3f;

// Undistort points using fisheye model
// Returns array of undistorted points (caller must free)
Point2f* fisheye_undistort_points(Point2f* points, int num_points,
                                   float* camera_matrix,  // 3x3 matrix
                                   float* dist_coeffs,     // 4 coefficients
                                   float* output_camera_matrix);  // 3x3 matrix (can be NULL)

// Find homography matrix between two sets of points
// Returns 3x3 homography matrix (caller must free)
float* find_homography(Point2f* src_points, Point2f* dst_points, int num_points);

// Apply perspective transform to points
Point2f* perspective_transform(Point2f* points, int num_points, float* homography);

// ===== Pose Estimation =====

// SolvePnP result structure
typedef struct {
    float rvec[3];  // Rotation vector
    float tvec[3];  // Translation vector
    int success;    // 1 if successful, 0 otherwise
} PnPResult;

// Solve PnP (Perspective-n-Point) problem
PnPResult solve_pnp(Point3f* object_points, Point2f* image_points, int num_points,
                    float* camera_matrix, float* dist_coeffs);

// Free allocated arrays
void free_points_2f(Point2f* points);
void free_points_3f(Point3f* points);
void free_matrix(float* matrix);

#ifdef __cplusplus
}
#endif

#endif // OPENCV_WRAPPER_H