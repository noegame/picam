/**
 * test_aruco_detection_pipeline.c
 * 
 * Test program for ArUco detection pipeline using OpenCV wrapper.
 * This follows the same detection pipeline as the Python implementation:
 * - Load image
 * - Apply sharpening filter
 * - Resize image (scale 1.5x)
 * - Detect ArUco markers
 * - Calculate marker centers
 * - Annotate image with IDs, centers, and counter
 * - Save annotated image
 */

#include "opencv_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Camera calibration parameters (from get_camera_matrix and get_distortion_matrix in aruco.py)
static float CAMERA_MATRIX[9] = {
    2493.62477, 0.0, 1977.18701,
    0.0, 2493.11358, 2034.91176,
    0.0, 0.0, 1.0
};

static float DIST_COEFFS[4] = {
    -0.1203345, 0.06802544, -0.13779641, 0.08243704
};

// Detection parameters (from get_aruco_detector in aruco.py)
typedef struct {
    int adaptiveThreshWinSizeMin;
    int adaptiveThreshWinSizeMax;
    int adaptiveThreshWinSizeStep;
    float minMarkerPerimeterRate;
    float maxMarkerPerimeterRate;
    float polygonalApproxAccuracyRate;
    int cornerRefinementWinSize;
    int cornerRefinementMaxIterations;
    int minDistanceToBorder;
    float minOtsuStdDev;
    float perspectiveRemoveIgnoredMarginPerCell;
} ArucoParams;

// Structure to hold marker center coordinates
typedef struct {
    float x;
    float y;
} MarkerCenter;

// Structure to hold marker counts by category
typedef struct {
    int black_markers;   // ID 41
    int blue_markers;    // ID 36
    int yellow_markers;  // ID 47
    int robot_markers;   // IDs 1-10
    int fixed_markers;   // IDs 20-23
    int total;
} MarkerCounts;

/**
 * Configure ArUco detector parameters matching Python implementation.
 * These optimized parameters enable detection of ~40 markers instead of just 7.
 * Values from get_aruco_detector() in Python code.
 */
void configure_detector_parameters(DetectorParametersHandle* params) {
    // Adaptive thresholding parameters
    setAdaptiveThreshWinSizeMin(params, 3);
    setAdaptiveThreshWinSizeMax(params, 53);
    setAdaptiveThreshWinSizeStep(params, 4);
    
    // Marker size constraints
    setMinMarkerPerimeterRate(params, 0.01);
    setMaxMarkerPerimeterRate(params, 4.0);
    
    // Polygon approximation accuracy
    setPolygonalApproxAccuracyRate(params, 0.05);
    
    // Corner refinement for sub-pixel accuracy
    setCornerRefinementMethod(params, CORNER_REFINE_SUBPIX);
    setCornerRefinementWinSize(params, 5);
    setCornerRefinementMaxIterations(params, 50);
    
    // Detection constraints
    setMinDistanceToBorder(params, 0);
    setMinOtsuStdDev(params, 2.0);
    
    // Perspective removal
    setPerspectiveRemoveIgnoredMarginPerCell(params, 0.15);
}

/**
 * Check if a marker ID is valid according to Eurobot 2026 rules.
 * Valid IDs:
 * - 1-5: Blue team robots
 * - 6-10: Yellow team robots
 * - 20-23: Fixed markers on field
 * - 36: Blue box
 * - 41: Empty box (black)
 * - 47: Yellow box
 */
int is_valid_marker_id(int id) {
    return (id >= 1 && id <= 10) ||    // Robots
           (id >= 20 && id <= 23) ||   // Fixed markers
           (id == 36) ||                // Blue box
           (id == 41) ||                // Empty box
           (id == 47);                  // Yellow box
}

/**
 * Filter detection results to keep only valid marker IDs.
 * Returns a new DetectionResult with only valid markers.
 * Original result should be freed by caller after using the filtered result.
 */
DetectionResult* filter_valid_markers(DetectionResult* result) {
    if (result == NULL || result->count == 0) return result;
    
    // Count valid markers
    int valid_count = 0;
    for (int i = 0; i < result->count; i++) {
        if (is_valid_marker_id(result->markers[i].id)) {
            valid_count++;
        }
    }
    
    // If all markers are valid, return original result
    if (valid_count == result->count) return result;
    
    // Create new filtered result
    DetectionResult* filtered = (DetectionResult*)malloc(sizeof(DetectionResult));
    if (filtered == NULL) return result;
    
    filtered->count = valid_count;
    if (valid_count == 0) {
        filtered->markers = NULL;
        return filtered;
    }
    
    filtered->markers = (DetectedMarker*)malloc(sizeof(DetectedMarker) * valid_count);
    if (filtered->markers == NULL) {
        free(filtered);
        return result;
    }
    
    // Copy valid markers
    int idx = 0;
    for (int i = 0; i < result->count; i++) {
        if (is_valid_marker_id(result->markers[i].id)) {
            filtered->markers[idx] = result->markers[i];
            idx++;
        }
    }
    
    return filtered;
}

/**
 * Categorize and count markers by type based on ArUco IDs
 * - ID 36: blue box
 * - ID 41: empty box (black)
 * - ID 47: yellow box
 * - IDs 1-10: robots
 * - IDs 20-23: fixed markers
 */
MarkerCounts count_markers_by_category(DetectionResult* result) {
    MarkerCounts counts = {0, 0, 0, 0, 0, 0};
    
    for (int i = 0; i < result->count; i++) {
        int id = result->markers[i].id;
        
        if (id == 41) {
            counts.black_markers++;
        } else if (id == 36) {
            counts.blue_markers++;
        } else if (id == 47) {
            counts.yellow_markers++;
        } else if (id >= 1 && id <= 10) {
            counts.robot_markers++;
        } else if (id >= 20 && id <= 23) {
            counts.fixed_markers++;
        }
        counts.total++;
    }
    
    return counts;
}

/**
 * Calculate the center coordinates of a marker from its corners
 */
MarkerCenter calculate_marker_center(DetectedMarker* marker) {
    MarkerCenter center;
    center.x = 0.0f;
    center.y = 0.0f;
    
    // Average of all 4 corners
    for (int i = 0; i < 4; i++) {
        center.x += marker->corners[i][0];
        center.y += marker->corners[i][1];
    }
    center.x /= 4.0f;
    center.y /= 4.0f;
    
    return center;
}

/**
 * Annotate image with marker IDs
 * Follows the logic from annotate_img_with_ids in aruco.py
 */
void annotate_with_ids(ImageHandle* image, DetectionResult* result, MarkerCenter* centers) {
    Color black = {0, 0, 0};
    Color green = {0, 255, 0};
    double font_scale = 0.5;
    
    for (int i = 0; i < result->count; i++) {
        char text[32];
        snprintf(text, sizeof(text), "ID:%d", result->markers[i].id);
        
        int x = (int)centers[i].x;
        int y = (int)centers[i].y;
        
        // Black outline
        put_text(image, text, x, y, font_scale, black, 3);
        // Green text
        put_text(image, text, x, y, font_scale, green, 1);
    }
}

/**
 * Annotate image with center coordinates
 * Follows the logic from annotate_img_with_centers in aruco.py
 */
void annotate_with_centers(ImageHandle* image, MarkerCenter* centers, int count) {
    Color black = {0, 0, 0};
    Color blue = {255, 0, 0};
    double font_scale = 0.5;
    
    for (int i = 0; i < count; i++) {
        char text[64];
        snprintf(text, sizeof(text), "(%d,%d)", (int)centers[i].x, (int)centers[i].y);
        
        int x = (int)centers[i].x;
        int y = (int)centers[i].y - 20;
        
        // Black outline
        put_text(image, text, x, y, font_scale, black, 3);
        // Blue text
        put_text(image, text, x, y, font_scale, blue, 1);
    }
}

/**
 * Annotate image with categorized marker counts
 * Displays counts for black, blue, yellow, robots, and total markers
 */
void annotate_with_counter(ImageHandle* image, MarkerCounts counts) {
    Color black = {0, 0, 0};
    Color green = {0, 255, 0};
    double font_scale = 0.8;  // Reduced font size
    int line_height = 35;     // Spacing between lines
    int start_x = 30;
    int start_y = 40;
    
    char text[64];
    
    // Black markers
    snprintf(text, sizeof(text), "black markers : %d", counts.black_markers);
    put_text(image, text, start_x, start_y, font_scale, black, 3);
    put_text(image, text, start_x, start_y, font_scale, green, 2);
    
    // Blue markers
    snprintf(text, sizeof(text), "blue markers : %d", counts.blue_markers);
    put_text(image, text, start_x, start_y + line_height, font_scale, black, 3);
    put_text(image, text, start_x, start_y + line_height, font_scale, green, 2);
    
    // Yellow markers
    snprintf(text, sizeof(text), "yellow markers : %d", counts.yellow_markers);
    put_text(image, text, start_x, start_y + line_height * 2, font_scale, black, 3);
    put_text(image, text, start_x, start_y + line_height * 2, font_scale, green, 2);
    
    // Robot markers
    snprintf(text, sizeof(text), "robots markers : %d", counts.robot_markers);
    put_text(image, text, start_x, start_y + line_height * 3, font_scale, black, 3);
    put_text(image, text, start_x, start_y + line_height * 3, font_scale, green, 2);
    
    // Fixed markers
    snprintf(text, sizeof(text), "fixed markers : %d", counts.fixed_markers);
    put_text(image, text, start_x, start_y + line_height * 4, font_scale, black, 3);
    put_text(image, text, start_x, start_y + line_height * 4, font_scale, green, 2);
    
    // Total
    snprintf(text, sizeof(text), "total : %d", counts.total);
    put_text(image, text, start_x, start_y + line_height * 5, font_scale, black, 3);
    put_text(image, text, start_x, start_y + line_height * 5, font_scale, green, 2);
}

/**
 * Annotate image with real-world coordinates
 * Follows the logic from annotate_img_with_real_coords in aruco.py
 */
void annotate_with_real_coords(ImageHandle* image, MarkerCenter* centers, 
                                Point3f* real_coords, int count) {
    Color black = {0, 0, 0};
    Color cyan = {255, 255, 0};
    double font_scale = 0.4;
    
    for (int i = 0; i < count; i++) {
        char text[64];
        snprintf(text, sizeof(text), "(%d,%d,%d)mm", 
                 (int)real_coords[i].x, (int)real_coords[i].y, (int)real_coords[i].z);
        
        int x = (int)centers[i].x + 50;
        int y = (int)centers[i].y;
        
        // Black outline
        put_text(image, text, x, y, font_scale, black, 3);
        // Cyan text
        put_text(image, text, x, y, font_scale, cyan, 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <image_path> [output_path]\n", argv[0]);
        printf("  image_path: Input image to process\n");
        printf("  output_path: Path to save annotated image (default: output_annotated.jpg)\n");
        printf("\nThis program follows the Python ArUco detection pipeline:\n");
        printf("  1. Load image\n");
        printf("  2. Apply sharpening filter\n");
        printf("  3. Resize image (1.5x scale)\n");
        printf("  4. Detect ArUco markers (DICT_4X4_50)\n");
        printf("  5. Calculate marker centers\n");
        printf("  6. Annotate image with IDs, centers, and counter\n");
        printf("  7. Save annotated image\n");
        return -1;
    }

    const char* input_path = argv[1];
    const char* output_path = (argc >= 3) ? argv[2] : "output_annotated.jpg";
    
    printf("=== ArUco Detection Pipeline Test ===\n\n");
    
    // ========== STEP 1: LOAD IMAGE ==========
    printf("[1/7] Loading image: %s\n", input_path);
    ImageHandle* image = load_image(input_path);
    if (image == NULL) {
        fprintf(stderr, "Error: Could not load image from %s\n", input_path);
        return -1;
    }
    
    int orig_width = get_image_width(image);
    int orig_height = get_image_height(image);
    printf("      Image loaded: %dx%d pixels\n", orig_width, orig_height);
    
    // ========== STEP 2: APPLY SHARPENING ==========
    printf("[2/7] Applying sharpening filter...\n");
    ImageHandle* sharpened = sharpen_image(image);
    if (sharpened == NULL) {
        fprintf(stderr, "Error: Failed to sharpen image\n");
        release_image(image);
        return -1;
    }
    release_image(image);  // Free original image
    printf("      Sharpening applied\n");
    
    // ========== STEP 3: RESIZE IMAGE ==========
    float scale = 1.5f;
    int new_width = (int)(orig_width * scale);
    int new_height = (int)(orig_height * scale);
    
    printf("[3/7] Resizing image (scale: %.1fx)\n", scale);
    ImageHandle* resized = resize_image(sharpened, new_width, new_height);
    if (resized == NULL) {
        fprintf(stderr, "Error: Failed to resize image\n");
        release_image(sharpened);
        return -1;
    }
    release_image(sharpened);  // Free sharpened image
    printf("      Resized to: %dx%d pixels\n", new_width, new_height);
    
    // ========== STEP 4: DETECT MARKERS ==========
    printf("[4/7] Detecting ArUco markers (DICT_4X4_50)...\n");
    
    // Create detector with DICT_4X4_50 (same as Python code)
    ArucoDictionaryHandle* dictionary = getPredefinedDictionary(DICT_4X4_50);
    if (dictionary == NULL) {
        fprintf(stderr, "Error: Could not create ArUco dictionary\n");
        release_image(resized);
        return -1;
    }
    
    DetectorParametersHandle* params = createDetectorParameters();
    if (params == NULL) {
        fprintf(stderr, "Error: Could not create detector parameters\n");
        releaseArucoDictionary(dictionary);
        release_image(resized);
        return -1;
    }
    
    // Configure detector parameters to match Python implementation
    // This enables detection of ~40 markers instead of just 7
    configure_detector_parameters(params);
    printf("      Detector parameters configured (matching Python)\n");
    
    ArucoDetectorHandle* detector = createArucoDetector(dictionary, params);
    if (detector == NULL) {
        fprintf(stderr, "Error: Could not create ArUco detector\n");
        releaseDetectorParameters(params);
        releaseArucoDictionary(dictionary);
        release_image(resized);
        return -1;
    }
    
    // Detect markers on resized image
    DetectionResult* result_raw = detectMarkersWithConfidence(detector, resized);
    if (result_raw == NULL) {
        fprintf(stderr, "Error: Detection failed\n");
        releaseArucoDetector(detector);
        releaseDetectorParameters(params);
        releaseArucoDictionary(dictionary);
        release_image(resized);
        return -1;
    }
    
    printf("      Detected %d marker(s) (raw)\n", result_raw->count);
    
    // Filter to keep only valid marker IDs (reject false positives)
    DetectionResult* result = filter_valid_markers(result_raw);
    int rejected_count = result_raw->count - result->count;
    
    if (rejected_count > 0) {
        printf("      Filtered out %d invalid marker(s)\n", rejected_count);
    }
    printf("      Valid markers: %d\n", result->count);
    
    // Free raw result if it was replaced by filtered result
    if (result != result_raw) {
        releaseDetectionResult(result_raw);
    }
    
    // ========== STEP 5: CALCULATE CENTERS ==========
    printf("[5/7] Calculating marker centers...\n");
    
    MarkerCenter* centers = NULL;
    if (result->count > 0) {
        centers = (MarkerCenter*)malloc(sizeof(MarkerCenter) * result->count);
        if (centers == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            releaseDetectionResult(result);
            releaseArucoDetector(detector);
            releaseDetectorParameters(params);
            releaseArucoDictionary(dictionary);
            release_image(resized);
            return -1;
        }
        
        for (int i = 0; i < result->count; i++) {
            centers[i] = calculate_marker_center(&result->markers[i]);
            
            // Scale centers back to original coordinates (divide by scale)
            centers[i].x /= scale;
            centers[i].y /= scale;
            
            printf("      Marker ID %d: center at (%.1f, %.1f)\n", 
                   result->markers[i].id, centers[i].x, centers[i].y);
        }
    }
    
    // Release resized image as we need to work with original size
    release_image(resized);
    
    // Reload original image for annotation
    image = load_image(input_path);
    if (image == NULL) {
        fprintf(stderr, "Error: Could not reload original image\n");
        if (centers) free(centers);
        releaseDetectionResult(result);
        releaseArucoDetector(detector);
        releaseDetectorParameters(params);
        releaseArucoDictionary(dictionary);
        return -1;
    }
    
    // ========== STEP 6: ANNOTATE IMAGE ==========
    printf("[6/7] Annotating image...\n");
    
    // Count markers by category
    MarkerCounts counts = count_markers_by_category(result);
    
    if (result->count > 0) {
        annotate_with_counter(image, counts);
        annotate_with_ids(image, result, centers);
        annotate_with_centers(image, centers, result->count);
        printf("      Annotations added: categorized counts, IDs, centers\n");
    } else {
        annotate_with_counter(image, counts);  // Still show counts (all zeros)
        printf("      No markers to annotate\n");
    }
    
    // ========== STEP 7: SAVE ANNOTATED IMAGE ==========
    printf("[7/7] Saving annotated image to: %s\n", output_path);
    
    if (save_image(output_path, image)) {
        printf("      Annotated image saved successfully!\n");
    } else {
        fprintf(stderr, "Error: Failed to save annotated image\n");
    }
    
    // ========== RESULTS SUMMARY ==========
    printf("\n=== Detection Results Summary ===\n");
    printf("Black markers  : %d\n", counts.black_markers);
    printf("Blue markers   : %d\n", counts.blue_markers);
    printf("Yellow markers : %d\n", counts.yellow_markers);
    printf("Robots markers : %d\n", counts.robot_markers);
    printf("Fixed markers  : %d\n", counts.fixed_markers);
    printf("Total markers  : %d\n\n", counts.total);
    
    for (int i = 0; i < result->count; i++) {
        printf("Marker #%d:\n", i + 1);
        printf("  ID: %d\n", result->markers[i].id);
        printf("  Center: (%.1f, %.1f)\n", centers[i].x, centers[i].y);
        printf("  Confidence: %.3f\n", result->markers[i].confidence);
        printf("  Corners (scaled back):\n");
        for (int j = 0; j < 4; j++) {
            printf("    [%d] (%.1f, %.1f)\n", j,
                   result->markers[i].corners[j][0] / scale,
                   result->markers[i].corners[j][1] / scale);
        }
        printf("\n");
    }
    
    // ========== CLEANUP ==========
    if (centers) free(centers);
    releaseDetectionResult(result);
    releaseArucoDetector(detector);
    releaseDetectorParameters(params);
    releaseArucoDictionary(dictionary);
    release_image(image);
    
    printf("Pipeline test complete!\n");
    return 0;
}
