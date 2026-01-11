#include "opencv_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <image_path> [dict_id] [output_path]\n", argv[0]);
        printf("  dict_id: ArUco dictionary (default: 10 = DICT_6X6_250)\n");
        printf("    0=DICT_4X4_50, 6=DICT_5X5_1000, 10=DICT_6X6_250, etc.\n");
        printf("  output_path: Path to save annotated image (optional)\n");
        return -1;
    }

    const char* input_path = argv[1];
    int dict_id = (argc >= 3) ? atoi(argv[2]) : DICT_4X4_50;
    const char* output_path = (argc >= 4) ? argv[3] : NULL;
    
    // Load image
    ImageHandle* image = load_image(input_path);
    if (image == NULL) {
        fprintf(stderr, "Error: Could not open or find the image: %s\n", input_path);
        return -1;
    }

    printf("Image loaded successfully: %dx%d pixels\n", 
           get_image_width(image), get_image_height(image));

    // Create ArUco detector
    printf("Initializing ArUco detector with dictionary ID %d...\n", dict_id);
    ArucoDictionaryHandle* dictionary = getPredefinedDictionary(dict_id);
    if (dictionary == NULL) {
        fprintf(stderr, "Error: Could not create ArUco dictionary\n");
        release_image(image);
        return -1;
    }

    DetectorParametersHandle* params = createDetectorParameters();
    if (params == NULL) {
        fprintf(stderr, "Error: Could not create detector parameters\n");
        releaseArucoDictionary(dictionary);
        release_image(image);
        return -1;
    }

    ArucoDetectorHandle* detector = createArucoDetector(dictionary, params);
    if (detector == NULL) {
        fprintf(stderr, "Error: Could not create ArUco detector\n");
        releaseDetectorParameters(params);
        releaseArucoDictionary(dictionary);
        release_image(image);
        return -1;
    }

    // Detect ArUco markers
    printf("Detecting ArUco markers...\n");
    DetectionResult* result = detectMarkersWithConfidence(detector, image);
    
    if (result == NULL) {
        fprintf(stderr, "Error: Detection failed\n");
        releaseArucoDetector(detector);
        releaseDetectorParameters(params);
        releaseArucoDictionary(dictionary);
        release_image(image);
        return -1;
    }

    // Display results
    printf("\n=== Detection Results ===\n");
    printf("Found %d marker(s)\n\n", result->count);
    
    for (int i = 0; i < result->count; i++) {
        printf("Marker #%d:\n", i + 1);
        printf("  ID: %d\n", result->markers[i].id);
        printf("  Confidence: %.3f\n", result->markers[i].confidence);
        printf("  Corners:\n");
        for (int j = 0; j < 4; j++) {
            printf("    [%d] (%.2f, %.2f)\n", j,
                   result->markers[i].corners[j][0],
                   result->markers[i].corners[j][1]);
        }
        printf("\n");
    }

    // Draw markers on image if output path is provided
    if (output_path != NULL && result->count > 0) {
        printf("Drawing markers on image...\n");
        ImageHandle* annotated = drawDetectedMarkers(image, result);
        
        if (annotated != NULL) {
            if (save_image(output_path, annotated)) {
                printf("Annotated image saved to: %s\n", output_path);
            } else {
                fprintf(stderr, "Error: Failed to save annotated image\n");
            }
            release_image(annotated);
        } else {
            fprintf(stderr, "Error: Failed to draw markers\n");
        }
    }

    // Clean up
    releaseDetectionResult(result);
    releaseArucoDetector(detector);
    releaseDetectorParameters(params);
    releaseArucoDictionary(dictionary);
    release_image(image);

    printf("Detection complete!\n");
    return 0;
}