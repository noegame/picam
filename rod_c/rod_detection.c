/**
 * @file rod_detection.c
 * @brief Computer vision thread for ArUco marker detection
 * @author Noé Game
 * @date 15/02/2026
 * @see rod_detection.c
 * @copyright Cecill-C (Cf. LICENCE.txt)
 * 
 * This program implements the computer vision thread that:
 * - Captures images using the emulated camera
 * - Detects ArUco markers on game elements
 * - Sends detected positions to the IPC thread via socket communication
 */


/* ******************************************************* Includes ****************************************************** */

#include "emulated_camera.h"
#include "opencv_wrapper.h"
#include "rod_cv.h"
#include "rod_config.h"
#include "rod_visualization.h"
#include "rod_socket.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <stdbool.h>

/* ***************************************************** Public macros *************************************************** */

// Default image folder path for emulated camera
#define DEFAULT_IMAGE_FOLDER ROD_DEFAULT_IMAGE_FOLDER

// Debug image output folder
#define DEBUG_OUTPUT_FOLDER ROD_DEBUG_OUTPUT_FOLDER

// Socket configuration
#define SOCKET_PATH ROD_SOCKET_PATH
#define MAX_DETECTION_SIZE ROD_MAX_DETECTION_SIZE

// Debug image saving (save one annotated image every N frames)
#define SAVE_DEBUG_IMAGE_INTERVAL ROD_SAVE_DEBUG_IMAGE_INTERVAL

/* ************************************************** Public types definition ******************************************** */

/**
 * @brief Application context
 */
typedef struct {
    EmulatedCameraContext* camera;
    ArucoDetectorHandle* detector;
    ArucoDictionaryHandle* dictionary;
    DetectorParametersHandle* params;
    RodSocketServer* socket_server;
    bool running;
} AppContext;

/* *********************************************** Public functions declarations ***************************************** */

/**
 * @brief Initialize application context
 * @param ctx Application context
 * @param image_folder Path to image folder
 * @return 0 on success, -1 on failure
 */
static int init_app_context(AppContext* ctx, const char* image_folder);

/**
 * @brief Cleanup application context
 * @param ctx Application context
 */
static void cleanup_app_context(AppContext* ctx);



/**
 * @brief Signal handler for graceful shutdown
 * @param signum Signal number
 */
static void signal_handler(int signum);

/* ******************************************* Global variables ******************************************************* */

static volatile bool g_running = true;

/* ******************************************* Public callback functions declarations ************************************ */

static void signal_handler(int signum) {
    (void)signum;  // Unused parameter
    g_running = false;
    printf("\nReceived interrupt signal, shutting down...\n");
}

static int init_app_context(AppContext* ctx, const char* image_folder) {
    memset(ctx, 0, sizeof(AppContext));
    ctx->socket_server = NULL;
    ctx->running = true;
    
    // Initialize emulated camera
    printf("Initializing emulated camera...\n");
    ctx->camera = emulated_camera_init();
    if (!ctx->camera) {
        fprintf(stderr, "Failed to initialize emulated camera\n");
        return -1;
    }
    
    // Set camera folder
    if (emulated_camera_set_folder(ctx->camera, image_folder) != 0) {
        fprintf(stderr, "Failed to set image folder: %s\n", image_folder);
        return -1;
    }
    
    // Start camera
    if (emulated_camera_start(ctx->camera) != 0) {
        fprintf(stderr, "Failed to start emulated camera\n");
        return -1;
    }
    printf("Emulated camera started with folder: %s\n", image_folder);
    
    // Initialize ArUco detector
    printf("Initializing ArUco detector...\n");
    ctx->dictionary = getPredefinedDictionary(rod_config_get_aruco_dictionary_type());
    if (!ctx->dictionary) {
        fprintf(stderr, "Failed to create ArUco dictionary\n");
        return -1;
    }
    
    ctx->params = createDetectorParameters();
    if (!ctx->params) {
        fprintf(stderr, "Failed to create detector parameters\n");
        return -1;
    }
    
    // Configure detector with optimized parameters
    rod_config_configure_detector_parameters(ctx->params);
    
    ctx->detector = createArucoDetector(ctx->dictionary, ctx->params);
    if (!ctx->detector) {
        fprintf(stderr, "Failed to create ArUco detector\n");
        return -1;
    }
    printf("ArUco detector initialized (DICT_4X4_50)\n");
    
    return 0;
}

static void cleanup_app_context(AppContext* ctx) {
    if (!ctx) return;
    
    // Cleanup ArUco detector
    if (ctx->detector) {
        releaseArucoDetector(ctx->detector);
        ctx->detector = NULL;
    }
    
    if (ctx->dictionary) {
        releaseArucoDictionary(ctx->dictionary);
        ctx->dictionary = NULL;
    }
    
    if (ctx->params) {
        releaseDetectorParameters(ctx->params);
        ctx->params = NULL;
    }
    
    // Cleanup camera
    if (ctx->camera) {
        emulated_camera_stop(ctx->camera);
        emulated_camera_cleanup(ctx->camera);
        ctx->camera = NULL;
    }
    
    // Close socket
    if (ctx->socket_server) {
        rod_socket_server_destroy(ctx->socket_server);
        ctx->socket_server = NULL;
    }
}

/**
 * @brief Main function of the program
 * Takes a picture with the emulated camera, find the aruco markers position with rod-cv, 
 * and send the position to rod-com via socket.
 */
int main(int argc, char* argv[]) {
    AppContext ctx;
    const char* image_folder = DEFAULT_IMAGE_FOLDER;
    
    // Parse command line arguments
    if (argc > 1) {
        image_folder = argv[1];
    }
    
    printf("=== ROD Detection - Computer Vision Thread ===\n");
    printf("Image folder: %s\n\n", image_folder);
    
    // Setup signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Initialize application context
    if (init_app_context(&ctx, image_folder) != 0) {
        fprintf(stderr, "Failed to initialize application\n");
        cleanup_app_context(&ctx);
        return 1;
    }
    
    // Initialize socket server
    ctx.socket_server = rod_socket_server_create(SOCKET_PATH);
    if (!ctx.socket_server) {
        fprintf(stderr, "Failed to initialize socket server\n");
        cleanup_app_context(&ctx);
        return 1;
    }
    
    printf("\nStarting detection loop (Ctrl+C to stop)...\n");
    
    // Main detection loop
    int frame_count = 0;
    while (g_running && ctx.running) {
        frame_count++;
        
        // Try to accept a client connection if not already connected
        rod_socket_server_accept(ctx.socket_server);
        
        // Capture image from emulated camera
        uint8_t* image_buffer = NULL;
        int width, height;
        size_t size;
        
        if (emulated_camera_take_picture(ctx.camera, &image_buffer, 
                                         &width, &height, &size) != 0) {
            fprintf(stderr, "Failed to capture image\n");
            usleep(10000);  // Wait 10ms before retry
            continue;
        }
        
        // Create OpenCV image handle from RGB buffer
        // emulated_camera returns RGB format (format=1)
        ImageHandle* image = create_image_from_buffer(image_buffer, width, height, 3, 1);
        free(image_buffer);  // Buffer is copied by create_image_from_buffer
        
        if (!image) {
            fprintf(stderr, "Failed to create image from buffer\n");
            usleep(10000);  // Wait 10ms before retry
            continue;
        }
        
        // Detect ArUco markers
        DetectionResult* detection = detectMarkersWithConfidence(ctx.detector, image);
        
        if (detection && detection->count > 0) {
            // Filter and process detected markers using rod_cv module
            MarkerData markers[100];  // Max 100 markers
            int valid_count = filter_valid_markers(detection, markers, 100);
            
            // Send detection results
            if (valid_count > 0) {
                rod_socket_server_send_detections(ctx.socket_server, markers, valid_count);
            }
            
            // Save debug image periodically using rod_visualization module
            if (frame_count % SAVE_DEBUG_IMAGE_INTERVAL == 0) {
                rod_viz_save_debug_image(image, markers, valid_count, frame_count, DEBUG_OUTPUT_FOLDER);
            }
            
            releaseDetectionResult(detection);
        } else {
            // No markers detected
            if (frame_count % 10 == 0) {
                printf("Frame %d: No markers detected\n", frame_count);
            }
            
            // Save debug image periodically even when no markers detected
            if (frame_count % SAVE_DEBUG_IMAGE_INTERVAL == 0) {
                MarkerData empty_markers[1];
                rod_viz_save_debug_image(image, empty_markers, 0, frame_count, DEBUG_OUTPUT_FOLDER);
            }
        }
        
        // Release image
        release_image(image);
    }
    
    printf("\nShutting down...\n");
    printf("Total frames processed: %d\n", frame_count);
    
    // Cleanup
    cleanup_app_context(&ctx);
    
    printf("ROD Detection stopped successfully\n");
    return 0;
}