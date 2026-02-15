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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <fcntl.h>

/* ***************************************************** Public macros *************************************************** */

// Default image folder path for emulated camera
#define DEFAULT_IMAGE_FOLDER "pictures/2026-01-16-playground-ready"

// Socket configuration
#define SOCKET_PATH "/tmp/rod_detection.sock"
#define MAX_DETECTION_SIZE 1024

// Detection loop timing (microseconds)
#define DETECTION_LOOP_DELAY 100000  // 100ms = 10 FPS

/* ************************************************** Public types definition ******************************************** */

/**
 * @brief Structure to hold marker detection data for transmission
 */
typedef struct {
    int id;
    float x;
    float y;
    float angle;
} MarkerData;

/**
 * @brief Application context
 */
typedef struct {
    EmulatedCameraContext* camera;
    ArucoDetectorHandle* detector;
    ArucoDictionaryHandle* dictionary;
    DetectorParametersHandle* params;
    int socket_fd;
    int client_fd;
    bool running;
} AppContext;

/* *********************************************** Public functions declarations ***************************************** */

/**
 * @brief Configure ArUco detector with optimized parameters
 * @param params Detector parameters handle
 * 
 * These parameters are optimized for Eurobot 2026 detection:
 * - Enables detection of ~40 markers instead of just 7
 * - Matches the Python implementation parameters
 */
static void configure_detector_parameters(DetectorParametersHandle* params);

/**
 * @brief Check if a marker ID is valid for Eurobot 2026
 * @param id Marker ID
 * @return 1 if valid, 0 otherwise
 * 
 * Valid IDs:
 * - 1-5: Blue team robots
 * - 6-10: Yellow team robots  
 * - 20-23: Fixed markers on field
 * - 36: Blue box
 * - 41: Empty box (black)
 * - 47: Yellow box
 */
static int is_valid_marker_id(int id);

/**
 * @brief Calculate marker center from corners
 * @param corners Array of 4 corner points [x,y]
 * @return Center point as Point2f
 */
static Point2f calculate_marker_center(float corners[4][2]);

/**
 * @brief Calculate marker angle from corners
 * @param corners Array of 4 corner points [x,y]
 * @return Angle in radians
 */
static float calculate_marker_angle(float corners[4][2]);

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
 * @brief Initialize socket connection to IPC thread
 * @param ctx Application context
 * @return 0 on success, -1 on failure
 */
static int init_socket(AppContext* ctx);

/**
 * @brief Try to accept a client connection (non-blocking)
 * @param ctx Application context
 * @return 0 if accepted or already connected, -1 on error
 */
static int try_accept_client(AppContext* ctx);

/**
 * @brief Send detection results via socket
 * @param ctx Application context
 * @param markers Array of marker data
 * @param count Number of markers
 * @return 0 on success, -1 on failure
 */
static int send_detection_results(AppContext* ctx, MarkerData* markers, int count);

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

static void configure_detector_parameters(DetectorParametersHandle* params) {
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

static int is_valid_marker_id(int id) {
    return (id >= 1 && id <= 10) ||    // Robots
           (id >= 20 && id <= 23) ||   // Fixed markers
           (id == 36) ||                // Blue box
           (id == 41) ||                // Empty box (black)
           (id == 47);                  // Yellow box
}

static Point2f calculate_marker_center(float corners[4][2]) {
    Point2f center;
    center.x = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4.0f;
    center.y = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4.0f;
    return center;
}

static float calculate_marker_angle(float corners[4][2]) {
    // Calculate angle from corner 0 to corner 1 (top edge of marker)
    float dx = corners[1][0] - corners[0][0];
    float dy = corners[1][1] - corners[0][1];
    return atan2f(dy, dx);
}

static int init_app_context(AppContext* ctx, const char* image_folder) {
    memset(ctx, 0, sizeof(AppContext));
    ctx->socket_fd = -1;
    ctx->client_fd = -1;
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
    ctx->dictionary = getPredefinedDictionary(DICT_4X4_50);
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
    configure_detector_parameters(ctx->params);
    
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
    if (ctx->client_fd >= 0) {
        close(ctx->client_fd);
        ctx->client_fd = -1;
    }
    
    if (ctx->socket_fd >= 0) {
        close(ctx->socket_fd);
        unlink(SOCKET_PATH);
        ctx->socket_fd = -1;
    }
}

static int init_socket(AppContext* ctx) {
    struct sockaddr_un addr;
    
    // Create Unix domain socket
    ctx->socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (ctx->socket_fd < 0) {
        fprintf(stderr, "Failed to create socket: %s\n", strerror(errno));
        return -1;
    }
    
    // Remove existing socket file if it exists
    unlink(SOCKET_PATH);
    
    // Setup socket address
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    
    // Bind socket
    if (bind(ctx->socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Failed to bind socket to %s: %s\n", SOCKET_PATH, strerror(errno));
        close(ctx->socket_fd);
        ctx->socket_fd = -1;
        return -1;
    }
    
    // Listen for connections
    if (listen(ctx->socket_fd, 1) < 0) {
        fprintf(stderr, "Failed to listen on socket: %s\n", strerror(errno));
        close(ctx->socket_fd);
        unlink(SOCKET_PATH);
        ctx->socket_fd = -1;
        return -1;
    }
    
    printf("Socket initialized and listening on %s\n", SOCKET_PATH);
    return 0;
}

static int try_accept_client(AppContext* ctx) {
    // If already have a client, nothing to do
    if (ctx->client_fd >= 0) {
        return 0;
    }
    
    // Try to accept a connection (non-blocking)
    struct sockaddr_un client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    // Set socket to non-blocking mode
    int flags = fcntl(ctx->socket_fd, F_GETFL, 0);
    fcntl(ctx->socket_fd, F_SETFL, flags | O_NONBLOCK);
    
    ctx->client_fd = accept(ctx->socket_fd, (struct sockaddr*)&client_addr, &client_len);
    
    // Restore blocking mode
    fcntl(ctx->socket_fd, F_SETFL, flags);
    
    if (ctx->client_fd >= 0) {
        printf("Client connected to detection socket\n");
        return 0;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // No client waiting, not an error
        return 0;
    } else {
        fprintf(stderr, "Error accepting client: %s\n", strerror(errno));
        return -1;
    }
}

static int send_detection_results(AppContext* ctx, MarkerData* markers, int count) {
    // Format detection results as JSON-like array: [[id, x, y, angle], ...]
    char buffer[MAX_DETECTION_SIZE];
    int offset = 0;
    
    // Start array
    offset += snprintf(buffer + offset, MAX_DETECTION_SIZE - offset, "[");
    
    // Add each marker
    for (int i = 0; i < count; i++) {
        if (i > 0) {
            offset += snprintf(buffer + offset, MAX_DETECTION_SIZE - offset, ",");
        }
        offset += snprintf(buffer + offset, MAX_DETECTION_SIZE - offset,
                          "[%d,%.2f,%.2f,%.4f]",
                          markers[i].id, markers[i].x, markers[i].y, markers[i].angle);
        
        // Check buffer overflow
        if (offset >= MAX_DETECTION_SIZE - 100) {
            fprintf(stderr, "Warning: Detection buffer too small, truncating results\n");
            break;
        }
    }
    
    // Close array and add newline
    offset += snprintf(buffer + offset, MAX_DETECTION_SIZE - offset, "]\n");
    
    // Print to console for debugging
    printf("Detected %d markers: %s", count, buffer);
    
    // Send via socket if client is connected
    if (ctx->client_fd >= 0) {
        ssize_t total_sent = 0;
        ssize_t bytes_to_send = strlen(buffer);
        
        while (total_sent < bytes_to_send) {
            ssize_t sent = send(ctx->client_fd, buffer + total_sent, 
                              bytes_to_send - total_sent, MSG_NOSIGNAL);
            
            if (sent < 0) {
                if (errno == EPIPE || errno == ECONNRESET) {
                    // Client disconnected
                    printf("Client disconnected\n");
                    close(ctx->client_fd);
                    ctx->client_fd = -1;
                    return -1;
                } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // Would block, try again
                    usleep(1000);
                    continue;
                } else {
                    fprintf(stderr, "Error sending data: %s\n", strerror(errno));
                    close(ctx->client_fd);
                    ctx->client_fd = -1;
                    return -1;
                }
            }
            
            total_sent += sent;
        }
    }
    
    return 0;
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
    
    // Initialize socket (currently a stub)
    if (init_socket(&ctx) != 0) {
        fprintf(stderr, "Failed to initialize socket\n");
        cleanup_app_context(&ctx);
        return 1;
    }
    
    printf("\nStarting detection loop (Ctrl+C to stop)...\n");
    
    // Main detection loop
    int frame_count = 0;
    while (g_running && ctx.running) {
        frame_count++;
        
        // Try to accept a client connection if not already connected
        try_accept_client(&ctx);
        
        // Capture image from emulated camera
        uint8_t* image_buffer = NULL;
        int width, height;
        size_t size;
        
        if (emulated_camera_take_picture(ctx.camera, &image_buffer, 
                                         &width, &height, &size) != 0) {
            fprintf(stderr, "Failed to capture image\n");
            usleep(DETECTION_LOOP_DELAY);
            continue;
        }
        
        // Create OpenCV image handle from RGB buffer
        // emulated_camera returns RGB format (format=1)
        ImageHandle* image = create_image_from_buffer(image_buffer, width, height, 3, 1);
        free(image_buffer);  // Buffer is copied by create_image_from_buffer
        
        if (!image) {
            fprintf(stderr, "Failed to create image from buffer\n");
            usleep(DETECTION_LOOP_DELAY);
            continue;
        }
        
        // Detect ArUco markers
        DetectionResult* detection = detectMarkersWithConfidence(ctx.detector, image);
        
        if (detection && detection->count > 0) {
            // Filter and process detected markers
            int valid_count = 0;
            MarkerData markers[100];  // Max 100 markers
            
            for (int i = 0; i < detection->count && valid_count < 100; i++) {
                DetectedMarker* marker = &detection->markers[i];
                
                // Only keep valid marker IDs
                if (!is_valid_marker_id(marker->id)) {
                    continue;
                }
                
                // Calculate center and angle
                Point2f center = calculate_marker_center(marker->corners);
                float angle = calculate_marker_angle(marker->corners);
                
                // Store marker data
                markers[valid_count].id = marker->id;
                markers[valid_count].x = center.x;
                markers[valid_count].y = center.y;
                markers[valid_count].angle = angle;
                valid_count++;
            }
            
            // Send detection results
            if (valid_count > 0) {
                send_detection_results(&ctx, markers, valid_count);
            }
            
            releaseDetectionResult(detection);
        } else {
            // No markers detected
            if (frame_count % 10 == 0) {
                printf("Frame %d: No markers detected\n", frame_count);
            }
        }
        
        // Release image
        release_image(image);
        
        // Control frame rate
        usleep(DETECTION_LOOP_DELAY);
    }
    
    printf("\nShutting down...\n");
    printf("Total frames processed: %d\n", frame_count);
    
    // Cleanup
    cleanup_app_context(&ctx);
    
    printf("ROD Detection stopped successfully\n");
    return 0;
}