/**
 * test_emulated_camera.c
 * 
 * Simple test program to demonstrate the emulated camera functionality.
 * Reads images from a folder and cycles through them.
 */

#include "emulated_camera.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_folder_path> [width] [height]\n", argv[0]);
        fprintf(stderr, "Example: %s /path/to/images 640 480\n", argv[0]);
        return 1;
    }
    
    const char* folder_path = argv[1];
    int width = 0;
    int height = 0;
    
    // Parse optional width and height
    if (argc >= 4) {
        width = atoi(argv[2]);
        height = atoi(argv[3]);
        printf("Will resize images to: %dx%d\n", width, height);
    }
    
    // Initialize emulated camera
    printf("Initializing emulated camera...\n");
    EmulatedCameraContext* camera = emulated_camera_init();
    if (!camera) {
        fprintf(stderr, "Failed to initialize emulated camera\n");
        return 1;
    }
    
    // Set folder path
    if (emulated_camera_set_folder(camera, folder_path) != 0) {
        fprintf(stderr, "Failed to set folder path\n");
        emulated_camera_cleanup(camera);
        return 1;
    }
    
    // Set size if specified
    if (width > 0 && height > 0) {
        if (emulated_camera_set_size(camera, width, height) != 0) {
            fprintf(stderr, "Failed to set camera size\n");
            emulated_camera_cleanup(camera);
            return 1;
        }
    }
    
    // Start camera (loads image list)
    if (emulated_camera_start(camera) != 0) {
        fprintf(stderr, "Failed to start emulated camera\n");
        emulated_camera_cleanup(camera);
        return 1;
    }
    
    // Capture a few images to demonstrate cycling
    printf("\nCapturing images...\n");
    for (int i = 0; i < 5; i++) {
        uint8_t* buffer = NULL;
        int img_width, img_height;
        size_t img_size;
        
        if (emulated_camera_take_picture(camera, &buffer, &img_width, 
                                         &img_height, &img_size) == 0) {
            printf("  Image %d: %dx%d, %zu bytes (RGB format)\n", 
                   i + 1, img_width, img_height, img_size);
            
            // Calculate some statistics on the image data
            if (buffer && img_size > 0) {
                unsigned long sum_r = 0, sum_g = 0, sum_b = 0;
                int num_pixels = img_width * img_height;
                
                for (int p = 0; p < num_pixels; p++) {
                    sum_r += buffer[p * 3 + 0];
                    sum_g += buffer[p * 3 + 1];
                    sum_b += buffer[p * 3 + 2];
                }
                
                printf("    Average RGB: (%lu, %lu, %lu)\n", 
                       sum_r / num_pixels, sum_g / num_pixels, sum_b / num_pixels);
            }
            
            // Free the buffer
            free(buffer);
        } else {
            fprintf(stderr, "Failed to capture image %d\n", i + 1);
        }
    }
    
    // Stop and cleanup
    printf("\nStopping emulated camera...\n");
    emulated_camera_stop(camera);
    
    printf("Cleaning up...\n");
    emulated_camera_cleanup(camera);
    
    printf("Test completed successfully!\n");
    return 0;
}
