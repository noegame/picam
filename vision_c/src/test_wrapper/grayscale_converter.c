#include "opencv_wrapper.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <image_path> [output_path]\n", argv[0]);
        return -1;
    }

    const char* input_path = argv[1];
    
    // Load image
    ImageHandle* image = load_image(input_path);
    if (image == NULL) {
        fprintf(stderr, "Error: Could not open or find the image: %s\n", input_path);
        return -1;
    }

    printf("Image loaded successfully: %dx%d pixels\n", 
           get_image_width(image), get_image_height(image));

    // Convert to grayscale
    ImageHandle* gray_image = convert_to_grayscale(image);
    printf("Image converted to grayscale\n");

    // Determine output path
    char output_path[512];
    if (argc >= 3) {
        strncpy(output_path, argv[2], sizeof(output_path) - 1);
        output_path[sizeof(output_path) - 1] = '\0';
    } else {
        const char* dot_pos = strrchr(input_path, '.');
        if (dot_pos != NULL) {
            size_t base_len = dot_pos - input_path;
            snprintf(output_path, sizeof(output_path), "%.*s_gray%s", 
                    (int)base_len, input_path, dot_pos);
        } else {
            snprintf(output_path, sizeof(output_path), "%s_gray.jpg", input_path);
        }
    }

    // Save the grayscale image
    if (save_image(output_path, gray_image)) {
        printf("Grayscale image saved to: %s\n", output_path);
    } else {
        fprintf(stderr, "Error: Could not save the image to: %s\n", output_path);
        release_image(gray_image);
        release_image(image);
        return -1;
    }

    // Clean up
    release_image(gray_image);
    release_image(image);

    return 0;
}