#ifndef OPENCV_WRAPPER_H
#define OPENCV_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for image data
typedef struct ImageHandle ImageHandle;

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

#ifdef __cplusplus
}
#endif

#endif // OPENCV_WRAPPER_H