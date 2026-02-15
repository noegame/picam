# Emulated Camera

The emulated camera module provides a way to simulate camera captures by reading images from a folder. This is useful for testing and development without requiring physical camera hardware.

## Features

- Reads image files (JPG, PNG, JPEG) from a specified folder
- Automatically cycles through images in alphabetical order
- Optional image resizing
- Returns images in RGB format (consistent with the Python implementation)
- Simple C API matching the camera interface pattern

## API Overview

### Initialization

```c
EmulatedCameraContext* emulated_camera_init();
```

Creates and initializes a new emulated camera context.

### Configuration

```c
int emulated_camera_set_folder(EmulatedCameraContext* ctx, const char* folder_path);
```

Sets the folder path containing images to use. Must be called before `emulated_camera_start()`.

```c
int emulated_camera_set_size(EmulatedCameraContext* ctx, int width, int height);
```

(Optional) Sets the desired output image size. Images will be resized to this resolution. If not called, images will be returned at their original size.

### Operation

```c
int emulated_camera_start(EmulatedCameraContext* ctx);
```

Starts the camera by loading the list of image files from the configured folder.

```c
int emulated_camera_take_picture(EmulatedCameraContext* ctx, 
                                  uint8_t** out_buffer,
                                  int* out_width,
                                  int* out_height,
                                  size_t* out_size);
```

Captures the next image from the folder. Returns image data in RGB format (3 bytes per pixel). The caller is responsible for freeing the returned buffer using `free()`.

```c
void emulated_camera_stop(EmulatedCameraContext* ctx);
```

Stops the camera and releases image list.

```c
void emulated_camera_cleanup(EmulatedCameraContext* ctx);
```

Cleans up and frees all resources associated with the camera context.

## Usage Example

```c
#include "emulated_camera.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize
    EmulatedCameraContext* camera = emulated_camera_init();
    
    // Configure
    emulated_camera_set_folder(camera, "/path/to/images");
    emulated_camera_set_size(camera, 640, 480);  // Optional
    
    // Start
    if (emulated_camera_start(camera) == 0) {
        // Capture an image
        uint8_t* buffer;
        int width, height;
        size_t size;
        
        if (emulated_camera_take_picture(camera, &buffer, &width, &height, &size) == 0) {
            printf("Captured: %dx%d, %zu bytes\n", width, height, size);
            
            // Process buffer (RGB format)...
            
            // Free the buffer when done
            free(buffer);
        }
        
        // Stop and cleanup
        emulated_camera_stop(camera);
    }
    
    emulated_camera_cleanup(camera);
    return 0;
}
```

## Building

The emulated camera is automatically included in the `rod_camera` library when you build the project:

```bash
cd rod_c
./build.sh
```

To build and run the test:

```bash
cd rod_c/build
make test_emulated_camera
./tests/test_emulated_camera /path/to/image/folder
```

Or with custom size:

```bash
./tests/test_emulated_camera /path/to/image/folder 640 480
```

## Image Format

- **Input**: Accepts JPG, PNG, and JPEG files
- **Output**: RGB format (3 bytes per pixel: Red, Green, Blue)
- **Ordering**: Images are processed in alphabetical order
- **Cycling**: After the last image, it wraps back to the first image

## Implementation Details

- Uses OpenCV wrapper for image loading and processing
- Automatically converts from OpenCV's BGR format to RGB for consistency with the Python implementation
- Supports optional image resizing using OpenCV's resize function
- Thread-safe as long as each thread uses its own context
- Maximum of 1000 images per folder (configurable in code)

## Comparison with Python Implementation

The C implementation closely mirrors the Python `EmulatedCamera` class:

| Feature | Python | C |
|---------|--------|---|
| Folder scanning | `Path.glob()` | `opendir()/readdir()` |
| Image loading | `cv2.imread()` | OpenCV wrapper |
| Format conversion | `cv2.cvtColor()` | OpenCV wrapper |
| Cycling | Index modulo | Index modulo |
| Memory management | Automatic | Manual (caller frees buffer) |

## Notes

- The caller must `free()` the buffer returned by `emulated_camera_take_picture()`
- Ensure the image folder exists and contains supported image files before starting
- Images are loaded on-demand, so the first capture may be slightly slower
- For optimal performance with large images, consider setting a smaller output size
