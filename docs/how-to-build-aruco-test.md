# How to Build test_aruco_detection_pipeline

This guide explains how to compile and run the ArUco detection pipeline test program in C.

## Prerequisites

Before building, you need to install CMake and OpenCV:

```bash
# Install CMake and OpenCV development libraries
sudo apt update
sudo apt install cmake libopencv-dev
```

To verify installation:
```bash
cmake --version
pkg-config --modversion opencv4
```

## Building the Program

### Option 1: Using the CMake build script (recommended)

The easiest way to build the program is to use the provided CMake build script:

```bash
cd vision_c
./build.sh
```

The executable will be created at `vision_c/build/apps/test_aruco_detection_pipeline`

### Option 2: Manual CMake build

If you prefer to build manually with CMake:

```bash
cd vision_c
mkdir -p build
cd build

# Configure the project
cmake ..

# Build specific target
make test_aruco_detection_pipeline

# Or build everything
make
```

### Option 3: Build other programs

```bash
cd vision_c/build
make detect_aruco           # Simple ArUco detector
make grayscale_converter    # Image format converter
make                        # Build all programs
```

## Running the Program

```bash
cd vision_c
./build/apps/test_aruco_detection_pipeline <image_path> [output_path]
```

### Arguments:
- `image_path` (required): Path to the input image to process
- `output_path` (optional): Path where to save the annotated image (default: `output_annotated.jpg`)

### Example:

```bash
# Using a test image from the pictures directory
./build/apps/test_aruco_detection_pipeline ../pictures/test_image.jpg result.jpg
```

## What the Program Does

The program follows the same pipeline as the Python implementation:

1. **Load image** - Loads the input image
2. **Sharpen** - Applies a sharpening filter to enhance marker edges
3. **Resize** - Scales the image by 1.5x to improve detection
4. **Detect** - Detects ArUco markers using DICT_4X4_50 dictionary
5. **Calculate centers** - Computes the center coordinates of each marker
6. **Annotate** - Adds visual annotations:
   - Counter showing total markers detected (top-left)
   - Marker IDs (at each marker center)
   - Center coordinates (above each marker)
7. **Save** - Saves the annotated image

## Output

The program displays:
- Progress messages for each step
- Detection results summary with:
  - Marker ID
  - Center coordinates
  - Confidence score
  - Corner positions

Example output:
```
=== ArUco Detection Pipeline Test ===

[1/7] Loading image: test.jpg
      Image loaded: 4056x4056 pixels
[2/7] Applying sharpening filter...
      Sharpening applied
[3/7] Resizing image (scale: 1.5x)
      Resized to: 6084x6084 pixels
[4/7] Detecting ArUco markers (DICT_4X4_50)...
      Detected 5 marker(s)
[5/7] Calculating marker centers...
      Marker ID 20: center at (1234.5, 2345.6)
      Marker ID 21: center at (2345.6, 3456.7)
      ...
[6/7] Annotating image...
      Annotations added: counter, IDs, centers
[7/7] Saving annotated image to: output.jpg
      Annotated image saved successfully!

=== Detection Results Summary ===
Total markers detected: 5
...
```

## Troubleshooting

### OpenCV not found
If you get errors about OpenCV headers not found:
```bash
# Check if OpenCV is installed
dpkg -l | grep opencv

# If not installed, install it
sudo apt install libopencv-dev
```

### Wrong OpenCV version
The code is written for OpenCV 4.x. Check your version:
```bash
pkg-config --modversion opencv4
```

If you have OpenCV 3.x, you may need to adjust the include paths in the build script.
