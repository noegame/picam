#!/bin/bash
# CMake build script for vision_c project
# Automatically configures and builds the project

set -e

echo "=== Building Vision C Project with CMake ==="
echo ""

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "[1/2] Configuring project with CMake..."
cmake .. > /dev/null 2>&1

# Build the project
echo "[2/2] Building test_aruco_detection_pipeline..."
make test_aruco_detection_pipeline

echo ""
echo "✓ Build successful!"
echo ""
echo "Executable location:"
echo "  ./build/apps/test_aruco_detection_pipeline"
echo ""
echo "To run the program:"
echo "  ./build/apps/test_aruco_detection_pipeline <image_path> [output_path]"
echo ""
echo "Example:"
echo "  ./build/apps/test_aruco_detection_pipeline ../pictures/test_image.jpg output.jpg"
echo ""
echo "To build other programs:"
echo "  cd build && make detect_aruco"
echo "  cd build && make grayscale_converter"
