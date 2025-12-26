#!/usr/bin/env python3

"""
test_detection.py
Detects ArUco markers in images.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
from vision_python.config import config
from raspberry.vision_python.src.img_processing import detect_aruco

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_IDS = {20, 21, 22, 23}

# ---------------------------------------------------------------------------
# Main Test Code
# ---------------------------------------------------------------------------

# Load configuration parameters
image_width = config.CAMERA_WIDTH
image_height = config.CAMERA_HEIGHT

# Prepare input/output directories
image_path = (
    config.RASPBERRY_DIR
    / "vision_python"
    / "tests"
    / "fixtures"
    / "camera"
    / "image.jpg"
)

# Getting image size
image_size = (image_width, image_height)

# Load image
img = cv2.imread(str(image_path))
if img is None:
    raise ValueError(f"Failed to load image from {image_path}")

# Initialize ArUco detector
aruco_detector = detect_aruco.init_aruco_detector()

# Detect ArUco markers sources points
tags_from_img = detect_aruco.detect_aruco_in_img(img, aruco_detector)

for tags in tags_from_img:
    print(f"Detected tag ID: {tags.aruco_id} at image position: {tags.x}, {tags.y}")
