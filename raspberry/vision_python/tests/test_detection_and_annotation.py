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
from vision_python.src.img_processing import detect_aruco
from vision_python.src.img_processing import unround_img

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
    config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures" / "data3" / "img.jpg"
)
annotated_img_dir = config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures"

# Getting image size
image_size = (image_width, image_height)

# Initialize ArUco detector
aruco_detector = detect_aruco.init_aruco_detector()

# Load image
img = cv2.imread(str(image_path))
if img is None:
    raise ValueError(f"Failed to load image from {image_path}")

# Detect ArUco markers sources points
tags_from_img, rejected_markers = detect_aruco.detect_aruco_in_img(img, aruco_detector)

# Create annotated image
annotated_img = detect_aruco.create_annotated_image(
    img, tags_from_img, rejected_markers
)

for tags in tags_from_img:
    print(f"Detected tag ID: {tags.aruco_id} at image position: {tags.x}, {tags.y}")

print(f"Number of rejected markers: {len(rejected_markers)}")

# Save annotated image
annotated_img_path = annotated_img_dir / "annotated_image.jpg"
cv2.imwrite(str(annotated_img_path), annotated_img)
print(f"Annotated image saved to {annotated_img_path}")
