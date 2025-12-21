#!/usr/bin/env python3

"""
test_detection.py
Detects ArUco markers in images.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
from pathlib import Path
from raspberry.config.env_loader import EnvConfig
from raspberry.src import detect_aruco as detect_aruco, unround_img


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_IDS = {20, 21, 22, 23}

# ---------------------------------------------------------------------------
# Main Test Code
# ---------------------------------------------------------------------------

# Load environment configuration
EnvConfig()

# Load configuration from .env
image_width = EnvConfig.get_camera_width()
image_height = EnvConfig.get_camera_height()
calibration_filename = EnvConfig.get_calibration_filename()

# Prepare input/output directories
repo_root = Path(__file__).resolve().parents[1]
camera_pictures_dir = repo_root / "tests" / "fixtures" / "camera"
unrounded_pictures_dir = repo_root / "tests" / "fixtures" / "unrounded"
straightened_pictures_dir = repo_root / "tests" / "fixtures" / "straightened"
annotated_img_dir = repo_root / "tests" / "fixtures" / "annotated"
image_path = "tests/fixtures/camera/image.jpg"

# Getting image size
image_size = (image_width, image_height)

# Import coefficients for unrounding
config_dir = repo_root / "raspberry" / "config"
calibration_file = config_dir / calibration_filename
camera_matrix, dist_coeffs = unround_img.import_camera_calibration(
    str(calibration_file)
)

# Calculate a new optimal camera matrix for distortion correction.
newcameramtx = unround_img.process_new_camera_matrix(
    camera_matrix, dist_coeffs, image_size
)

# Initialize ArUco detector
aruco_detector = detect_aruco.init_aruco_detector()

# Load image
img = cv2.imread(str(image_path))
if img is None:
    raise ValueError(f"Failed to load image from {image_path}")

# Unround the image
img = unround_img.unround(
    img=img,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    newcameramtx=newcameramtx,
)

# Detect ArUco markers sources points
tags_from_img = detect_aruco.detect_aruco_in_img(img, aruco_detector)

# Create annotated image
annotated_img = detect_aruco.create_annotated_image(img, tags_from_img)

for tags in tags_from_img:
    print(f"Detected tag ID: {tags.aruco_id} at image position: {tags.x}, {tags.y}")

# Save annotated image
annotated_img_path = annotated_img_dir / "annotated_image.jpg"
cv2.imwrite(str(annotated_img_path), annotated_img)
print(f"Annotated image saved to {annotated_img_path}")
