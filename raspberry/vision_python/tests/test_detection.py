#!/usr/bin/env python3

"""
test_detection.py
Detects ArUco markers in images.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
from pathlib import Path
from vision_python.config.env_loader import EnvConfig
from vision_python.src import detect_aruco

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
image_path = camera_pictures_dir / "image.jpg"

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
