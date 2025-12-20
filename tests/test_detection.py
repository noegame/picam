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
from raspberry.src.aruco import Aruco
from raspberry.src import detect_aruco as detect_aruco
from raspberry.src.aruco import find_aruco_by_id

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
image_path = "tests/fixtures/camera/image.jpg"

# Getting image size
image_size = (image_width, image_height)

# Load image
img = cv2.imread(str(image_path))
if img is None:
    raise ValueError(f"Failed to load image from {image_path}")

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Detect ArUco markers sources points
tags_from_img, img_annotated = detect_aruco.detect_in_image(
    img, aruco_dict, None, aruco_params, draw=False
)

# Temporarily convert detected tags to Aruco objects (should be done inside detect_aruco)
tags_from_img = detect_aruco.convert_detected_tags(tags_from_img)

for tags in tags_from_img:
    print(f"Detected tag ID: {tags.aruco_id} at image position: {tags.x}, {tags.y}")
