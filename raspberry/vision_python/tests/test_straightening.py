#!/usr/bin/env python3

"""
test_straightening.py
Applies perspective transformation to straighten images.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
from pathlib import Path
from vision_python.config.env_loader import EnvConfig
from vision_python.src import aruco
from vision_python.src import detect_aruco

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

# Find source points by their ArUco IDs
A2 = aruco.find_aruco_by_id(tags_from_img, 20)
B2 = aruco.find_aruco_by_id(tags_from_img, 22)
C2 = aruco.find_aruco_by_id(tags_from_img, 21)
D2 = aruco.find_aruco_by_id(tags_from_img, 23)

# Verify all markers were found
if A2 is None or B2 is None or C2 is None or D2 is None:
    raise ValueError(
        "One or more required ArUco markers (20, 21, 22, 23) were not found in the image"
    )

print(f"src points : {[(A2.x, A2.y), (B2.x, B2.y), (D2.x, D2.y), (C2.x, C2.y)]}")

# Define destination points
A1 = aruco.Aruco(53, 53, 1, 20)  # SO
B1 = aruco.Aruco(123, 53, 1, 22)  # SE
C1 = aruco.Aruco(53, 213, 1, 21)  # NO
D1 = aruco.Aruco(123, 213, 1, 23)  # NE

# Define source points (corners of the area to be straightened)
src_points = np.array(
    [[A2.x, A2.y], [B2.x, B2.y], [D2.x, D2.y], [C2.x, C2.y]],
    dtype=np.float32,
)

# Define destination points (where the corners should map to)
dst_points = np.array(
    [[A1.x, A1.y], [B1.x, B1.y], [D1.x, D1.y], [C1.x, C1.y]], dtype=np.float32
)

# Calculate the perspective transformation matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation to straighten the image
w, h = img.shape[:2]
print(f"  w : {w} h : {h} ")
straightened_img = cv2.warpPerspective(img, matrix, (w, h))

# Save images for verification
straightened_picture_path = straightened_pictures_dir / "straightened.jpg"
cv2.imwrite(str(straightened_picture_path), straightened_img)
print(f"Images saved to {straightened_picture_path}")
