#!/usr/bin/env python3

"""
test_unround.py
Corrects image round distortion using camera calibration parameters.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
from pathlib import Path
from raspberry.config.env_loader import EnvConfig
from raspberry.src import unround_image as unround_image

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
warped_pictures_dir = repo_root / "tests" / "fixtures" / "straightened"
image_path = "tests/fixtures/camera/image.jpg"

# Import coefficients for unrounding
config_dir = repo_root / "raspberry" / "config"
calibration_file = config_dir / calibration_filename
camera_matrix, dist_coeffs = unround_image.import_camera_calibration(
    str(calibration_file)
)

# Getting image size
image_size = (image_width, image_height)

# Calculate a new optimal camera matrix for distortion correction.
newcameramtx = unround_image.process_new_camera_matrix(
    camera_matrix, dist_coeffs, image_size
)

# Load image
img = cv2.imread(str(image_path))
if img is None:
    raise ValueError(f"Failed to load image from {image_path}")

# Unround the image
img_unrounded = unround_image.unround(
    img=img,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    newcameramtx=newcameramtx,
)

# Save images for verification
unrounded_picture_path = unrounded_pictures_dir / "unrounded.jpg"
cv2.imwrite(str(unrounded_picture_path), img_unrounded)
print(f"Images saved to {unrounded_picture_path}")
