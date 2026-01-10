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
from vision_python.config import config
from vision_python.src.img_processing import unround_img

# ---------------------------------------------------------------------------
# Main Test Code
# ---------------------------------------------------------------------------

# Load configuration parameters
image_width = config.get_camera_width()
image_height = config.get_camera_height()
calibration_file = config.get_camera_calibration_file()
pictures_dir = config.get_pictures_directory()
input_image_path = pictures_dir / "test" / "test.jpg"
output_image_path = pictures_dir / "debug" / "unround" / "unrounded_image.jpg"

# Import coefficients for unrounding
camera_matrix, dist_coeffs = unround_img.import_camera_calibration(
    str(calibration_file)
)

# Getting image size
image_size = (image_width, image_height)

# Calculate a new optimal camera matrix for distortion correction.
newcameramtx = unround_img.process_new_camera_matrix(
    camera_matrix, dist_coeffs, image_size
)

# Load image
img = cv2.imread(str(input_image_path))
if img is None:
    raise ValueError(f"Failed to load image from {input_image_path}")

# Unround the image
img_unrounded = unround_img.unround(
    img=img,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    newcameramtx=newcameramtx,
)

# Save images for verification
cv2.imwrite(str(output_image_path), img_unrounded)
print(f"Images saved to {output_image_path}")
