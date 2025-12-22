#!/usr/bin/env python3

"""
test_detection_flow.py
Goal : Debug process of ArUco markers detection in imgs.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
from vision_python.config import config
from vision_python.src import detect_aruco
from raspberry.vision_python.src.aruco.aruco_data import get_aruco_smiley

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

img_width = config.CAMERA_WIDTH
img_height = config.CAMERA_HEIGHT
input_img_path = config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures" / "data1"
input_img_path = config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures"
img_size = (img_width, img_height)

# ---------------------------------------------------------------------------
# Main Test Code
# ---------------------------------------------------------------------------


def main():
    # Load img
    img = cv2.imread(str(input_img_path))
    if img is None:
        raise ValueError(f"Failed to load img from {input_img_path}")

    # Initialize ArUco detector
    aruco_detector = detect_aruco.init_aruco_detector()

    # Detect ArUco markers sources points
    tags_from_img = detect_aruco.detect_aruco_in_img(img, aruco_detector)

    for tags in tags_from_img:
        print(
            f"{get_aruco_smiley(tags.aruco_id)} Detected tag ID: {tags.aruco_id} at img position: {tags.x}, {tags.y}"
        )


if __name__ == "__main__":
    main()
