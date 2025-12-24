#!/usr/bin/env python3

"""
test_detection_flow.py
Goal : Debug process of ArUco markers detection in imgs.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
from pathlib import Path
from vision_python.config import config
from vision_python.src import detect_aruco
from vision_python.src.aruco.aruco_data import get_aruco_smiley

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

img_width = config.CAMERA_WIDTH
img_height = config.CAMERA_HEIGHT
input_img_dir = config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures" / "data1"
img_size = (img_width, img_height)

# ---------------------------------------------------------------------------
# Main Test Code
# ---------------------------------------------------------------------------


def main():
    # Initialize ArUco detector
    aruco_detector = detect_aruco.init_aruco_detector()

    # Get all image files from data1 directory
    image_files = sorted(input_img_dir.glob("*.jpg"))

    if not image_files:
        print(f"No images found in {input_img_dir}")
        return

    print(f"Processing {len(image_files)} images from {input_img_dir}\n")

    # Process each image
    for img_path in image_files:
        print(f"--- Processing: {img_path.name} ---")

        # Load img
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Failed to load image\n")
            continue

        # Detect ArUco markers
        tags_from_img = detect_aruco.detect_aruco_in_img(img, aruco_detector)

        if tags_from_img:
            for tags in tags_from_img:
                print(
                    f"  {get_aruco_smiley(tags.aruco_id)} Detected tag ID: {tags.aruco_id} at img position: {tags.x}, {tags.y}"
                )
        else:
            print("  No tags detected")
        print()


if __name__ == "__main__":
    main()
