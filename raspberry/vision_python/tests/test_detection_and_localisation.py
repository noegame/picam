#!/usr/bin/env python3

"""
test_detection_and_localisation.py
Detects ArUco markers in images and process real world coordinates.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
from pathlib import Path
from raspberry.vision_python.src.aruco import aruco as aruco
from vision_python.src import detect_aruco as detect_aruco
from vision_python.src import unround_img as unround_img
from vision_python.config import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Define destination points (real world coordinates in mm)
A1 = aruco.Aruco(53, 53, 1, 20)  # SO
B1 = aruco.Aruco(124, 53, 1, 22)  # SE
C1 = aruco.Aruco(53, 212, 1, 21)  # NO
D1 = aruco.Aruco(124, 212, 1, 23)  # NE

FIXED_IDS = {20, 21, 22, 23}
EXPECTED_TAG_36_COORDS = (159, 247)  # Expected coordinates in mm
PIXEL_TO_MM_RATIO = 1.0  # Conversion factor (adjust based on your calibration)

# ---------------------------------------------------------------------------
# Main Test Code
# ---------------------------------------------------------------------------


def main():

    # Load configuration parameters
    image_width = config.CAMERA_WIDTH
    image_height = config.CAMERA_HEIGHT

    # Prepare input/output directories
    fixtures_dir = (
        config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures" / "camera"
    )
    image_files = sorted([f for f in fixtures_dir.glob("*.jpg") if f.is_file()])

    if not image_files:
        raise ValueError(f"No images found in {fixtures_dir}")

    print(f"Found {len(image_files)} images to process\n")

    # Load camera calibration for unrounding
    camera_matrix, dist_coeffs = unround_img.import_camera_calibration(
        str(config.CALIBRATION_FILE)
    )
    image_size = (image_width, image_height)
    newcameramtx = unround_img.process_new_camera_matrix(
        camera_matrix, dist_coeffs, image_size
    )

    # Initialize ArUco detector
    aruco_detector = detect_aruco.init_aruco_detector()

    # Expected point for tag 36 (in real world coordinates)
    expected_tag_36 = aruco.Aruco(
        EXPECTED_TAG_36_COORDS[0], EXPECTED_TAG_36_COORDS[1], 1, 36
    )

    # Storage for results
    errors = []
    detected_count = 0

    # Process each image
    for image_path in image_files:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️  Failed to load image: {image_path.name}")
            continue

        # Unround the image before detection
        img = unround_img.unround(
            img=img,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            newcameramtx=newcameramtx,
        )

        # Detect ArUco markers sources points
        tags_from_img = detect_aruco.detect_aruco_in_img(img, aruco_detector)

        # Find source points by their ArUco IDs
        A2 = aruco.find_aruco_by_id(tags_from_img, 20)
        B2 = aruco.find_aruco_by_id(tags_from_img, 22)
        C2 = aruco.find_aruco_by_id(tags_from_img, 21)
        D2 = aruco.find_aruco_by_id(tags_from_img, 23)

        # Verify all reference markers were found
        if A2 is None or B2 is None or C2 is None or D2 is None:
            print(f"⚠️  Missing reference markers in: {image_path.name}")
            continue

        # Define source points (corners of the area to be straightened in image coordinates)
        src_points = np.array(
            [[A2.x, A2.y], [B2.x, B2.y], [D2.x, D2.y], [C2.x, C2.y]],
            dtype=np.float32,
        )

        # Define destination points (where the corners should map to in real world coordinates)
        dst_points = np.array(
            [[A1.x, A1.y], [B1.x, B1.y], [D1.x, D1.y], [C1.x, C1.y]], dtype=np.float32
        )

        # Calculate the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Convert detected tags image coordinates to real world coordinates
        tags_from_real_world = []
        for tag in tags_from_img:
            # Create homogeneous coordinate [x, y, z]
            img_point = np.array([tag.x, tag.y, tag.z], dtype=np.float32).reshape(3, 1)

            # Apply perspective transformation matrix to convert to real world coordinates
            real_world_point = matrix @ img_point

            # Normalize homogeneous coordinates
            real_x = real_world_point[0, 0] / real_world_point[2, 0]
            real_y = real_world_point[1, 0] / real_world_point[2, 0]

            # Create new Aruco object with transformed coordinates
            transformed_tag = aruco.Aruco(
                real_x, real_y, tag.z, tag.aruco_id, tag.angle
            )
            tags_from_real_world.append(transformed_tag)

        # Find tag 36 in real world coordinates
        tag_36 = aruco.find_aruco_by_id(tags_from_real_world, 36)

        if tag_36 is not None:
            detected_count += 1
            # Calculate error in mm
            error = aruco.distance(tag_36, expected_tag_36) * PIXEL_TO_MM_RATIO
            errors.append(error)
            print(
                f"✓ {image_path.name}: Tag 36 found at ({tag_36.x:.2f}, {tag_36.y:.2f}) mm | Error: {error:.2f} mm"
            )
        else:
            print(f"✗ {image_path.name}: Tag 36 not found")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {len(image_files)}")
    print(f"Tag 36 detected in: {detected_count} images")

    if errors:
        print(
            f"\nExpected Tag 36 location: ({EXPECTED_TAG_36_COORDS[0]}, {EXPECTED_TAG_36_COORDS[1]}) mm"
        )
        print(f"\nError Statistics (in mm):")
        print(f"  Mean error:     {np.mean(errors):.2f} mm")
        print(f"  Std deviation:  {np.std(errors):.2f} mm")
        print(f"  Min error:      {np.min(errors):.2f} mm")
        print(f"  Max error:      {np.max(errors):.2f} mm")
    else:
        print("No detections found for tag 36!")


if __name__ == "__main__":
    main()
