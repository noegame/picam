#!/usr/bin/env python3

"""
test_img_processing.py
Tests the entire img processing pipeline and aruco detection using directly opencv.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import cv2
from vision_python.config import config
from vision_python.src.aruco import aruco

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Define destination points (real world coordinates in mm)
A1 = aruco.Aruco(600, 600, 1, 20)
B1 = aruco.Aruco(1400, 600, 1, 22)
C1 = aruco.Aruco(600, 2400, 1, 21)
D1 = aruco.Aruco(1400, 2400, 1, 23)

FIXED_IDS = {20, 21, 22, 23}

# Load configuration parameters
img_width = config.CAMERA_WIDTH
img_height = config.CAMERA_HEIGHT
img_size = (img_width, img_height)

# Prepare input/output directories
input_img_dir = config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures" / "data2"
output_img_dir = config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ArUco detection preparation
# ---------------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Set custom detection parameters
parameters = cv2.aruco.DetectorParameters()

# Use subpixel corner refinement for better accuracy
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Adjust parameters only if you have specific detection issues:
# parameters.adaptiveThreshConstant = 7  # Adjust for lighting conditions
# parameters.minMarkerPerimeterRate = 0.03  # Minimum marker size (default: 0.03)
# parameters.maxMarkerPerimeterRate = 4.0  # Maximum marker size (default: 4.0)

# Create ArUco detector
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ---------------------------------------------------------------------------
# Get all image files from input directory
# ---------------------------------------------------------------------------

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
image_files = [
    f
    for f in input_img_dir.iterdir()
    if f.is_file() and f.suffix.lower() in image_extensions
]

if not image_files:
    raise ValueError(f"No image files found in {input_img_dir}")

print(f"Found {len(image_files)} image(s) to process")

# ---------------------------------------------------------------------------
# Process each image
# ---------------------------------------------------------------------------

for img_file in image_files:
    print(f"\n{'='*80}")
    print(f"Processing: {img_file.name}")
    print(f"{'='*80}")

    # ---------------------------------------------------------------------------
    # img preprocessing
    # ---------------------------------------------------------------------------

    # Load img
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"Failed to load img from {img_file}, skipping...")
        continue

    # Get base filename without extension for output files
    base_name = img_file.stem

    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_img_dir / f"{base_name}_grayscale.jpg"), img)

    # Optional: Adaptive thresholding preview
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)

    # Apply sharpening
    img = cv2.addWeighted(img, 1.5, img, -0.5, 0)
    cv2.imwrite(str(output_img_dir / f"{base_name}_sharpened.jpg"), img)

    # ---------------------------------------------------------------------------
    # Aruco detection
    # ---------------------------------------------------------------------------

    corners_list, ids, rejected = aruco_detector.detectMarkers(img)

    detected_markers = []
    if ids is None or len(ids) == 0:
        print("No ArUco markers detected.")
    else:

        ids = ids.flatten()
        for corners, id_val in zip(corners_list, ids):
            corners = corners.reshape((4, 2)).astype(np.float32)
            cX = float(np.mean(corners[:, 0]))
            cY = float(np.mean(corners[:, 1]))

            # Calculate angle from top edge (corners are ordered: TL, TR, BR, BL)
            dx = corners[1][0] - corners[0][0]  # Top-right X - Top-left X
            dy = corners[1][1] - corners[0][1]  # Top-right Y - Top-left Y
            angle = float(np.degrees(np.arctan2(dy, dx)))

            # Normalize angle to [-180, 180] range
            if angle > 180:
                angle -= 360
            elif angle < -180:
                angle += 360

            a = aruco.Aruco(
                cX,
                cY,
                1,
                id_val,
                angle,
            )
            detected_markers.append(a)

    tags_from_img = detected_markers

    # ---------------------------------------------------------------------------
    # Annotate detected markers on image
    # ---------------------------------------------------------------------------

    annotated_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedMarkers(annotated_img, corners_list, ids)
    for marker in detected_markers:
        center = (int(marker.x), int(marker.y))
        cv2.circle(annotated_img, center, 5, (0, 255, 0), -1)
        cv2.putText(
            annotated_img,
            f"ID:{marker.aruco_id}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(str(output_img_dir / f"{base_name}_annotated.jpg"), annotated_img)

    # ---------------------------------------------------------------------------
    # Perspective transformation for real world coordinates
    # ---------------------------------------------------------------------------

    # Source points from detected markers
    src_points = []
    for fixed_id in FIXED_IDS:
        for tag in tags_from_img:
            if tag.aruco_id == fixed_id:
                src_points.append([tag.x, tag.y])
                break

    if len(src_points) != 4:
        print(
            f"Warning: Not all fixed ArUco markers were detected for perspective transform. Found {len(src_points)}/4 markers."
        )
        print("Skipping perspective transformation for this image.")
    else:
        src_points = np.array(src_points, dtype=np.float32)

        # Destination points in real world coordinates
        dst_points = np.array(
            [
                [A1.x, A1.y],
                [B1.x, B1.y],
                [C1.x, C1.y],
                [D1.x, D1.y],
            ],
            dtype=np.float32,
        )
        # Compute perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Transform detected tags to real world coordinates
        for tag in tags_from_img:
            img_point = np.array([[tag.x, tag.y]], dtype=np.float32)
            img_point = np.array([img_point])  # Reshape for perspectiveTransform
            real_point = cv2.perspectiveTransform(img_point, perspective_matrix)
            real_x, real_y = real_point[0][0]
            tag.real_x = real_x
            tag.real_y = real_y

    # ---------------------------------------------------------------------------
    # Print detected tags information
    # ---------------------------------------------------------------------------

    # sort list of tags by id
    tags_from_img.sort(key=lambda tag: tag.aruco_id)

    for tag in tags_from_img:
        tag.print()

print(f"\n{'='*80}")
print(f"Processing complete! Processed {len(image_files)} image(s)")
print(f"{'='*80}")
