#!/usr/bin/env python3

"""
test_img_preprocessing_and_detection.py

Tests the entire pictures processing pipeline and aruco detection using directly opencv.

- Take a folder of pictures as input, process each picture to detect aruco markers, and print the detected markers' IDs and positions.
- save the different steps images in an output folder for visual verification.
- save annotaded pictures with detected markers highlighted.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
from vision_python.config import config
from vision_python.src.aruco import aruco
from vision_python.src.img_processing import unround_img


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

# Désarrondissement de l'image (correction de l'effet fish-eye)
unround = False

# Prétraitement optimal
sharpen_alpha = 1.5  # Contraste
sharpen_beta = -0.5  # Netteté
sharpen_gamma = 0  # Luminosité
use_clahe = False  # Égalisation d'histogramme adaptative, False sauf éclairage variable

# Détection ArUco optimale
adaptive_thresh_constant = 7  # Ajustement aux conditions d'éclairage
min_marker_perimeter_rate = 0.03  # Taille minimale du marqueur
max_marker_perimeter_rate = 4.0  # Taille maximale du marqueur
polygonal_approx_accuracy_rate = 0.03  # Précision de détection des coins

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

calibration_file = (
    config.get_camera_calibration_directory() / "camera_calibration_2000x2000.npz"
)
img_width = config.get_camera_width()
img_height = config.get_camera_height()
input_pict_dir = config.get_camera_directory() / "2026-01-09-playground-ready"
debug_pict_dir = config.get_debug_directory() / "2026-01-09-playground-ready"
img_size = (img_width, img_height)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Define destination points (real world coordinates in mm)
A1 = aruco.Aruco(600, 600, 1, 20)
B1 = aruco.Aruco(1400, 600, 1, 22)
C1 = aruco.Aruco(600, 2400, 1, 21)
D1 = aruco.Aruco(1400, 2400, 1, 23)
FIXED_IDS = {20, 21, 22, 23}

# ---------------------------------------------------------------------------
# ArUco detection preparation
# ---------------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Set custom detection parameters
parameters = cv2.aruco.DetectorParameters()

# Use subpixel corner refinement for better accuracy
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Apply configurable detection parameters
parameters.adaptiveThreshConstant = adaptive_thresh_constant
parameters.minMarkerPerimeterRate = min_marker_perimeter_rate
parameters.maxMarkerPerimeterRate = max_marker_perimeter_rate
parameters.polygonalApproxAccuracyRate = polygonal_approx_accuracy_rate

# Create ArUco detector
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

img_processing_step = 0

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def save_debug_image(
    base_name: str, comment: str, img_processing_step: int, img: np.ndarray
) -> int:
    dir = debug_pict_dir
    filename = build_filename(
        dir,
        base_name,
        comment,
        img_processing_step,
    )
    save(filename, img)
    return img_processing_step + 1


def build_filename(
    dir,
    base_name: str,
    comment: str,
    img_processing_step: int,
) -> str:
    return f"{dir}/{base_name}_{img_processing_step}_{comment}.jpg"


def save(filename: str, img: np.ndarray) -> None:
    cv2.imwrite(filename, img)


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------


def main():

    # ---------------------------------------------------------------------------
    # Ensure debug directory exists
    # ---------------------------------------------------------------------------

    debug_pict_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Get all image files from input directory
    # ---------------------------------------------------------------------------

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [
        f
        for f in input_pict_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        raise ValueError(f"No image files found in {input_pict_dir}")

    print(f"\n{'='*80}")
    print(f"Found {len(image_files)} image(s) to process")
    print("Detection Parameters:")
    print(f"  - Adaptive Threshold Constant: {adaptive_thresh_constant}")
    print(f"  - Min Marker Perimeter Rate: {min_marker_perimeter_rate}")
    print(f"  - Max Marker Perimeter Rate: {max_marker_perimeter_rate}")
    print(f"  - Polygon Approx Accuracy Rate: {polygonal_approx_accuracy_rate}")
    print(f"\nSharpening Parameters:")
    print(f"  - Alpha (contrast): {sharpen_alpha}")
    print(f"  - Beta (sharpness): {sharpen_beta}")
    print(f"  - Gamma (brightness): {sharpen_gamma}")
    print(f"{'='*80}")

    camera_matrix, dist_coeffs = unround_img.import_camera_calibration(
        str(calibration_file)
    )
    newcameramtx = unround_img.process_new_camera_matrix(
        camera_matrix, dist_coeffs, img_size
    )

    # ---------------------------------------------------------------------------
    # Process each image
    # ---------------------------------------------------------------------------

    for img_file in image_files:
        step = 0
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

        # Save original image for debugging
        step = save_debug_image(base_name, "original", step, img)

        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        step = save_debug_image(base_name, "grayscale", step, img)

        # Apply sharpening with configurable parameters
        img = cv2.addWeighted(img, sharpen_alpha, img, sharpen_beta, sharpen_gamma)
        step = save_debug_image(base_name, "sharpened", step, img)

        # Apply unrounding if enabled
        if unround:
            img = unround_img.unround(img, camera_matrix, dist_coeffs, newcameramtx)
            step = save_debug_image(base_name, "unrounded", step, img)

        # Apply CLAHE if enabled
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            step = save_debug_image(base_name, "clahe", step, img)

        # ---------------------------------------------------------------------------
        # Aruco detection
        # ---------------------------------------------------------------------------

        corners_list, ids, rejected = aruco_detector.detectMarkers(img)

        detected_markers = []  # List of (marker, corners) tuples
        if ids is None or len(ids) == 0:
            print("⚠️  No ArUco markers detected.")
        else:
            print(f"✓ Detected {len(ids)} ArUco marker(s)")

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
                # Store marker with its corners for later angle calculation
                detected_markers.append((a, corners))

        tags_from_img = [marker for marker, corners in detected_markers]

        # ---------------------------------------------------------------------------
        # Annotate detected markers on image
        # ---------------------------------------------------------------------------

        annotated_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Draw marker boundaries without IDs
        cv2.aruco.drawDetectedMarkers(annotated_img, corners_list)
        for marker, corners in detected_markers:
            center = (int(marker.x), int(marker.y))
            # Draw green dot at center
            cv2.circle(annotated_img, center, 5, (0, 255, 0), -1)
            # Draw ID text with offset from center (20 pixels to the right and down)
            text_pos = (center[0] + 20, center[1] + 20)
            cv2.putText(
                annotated_img,
                str(marker.aruco_id),
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),  # Blue color
                2,
            )

        # save annotated image
        step = save_debug_image(base_name, "annotated", step, annotated_img)

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
                f"⚠️  Warning: Not all fixed ArUco markers were detected for perspective transform. Found {len(src_points)}/4 markers."
            )
            print("   Skipping perspective transformation for this image.")
            print("   Real world coordinates will not be calculated.")
        else:
            print(
                "✓ All 4 fixed markers detected, computing perspective transformation..."
            )
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
            for tag, tag_corners in detected_markers:
                img_point = np.array([[tag.x, tag.y]], dtype=np.float32)
                img_point = np.array([img_point])  # Reshape for perspectiveTransform
                real_point = cv2.perspectiveTransform(img_point, perspective_matrix)
                real_x, real_y = real_point[0][0]
                tag.real_x = real_x
                tag.real_y = real_y

                # Transform corners to real world coordinates to calculate real angle
                corners_reshaped = tag_corners.reshape(
                    1, -1, 2
                )  # Shape for perspectiveTransform
                real_corners = cv2.perspectiveTransform(
                    corners_reshaped, perspective_matrix
                )
                real_corners = real_corners.reshape(4, 2)

                # Calculate angle from top edge in real world coordinates
                dx = real_corners[1][0] - real_corners[0][0]  # Top-right X - Top-left X
                dy = real_corners[1][1] - real_corners[0][1]  # Top-right Y - Top-left Y
                real_angle = float(np.degrees(np.arctan2(dy, dx)))

                # Normalize angle to [-180, 180] range
                if real_angle > 180:
                    real_angle -= 360
                elif real_angle < -180:
                    real_angle += 360

                tag.real_angle = real_angle

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


if __name__ == "__main__":
    main()
