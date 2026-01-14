#!/usr/bin/env python3

"""
benchmark.py

Il y a 38 tags Aruco à détecter dans le jeu d'images test (6 noirs, 16 jaunes, 16 bleues).
Ce script traite un dossier d'images, détecte les marqueurs Aruco dans chaque image,
et génère un rapport de performance indiquant le nombre total de marqueurs détectés,
ainsi que le temps moyen de traitement par image.

Le rapport final :
- nombre moyen de tags détectés par image
- nombre de tag noirs détectés par image
- nombre de tag jaunes détectés par image
- nombre de tag bleus détectés par image
- temps moyen de traitement par image.


dossier jeu d'image de test : 2026-01-09-playground-ready
"""


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import time
from vision_python.config import config
from vision_python.src.aruco import aruco
from vision_python.src.img_processing import unround_img

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

rejected_marker = True  # Draw rejected markers on debug images
resume = True  # Print summary of detections instead of detailed info
save_debug_images = False  # Save debug images at each processing step

# ---------------------------------------------------------------------------
# Image Processing Parameters
# ---------------------------------------------------------------------------

# Get image processing parameters from config
img_params = config.get_image_processing_params()

# Extract parameters
unround = img_params["use_unround"]
use_clahe = img_params["use_clahe"]
Thresholding = img_params["use_binarization"]
sharpen_alpha = img_params["sharpen_alpha"]
sharpen_beta = img_params["sharpen_beta"]
sharpen_gamma = img_params["sharpen_gamma"]
adaptive_thresh_constant = img_params["adaptive_thresh_constant"]
min_marker_perimeter_rate = img_params["min_marker_perimeter_rate"]
max_marker_perimeter_rate = img_params["max_marker_perimeter_rate"]
polygonal_approx_accuracy_rate = img_params["polygonal_approx_accuracy_rate"]


# ---------------------------------------------------------------------------
# Paths and Directories
# ---------------------------------------------------------------------------

calibration_file = config.get_camera_calibration_file()
img_width = config.get_camera_width()
img_height = config.get_camera_height()
input_pict_dir = (
    config.get_camera_directory() / "2026-01-09-playground-ready"
)  # "2026-01-09-playground-ready"
debug_pict_dir = config.get_debug_directory() / "CCC"  # "2026-01-09-playground-ready"
img_size = (img_width, img_height)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Get fixed ArUco marker positions from config
fixed_markers = config.get_fixed_aruco_positions()
fixed_ids = [marker.aruco_id for marker in fixed_markers]

A1 = aruco.Aruco(600, 600, 1, 20)
B1 = aruco.Aruco(1400, 600, 1, 22)
C1 = aruco.Aruco(600, 2400, 1, 21)
D1 = aruco.Aruco(1400, 2400, 1, 23)

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
    if save_debug_images:
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
# Image processing functions
# ---------------------------------------------------------------------------


def load_and_convert_to_grayscale(img_path: str) -> np.ndarray:
    """Load an image and convert it to grayscale.

    Args:
        img_path: Path to the image file

    Returns:
        Grayscale image as numpy array
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_sharpening(
    img: np.ndarray, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Apply sharpening to the image.

    Args:
        img: Input grayscale image
        alpha: Contrast parameter
        beta: Sharpness parameter
        gamma: Brightness parameter

    Returns:
        Sharpened image
    """
    return cv2.addWeighted(img, alpha, img, beta, gamma)


def apply_unrounding(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    newcameramtx: np.ndarray,
) -> np.ndarray:
    """Apply fish-eye distortion correction.

    Args:
        img: Input image
        camera_matrix: Camera calibration matrix
        dist_coeffs: Distortion coefficients
        newcameramtx: New camera matrix

    Returns:
        Corrected image
    """
    return unround_img.unround(img, camera_matrix, dist_coeffs, newcameramtx)


def apply_clahe(
    img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        img: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Image with enhanced contrast
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def apply_thresholding(img: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding to create binary image.

    Args:
        img: Input grayscale image

    Returns:
        Binary (black and white) image
    """
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img


def detect_aruco_markers(img: np.ndarray, detector: cv2.aruco.ArucoDetector) -> tuple:
    """Detect ArUco markers in the image.

    Args:
        img: Input grayscale image
        detector: ArUco detector with configured parameters

    Returns:
        Tuple of (corners_list, ids, rejected)
    """
    return detector.detectMarkers(img)


def create_marker_objects(corners_list: list, ids: np.ndarray) -> list:
    """Create Aruco marker objects from detected corners and IDs.

    Args:
        corners_list: List of detected marker corners
        ids: Array of detected marker IDs

    Returns:
        List of (marker, corners) tuples
    """
    detected_markers = []

    if ids is None or len(ids) == 0:
        return detected_markers

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

        marker = aruco.Aruco(cX, cY, 1, id_val, angle)
        detected_markers.append((marker, corners))

    return detected_markers


def annotate_image_with_markers(
    img: np.ndarray, corners_list: list, detected_markers: list
) -> np.ndarray:
    """Annotate image with detected markers.

    Args:
        img: Input grayscale image
        corners_list: List of detected marker corners
        detected_markers: List of (marker, corners) tuples

    Returns:
        Annotated BGR image
    """
    annotated_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw marker boundaries without IDs
    cv2.aruco.drawDetectedMarkers(annotated_img, corners_list)

    # Draw center points and IDs
    for marker, corners in detected_markers:
        center = (int(marker.x), int(marker.y))
        # Draw green dot at center
        cv2.circle(annotated_img, center, 5, (0, 255, 0), -1)
        # Draw ID text with offset from center
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

    return annotated_img


def annotate_image_with_rejected_markers(img: np.ndarray, rejected: list) -> np.ndarray:
    """Annotate image with rejected marker candidates.

    Args:
        img: Input grayscale image
        rejected: List of rejected marker corners

    Returns:
        Annotated BGR image with rejected markers in red
    """
    rejected_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for rejected_corners in rejected:
        # Draw the rejected marker boundary in red
        pts = rejected_corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(rejected_img, [pts], True, (0, 0, 255), 2)

        # Calculate and draw center point in red
        corners_reshaped = rejected_corners.reshape((4, 2))
        cX = int(np.mean(corners_reshaped[:, 0]))
        cY = int(np.mean(corners_reshaped[:, 1]))
        cv2.circle(rejected_img, (cX, cY), 5, (0, 0, 255), -1)

        # Add "REJECTED" text
        text_pos = (cX + 20, cY + 20)
        cv2.putText(
            rejected_img,
            "REJECTED",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    return rejected_img


def compute_perspective_transform(detected_markers: list, fixed_ids: set) -> tuple:
    """Compute perspective transformation matrix from fixed markers.

    Args:
        detected_markers: List of (marker, corners) tuples
        fixed_ids: Set of fixed marker IDs to use for transformation

    Returns:
        Tuple of (perspective_matrix, found_all_markers)
    """
    tags_from_img = [marker for marker, corners in detected_markers]

    # Source points from detected markers
    src_points = []
    for fixed_id in fixed_ids:
        for tag in tags_from_img:
            if tag.aruco_id == fixed_id:
                src_points.append([tag.x, tag.y])
                break

    if len(src_points) != 4:
        return None, False

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
    return perspective_matrix, True


def transform_markers_to_real_world(
    detected_markers: list, perspective_matrix: np.ndarray
) -> None:
    """Transform marker coordinates to real world coordinates.

    Args:
        detected_markers: List of (marker, corners) tuples (modified in place)
        perspective_matrix: Perspective transformation matrix
    """
    for tag, tag_corners in detected_markers:
        # Transform center point
        img_point = np.array([[tag.x, tag.y]], dtype=np.float32)
        img_point = np.array([img_point])
        real_point = cv2.perspectiveTransform(img_point, perspective_matrix)
        real_x, real_y = real_point[0][0]
        tag.real_x = real_x
        tag.real_y = real_y

        # Transform corners to calculate real angle
        corners_reshaped = tag_corners.reshape(1, -1, 2)
        real_corners = cv2.perspectiveTransform(corners_reshaped, perspective_matrix)
        real_corners = real_corners.reshape(4, 2)

        # Calculate angle from top edge in real world coordinates
        dx = real_corners[1][0] - real_corners[0][0]
        dy = real_corners[1][1] - real_corners[0][1]
        real_angle = float(np.degrees(np.arctan2(dy, dx)))

        # Normalize angle to [-180, 180] range
        if real_angle > 180:
            real_angle -= 360
        elif real_angle < -180:
            real_angle += 360

        tag.real_angle = real_angle


def print_detection_summary(detected_markers: list) -> None:
    """Print summary of detected markers by color.

    Args:
        detected_markers: List of (marker, corners) tuples
    """
    tags_from_img = [marker for marker, corners in detected_markers]

    # Categorize tags by color based on ID ranges
    # Fixed markers: 20-23 (excluded from color count)
    # Yellow: 47
    # Blue: 36
    # Black: 41
    yellow_tags = [tag for tag in tags_from_img if tag.aruco_id == 47]
    blue_tags = [tag for tag in tags_from_img if tag.aruco_id == 36]
    black_tags = [tag for tag in tags_from_img if tag.aruco_id == 41]

    print(f"\nRÉSUMÉ DE DÉTECTION:")
    print(f"📊  - Nombre total de tags détectés: {len(tags_from_img)}")
    print(f"🟡  - Nombre de tags jaunes: {len(yellow_tags)}")
    print(f"🔵  - Nombre de tags bleus: {len(blue_tags)}")
    print(f"⚫  - Nombre de tags noirs: {len(black_tags)}")


def print_detailed_markers(detected_markers: list) -> None:
    """Print detailed information for each detected marker.

    Args:
        detected_markers: List of (marker, corners) tuples
    """
    tags_from_img = [marker for marker, corners in detected_markers]
    tags_from_img.sort(key=lambda tag: tag.aruco_id)

    for tag in tags_from_img:
        tag.print()


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
    newcameramtx, roi = unround_img.process_new_camera_matrix(
        camera_matrix, dist_coeffs, img_size
    )

    # ---------------------------------------------------------------------------
    # Statistics tracking
    # ---------------------------------------------------------------------------

    total_tags_detected = 0
    total_yellow_tags = 0
    total_blue_tags = 0
    total_black_tags = 0
    total_processing_time = 0.0
    images_processed = 0

    # ---------------------------------------------------------------------------
    # Process each image
    # ---------------------------------------------------------------------------

    for img_file in image_files:
        step = 0
        print(f"\n{'='*80}")
        print(f"Processing: {img_file.name}")
        print(f"{'='*80}")

        # Get base filename without extension for output files
        base_name = img_file.stem

        # Start timing
        start_time = time.time()

        try:
            # ---------------------------------------------------------------------------
            # Image preprocessing
            # ---------------------------------------------------------------------------

            # Load and convert to grayscale
            img = load_and_convert_to_grayscale(str(img_file))

            # Save original (converted to BGR for visualization)
            step = save_debug_image(
                base_name, "original", step, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            )
            step = save_debug_image(base_name, "grayscale", step, img)

            # Apply sharpening
            img = apply_sharpening(img, sharpen_alpha, sharpen_beta, sharpen_gamma)
            step = save_debug_image(base_name, "sharpened", step, img)

            # Apply unrounding if enabled
            if unround:
                img = apply_unrounding(img, camera_matrix, dist_coeffs, newcameramtx)
                step = save_debug_image(base_name, "unrounded", step, img)

            # Apply CLAHE if enabled
            if use_clahe:
                img = apply_clahe(img)
                step = save_debug_image(base_name, "clahe", step, img)

            # Apply thresholding if enabled
            if Thresholding:
                img = apply_thresholding(img)
                step = save_debug_image(base_name, "thresholded", step, img)

            # ---------------------------------------------------------------------------
            # ArUco detection
            # ---------------------------------------------------------------------------

            corners_list, ids, rejected = detect_aruco_markers(img, aruco_detector)

            if ids is None or len(ids) == 0:
                detected_markers = []
            else:
                detected_markers = create_marker_objects(corners_list, ids)

            tags_from_img = [marker for marker, corners in detected_markers]

            # ---------------------------------------------------------------------------
            # Annotate detected markers on image
            # ---------------------------------------------------------------------------

            if detected_markers:
                annotated_img = annotate_image_with_markers(
                    img, corners_list, detected_markers
                )
                step = save_debug_image(base_name, "annotated", step, annotated_img)

            # Draw rejected markers if enabled
            if rejected_marker and rejected is not None and len(rejected) > 0:
                rejected_img = annotate_image_with_rejected_markers(img, rejected)
                step = save_debug_image(base_name, "rejected", step, rejected_img)

            # ---------------------------------------------------------------------------
            # Perspective transformation for real world coordinates
            # ---------------------------------------------------------------------------

            perspective_matrix, found_all = compute_perspective_transform(
                detected_markers, fixed_ids
            )

            if not found_all:
                print(
                    f"⚠️  Warning: Not all fixed ArUco markers were detected for perspective transform."
                )
                print("   Skipping perspective transformation for this image.")
                print("   Real world coordinates will not be calculated.")
            else:
                transform_markers_to_real_world(detected_markers, perspective_matrix)

            # ---------------------------------------------------------------------------
            # Print detected tags information
            # ---------------------------------------------------------------------------

            if resume:
                print_detection_summary(detected_markers)
            else:
                print_detailed_markers(detected_markers)

            # ---------------------------------------------------------------------------
            # Update statistics
            # ---------------------------------------------------------------------------

            # End timing
            end_time = time.time()
            processing_time = end_time - start_time

            # Count tags by color
            tags_from_img = [marker for marker, corners in detected_markers]
            yellow_count = len([tag for tag in tags_from_img if tag.aruco_id == 47])
            blue_count = len([tag for tag in tags_from_img if tag.aruco_id == 36])
            black_count = len([tag for tag in tags_from_img if tag.aruco_id == 41])

            # Update totals
            total_tags_detected += len(tags_from_img)
            total_yellow_tags += yellow_count
            total_blue_tags += blue_count
            total_black_tags += black_count
            total_processing_time += processing_time
            images_processed += 1

            print(f"\n⏱️  Temps de traitement: {processing_time:.3f}s")

        except ValueError as e:
            print(f"❌ Error processing {img_file.name}: {e}")
            continue

    # ---------------------------------------------------------------------------
    # Print final benchmark report
    # ---------------------------------------------------------------------------

    print(f"\n{'='*80}")
    print(f"RAPPORT FINAL DE PERFORMANCE")
    print(f"{'='*80}")

    if images_processed > 0:
        avg_tags = total_tags_detected / images_processed
        avg_yellow = total_yellow_tags / images_processed
        avg_blue = total_blue_tags / images_processed
        avg_black = total_black_tags / images_processed
        avg_time = total_processing_time / images_processed

        print(f"\n📊 Images traitées: {images_processed}")
        print(f"\n📈 DÉTECTIONS MOYENNES PAR IMAGE:")
        print(f"  - Nombre moyen de tags détectés: {avg_tags:.2f}")
        print(f"  - Nombre de tags jaunes: {avg_yellow:.2f}")
        print(f"  - Nombre de tags bleus: {avg_blue:.2f}")
        print(f"  - Nombre de tags noirs: {avg_black:.2f}")
        print(f"\n⏱️  TEMPS MOYEN DE TRAITEMENT:")
        print(f"  - Temps moyen par image: {avg_time:.3f}s")
        print(f"  - Temps total: {total_processing_time:.3f}s")

        print(f"\n📊 TOTAUX:")
        print(f"  - Total tags détectés: {total_tags_detected}")
        print(f"  - Total tags jaunes: {total_yellow_tags}")
        print(f"  - Total tags bleus: {total_blue_tags}")
        print(f"  - Total tags noirs: {total_black_tags}")
    else:
        print("\n⚠️  Aucune image n'a été traitée avec succès.")

    print(f"\n{'='*80}")
    print(f"Benchmark terminé!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
