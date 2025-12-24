#!/usr/bin/env python3

"""
magic_param_finder.py

Information about this program :
- the program use a set of images with arucos on a playground at initial known positions
- the program tries to detect and all locate all the arucos
- the program evaluate the detection quality by comparing the detected arucos number and the undetected arucos number
  and the position error of the detected arucos compared to the known initial positions
- the program try to improve itself by adjusting parameters (preprocessing, aruco detection)
- once the better parameters are found the program print the new parameters to be used in the main code
- the final goal of this program is to find the best parameters to detect and locate arucos on the playground that will be used during a match

Information about eurobot competition 2026 rules :
- aruco markers are placed on game element to identify them
- there are 2 types of game elements : boxes and empty boxes
- the boxes have two faces : a blue face with aruco marker 36 and a yellow face with aruco marker 47
- the empty boxes have aruco marker 41 on both faces
- at the beginning of the match the game elements are placed on the playground at known positions and orientations but we don't know which face is visible

"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import cv2
from typing import List, Dict, Tuple
from dataclasses import dataclass
from vision_python.config import config
from vision_python.src.aruco import aruco
from vision_python.src.aruco import aruco_initial_position

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Define destination points (real world coordinates in mm)
A1 = aruco.Aruco(600, 600, 1, 20)
B1 = aruco.Aruco(1400, 600, 1, 22)
C1 = aruco.Aruco(600, 2400, 1, 21)
D1 = aruco.Aruco(1400, 2400, 1, 23)

FIXED_IDS = {20, 21, 22, 23}

# Game element ArUco marker IDs
BOX_BLUE_FACE_ID = 36
BOX_YELLOW_FACE_ID = 47
EMPTY_BOX_ID = 41
GAME_ELEMENT_IDS = {BOX_BLUE_FACE_ID, BOX_YELLOW_FACE_ID, EMPTY_BOX_ID}

# Ground truth positions for game element ArUco markers
# Since we don't know which face is visible or which elements are boxes vs empty boxes,
# we create a list of all possible positions from initial_position data
# The program will match detected markers to nearest expected positions

# Build a list of all expected positions (for any game element ID)
EXPECTED_GAME_ELEMENT_POSITIONS = []
for pos in aruco_initial_position.initial_position:
    possible_ids = pos[0]  # List of possible IDs at this position
    zone = pos[1]
    x, y, z, angle = pos[2], pos[3], pos[4], pos[5]
    EXPECTED_GAME_ELEMENT_POSITIONS.append(
        {
            "possible_ids": possible_ids,
            "zone": zone,
            "x": x,
            "y": y,
            "z": z,
            "angle": angle,
        }
    )

# Ground truth for markers with known fixed positions
GROUND_TRUTH_POSITIONS = {}

# Add fixed calibration markers to ground truth (these have exact known positions)
GROUND_TRUTH_POSITIONS[20] = {"x": 600, "y": 600, "z": 1, "angle": 0}
GROUND_TRUTH_POSITIONS[21] = {"x": 600, "y": 2400, "z": 1, "angle": 0}
GROUND_TRUTH_POSITIONS[22] = {"x": 1400, "y": 600, "z": 1, "angle": 0}
GROUND_TRUTH_POSITIONS[23] = {"x": 1400, "y": 2400, "z": 1, "angle": 0}

# Load configuration parameters
img_width = config.CAMERA_WIDTH
img_height = config.CAMERA_HEIGHT
img_size = (img_width, img_height)

# Prepare input/output directories
input_img_dir = (
    config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures" / "initial_playground"
)
output_img_dir = (
    config.RASPBERRY_DIR / "vision_python" / "tests" / "fixtures" / "output"
)

# ---------------------------------------------------------------------------
# Parameter Search Space
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingParams:
    """Parameters for image preprocessing"""

    sharpen_alpha: float = 1.5  # Weight for original image in sharpening
    sharpen_beta: float = -0.5  # Weight for blurred image in sharpening
    use_clahe: bool = (
        False  # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
    )
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8


@dataclass
class ArucoDetectionParams:
    """Parameters for ArUco detection"""

    adaptive_thresh_constant: int = 7
    min_marker_perimeter_rate: float = 0.03
    max_marker_perimeter_rate: float = 4.0
    polygonal_approx_accuracy_rate: float = 0.03
    min_corner_distance_rate: float = 0.05
    min_distance_to_border: int = 3
    corner_refinement_method: int = cv2.aruco.CORNER_REFINE_SUBPIX


@dataclass
class DetectionResult:
    """Results from a single detection attempt"""

    num_detected: int
    num_missed: int
    avg_position_error: float
    max_position_error: float
    score: float
    preprocessing_params: PreprocessingParams
    detection_params: ArucoDetectionParams


# Define parameter search space
PREPROCESSING_SEARCH_SPACE = {
    "sharpen_alpha": [1.0, 1.5, 2.0],
    "sharpen_beta": [-0.5, -0.7, -1.0],
    "use_clahe": [False, True],
    "clahe_clip_limit": [2.0, 3.0],
    "clahe_tile_size": [8, 16],
}

ARUCO_DETECTION_SEARCH_SPACE = {
    "adaptive_thresh_constant": [5, 7, 10],
    "min_marker_perimeter_rate": [0.01, 0.03, 0.05],
    "max_marker_perimeter_rate": [3.0, 4.0, 5.0],
    "polygonal_approx_accuracy_rate": [0.03, 0.05, 0.1],
    "min_corner_distance_rate": [0.05, 0.1],
    "min_distance_to_border": [3, 5],
}

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def calculate_detection_score(
    detected_markers: List[aruco.Aruco], ground_truth: Dict[int, Dict]
) -> Tuple[int, int, float, float, float]:
    """
    Calculate detection quality metrics.

    Focuses on:
    - Detection of fixed calibration markers (IDs 20, 21, 22, 23) - critical for perspective transform
    - Detection of game element markers (IDs 36, 41, 47) - boxes and empty boxes
    - Position accuracy for markers with known ground truth

    Returns:
        - num_detected: Number of markers detected
        - num_missed: Number of expected markers that weren't detected
        - avg_position_error: Average position error for detected markers with ground truth (mm)
        - max_position_error: Maximum position error for detected markers with ground truth (mm)
        - score: Overall quality score (higher is better)
    """
    detected_ids = {marker.aruco_id for marker in detected_markers}

    # Count fixed markers (critical for calibration)
    detected_fixed = len(detected_ids & FIXED_IDS)
    missed_fixed = len(FIXED_IDS - detected_ids)

    # Count game element markers
    detected_game = len(detected_ids & GAME_ELEMENT_IDS)

    # Total detection counts
    num_detected = len(detected_ids)
    expected_ids = set(ground_truth.keys())
    num_missed = len(expected_ids - detected_ids)

    # Calculate position errors for markers with known ground truth (fixed markers)
    position_errors = []
    for marker in detected_markers:
        if (
            marker.aruco_id in ground_truth
            and hasattr(marker, "real_x")
            and hasattr(marker, "real_y")
        ):
            gt = ground_truth[marker.aruco_id]
            error = np.sqrt(
                (marker.real_x - gt["x"]) ** 2 + (marker.real_y - gt["y"]) ** 2
            )
            position_errors.append(error)

    # Also calculate position errors for game elements by matching to nearest expected position
    game_element_errors = []
    for marker in detected_markers:
        if (
            marker.aruco_id in GAME_ELEMENT_IDS
            and hasattr(marker, "real_x")
            and hasattr(marker, "real_y")
        ):
            # Find nearest expected position that could have this marker ID
            min_dist = float("inf")
            for exp_pos in EXPECTED_GAME_ELEMENT_POSITIONS:
                if marker.aruco_id in exp_pos["possible_ids"]:
                    dist = np.sqrt(
                        (marker.real_x - exp_pos["x"]) ** 2
                        + (marker.real_y - exp_pos["y"]) ** 2
                    )
                    min_dist = min(min_dist, dist)
            if (
                min_dist != float("inf") and min_dist < 200
            ):  # Only count if within 200mm
                game_element_errors.append(min_dist)

    # Combine all position errors
    all_errors = position_errors + game_element_errors
    position_errors = all_errors

    avg_position_error = np.mean(position_errors) if position_errors else float("inf")
    max_position_error = np.max(position_errors) if position_errors else float("inf")

    # Calculate score with emphasis on:
    # 1. All 4 fixed markers must be detected (critical for perspective transform)
    # 2. Maximum number of game elements detected
    # 3. Position accuracy for markers with ground truth

    # Penalty if not all fixed markers detected (makes perspective transform impossible)
    if detected_fixed < 4:
        fixed_penalty = 50 * (4 - detected_fixed)  # Heavy penalty
    else:
        fixed_penalty = 0

    # Reward for detecting game elements (main optimization target)
    game_element_score = detected_game * 10

    # Reward for overall detection rate
    expected_game_elements = len(EXPECTED_GAME_ELEMENT_POSITIONS)
    detection_rate = detected_game / max(expected_game_elements, 1)
    detection_score = 50 * detection_rate

    # Position accuracy bonus (lower error = higher score)
    if position_errors:
        position_penalty = avg_position_error / 50.0  # Normalize by 50mm
    else:
        position_penalty = 0

    # Final score: maximize detections, minimize position errors, heavily penalize missing fixed markers
    score = detection_score + game_element_score - position_penalty - fixed_penalty

    return num_detected, num_missed, avg_position_error, max_position_error, score


def preprocess_image(img: np.ndarray, params: PreprocessingParams) -> np.ndarray:
    """Apply preprocessing to the image based on parameters"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE if enabled
    if params.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=params.clahe_clip_limit,
            tileGridSize=(params.clahe_tile_size, params.clahe_tile_size),
        )
        img = clahe.apply(img)

    # Apply sharpening
    img = cv2.addWeighted(img, params.sharpen_alpha, img, params.sharpen_beta, 0)

    return img


def detect_markers(
    img: np.ndarray, params: ArucoDetectionParams
) -> Tuple[List, np.ndarray, List]:
    """Detect ArUco markers with given parameters"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Set detection parameters
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = params.corner_refinement_method
    parameters.adaptiveThreshConstant = params.adaptive_thresh_constant
    parameters.minMarkerPerimeterRate = params.min_marker_perimeter_rate
    parameters.maxMarkerPerimeterRate = params.max_marker_perimeter_rate
    parameters.polygonalApproxAccuracyRate = params.polygonal_approx_accuracy_rate
    parameters.minCornerDistanceRate = params.min_corner_distance_rate
    parameters.minDistanceToBorder = params.min_distance_to_border

    # Create detector and detect markers
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners_list, ids, rejected = aruco_detector.detectMarkers(img)

    return corners_list, ids, rejected


def process_detected_markers(corners_list, ids) -> List[aruco.Aruco]:
    """Convert detected markers to Aruco objects"""
    detected_markers = []

    if ids is None or len(ids) == 0:
        return detected_markers

    ids = ids.flatten()
    for corners, id_val in zip(corners_list, ids):
        corners = corners.reshape((4, 2)).astype(np.float32)
        cX = float(np.mean(corners[:, 0]))
        cY = float(np.mean(corners[:, 1]))

        # Calculate angle from top edge
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # Normalize angle to [-180, 180] range
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360

        a = aruco.Aruco(cX, cY, 1, id_val, angle)
        detected_markers.append(a)

    return detected_markers


def apply_perspective_transform(tags_from_img: List[aruco.Aruco]) -> bool:
    """Apply perspective transformation to get real-world coordinates"""
    # Source points from detected markers
    src_points = []
    for fixed_id in FIXED_IDS:
        for tag in tags_from_img:
            if tag.aruco_id == fixed_id:
                src_points.append([tag.x, tag.y])
                break

    if len(src_points) != 4:
        return False

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
        img_point = np.array([img_point])
        real_point = cv2.perspectiveTransform(img_point, perspective_matrix)
        real_x, real_y = real_point[0][0]
        tag.real_x = real_x
        tag.real_y = real_y

    return True


# ---------------------------------------------------------------------------
# Parameter Search Implementation
# ---------------------------------------------------------------------------

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
# Print ArUco Marker Information
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("ARUCO MARKER CONFIGURATION")
print("=" * 80)
print("\nFixed Calibration Markers (for perspective transform):")
print(f"  IDs: {sorted(FIXED_IDS)}")
print("\nGame Element Markers (Eurobot 2026):")
print(f"  ID {BOX_BLUE_FACE_ID}: Box - Blue Face")
print(f"  ID {BOX_YELLOW_FACE_ID}: Box - Yellow Face")
print(f"  ID {EMPTY_BOX_ID}: Empty Box (both faces)")
print("\nOptimization Goal:")
print("  - Detect all 4 fixed markers (required for perspective transform)")
print("  - Maximize detection of game element markers")
print("  - Minimize position errors for markers with known ground truth")

# ---------------------------------------------------------------------------
# Parameter Search
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STARTING PARAMETER OPTIMIZATION")
print("=" * 80)

# Generate test configurations
test_configs = []

# Test default configuration first
test_configs.append((PreprocessingParams(), ArucoDetectionParams()))

# Test variations of preprocessing
for sharpen_alpha in [1.0, 1.5, 2.0]:
    for sharpen_beta in [-0.5, -0.7]:
        test_configs.append(
            (
                PreprocessingParams(
                    sharpen_alpha=sharpen_alpha, sharpen_beta=sharpen_beta
                ),
                ArucoDetectionParams(),
            )
        )

# Test variations of CLAHE
for use_clahe in [True]:
    for clip_limit in [2.0, 3.0]:
        test_configs.append(
            (
                PreprocessingParams(use_clahe=use_clahe, clahe_clip_limit=clip_limit),
                ArucoDetectionParams(),
            )
        )

# Test variations of ArUco detection parameters
for adaptive_const in [5, 7, 10]:
    for min_perim in [0.01, 0.03, 0.05]:
        test_configs.append(
            (
                PreprocessingParams(),
                ArucoDetectionParams(
                    adaptive_thresh_constant=adaptive_const,
                    min_marker_perimeter_rate=min_perim,
                ),
            )
        )

print(f"\nTesting {len(test_configs)} parameter configurations...\n")

best_results = []
best_overall_score = -float("inf")
best_config = None

# Test each configuration
for config_idx, (preproc_params, detect_params) in enumerate(test_configs):
    print(f"\n[{config_idx + 1}/{len(test_configs)}] Testing configuration:")
    print(
        f"  Preprocessing: sharpen=({preproc_params.sharpen_alpha}, {preproc_params.sharpen_beta}), "
        f"CLAHE={preproc_params.use_clahe}"
    )
    print(
        f"  Detection: adaptive_const={detect_params.adaptive_thresh_constant}, "
        f"min_perim={detect_params.min_marker_perimeter_rate}"
    )

    total_detected = 0
    total_missed = 0
    total_score = 0.0
    all_position_errors = []

    # Test configuration on all images
    for img_file in image_files:
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # Preprocess
        processed_img = preprocess_image(img, preproc_params)

        # Detect markers
        corners_list, ids, rejected = detect_markers(processed_img, detect_params)

        # Process detected markers
        detected_markers = process_detected_markers(corners_list, ids)

        # Apply perspective transform
        transform_success = apply_perspective_transform(detected_markers)

        if transform_success:
            # Calculate metrics
            num_det, num_miss, avg_err, max_err, score = calculate_detection_score(
                detected_markers, GROUND_TRUTH_POSITIONS
            )

            total_detected += num_det
            total_missed += num_miss
            total_score += score
            if avg_err != float("inf"):
                all_position_errors.append(avg_err)

    # Calculate average metrics across all images
    avg_score = total_score / len(image_files) if image_files else 0
    avg_position_error = (
        np.mean(all_position_errors) if all_position_errors else float("inf")
    )

    result = DetectionResult(
        num_detected=total_detected,
        num_missed=total_missed,
        avg_position_error=avg_position_error,
        max_position_error=(
            np.max(all_position_errors) if all_position_errors else float("inf")
        ),
        score=avg_score,
        preprocessing_params=preproc_params,
        detection_params=detect_params,
    )

    best_results.append(result)

    print(
        f"  Results: detected={total_detected}, missed={total_missed}, "
        f"avg_error={avg_position_error:.2f}mm, score={avg_score:.2f}"
    )

    # Track best configuration
    if avg_score > best_overall_score:
        best_overall_score = avg_score
        best_config = result

# ---------------------------------------------------------------------------
# Print Best Results
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE - BEST PARAMETERS FOUND")
print("=" * 80)

if best_config:
    print(f"\nBest Score: {best_config.score:.2f}")
    print(f"Total Detected: {best_config.num_detected}")
    print(f"Total Missed: {best_config.num_missed}")
    if best_config.avg_position_error != float("inf"):
        print(f"Average Position Error: {best_config.avg_position_error:.2f} mm")
        print(f"Max Position Error: {best_config.max_position_error:.2f} mm")

    print("\n--- BEST PREPROCESSING PARAMETERS ---")
    print(f"sharpen_alpha = {best_config.preprocessing_params.sharpen_alpha}")
    print(f"sharpen_beta = {best_config.preprocessing_params.sharpen_beta}")
    print(f"use_clahe = {best_config.preprocessing_params.use_clahe}")
    if best_config.preprocessing_params.use_clahe:
        print(f"clahe_clip_limit = {best_config.preprocessing_params.clahe_clip_limit}")
        print(f"clahe_tile_size = {best_config.preprocessing_params.clahe_tile_size}")

    print("\n--- BEST ARUCO DETECTION PARAMETERS ---")
    print(
        f"adaptive_thresh_constant = {best_config.detection_params.adaptive_thresh_constant}"
    )
    print(
        f"min_marker_perimeter_rate = {best_config.detection_params.min_marker_perimeter_rate}"
    )
    print(
        f"max_marker_perimeter_rate = {best_config.detection_params.max_marker_perimeter_rate}"
    )
    print(
        f"polygonal_approx_accuracy_rate = {best_config.detection_params.polygonal_approx_accuracy_rate}"
    )
    print(
        f"min_corner_distance_rate = {best_config.detection_params.min_corner_distance_rate}"
    )
    print(
        f"min_distance_to_border = {best_config.detection_params.min_distance_to_border}"
    )

    print("\n--- CODE TO USE IN MAIN PROGRAM ---")
    print(
        """
# Preprocessing
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"""
    )

    if best_config.preprocessing_params.use_clahe:
        print(
            f"""
clahe = cv2.createCLAHE(clipLimit={best_config.preprocessing_params.clahe_clip_limit}, 
                        tileGridSize=({best_config.preprocessing_params.clahe_tile_size}, 
                                     {best_config.preprocessing_params.clahe_tile_size}))
img = clahe.apply(img)"""
        )

    print(
        f"""
img = cv2.addWeighted(img, {best_config.preprocessing_params.sharpen_alpha}, 
                     img, {best_config.preprocessing_params.sharpen_beta}, 0)

# ArUco Detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.adaptiveThreshConstant = {best_config.detection_params.adaptive_thresh_constant}
parameters.minMarkerPerimeterRate = {best_config.detection_params.min_marker_perimeter_rate}
parameters.maxMarkerPerimeterRate = {best_config.detection_params.max_marker_perimeter_rate}
parameters.polygonalApproxAccuracyRate = {best_config.detection_params.polygonal_approx_accuracy_rate}
parameters.minCornerDistanceRate = {best_config.detection_params.min_corner_distance_rate}
parameters.minDistanceToBorder = {best_config.detection_params.min_distance_to_border}
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
"""
    )

# Sort and print top 5 configurations
print("\n" + "=" * 80)
print("TOP 5 CONFIGURATIONS")
print("=" * 80)

best_results.sort(key=lambda x: x.score, reverse=True)
for idx, result in enumerate(best_results[:5], 1):
    print(f"\n#{idx} - Score: {result.score:.2f}")
    print(f"  Detected: {result.num_detected}, Missed: {result.num_missed}")
    if result.avg_position_error != float("inf"):
        print(f"  Avg Error: {result.avg_position_error:.2f}mm")
    print(
        f"  Preproc: sharpen=({result.preprocessing_params.sharpen_alpha}, "
        f"{result.preprocessing_params.sharpen_beta}), CLAHE={result.preprocessing_params.use_clahe}"
    )
    print(
        f"  Detect: adaptive={result.detection_params.adaptive_thresh_constant}, "
        f"min_perim={result.detection_params.min_marker_perimeter_rate}"
    )

# ---------------------------------------------------------------------------
# Optional: Process images with best parameters and save annotated output
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PROCESSING IMAGES WITH BEST PARAMETERS")
print("=" * 80)

if best_config:
    for img_file in image_files:
        print(f"\n{img_file.name}:")

        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  Failed to load image, skipping...")
            continue

        # Get base filename without extension for output files
        base_name = img_file.stem

        # Preprocess with best parameters
        processed_img = preprocess_image(img, best_config.preprocessing_params)
        cv2.imwrite(
            str(output_img_dir / f"{base_name}_best_preprocessed.jpg"), processed_img
        )

        # Detect markers with best parameters
        corners_list, ids, rejected = detect_markers(
            processed_img, best_config.detection_params
        )

        # Process detected markers
        detected_markers = process_detected_markers(corners_list, ids)

        # Apply perspective transform
        transform_success = apply_perspective_transform(detected_markers)

        # Annotate image
        annotated_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        if ids is not None and len(ids) > 0:
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
        cv2.imwrite(
            str(output_img_dir / f"{base_name}_best_annotated.jpg"), annotated_img
        )

        # Calculate and print metrics
        if transform_success:
            num_det, num_miss, avg_err, max_err, score = calculate_detection_score(
                detected_markers, GROUND_TRUTH_POSITIONS
            )

            # Categorize detected markers
            detected_ids = {m.aruco_id for m in detected_markers}
            fixed_detected = detected_ids & FIXED_IDS
            game_detected = detected_ids & GAME_ELEMENT_IDS
            other_detected = detected_ids - FIXED_IDS - GAME_ELEMENT_IDS

            print(f"  Total Detected: {num_det}, Missed: {num_miss}")
            print(
                f"  Fixed markers (20-23): {len(fixed_detected)}/4 detected: {sorted(fixed_detected)}"
            )
            print(
                f"  Game elements (36,41,47): {len(game_detected)} detected: {sorted(game_detected)}"
            )
            if other_detected:
                print(f"  Other markers: {sorted(other_detected)}")
            if avg_err != float("inf"):
                print(f"  Avg Position Error: {avg_err:.2f}mm, Max: {max_err:.2f}mm")
            print(f"  Score: {score:.2f}")

            # Print detected markers sorted by ID
            detected_markers.sort(key=lambda tag: tag.aruco_id)
            for marker in detected_markers:
                marker_type = ""
                if marker.aruco_id in FIXED_IDS:
                    marker_type = " [FIXED]"
                elif marker.aruco_id == BOX_BLUE_FACE_ID:
                    marker_type = " [BOX-BLUE]"
                elif marker.aruco_id == BOX_YELLOW_FACE_ID:
                    marker_type = " [BOX-YELLOW]"
                elif marker.aruco_id == EMPTY_BOX_ID:
                    marker_type = " [EMPTY-BOX]"

                if hasattr(marker, "real_x") and hasattr(marker, "real_y"):
                    print(
                        f"    ID {marker.aruco_id}{marker_type}: ({marker.real_x:.1f}, {marker.real_y:.1f}) mm, angle: {marker.angle:.1f}°"
                    )
        else:
            print(f"  Could not apply perspective transform (missing fixed markers)")
            detected_ids = {m.aruco_id for m in detected_markers}
            fixed_detected = detected_ids & FIXED_IDS
            print(
                f"  Fixed markers detected: {len(fixed_detected)}/4: {sorted(fixed_detected)}"
            )

print("\n" + "=" * 80)
print("ALL PROCESSING COMPLETE")
print("=" * 80)
