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

image preprocessing parameters to optimize :
- sharpening (alpha, beta)
- CLAHE (use or not, clip limit, tile size)

aruco detection parameters to optimize :
- adaptive threshold constant
- min marker perimeter rate
- max marker perimeter rate
- polygonal approx accuracy rateZ
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
from vision_python.src.img_processing import processing_pipeline as pipeline

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
input_img_dir = config.PICTURES_DIR / "camera" / "2026-01-09"
output_img_dir = config.get_debug_directory() / "magic_param_finder_output"
use_best_config = False

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

    def create_detector(self) -> cv2.aruco.ArucoDetector:
        """Create an ArUco detector with these parameters"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = self.corner_refinement_method
        parameters.adaptiveThreshConstant = self.adaptive_thresh_constant
        parameters.minMarkerPerimeterRate = self.min_marker_perimeter_rate
        parameters.maxMarkerPerimeterRate = self.max_marker_perimeter_rate
        parameters.polygonalApproxAccuracyRate = self.polygonal_approx_accuracy_rate
        parameters.minCornerDistanceRate = self.min_corner_distance_rate
        parameters.minDistanceToBorder = self.min_distance_to_border
        return cv2.aruco.ArucoDetector(aruco_dict, parameters)


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
                position_errors.append(min_dist)

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


def detect_and_process_with_pipeline(
    img: np.ndarray,
    preproc_params: PreprocessingParams,
    detect_params: ArucoDetectionParams,
) -> Tuple[List[aruco.Aruco], bool]:
    """Use the pipeline to detect and process ArUco markers"""
    # Create ArUco detector
    aruco_detector = detect_params.create_detector()

    # Define fixed markers for perspective transform
    fixed_markers = [A1, B1, C1, D1]

    # Define playground corners (in real world coordinates)
    playground_corners = [
        (A1.x, A1.y),
        (B1.x, B1.y),
        (D1.x, D1.y),
        (C1.x, C1.y),
    ]

    # Call the pipeline
    detected_markers, final_img, perspective_matrix, metadata = (
        pipeline.process_image_for_aruco_detection(
            img=img,
            aruco_detector=aruco_detector,
            camera_matrix=None,
            dist_coeffs=None,
            newcameramtx=None,
            fixed_markers=fixed_markers,
            playground_corners=playground_corners,
            use_unround=False,
            use_clahe=preproc_params.use_clahe,
            use_thresholding=False,
            sharpen_alpha=preproc_params.sharpen_alpha,
            sharpen_beta=preproc_params.sharpen_beta,
            sharpen_gamma=0.0,
            use_mask_playground=False,
            use_straighten_image=False,
            save_debug_images=False,
            debug_dir=None,
            base_name="image",
            apply_contrast_boost=False,
            contrast_alpha=1.1,
        )
    )

    # Extract Aruco objects from detected_markers
    # detected_markers is a list of (marker, corners) tuples
    tags_from_img = [marker for marker, corners in detected_markers]

    # Check if perspective transform was successful
    transform_success = metadata["perspective_transform_computed"]

    return tags_from_img, transform_success


# ---------------------------------------------------------------------------
# Parameter Search Implementation
# ---------------------------------------------------------------------------


def load_image_files(img_dir):
    """Load all image files from the specified directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [
        f
        for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        raise ValueError(f"No image files found in {img_dir}")

    print(f"Found {len(image_files)} image(s) to process")
    return image_files


def print_marker_configuration():
    """Print information about ArUco marker configuration."""
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


def generate_test_configurations():
    """Generate all parameter configurations to test."""
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
                    PreprocessingParams(
                        use_clahe=use_clahe, clahe_clip_limit=clip_limit
                    ),
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

    return test_configs


def run_parameter_search(test_configs, image_files):
    """Run parameter optimization search across all configurations."""
    print("\n" + "=" * 80)
    print("STARTING PARAMETER OPTIMIZATION")
    print("=" * 80)
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
        num_images_processed = 0

        # Test configuration on all images
        for img_file in image_files:
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            # Use pipeline to detect and process markers
            detected_markers, transform_success = detect_and_process_with_pipeline(
                img, preproc_params, detect_params
            )

            if transform_success:
                # Calculate metrics
                num_det, num_miss, avg_err, max_err, score = calculate_detection_score(
                    detected_markers, GROUND_TRUTH_POSITIONS
                )

                total_detected += num_det
                total_missed += num_miss
                total_score += score
                num_images_processed += 1
                if avg_err != float("inf"):
                    all_position_errors.append(avg_err)

        # Calculate average metrics across all images
        avg_score = total_score / len(image_files) if image_files else 0
        avg_position_error = (
            np.mean(all_position_errors) if all_position_errors else float("inf")
        )
        avg_tags_per_photo = (
            total_detected / num_images_processed if num_images_processed > 0 else 0
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
            f"avg_tags_per_photo={avg_tags_per_photo:.1f}, "
            f"avg_error={avg_position_error:.2f}mm, score={avg_score:.2f}"
        )

        # Track best configuration
        if avg_score > best_overall_score:
            best_overall_score = avg_score
            best_config = result

    return best_results, best_config


def print_best_parameters(best_config):
    """Print the best parameters found during optimization."""
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
            print(
                f"clahe_clip_limit = {best_config.preprocessing_params.clahe_clip_limit}"
            )
            print(
                f"clahe_tile_size = {best_config.preprocessing_params.clahe_tile_size}"
            )

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
blur = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.addWeighted(img, {best_config.preprocessing_params.sharpen_alpha}, 
                     blur, {best_config.preprocessing_params.sharpen_beta}, 0)

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


def print_top_configurations(best_results, image_files):
    """Print the top 5 parameter configurations."""
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 80)

    best_results.sort(key=lambda x: x.score, reverse=True)
    for idx, result in enumerate(best_results[:5], 1):
        # Calculate avg tags per photo for this result
        avg_tags = result.num_detected / len(image_files) if image_files else 0
        print(f"\n#{idx} - Score: {result.score:.2f}")
        print(
            f"  Detected: {result.num_detected}, Missed: {result.num_missed}, Avg tags/photo: {avg_tags:.1f}"
        )
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


def process_images_with_best_config(best_config, image_files, output_img_dir):
    """Process all images with the best configuration and save debug outputs."""
    for img_file in image_files:
        print(f"\n{img_file.name}:")

        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  Failed to load image, skipping...")
            continue

        # Get base filename without extension for output files
        base_name = img_file.stem

        # Use pipeline with best parameters to detect and process
        aruco_detector = best_config.detection_params.create_detector()
        fixed_markers = [A1, B1, C1, D1]
        playground_corners = [
            (A1.x, A1.y),
            (B1.x, B1.y),
            (D1.x, D1.y),
            (C1.x, C1.y),
        ]

        detected_markers_tuples, processed_img, perspective_matrix, metadata = (
            pipeline.process_image_for_aruco_detection(
                img=img,
                aruco_detector=aruco_detector,
                camera_matrix=None,
                dist_coeffs=None,
                newcameramtx=None,
                fixed_markers=fixed_markers,
                playground_corners=playground_corners,
                use_unround=False,
                use_clahe=best_config.preprocessing_params.use_clahe,
                use_thresholding=False,
                sharpen_alpha=best_config.preprocessing_params.sharpen_alpha,
                sharpen_beta=best_config.preprocessing_params.sharpen_beta,
                sharpen_gamma=0.0,
                use_mask_playground=False,
                use_straighten_image=False,
                save_debug_images=True,
                debug_dir=output_img_dir,
                base_name=base_name,
                apply_contrast_boost=False,
                contrast_alpha=1.1,
            )
        )

        # Extract markers and corners
        detected_markers = [marker for marker, corners in detected_markers_tuples]
        corners_list = [corners for marker, corners in detected_markers_tuples]
        ids = (
            np.array([[marker.aruco_id] for marker in detected_markers])
            if detected_markers
            else None
        )

        transform_success = metadata["perspective_transform_computed"]

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


def main():
    """Main function to run the parameter optimization."""
    # Load image files
    image_files = load_image_files(input_img_dir)

    # Print marker configuration
    print_marker_configuration()

    # Generate test configurations
    test_configs = generate_test_configurations()

    # Run parameter search
    best_results, best_config = run_parameter_search(test_configs, image_files)

    # Print best parameters
    print_best_parameters(best_config)

    # Print top configurations
    print_top_configurations(best_results, image_files)

    # Optional: Process images with best parameters and save annotated output
    if use_best_config:
        process_images_with_best_config(best_config, image_files, output_img_dir)

    print("\n" + "=" * 80)
    print("ALL PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
