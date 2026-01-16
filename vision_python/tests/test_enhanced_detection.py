#!/usr/bin/env python3

"""
test_enhanced_detection.py

Enhanced ArUco detection pipeline using multi-pass and multi-scale strategies
to maximize detection rate in challenging conditions.

Features:
- Multi-pass detection: 3 passes with aggressive, standard, and conservative parameters
- Multi-scale detection: Tests detection at 0.75x, 1.0x, and 1.25x scales
- Enhanced preprocessing: CLAHE, bilateral filtering, sharpening, morphological operations
- Comparison with standard pipeline to measure improvement
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
from vision_python.config import config
from vision_python.src.aruco import aruco
from vision_python.src.img_processing import unround_img
from vision_python.src.img_processing import detect_aruco
from vision_python.src.img_processing import processing_pipeline as pipeline


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

rejected_marker = True  # Draw rejected markers on debug images
resume = True  # Print summary of detections instead of detailed info
save_debug_images = True  # Save debug images at each processing step
mask_playground = True  # Mask to keep only playground area
use_multi_pass = True  # Use multiple detection passes with different parameters
use_multi_scale = True  # Try detection at different image scales
run_standard_comparison = True  # Also run standard pipeline for comparison
input_pict_dir = config.get_camera_directory() / "2026-01-09-playground-ready"
debug_pict_dir = config.get_debug_directory() / "enhanced_detection_output"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Get camera parameters from config
camera_params = config.get_camera_params()

# Get image processing parameters from config
img_params = config.get_image_processing_params()

# Extract image processing parameters
use_unround = img_params["use_unround"]
use_clahe = img_params["use_clahe"]
use_thresholding = img_params["use_thresholding"]
sharpen_alpha = img_params["sharpen_alpha"]
sharpen_beta = img_params["sharpen_beta"]
sharpen_gamma = img_params["sharpen_gamma"]
adaptive_thresh_constant = img_params["adaptive_thresh_constant"]
min_marker_perimeter_rate = img_params["min_marker_perimeter_rate"]
max_marker_perimeter_rate = img_params["max_marker_perimeter_rate"]
polygonal_approx_accuracy_rate = img_params["polygonal_approx_accuracy_rate"]

# Extract camera parameters
calibration_file = camera_params["calibration_file"]
img_width = camera_params["img_width"]
img_height = camera_params["img_height"]
img_size = (img_width, img_height)

# Get fixed ArUco marker positions from config
FIXED_MARKERS = config.get_fixed_aruco_markers()
fixed_ids = [marker.aruco_id for marker in FIXED_MARKERS]

# Playground corners in real world coordinates (mm)
PLAYGROUND_CORNERS = config.get_playground_corners()

# ---------------------------------------------------------------------------
# Enhanced Detection Functions
# ---------------------------------------------------------------------------


def multi_pass_detection(img, aruco_dict=None):
    """
    Perform multiple detection passes with different parameters.
    Returns merged results from all passes.

    Args:
        img: Input image (grayscale or BGR)
        aruco_dict: ArUco dictionary to use

    Returns:
        dict: {marker_id: (corners, detection_method)}
    """
    if aruco_dict is None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    detected_markers = {}  # id -> (corners, params_used)

    # Pass 1: Aggressive - find small/faint markers
    params1 = cv2.aruco.DetectorParameters()
    params1.adaptiveThreshConstant = 5
    params1.minMarkerPerimeterRate = 0.01
    params1.maxMarkerPerimeterRate = 5.0
    params1.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params1.polygonalApproxAccuracyRate = 0.05
    params1.minCornerDistanceRate = 0.05
    params1.minDistanceToBorder = 3
    detector1 = cv2.aruco.ArucoDetector(aruco_dict, params1)
    corners1, ids1, _ = detector1.detectMarkers(gray)
    if ids1 is not None:
        for i, marker_id in enumerate(ids1):
            detected_markers[marker_id[0]] = (corners1[i], "aggressive")

    # Pass 2: Standard - balanced parameters
    params2 = cv2.aruco.DetectorParameters()
    params2.adaptiveThreshConstant = 7
    params2.minMarkerPerimeterRate = 0.03
    params2.maxMarkerPerimeterRate = 4.0
    params2.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params2.polygonalApproxAccuracyRate = 0.03
    params2.minCornerDistanceRate = 0.05
    params2.minDistanceToBorder = 3
    detector2 = cv2.aruco.ArucoDetector(aruco_dict, params2)
    corners2, ids2, _ = detector2.detectMarkers(gray)
    if ids2 is not None:
        for i, marker_id in enumerate(ids2):
            if marker_id[0] not in detected_markers:
                detected_markers[marker_id[0]] = (corners2[i], "standard")

    # Pass 3: Conservative - reduce false positives
    params3 = cv2.aruco.DetectorParameters()
    params3.adaptiveThreshConstant = 10
    params3.minMarkerPerimeterRate = 0.05
    params3.maxMarkerPerimeterRate = 4.0
    params3.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    params3.polygonalApproxAccuracyRate = 0.03
    params3.minCornerDistanceRate = 0.05
    params3.minDistanceToBorder = 3
    detector3 = cv2.aruco.ArucoDetector(aruco_dict, params3)
    corners3, ids3, _ = detector3.detectMarkers(gray)
    if ids3 is not None:
        for i, marker_id in enumerate(ids3):
            if marker_id[0] not in detected_markers:
                detected_markers[marker_id[0]] = (corners3[i], "conservative")

    return detected_markers


def multi_scale_detection(img, aruco_dict=None, scales=[0.75, 1.0, 1.25]):
    """
    Try detection at multiple image scales.
    Returns merged results from all scales.

    Args:
        img: Input image (grayscale or BGR)
        aruco_dict: ArUco dictionary to use
        scales: List of scale factors to try

    Returns:
        dict: {marker_id: (corners, detection_method)}
    """
    if aruco_dict is None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    detected_markers = {}  # id -> (corners, scale_used)

    for scale in scales:
        # Resize image
        if scale != 1.0:
            h, w = gray.shape[:2]
            scaled_img = cv2.resize(gray, (int(w * scale), int(h * scale)))
        else:
            scaled_img = gray

        # Detect on scaled image
        if use_multi_pass:
            scale_detections = multi_pass_detection(scaled_img, aruco_dict)
            for marker_id, (corners, method) in scale_detections.items():
                # Scale corners back to original size
                if scale != 1.0:
                    corners = corners / scale
                if marker_id not in detected_markers:
                    detected_markers[marker_id] = (
                        corners,
                        f"scale_{scale}_{method}",
                    )
        else:
            # Use standard detector
            params = cv2.aruco.DetectorParameters()
            params.adaptiveThreshConstant = adaptive_thresh_constant
            params.minMarkerPerimeterRate = min_marker_perimeter_rate
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(scaled_img)

            if ids is not None:
                for i, marker_id in enumerate(ids):
                    # Scale corners back to original size
                    scaled_corners = corners[i] / scale if scale != 1.0 else corners[i]
                    if marker_id[0] not in detected_markers:
                        detected_markers[marker_id[0]] = (
                            scaled_corners,
                            f"scale_{scale}",
                        )

    return detected_markers


def enhanced_preprocessing(img):
    """
    Apply enhanced preprocessing for difficult detection scenarios.

    Args:
        img: Input image (BGR or grayscale)

    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 1. Illumination normalization using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 2. Bilateral filter (preserves edges better than Gaussian)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3. Sharpening
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # 4. Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return gray


def visualize_detections(img, detected_dict, title="Detected Markers"):
    """
    Create annotated image with detected markers.

    Args:
        img: Original image
        detected_dict: Dictionary of {marker_id: (corners, method)}
        title: Title for the visualization

    Returns:
        Annotated image
    """
    annotated = img.copy()

    ids_list = []
    corners_list = []
    for marker_id, (corners, method) in detected_dict.items():
        ids_list.append([marker_id])
        corners_list.append(corners)

    if ids_list:
        ids_array = np.array(ids_list, dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(annotated, corners_list, ids_array)

        # Add text showing detection method for each marker
        for i, (marker_id, (corners, method)) in enumerate(detected_dict.items()):
            # Get center of marker
            center = corners[0].mean(axis=0).astype(int)
            # Draw method text above marker
            cv2.putText(
                annotated,
                method,
                (center[0], center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

    return annotated


# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------


def main():
    """Main function to run enhanced detection tests."""

    # Ensure debug directory exists
    debug_pict_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files from input directory
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [
        f
        for f in input_pict_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        raise ValueError(f"No image files found in {input_pict_dir}")

    print(f"\n{'='*80}")
    print(f"ENHANCED ARUCO DETECTION TEST")
    print(f"{'='*80}")
    print(f"Found {len(image_files)} image(s) to process")
    print("\nEnhanced Detection Options:")
    print(
        f"  - Multi-pass detection: {use_multi_pass} (aggressive + standard + conservative)"
    )
    print(f"  - Multi-scale detection: {use_multi_scale} (0.75x, 1.0x, 1.25x)")
    print(
        f"  - Enhanced preprocessing: CLAHE + bilateral filter + sharpening + morphology"
    )
    print(f"  - Save debug images: {save_debug_images}")
    print(f"  - Run standard comparison: {run_standard_comparison}")
    print(f"{'='*80}")

    # Load camera calibration matrices
    camera_matrix, dist_coeffs, newcameramtx, roi = (
        config.get_camera_calibration_matrices()
    )

    # Initialize ArUco detector for standard pipeline
    aruco_detector = detect_aruco.init_aruco_detector()

    # Statistics tracking
    total_enhanced = 0
    total_standard = 0
    total_enhanced_only = 0
    total_standard_only = 0

    # Process each image
    for img_file in image_files:
        print(f"\n{'='*80}")
        print(f"Processing: {img_file.name}")
        print(f"{'='*80}")

        base_name = img_file.stem

        try:
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                raise ValueError(f"Failed to load image from {img_file}")

            # Save original for debugging
            if save_debug_images:
                pipeline.save_debug_image(
                    base_name, "00_original", 0, img, debug_pict_dir
                )

            # ==================== ENHANCED DETECTION ====================
            print(f"\n🔍 Running ENHANCED detection...")

            # Apply enhanced preprocessing
            preprocessed = enhanced_preprocessing(img)

            if save_debug_images:
                pipeline.save_debug_image(
                    base_name, "01_enhanced_preproc", 1, preprocessed, debug_pict_dir
                )

            # Run multi-scale detection (which includes multi-pass if enabled)
            if use_multi_scale:
                detected_dict = multi_scale_detection(
                    preprocessed, scales=[0.75, 1.0, 1.25]
                )
            else:
                detected_dict = multi_pass_detection(preprocessed)

            print(f"✅ Enhanced detection found {len(detected_dict)} unique markers")

            # Print detection method breakdown
            if detected_dict:
                method_counts = {}
                for marker_id, (corners, method) in detected_dict.items():
                    method_counts[method] = method_counts.get(method, 0) + 1

                print(f"\n   Detection breakdown by method:")
                for method, count in sorted(method_counts.items()):
                    print(f"     - {method}: {count} marker(s)")

                print(f"\n   Detected marker IDs: {sorted(detected_dict.keys())}")
            else:
                print(f"   No markers detected")

            # Create annotated image with detected markers
            if detected_dict and save_debug_images:
                annotated_img = visualize_detections(
                    img, detected_dict, "Enhanced Detection"
                )
                pipeline.save_debug_image(
                    base_name,
                    "02_enhanced_detected",
                    2,
                    annotated_img,
                    debug_pict_dir,
                )

            # ==================== STANDARD DETECTION (for comparison) ====================
            if run_standard_comparison:
                print(f"\n🔍 Running STANDARD pipeline for comparison...")

                detected_markers, final_img, perspective_matrix, metadata = (
                    pipeline.process_image_for_aruco_detection(
                        img=img,
                        aruco_detector=aruco_detector,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                        newcameramtx=newcameramtx,
                        fixed_markers=FIXED_MARKERS,
                        playground_corners=PLAYGROUND_CORNERS,
                        use_unround=use_unround,
                        use_clahe=use_clahe,
                        use_thresholding=use_thresholding,
                        sharpen_alpha=sharpen_alpha,
                        sharpen_beta=sharpen_beta,
                        sharpen_gamma=sharpen_gamma,
                        use_mask_playground=mask_playground,
                        use_straighten_image=mask_playground,
                        save_debug_images=save_debug_images,
                        debug_dir=debug_pict_dir,
                        base_name=base_name + "_standard",
                        apply_contrast_boost=False,
                        contrast_alpha=1.3,
                    )
                )

                standard_ids = {m[0].aruco_id for m in detected_markers}
                enhanced_ids = set(detected_dict.keys())

                print(f"✅ Standard pipeline found {len(standard_ids)} unique markers")
                if standard_ids:
                    print(f"   Detected marker IDs: {sorted(standard_ids)}")

                # ==================== COMPARISON ====================
                only_enhanced = enhanced_ids - standard_ids
                only_standard = standard_ids - enhanced_ids
                common = enhanced_ids & standard_ids

                print(f"\n📈 DETECTION COMPARISON:")
                print(f"{'─'*60}")
                print(
                    f"   Enhanced only:  {len(only_enhanced):2d} markers  {sorted(only_enhanced)}"
                )
                print(
                    f"   Standard only:  {len(only_standard):2d} markers  {sorted(only_standard)}"
                )
                print(f"   Both methods:   {len(common):2d} markers  {sorted(common)}")
                print(
                    f"   Total unique:   {len(enhanced_ids | standard_ids):2d} markers"
                )
                print(f"{'─'*60}")

                improvement = len(only_enhanced) - len(only_standard)
                if improvement > 0:
                    print(f"   ✅ Enhanced found {improvement} more marker(s)")
                elif improvement < 0:
                    print(f"   ⚠️  Standard found {-improvement} more marker(s)")
                else:
                    print(f"   ℹ️  Both methods found the same markers")

                # Update statistics
                total_enhanced += len(enhanced_ids)
                total_standard += len(standard_ids)
                total_enhanced_only += len(only_enhanced)
                total_standard_only += len(only_standard)

        except Exception as e:
            print(f"❌ Error processing {img_file.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # ==================== FINAL STATISTICS ====================
    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS ACROSS ALL IMAGES")
    print(f"{'='*80}")
    print(f"Processed {len(image_files)} image(s)")
    print(f"\nTotal detections:")
    print(f"  Enhanced: {total_enhanced} markers")
    print(f"  Standard: {total_standard} markers")
    print(f"\nUnique detections:")
    print(f"  Enhanced only: {total_enhanced_only} markers")
    print(f"  Standard only: {total_standard_only} markers")

    if total_standard > 0:
        improvement_pct = ((total_enhanced - total_standard) / total_standard) * 100
        print(f"\nOverall improvement: {improvement_pct:+.1f}%")

    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"Debug images saved to: {debug_pict_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
