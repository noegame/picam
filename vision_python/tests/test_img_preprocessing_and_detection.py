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
input_pict_dir = config.get_camera_directory() / "2026-01-09-playground-ready"
debug_pict_dir = config.get_debug_directory() / "2026-01-09-playground-ready"

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

# Get caméra parameters from config
camera_params = config.get_camera_params()

# Get image processing parameters from config
img_params = config.get_image_processing_params()

# extract image processing parameters
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

# extract camera parameters
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
# ArUco detection preparation
# ---------------------------------------------------------------------------

# Initialize ArUco detector with parameters from config
aruco_detector = detect_aruco.init_aruco_detector()


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
    print("\nOptions:")
    print(f"  - Mask playground area: {mask_playground}")
    print(f"  - Save debug images: {save_debug_images}")
    print("\nDetection Parameters:")
    print(f"  - Adaptive Threshold Constant: {adaptive_thresh_constant}")
    print(f"  - Min Marker Perimeter Rate: {min_marker_perimeter_rate}")
    print(f"  - Max Marker Perimeter Rate: {max_marker_perimeter_rate}")
    print(f"  - Polygon Approx Accuracy Rate: {polygonal_approx_accuracy_rate}")
    print(f"\nSharpening Parameters:")
    print(f"  - Alpha (contrast): {sharpen_alpha}")
    print(f"  - Beta (sharpness): {sharpen_beta}")
    print(f"  - Gamma (brightness): {sharpen_gamma}")
    print(f"{'='*80}")

    # Load camera calibration matrices
    camera_matrix, dist_coeffs, newcameramtx, roi = (
        config.get_camera_calibration_matrices()
    )

    # ---------------------------------------------------------------------------
    # Process each image
    # ---------------------------------------------------------------------------

    for img_file in image_files:
        print(f"\n{'='*80}")
        print(f"Processing: {img_file.name}")
        print(f"{'='*80}")

        # Get base filename without extension for output files
        base_name = img_file.stem

        try:
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                raise ValueError(f"Failed to load image from {img_file}")

            # Save original for debugging
            if save_debug_images:
                pipeline.save_debug_image(base_name, "original", 0, img, debug_pict_dir)

            # Use centralized processing pipeline
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
                    use_mask_playground=True,
                    use_straighten_image=True,  # Straighten if masking
                    save_debug_images=True,
                    debug_dir=debug_pict_dir,
                    base_name=base_name,
                    apply_contrast_boost=False,
                    contrast_alpha=1.3,
                )
            )

            # Print processing metadata
            if metadata["masking_applied"]:
                print(f"✅ Playground masking applied")
            if metadata["straightening_applied"]:
                print(f"✅ Image straightening applied")
            if metadata["perspective_transform_computed"]:
                print(
                    f"✅ Perspective transform computed, real-world coordinates calculated"
                )
            else:
                print(f"⚠️  Warning: Not all fixed ArUco markers detected")
                print("   Real world coordinates not calculated")

            # Print detected tags information
            if resume:
                pipeline.print_detection_summary(detected_markers)
            else:
                pipeline.print_detailed_markers(detected_markers)

        except ValueError as e:
            print(f"❌ Error processing {img_file.name}: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"Processing complete! Processed {len(image_files)} image(s)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
