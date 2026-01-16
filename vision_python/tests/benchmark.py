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
from vision_python.src.img_processing import straighten_img
from vision_python.src.img_processing import processing_pipeline as pipeline
from vision_python.src.img_processing import detect_aruco


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

rejected_marker = False  # Draw rejected markers on debug images
resume = True  # Print summary of detections instead of detailed info
save_debug_images = False  # Save debug images at each processing step
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
        print(f"\n{'='*80}")
        print(f"Processing: {img_file.name}")
        print(f"{'='*80}")

        # Get base filename without extension for output files
        base_name = img_file.stem

        # Start timing
        start_time = time.time()

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
                    use_mask_playground=mask_playground,
                    use_straighten_image=mask_playground,  # Straighten if masking
                    save_debug_images=save_debug_images,
                    debug_dir=debug_pict_dir,
                    base_name=base_name,
                    apply_contrast_boost=False,
                    contrast_alpha=1.1,
                )
            )

            # Print processing metadata
            if metadata["perspective_transform_computed"]:
                print(f"✅ Real-world coordinates calculated")
            else:
                print(f"⚠️  Warning: Not all fixed ArUco markers detected")

            # Print detected tags information
            if resume:
                pipeline.print_detection_summary(detected_markers)
            else:
                pipeline.print_detailed_markers(detected_markers)

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
