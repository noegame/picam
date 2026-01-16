#!/usr/bin/env python3

"""
processing_pipeline.py

Reusable image processing and ArUco detection functions for the vision pipeline.
Provides utilities for preprocessing, detection, annotation, and coordinate transformation.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from vision_python.src.aruco import aruco
from vision_python.src.img_processing import unround_img


# ---------------------------------------------------------------------------
# Image Loading and Conversion Functions
# ---------------------------------------------------------------------------


def load_and_convert_to_grayscale(img_path: str) -> np.ndarray:
    """Load an image and convert it to grayscale.

    Args:
        img_path: Path to the image file

    Returns:
        Grayscale image as numpy array

    Raises:
        ValueError: If image cannot be loaded
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Image Enhancement Functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ArUco Detection Functions
# ---------------------------------------------------------------------------


def detect_aruco_markers(img: np.ndarray, detector: cv2.aruco.ArucoDetector) -> tuple:
    """Detect ArUco markers in the image.

    Args:
        img: Input grayscale image
        detector: ArUco detector with configured parameters

    Returns:
        Tuple of (corners_list, ids, rejected)
    """
    return detector.detectMarkers(img)


def create_marker_objects(corners_list: list, ids: np.ndarray) -> List[Tuple]:
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


# ---------------------------------------------------------------------------
# Image Annotation Functions
# ---------------------------------------------------------------------------


def annotate_image_with_markers(
    img: np.ndarray, corners_list: list, detected_markers: List[Tuple]
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
            (255, 0, 0),
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
        corners = rejected_corners.reshape((4, 2))
        corners = corners.astype(int)

        # Draw polygon connecting all corners
        cv2.polylines(
            rejected_img,
            [corners],
            isClosed=True,
            color=(0, 0, 255),  # Red in BGR
            thickness=2,
        )

        # Draw corner points
        for corner in corners:
            cv2.circle(rejected_img, tuple(corner), 3, (0, 0, 255), -1)

    return rejected_img


# ---------------------------------------------------------------------------
# Perspective Transform Functions
# ---------------------------------------------------------------------------


def compute_perspective_transform(
    detected_markers: List[Tuple], fixed_markers: List[aruco.Aruco]
) -> Tuple[Optional[np.ndarray], bool]:
    """Compute perspective transformation matrix from fixed markers.

    Args:
        detected_markers: List of (marker, corners) tuples
        fixed_markers: List of fixed Aruco marker objects with real world coordinates

    Returns:
        Tuple of (perspective_matrix, found_all_markers)
        perspective_matrix is None if not all fixed markers are found
    """
    tags_from_img = [marker for marker, corners in detected_markers]

    # Extract fixed marker IDs and build mapping
    fixed_id_to_coords = {m.aruco_id: (m.x, m.y) for m in fixed_markers}
    fixed_ids = list(fixed_id_to_coords.keys())

    # Source points from detected markers
    src_points = []
    dst_points = []

    for fixed_id in sorted(fixed_ids):
        tag = aruco.find_aruco_by_id(tags_from_img, fixed_id)
        if tag is None:
            return None, False
        src_points.append([tag.x, tag.y])
        dst_points.append(list(fixed_id_to_coords[fixed_id]))

    if len(src_points) != len(fixed_markers):
        return None, False

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Compute perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return perspective_matrix, True


def transform_markers_to_real_world(
    detected_markers: List[Tuple], perspective_matrix: np.ndarray
) -> None:
    """Transform marker coordinates to real world coordinates.

    Args:
        detected_markers: List of (marker, corners) tuples (modified in place)
        perspective_matrix: Perspective transformation matrix
    """
    for tag, tag_corners in detected_markers:
        # Create homogeneous coordinate [x, y, 1]
        img_point = np.array([tag.x, tag.y, 1.0], dtype=np.float32).reshape(3, 1)

        # Apply perspective transformation
        real_world_point = perspective_matrix @ img_point

        # Normalize homogeneous coordinates
        real_x = real_world_point[0, 0] / real_world_point[2, 0]
        real_y = real_world_point[1, 0] / real_world_point[2, 0]

        # Update marker with real world coordinates
        tag.real_x = real_x
        tag.real_y = real_y
        tag.real_angle = tag.angle  # Angle remains the same


# ---------------------------------------------------------------------------
# Mask Functions
# ---------------------------------------------------------------------------


def mask_playground_area(
    img: np.ndarray,
    detected_markers: List[Tuple],
    playground_corners: List[Tuple[float, float]],
    fixed_markers: List[aruco.Aruco],
) -> Tuple[Optional[np.ndarray], bool, Optional[np.ndarray]]:
    """Create a mask that only keeps the playground area based on fixed markers.

    Args:
        img: Input image (grayscale or BGR)
        detected_markers: List of (marker, corners) tuples
        playground_corners: List of (x, y) tuples defining playground corners in real world
        fixed_markers: List of fixed Aruco marker objects

    Returns:
        Tuple of (masked_image, found_all_markers, mask)
        Returns (None, False, None) if not all fixed markers found
    """
    # Compute perspective transform
    perspective_matrix, found_all = compute_perspective_transform(
        detected_markers, fixed_markers
    )

    if not found_all:
        return None, False, None

    # Compute inverse transformation (real world -> image)
    inverse_matrix = np.linalg.inv(perspective_matrix)

    # Convert playground corners to numpy array
    playground_corners_real = np.array(
        [[[x, y]] for x, y in playground_corners], dtype=np.float32
    )

    # Transform playground corners to image coordinates
    playground_corners_img = cv2.perspectiveTransform(
        playground_corners_real, inverse_matrix
    )
    playground_corners_img = playground_corners_img.reshape(-1, 2).astype(np.int32)

    # Create mask with the same shape as the image
    if len(img.shape) == 2:
        # Grayscale image
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        # Color image
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Fill the playground area with white
    cv2.fillPoly(mask, [playground_corners_img], 255)

    # Apply mask to image
    if len(img.shape) == 2:
        # Grayscale image
        masked_img = cv2.bitwise_and(img, img, mask=mask)
    else:
        # Color image
        masked_img = cv2.bitwise_and(img, img, mask=mask)

    return masked_img, True, mask


# ---------------------------------------------------------------------------
# Display and Reporting Functions
# ---------------------------------------------------------------------------


def print_detection_summary(detected_markers: List[Tuple]) -> None:
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


def print_detailed_markers(detected_markers: List[Tuple]) -> None:
    """Print detailed information for each detected marker.

    Args:
        detected_markers: List of (marker, corners) tuples
    """
    tags_from_img = [marker for marker, corners in detected_markers]
    tags_from_img.sort(key=lambda tag: tag.aruco_id)

    for tag in tags_from_img:
        tag.print()


# ---------------------------------------------------------------------------
# Debug Image Saving Functions
# ---------------------------------------------------------------------------


def save_debug_image(
    base_name: str,
    comment: str,
    img_processing_step: int,
    img: np.ndarray,
    debug_dir: Path,
) -> int:
    """Save debug image with step numbering.

    Args:
        base_name: Base name for the file
        comment: Description of processing step
        img_processing_step: Current step number
        img: Image to save
        debug_dir: Directory to save debug images

    Returns:
        Next step number (incremented)
    """
    filename = build_filename(debug_dir, base_name, comment, img_processing_step)
    save(filename, img)
    return img_processing_step + 1


def build_filename(
    debug_dir: Path, base_name: str, comment: str, img_processing_step: int
) -> str:
    """Build filename for debug image.

    Args:
        debug_dir: Directory for debug images
        base_name: Base name for the file
        comment: Description of processing step
        img_processing_step: Current step number

    Returns:
        Full path as string
    """
    return f"{debug_dir}/{base_name}_{img_processing_step}_{comment}.jpg"


def save(filename: str, img: np.ndarray) -> None:
    """Save image to file.

    Args:
        filename: Full path to save image
        img: Image to save
    """
    cv2.imwrite(filename, img)


# ---------------------------------------------------------------------------
# High-level Processing Pipeline
# ---------------------------------------------------------------------------


def process_image_for_aruco_detection(
    img: np.ndarray,
    aruco_detector: cv2.aruco.ArucoDetector,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    newcameramtx: Optional[np.ndarray] = None,
    fixed_markers: Optional[List[aruco.Aruco]] = None,
    playground_corners: Optional[List[Tuple[float, float]]] = None,
    use_unround: bool = False,
    use_clahe: bool = False,
    use_thresholding: bool = False,
    use_sharpening: bool = False,
    sharpen_alpha: float = 1.5,
    sharpen_beta: float = -0.5,
    sharpen_gamma: float = 0.0,
    use_mask_playground: bool = False,
    use_straighten_image: bool = False,
    save_debug_images: bool = False,
    debug_dir: Optional[Path] = None,
    base_name: str = "image",
    apply_contrast_boost: bool = False,
    contrast_alpha: float = 1.1,
) -> Tuple[List[Tuple], np.ndarray, Optional[np.ndarray], dict]:
    """
    Complete pipeline for ArUco marker detection in an image.

    This function centralizes all preprocessing and detection steps:
    1. Grayscale conversion (if needed)
    2. Sharpening
    3. Unrounding (lens distortion correction)
    4. CLAHE (contrast enhancement)
    5. Thresholding
    6. First ArUco detection pass
    7. Playground masking (optional)
    8. Image straightening (optional)
    9. Contrast boost (optional)
    10. Second ArUco detection pass (if masking/straightening enabled)
    11. Perspective transformation to real-world coordinates

    Args:
        img: Input image (grayscale or BGR)
        aruco_detector: Configured ArUco detector
        camera_matrix: Camera calibration matrix (required if use_unround=True)
        dist_coeffs: Distortion coefficients (required if use_unround=True)
        newcameramtx: Optimized camera matrix (required if use_unround=True)
        fixed_markers: List of fixed markers for perspective transform
        playground_corners: Corners of playground in real world coordinates
        use_unround: Apply lens distortion correction
        use_clahe: Apply CLAHE contrast enhancement
        use_thresholding: Apply adaptive thresholding
        sharpen_alpha: Sharpening contrast parameter
        sharpen_beta: Sharpening beta parameter
        sharpen_gamma: Sharpening gamma parameter
        use_mask_playground: Mask area to playground only
        use_straighten_image: Apply perspective correction to straighten playground
        save_debug_images: Save intermediate processing steps
        debug_dir: Directory for debug images
        base_name: Base name for debug image files
        apply_contrast_boost: Apply final contrast boost before detection
        contrast_alpha: Contrast boost alpha parameter

    Returns:
        Tuple of:
        - detected_markers: List of (marker, corners) tuples
        - final_img: Final processed image used for detection
        - perspective_matrix: Matrix for real-world transformation (or None)
        - metadata: Dictionary with processing information
    """
    from vision_python.src.img_processing import straighten_img as straighten

    step = 0
    metadata = {
        "masking_applied": False,
        "straightening_applied": False,
        "all_fixed_markers_found": False,
        "perspective_transform_computed": False,
    }

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if save_debug_images and debug_dir:
            step = save_debug_image(base_name, "grayscale", step, img, debug_dir)

    # Apply sharpening
    if use_sharpening:
        img = apply_sharpening(img, sharpen_alpha, sharpen_beta, sharpen_gamma)
    if save_debug_images and debug_dir:
        step = save_debug_image(base_name, "sharpened", step, img, debug_dir)

    # Apply unrounding if enabled
    if use_unround:
        if camera_matrix is None or dist_coeffs is None or newcameramtx is None:
            raise ValueError("Camera calibration parameters required for unrounding")
        img = apply_unrounding(img, camera_matrix, dist_coeffs, newcameramtx)
        if save_debug_images and debug_dir:
            step = save_debug_image(base_name, "unrounded", step, img, debug_dir)

    # Apply CLAHE if enabled
    if use_clahe:
        img = apply_clahe(img)
        if save_debug_images and debug_dir:
            step = save_debug_image(base_name, "clahe", step, img, debug_dir)

    # Apply thresholding if enabled
    if use_thresholding:
        img = apply_thresholding(img)
        if save_debug_images and debug_dir:
            step = save_debug_image(base_name, "thresholded", step, img, debug_dir)

    # Apply contrast boost before first detection if requested
    if apply_contrast_boost:
        img = cv2.convertScaleAbs(img, alpha=contrast_alpha, beta=0)
        if save_debug_images and debug_dir:
            step = save_debug_image(base_name, "contrast_boost", step, img, debug_dir)

    # First ArUco detection pass
    corners_list, ids, rejected = detect_aruco_markers(img, aruco_detector)

    if ids is None or len(ids) == 0:
        detected_markers = []
    else:
        detected_markers = create_marker_objects(corners_list, ids)

    final_img = img

    # Mask playground area if enabled
    if use_mask_playground and playground_corners and fixed_markers:
        masked_img, found_all_fixed, mask = mask_playground_area(
            img, detected_markers, playground_corners, fixed_markers
        )

        if found_all_fixed:
            metadata["masking_applied"] = True
            metadata["all_fixed_markers_found"] = True

            if save_debug_images and debug_dir:
                step = save_debug_image(base_name, "mask", step, mask, debug_dir)
                step = save_debug_image(
                    base_name, "masked_playground", step, masked_img, debug_dir
                )

            # Straighten image if enabled
            if use_straighten_image:
                tags_from_img = [marker for marker, _ in detected_markers]
                marker_20 = aruco.find_aruco_by_id(tags_from_img, 20)
                marker_21 = aruco.find_aruco_by_id(tags_from_img, 21)
                marker_22 = aruco.find_aruco_by_id(tags_from_img, 22)
                marker_23 = aruco.find_aruco_by_id(tags_from_img, 23)

                if all([marker_20, marker_21, marker_22, marker_23]):
                    metadata["straightening_applied"] = True

                    # Source points from detected markers
                    src_points_straighten = np.array(
                        [
                            [marker_20.x, marker_20.y],
                            [marker_22.x, marker_22.y],
                            [marker_23.x, marker_23.y],
                            [marker_21.x, marker_21.y],
                        ],
                        dtype=np.float32,
                    )

                    # Destination points from fixed marker positions
                    # Scale factor for output image
                    scale_factor = 1.0
                    output_width = int(3000 * scale_factor)
                    output_height = int(2000 * scale_factor)

                    # Map to real positions
                    dst_points_straighten = np.array(
                        [
                            [
                                marker_20.aruco_id == 20 and 600 * scale_factor or 0,
                                marker_20.aruco_id == 20 and 600 * scale_factor or 0,
                            ],
                            [
                                marker_22.aruco_id == 22 and 600 * scale_factor or 0,
                                marker_22.aruco_id == 22 and 1400 * scale_factor or 0,
                            ],
                            [
                                marker_23.aruco_id == 23 and 2400 * scale_factor or 0,
                                marker_23.aruco_id == 23 and 1400 * scale_factor or 0,
                            ],
                            [
                                marker_21.aruco_id == 21 and 2400 * scale_factor or 0,
                                marker_21.aruco_id == 21 and 600 * scale_factor or 0,
                            ],
                        ],
                        dtype=np.float32,
                    )

                    # Use fixed marker coordinates from the list
                    if fixed_markers:
                        fixed_dict = {m.aruco_id: m for m in fixed_markers}
                        if all(id in fixed_dict for id in [20, 21, 22, 23]):
                            dst_points_straighten = np.array(
                                [
                                    [
                                        fixed_dict[20].x * scale_factor,
                                        fixed_dict[20].y * scale_factor,
                                    ],
                                    [
                                        fixed_dict[22].x * scale_factor,
                                        fixed_dict[22].y * scale_factor,
                                    ],
                                    [
                                        fixed_dict[23].x * scale_factor,
                                        fixed_dict[23].y * scale_factor,
                                    ],
                                    [
                                        fixed_dict[21].x * scale_factor,
                                        fixed_dict[21].y * scale_factor,
                                    ],
                                ],
                                dtype=np.float32,
                            )

                    straightened_img = straighten.straighten_image(
                        masked_img,
                        src_points_straighten,
                        dst_points_straighten,
                        output_width,
                        output_height,
                    )

                    if save_debug_images and debug_dir:
                        step = save_debug_image(
                            base_name, "straightened", step, straightened_img, debug_dir
                        )

                    # Apply contrast boost to straightened image if requested
                    if apply_contrast_boost:
                        straightened_img = cv2.convertScaleAbs(
                            straightened_img, alpha=contrast_alpha, beta=0
                        )
                        if save_debug_images and debug_dir:
                            step = save_debug_image(
                                base_name,
                                "straightened_contrast_boost",
                                step,
                                straightened_img,
                                debug_dir,
                            )

                    # Second detection pass on straightened image
                    corners_list, ids, rejected = detect_aruco_markers(
                        straightened_img, aruco_detector
                    )

                    if ids is None or len(ids) == 0:
                        detected_markers = []
                    else:
                        detected_markers = create_marker_objects(corners_list, ids)

                    final_img = straightened_img
            else:
                final_img = masked_img

    # Compute perspective transformation if fixed markers are provided
    perspective_matrix = None
    if fixed_markers and len(detected_markers) > 0:
        perspective_matrix, found_all = compute_perspective_transform(
            detected_markers, fixed_markers
        )

        if found_all:
            metadata["perspective_transform_computed"] = True
            # Transform markers to real world coordinates
            transform_markers_to_real_world(detected_markers, perspective_matrix)

    # Save annotated image if debug is enabled
    if save_debug_images and debug_dir and len(detected_markers) > 0:
        annotated_img = annotate_image_with_markers(
            final_img, corners_list, detected_markers
        )
        step = save_debug_image(base_name, "annotated", step, annotated_img, debug_dir)

        # Save rejected markers if any
        if rejected is not None and len(rejected) > 0:
            rejected_img = annotate_image_with_rejected_markers(final_img, rejected)
            step = save_debug_image(
                base_name, "rejected", step, rejected_img, debug_dir
            )

    return detected_markers, final_img, perspective_matrix, metadata
