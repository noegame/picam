"""
For images in a folder, perform:
- Detection of ArUco tags (coordinates in image space)
- Pose estimation (coordinates in real-world space)
- Annotation of images with detection results
- Save annotated images to output folder
"""

import cv2 as cv
import numpy as np
import time as t
import os
import glob

def get_camera_matrix() -> np.ndarray:
    """
    Returns the camera intrinsic matrix.
    Returns:
        np.ndarray: Camera intrinsic matrix
    """
    camera_matrix = np.array(
        [
            [2.49362477e03, 0.00000000e00, 1.97718701e03],
            [0.00000000e00, 2.49311358e03, 2.03491176e03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    return camera_matrix


def get_distortion_matrix() -> np.ndarray:
    """
    Returns the fisheye distortion coefficients.
    Returns:
        np.ndarray: Fisheye distortion coefficients
    """
    dist_matrix = np.array([[-0.1203345, 0.06802544, -0.13779641, 0.08243704]])

    return dist_matrix


def get_aruco_detector() -> cv.aruco.ArucoDetector:
    """
    Create and return an ArUco detector with specific parameters.
    Returns:
        cv.aruco.ArucoDetector: ArUco detector instance
    """
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    params = cv.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.minDistanceToBorder = 0
    params.minOtsuStdDev = 2.0
    params.perspectiveRemoveIgnoredMarginPerCell = 0.15
    detector = cv.aruco.ArucoDetector(aruco_dict, params)
    return detector


def get_output_folder() -> str:
    """
    Create and return the output folder path for debug images.

    Returns:
        str: output folder path
    """
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_folder = os.path.join("pictures/debug", script_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def get_input_folder() -> str:
    """
    Returns the input folder path containing images to process.

    Returns:
        str: input folder path
    """
    input_folder = "pictures/2026-01-16-playground-ready"
    return input_folder


def print_detection_stats(
    ids: np.ndarray, img_coords: np.ndarray, real_coords=None, expected_coords=None
):
    """
    Print statistics about the detected markers.

    Args:
        ids: Array of marker IDs (Nx1)
        img_coords: List of marker center coordinates in the image coordinate system (Nx2)
        real_coords: Optional list of real-world coordinates system (Nx3)
        expected_coords: Optional dictionary of expected positions by ID in the real-world coordinate system
    """

    if ids is None or len(ids) == 0:
        print("No valid tags detected")
        return

    # Prepare data for sorted display
    detections = []
    for i, marker_id in enumerate(ids):
        mid = marker_id[0] if isinstance(marker_id, np.ndarray) else marker_id
        detection = {
            "id": mid,
            "center": img_coords[i],
            "real_coords": real_coords[i] if real_coords else None,
            "expected_pos": None,
            "error_x": None,
            "error_y": None,
        }

        # Calculate errors if expected positions provided
        if real_coords and expected_coords and mid in expected_coords:
            closest = find_closest_expected_position(
                mid, (real_coords[i][0], real_coords[i][1])
            )
            if closest:
                z_exp = real_coords[i][2]  # Use same Z as detected
                detection["expected_pos"] = (closest[0], closest[1], z_exp)
                detection["error_x"] = real_coords[i][0] - closest[0]
                detection["error_y"] = real_coords[i][1] - closest[1]

        detections.append(detection)

    # Print header
    if real_coords is not None:
        print(
            f"\n{'#':<6}{'ID':<6}{'Image Position':<20}{'Detected (mm)':<25}{'Expected (mm)':<25}{'Error (mm)':<20}"
        )
        print("-" * 102)
    else:
        print(f"\n{'#':<6}{'ID':<10}{'Center Position':<25}")
        print("-" * 41)

    # Sort by ID then by position
    detections.sort(key=lambda x: (x["id"], x["center"][0], x["center"][1]))

    # Print each detection
    for idx, detection in enumerate(detections, 1):
        marker_id = detection["id"]
        center_x = int(detection["center"][0])
        center_y = int(detection["center"][1])

        if real_coords is not None and detection["real_coords"] is not None:
            real_x = int(detection["real_coords"][0])
            real_y = int(detection["real_coords"][1])
            real_z = int(detection["real_coords"][2])

            if detection["expected_pos"] is not None:
                exp_x = int(detection["expected_pos"][0])
                exp_y = int(detection["expected_pos"][1])
                exp_z = int(detection["expected_pos"][2])
                err_x = int(detection["error_x"])
                err_y = int(detection["error_y"])
                print(
                    f"{idx:<6}{marker_id:<6}({center_x},{center_y}){'':>6}({real_x},{real_y},{real_z}){'':>6}({exp_x},{exp_y},{exp_z}){'':>6}(Δx:{err_x:+4}, Δy:{err_y:+4})"
                )
            else:
                print(
                    f"{idx:<6}{marker_id:<6}({center_x},{center_y}){'':>6}({real_x},{real_y},{real_z}){'':>6}(N/A){'':>12}(N/A)"
                )
        else:
            print(f"{idx:<6}{marker_id:<10}({center_x}, {center_y})")


def print_dict(stats: dict):
    """
    Displays timing statistics for each processing step.

    Args:
        stats (dict): Timing statistics
    """
    width = 20
    print("\nTiming statistics per step:")

    total_time = 0.0
    for key, value in stats.items():
        # Skip non-timing entries (counters)
        if key in ["rejected_count", "valid_count"]:
            continue
        word_witdh = len(key)
        print(f"{key}{" "* (width-word_witdh)}{value:2f}")
        total_time += value

    # Print total_time at the end
    key = "total_time"
    word_witdh = len(key)
    print(f"{key}{" "* (width-width)}{total_time:.2f}")


def print_tab(tab: list, space: int):
    """
    Displays tab with same space.

    Args:
        tab (list): Values to display
    """
    string = ""
    for word in tab:
        word_witdh = len(word)
        string += f"{word}{" "* (space - word_witdh)}"
    return string


def annotate_img_with_ids(
    img: np.ndarray,
    centers: list[tuple[float, float]],
    ids: list[int],
) -> np.ndarray:
    """
    Add marker IDs annotations to the image.

    Args:
        img: Input image (BGR)
        centers: List of center coordinates (tuples)
        ids: List of marker IDs

    Returns:
        img: Annotated image
    """

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for i, tag in enumerate(centers):
        center_x = int(tag[0])
        center_y = int(tag[1])

        # Display tag ID
        text = f"ID:{ids[i]}"
        pos = (center_x, center_y)
        # Black outline then green text
        cv.putText(img, text, pos, font, font_scale, (0, 0, 0), thickness + 2)
        cv.putText(img, text, pos, font, font_scale, (0, 255, 0), thickness)

    return img


def annotate_img_with_centers(
    img: np.ndarray,
    centers: list[tuple[float, float]],
) -> np.ndarray:
    """
    Add center coordinates annotations to the image.

    Args:
        img: Input image (BGR)
        centers: List of center coordinates (tuples)

    Returns:
        img: Annotated image
    """

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for i, tag in enumerate(centers):
        center_x = int(tag[0])
        center_y = int(tag[1])

        # Display center coordinates
        text = f"({center_x},{center_y})"
        pos = (center_x, center_y - 20)
        # Black outline then blue text
        cv.putText(img, text, pos, font, font_scale, (0, 0, 0), thickness + 2)
        cv.putText(img, text, pos, font, font_scale, (255, 0, 0), thickness)

    return img


def annotate_img_with_counter(
    img: np.ndarray,
    count: int,
) -> np.ndarray:
    """
    Add counter annotation to the image.

    Args:
        img: Input image (BGR)
        count: Number of detected tags

    Returns:
        img: Annotated image
    """

    counter_text = f"Tags detectes: {count}"
    counter_pos = (50, 100)
    cv.putText(
        img,
        counter_text,
        counter_pos,
        cv.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 0),
        8,
    )
    cv.putText(
        img,
        counter_text,
        counter_pos,
        cv.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 255, 0),
        5,
    )

    return img


def annotate_img_with_real_coords(
    img: np.ndarray,
    centers: list[tuple[float, float]],
    real_coords: list[tuple[float, float, float]],
) -> np.ndarray:
    """
    Add real-world coordinates annotations to the image.

    Args:
        img: Input image (BGR)
        centers: List of center coordinates (tuples)
        real_coords: List of real-world coordinates (tuples)
    Returns:
        img: Annotated image
    """

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    for i, tag in enumerate(centers):
        center_x = int(tag[0])
        center_y = int(tag[1])

        # Display real coordinates
        real_x = int(real_coords[i][0])
        real_y = int(real_coords[i][1])
        real_z = int(real_coords[i][2])
        real_text = f"({real_x},{real_y},{real_z})mm"
        real_pos = (center_x + 50, center_y)

        # Black outline then cyan text
        cv.putText(
            img,
            real_text,
            real_pos,
            font,
            font_scale * 0.8,
            (0, 0, 0),
            thickness + 2,
        )
        cv.putText(
            img,
            real_text,
            real_pos,
            font,
            font_scale * 0.8,
            (255, 255, 0),
            thickness,
        )

    return img


def annotate_img_with_expected_pos(
    img: np.ndarray,
    centers: list[tuple[float, float]],
    expected_positions: list[tuple[float, float, float]],
    errors: list[tuple[float, float]],
) -> np.ndarray:
    """
    Add expected positions annotations to the image.

    Args:
        img: Input image (BGR)
        centers: List of center coordinates (tuples)
        expected_positions: List of expected positions (tuples)
        errors: List of errors (tuples)

    Returns:
        img: Annotated image
    """

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for i, tag in enumerate(centers):
        center_x = int(tag[0])
        center_y = int(tag[1])

        # Display expected position and error
        exp_x = int(expected_positions[i][0])
        exp_y = int(expected_positions[i][1])
        exp_z = int(expected_positions[i][2])
        err_x = int(errors[i][0])
        err_y = int(errors[i][1])

        # Expected position text
        exp_text = f"Att: ({exp_x},{exp_y},{exp_z})mm"
        exp_pos = (center_x + 50, center_y + 20)
        cv.putText(
            img,
            exp_text,
            exp_pos,
            font,
            font_scale * 0.8,
            (0, 0, 0),
            thickness + 2,
        )
        cv.putText(
            img,
            exp_text,
            exp_pos,
            font,
            font_scale * 0.8,
            (255, 0, 255),  # Magenta for expected position
            thickness,
        )

    return img


def apply_pose_correction(x_detected, y_detected):
    """
    Applique des corrections empiriques pour améliorer la précision de localisation.

    Basé sur l'analyse des erreurs systématiques observées:
    - Erreur X au centre (1000-2000mm): -71mm
    - Erreur Y en haut (>2400mm): +30mm
    - Erreur Y en bas (<400mm): +15mm

    Amélioration attendue:
    - Réduction d'erreur RMSE totale: ~30% (de 54mm à 38mm)
    - Réduction MAE X: ~46% (de 36mm à 20mm)
    - Réduction MAE Y: ~49% (de 22mm à 11mm)

    Args:
        x_detected: Position X détectée en mm (coordonnées terrain)
        y_detected: Position Y détectée en mm (coordonnées terrain)

    Returns:
        tuple: (x_corrected, y_corrected) en mm
    """
    x_corrected = x_detected
    y_corrected = y_detected

    # Corrections Y (axe vertical du terrain)
    if y_detected < 400:  # Zone basse du terrain
        y_corrected = y_detected - 15
    elif 400 <= y_detected < 900:  # Zone milieu-basse
        y_corrected = y_detected + 14
    elif 900 <= y_detected < 1600:  # Zone milieu
        y_corrected = y_detected + 16
    elif 1600 <= y_detected < 2400:  # Zone milieu-haute
        y_corrected = y_detected + 5
    else:  # y_detected >= 2400, Zone haute du terrain
        y_corrected = y_detected + 30

    # Corrections X (axe horizontal du terrain)
    if x_detected < 500:  # Zone gauche
        x_corrected = x_detected - 8
    elif 500 <= x_detected < 1000:  # Zone centre-gauche
        x_corrected = x_detected + 2
    elif 1000 <= x_detected < 2000:  # Zone centre (ERREUR CRITIQUE: -71mm)
        x_corrected = x_detected - 71
    else:  # x_detected >= 2000, Zone droite
        x_corrected = x_detected - 8

    return x_corrected, y_corrected


def find_mask(
    detector: cv.aruco.ArucoDetector,
    img: np.ndarray,
    scale_y=1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a binary mask corresponding to the field.
    Allows extending or reducing mask height via scale_y.

    - scale_y > 1.0 : taller mask
    - scale_y < 1.0 : shorter mask

    Args:
        detector (cv.aruco.ArucoDetector): ArUco detector instance
        img (np.ndarray): Input image (BGR)
        scale_y (float): Vertical scaling factor for mask height

    Returns:
        np.ndarray: mask
    """

    # Load reference image

    if img is None:
        raise RuntimeError(f"Unable to load image")

    h, w = img.shape[:2]

    # Detection of fixed tags
    corners, ids, rejected, timings = detect_markers(detector, img, mask=None)

    if corners is None:
        raise RuntimeError("No ArUco detection")

    # Known real-world positions of tags
    tag_irl = {
        20: (600, 600, 0),
        21: (600, 2400, 0),
        22: (1400, 600, 0),
        23: (1400, 2400, 0),
    }

    # Prepare points for homography calculation
    src_pts = []
    dst_pts = []

    # Collect corresponding points
    for i, marker_id in enumerate(ids):
        mid = marker_id[0]
        if mid in tag_irl:
            # Image points (center of detected tag)
            corner = corners[i].reshape(-1, 2)
            center_x = np.mean(corner[:, 0])
            center_y = np.mean(corner[:, 1])
            dst_pts.append((center_x, center_y))

            # Real-world points
            real_x, real_y, _ = tag_irl[mid]
            src_pts.append((real_x, real_y))

    if len(src_pts) != 4:
        raise RuntimeError(f"Fixed tags detected: {len(src_pts)}/4")

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Undistort image points before calculating homography
    dst_pts_reshaped = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    dst_pts_undistorted = cv.fisheye.undistortPoints(
        dst_pts_reshaped, camera_matrix, dist_matrix, None, camera_matrix
    )
    dst_pts = dst_pts_undistorted.reshape(-1, 2)

    # Homography real-world → image
    H, _ = cv.findHomography(src_pts, dst_pts)
    if H is None:
        raise RuntimeError("Homography not computable")

    # Calculate inverse homography (image → real-world)
    H_inv = np.linalg.inv(H)

    # Real-world corners of field
    terrain_irl = np.array(
        [
            [0, 0],
            [2000, 0],
            [2000, 3000],
            [0, 3000],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    # Image projection
    terrain_img = cv.perspectiveTransform(terrain_irl, H)
    terrain_img = terrain_img.reshape(-1, 2)

    # ----- VERTICAL EXTENSION OF MASK -----
    if scale_y != 1.0:
        center_y = np.mean(terrain_img[:, 1])
        terrain_img[:, 1] = center_y + (terrain_img[:, 1] - center_y) * scale_y

    # Clip to image bounds
    terrain_img[:, 0] = np.clip(terrain_img[:, 0], 0, w - 1)
    terrain_img[:, 1] = np.clip(terrain_img[:, 1], 0, h - 1)

    terrain_img = terrain_img.astype(np.int32)

    # Create mask (SAME SIZE AS IMAGE)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.fillPoly(mask, [terrain_img], 255)

    return mask, H_inv
    # return mask


def detect_markers(
    detector: cv.aruco.ArucoDetector, image: np.ndarray, mask: np.ndarray | None
) -> tuple:
    """
    Find the markers in the input image and return their corners coordinates and IDs.

    Args:
        detector: ArUco detector instance
        image: Input image (BGR)
        mask: Optional mask to restrict detection area
        scale: Scale factor for image resizing (default 1.5)
        sharpen: Whether to apply sharpening filter (default True)

    Returns:
        tuple: (corners, ids, rejected, timings)
            - corners: List of corner arrays (Nx4x2)
            - ids: Array of marker IDs (Nx1)
            - rejected: List of rejected marker candidates
            - timings: Dictionary with timing information
    """
    timings = {}
    scale = 1.5
    sharpen = True

    # Apply mask if provided
    if mask is not None:
        t0 = t.time()
        image = cv.bitwise_and(image, image, mask=mask)
        timings["mask_application"] = t.time() - t0

    # Preprocessing: Sharpness enhancement
    if sharpen:
        t0 = t.time()
        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image = cv.filter2D(image, -1, kernel_sharpening)
        timings["sharpening"] = t.time() - t0

    # Resize if scale != 1.0
    if scale != 1.0:
        t0 = t.time()
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv.resize(image, (width, height))
        timings["resizing"] = t.time() - t0

    # Detection
    t0 = t.time()
    corners, ids, rejected = detector.detectMarkers(image)
    timings["detection"] = t.time() - t0

    # Rescale corners back to original coordinates
    if scale != 1.0 and corners:
        corners = [corner / scale for corner in corners]

    return corners, ids, rejected, timings


def find_closest_expected_position(id, corners):
    """
    Finds the closest expected position to a detected position.
    Returns (expected_x, expected_y, distance) or None if no expected position.

    Args:
        id: Marker ID
        corners: Detected position of the marker (x, y)

    Returns:
        tuple: (x, y, d) the closest expected position and distance between the detected position and the expected position, or None
    """
    if id not in EXPECTED_POSITIONS:
        return None

    expected_positions = EXPECTED_POSITIONS[id]
    min_distance = float("inf")
    closest_pos = None

    for exp_x, exp_y, exp_z in expected_positions:
        distance = np.sqrt((corners[0] - exp_x) ** 2 + (corners[1] - exp_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_pos = (exp_x, exp_y)

    return (*closest_pos, min_distance) if closest_pos else None


def find_center_coord(corners):
    """
    Find the center coordinates of each detected marker from their corners coordinates.

    Args:
        corners: Single marker corners array (4x2) or list of corners arrays

    Returns:
        tuple: (center_x, center_y) for single marker or list of tuples for multiple markers
    """
    if isinstance(corners, list):
        # Multiple markers
        centers = []
        for corner in corners:
            corner_array = corner[0] if len(corner.shape) == 3 else corner
            center_x = np.mean(corner_array[:, 0])
            center_y = np.mean(corner_array[:, 1])
            centers.append((center_x, center_y))
        return centers
    else:
        # Single marker
        corner_array = corners[0] if len(corners.shape) == 3 else corners
        center_x = np.mean(corner_array[:, 0])
        center_y = np.mean(corner_array[:, 1])
        return (center_x, center_y)


def pose_estimation_homography(points, homography_inv, K, D, z_values=None):
    """
    Convert the coordinates from the image coordinate system to the terrain coordinate system.

    Args:
        points: Array of points in image coordinates (Nx2) or list of tuples
        homography_inv: Inverse homography matrix (image -> real world)
        K: Camera intrinsic matrix
        D: Fisheye distortion coefficients
        z_values: Optional Z coordinates for each point (default: 30mm for all)

    Returns:
        list: List of (x, y, z) tuples in real-world coordinates (mm)
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points, dtype=np.float32)

    if z_values is None:
        z_values = [30] * len(points)  # Default 30mm height

    real_coords = []
    for i, point in enumerate(points):
        # Reshape for undistortion
        point_distorted = np.array([[[point[0], point[1]]]], dtype=np.float32)

        # Undistort point before transforming it
        point_undistorted = cv.fisheye.undistortPoints(point_distorted, K, D, None, K)

        # Apply homography transformation
        point_real = cv.perspectiveTransform(point_undistorted, homography_inv)

        # Extract coordinates and add Z
        x = point_real[0][0][0]
        y = point_real[0][0][1]
        z = z_values[i]

        real_coords.append((x, y, z))

    return real_coords


def pose_estimation_pnp(corners, ids, K, D, marker_size=100):
    """
    Estimate the pose of each detected marker using solvePnP.

    Args:
        corners: List of corners arrays (Nx4x2)
        ids: Array of marker IDs (Nx1)
        K: Camera intrinsic matrix
        D: Fisheye distortion coefficients
        marker_size: Size of the marker in mm (default 100mm)

    Returns:
        dict: Dictionary mapping marker ID to (rvec, tvec)
    """
    pose_dict = {}

    # Define 3D points of the marker corners in its local coordinate system
    half_size = marker_size / 2.0
    obj_points = np.array(
        [
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0],
        ],
        dtype=np.float32,
    )

    for i, corner in enumerate(corners):
        marker_id = ids[i][0]
        img_points = corner.reshape(-1, 2).astype(np.float32)

        # Solve PnP
        success, rvec, tvec = cv.solvePnP(
            obj_points, img_points, K, D, flags=cv.SOLVEPNP_IPPE_SQUARE
        )

        if success:
            pose_dict[marker_id] = (rvec, tvec)

    return pose_dict


def process_img(
    img: str,
    detector: cv.aruco.ArucoDetector,
    output_path: str,
    verbose: bool = True,
    mask: np.ndarray | None = None,
    homography_inv: np.ndarray | None = None,
):
    """
    Orchestrates the complete ArUco detection and pose estimation pipeline.
    Returns list of detected tags with their real coordinates if homography_inv is provided

    Args:
        img: Input image file path (string)
        detector: ArUco detector instance
        output_path: Path to save debug images
        verbose: Whether to print detailed statistics (default True)
        mask: Optional mask to restrict detection area
        homography_inv: Optional inverse homography matrix for localization
    """
    # Image statistics
    timings = {
        "loading": 0,
        "color_conversion": 0,
        "mask_application": 0,
        "sharpening": 0,
        "resizing": 0,
        "detection": 0,
        "pose estimation": 0,
        "annotation": 0,
        "saving": 0,
        "rejected_count": 0,
        "valid_count": 0,
    }

    t_start = t.time()
    ALLOWED_IDS = [20, 21, 22, 23, 41, 36, 47]

    # Load image (PNG or JPG)
    t0 = t.time()
    image = cv.imread(img, cv.IMREAD_UNCHANGED)
    timings["loading"] = t.time() - t0

    if image is None:
        print(f"Error: Unable to load image {img}")
        return

    # Convert to BGR if image has alpha channel
    t0 = t.time()
    if image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    timings["color_conversion"] = t.time() - t0

    # ========== STEP 1: DETECT MARKERS ==========
    corners, ids, _, detection_timings = detect_markers(detector, image, mask=mask)

    # Merge detection timings into main timings
    timings.update(detection_timings)

    # ========== STEP 2: FILTER ALLOWED IDS ==========
    all_detections = []
    valid_ids = []
    valid_corners = []

    if ids is not None:
        for i, marker_id in enumerate(ids):
            mid = marker_id[0]

            # FILTERING: Ignore unauthorized IDs
            if mid not in ALLOWED_IDS:
                timings["rejected_count"] += 1
                continue

            valid_ids.append(mid)
            valid_corners.append(corners[i][0])

    # ========== STEP 3: CALCULATE CENTERS ==========
    centers = []
    if valid_corners:
        centers = find_center_coord(valid_corners)

    # ========== STEP 4: POSE ESTIMATION (if homography provided) ==========
    t0 = t.time()
    real_coords = None
    if homography_inv is not None and centers:
        # Determine Z values for each tag
        z_values = []
        for mid in valid_ids:
            z_expected = 30  # Default 30mm (common height of tags)
            if mid in EXPECTED_POSITIONS:
                # Retrieve Z from initial data
                for entry in initial_position:
                    if mid in entry[0]:
                        z_expected = entry[4]  # Index 4 = Z
                        break
            z_values.append(z_expected)

        # Convert image coordinates to real-world coordinates
        real_coords_raw = pose_estimation_homography(
            centers, homography_inv, camera_matrix, dist_matrix, z_values
        )

        # Apply empirical corrections to improve accuracy
        real_coords = []
        for x_raw, y_raw, z in real_coords_raw:
            x_corrected, y_corrected = apply_pose_correction(x_raw, y_raw)
            real_coords.append((x_corrected, y_corrected, z))
    timings["localization"] = t.time() - t0

    # ========== STEP 5: BUILD DETECTION RESULTS ==========
    for i, mid in enumerate(valid_ids):
        detection = {
            "id": mid,
            "corners": valid_corners[i],
            "center": centers[i],
            "real_coords": real_coords[i] if real_coords else None,
            "expected_pos": None,
            "error_x": None,
            "error_y": None,
            "error_distance": None,
        }

        # Calculate errors if real coordinates available
        if real_coords:
            closest = find_closest_expected_position(
                mid, (real_coords[i][0], real_coords[i][1])
            )
            if closest:
                z_exp = real_coords[i][2]
                detection["expected_pos"] = (closest[0], closest[1], z_exp)
                detection["error_x"] = real_coords[i][0] - closest[0]
                detection["error_y"] = real_coords[i][1] - closest[1]
                detection["error_distance"] = closest[2]

        all_detections.append(detection)

    # Calculate total time before display
    timings["total_detection"] = t.time() - t_start

    # ========== STEP 6: ANNOTATE IMAGE ==========
    t0 = t.time()
    output_image = image.copy()

    if len(all_detections) > 0:
        # Extract data for annotation functions
        all_centers = [d["center"] for d in all_detections]
        all_ids = [d["id"] for d in all_detections]
        all_real_coords = [d["real_coords"] for d in all_detections if d["real_coords"]]
        all_expected_pos = [
            d["expected_pos"] for d in all_detections if d["expected_pos"]
        ]
        all_errors = [
            (d["error_x"], d["error_y"])
            for d in all_detections
            if d["error_x"] is not None
        ]
        # Centers with expected positions (for matching indices)
        centers_with_expected = [
            d["center"] for d in all_detections if d["expected_pos"]
        ]

        # Apply annotations using already defined functions
        output_image = annotate_img_with_counter(output_image, len(all_detections))
        output_image = annotate_img_with_ids(output_image, all_centers, all_ids)
        # output_image = annotate_img_with_centers(output_image, all_centers)

        if all_real_coords:
            output_image = annotate_img_with_real_coords(
                output_image, all_centers, all_real_coords
            )

        if all_expected_pos and all_errors:
            output_image = annotate_img_with_expected_pos(
                output_image, centers_with_expected, all_expected_pos, all_errors
            )

    timings["annotation"] = t.time() - t0

    # Save with optimized compression
    t0 = t.time()
    # Fast PNG compression (level 1 instead of default 3)
    cv.imwrite(output_path, output_image, [cv.IMWRITE_PNG_COMPRESSION, 1])
    timings["saving"] = t.time() - t0

    if verbose:
        print(f"\nAnnotated image saved: {output_path}")

    # ========== STEP 7: PRINT DETECTION STATISTICS ==========
    if verbose:
        print(f"\nUnauthorized IDs rejected: \t{timings['rejected_count']}")
        print(f"Valid tags detected: \t\t{len(all_detections)}")
        print(f"Detection duration: \t\t{timings['total_detection']:.2f} seconds")

        if len(all_detections) > 0:
            # Prepare data for print_detection_stats
            stats_ids = np.array([[d["id"]] for d in all_detections])
            stats_centers = [d["center"] for d in all_detections]
            stats_real_coords = (
                [d["real_coords"] for d in all_detections] if real_coords else None
            )

            print_detection_stats(
                stats_ids, stats_centers, stats_real_coords, EXPECTED_POSITIONS  # type: ignore
            )
        else:
            print("No valid tags detected")

        # Print timing breakdown
        print_dict(timings)


def process_folder(input_folder: str, output_folder: str):
    """
    Process every image in a folder and print statistics.

    Args:
        input_folder: Path to the input folder containing images
        output_folder: Path to the output folder to save annotated images
    """

    # Initialize timing for total script execution
    t_init = t.time()

    # Get all images (PNG and JPG)
    image_files = []
    for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        print(f"No images found in folder: {input_folder}")
        return

    image_files.sort()
    total_images = len(image_files)

    print("=" * 80)
    print(f"PROCESSING {total_images} IMAGES")
    print("=" * 80)

    # Get ArUco detector
    detector = get_aruco_detector()

    # Load first image to find mask
    first_image = cv.imread(image_files[0])

    if first_image is None:
        print(f"Error: Unable to load first image: {image_files[0]}")
        return

    # Find mask and calculate inverse homography
    mask, H_inv = find_mask(detector, first_image, scale_y=1.1)

    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_name)

        print("\n" + "=" * 80)
        print(f"[{idx}/{total_images}] Processing: {image_name}")

        process_img(
            image_path,
            detector,
            output_path,
            verbose=True,
            mask=mask,
            homography_inv=H_inv,
        )
