"""
Detection of ArUco tags
Localization of ArUcO tags in real-world coordinates
"""

import cv2
import numpy as np
import time as t
import os
import glob
import aruco_initial_positions


TAG_SIZE = 40.0


def find_closest_expected_position(tag_id, detected_pos):
    """
    Finds the closest expected position to a detected position.
    Returns (expected_x, expected_y, distance) or None if no expected position.
    """
    if tag_id not in EXPECTED_POSITIONS:
        return None

    expected_positions = EXPECTED_POSITIONS[tag_id]
    min_distance = float("inf")
    closest_pos = None

    for exp_x, exp_y in expected_positions:
        distance = np.sqrt(
            (detected_pos[0] - exp_x) ** 2 + (detected_pos[1] - exp_y) ** 2
        )
        if distance < min_distance:
            min_distance = distance
            closest_pos = (exp_x, exp_y)

    return (*closest_pos, min_distance) if closest_pos else None


def detect_aruco_tags(
    image_path,
    detector,
    output_path="output.png",
    verbose=True,
    mask=None,
    homography_inv=None,
):
    """
    Optimized detection with unique configuration: 1.25 scale × Sharpened × Aggressive small tags
    Filters only allowed IDs: 20, 21, 22, 23, 41, 36, 47
    Returns list of detected tags with their real coordinates if homography_inv is provided
    """

    t_start = t.time()  # Time for processing one image
    timings = {}  # Dictionary to store time for each step
    rejected_count = 0  # Counter for rejected IDs
    ALLOWED_IDS = [20, 21, 22, 23, 41, 36, 47]  # ID of existing tags

    # Load image (PNG or JPG)
    t0 = t.time()
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    timings["loading"] = t.time() - t0

    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Convert to BGR if image has alpha channel
    t0 = t.time()
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    timings["color_conversion"] = t.time() - t0

    # mask
    if mask is not None:
        t0 = t.time()
        # Resize mask to the size of the resized image
        image = cv2.bitwise_and(image, image, mask=mask)
        timings["mask_application"] = t.time() - t0

    # List to store all detected markers
    all_detections = []

    # Preprocessing: Sharpness enhancement
    t0 = t.time()
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    timings["sharpening"] = t.time() - t0

    # Resize to 1.5 scale
    t0 = t.time()
    scale = 1.5
    width = int(sharpened.shape[1] * scale)
    height = int(sharpened.shape[0] * scale)
    img_scaled = cv2.resize(sharpened, (width, height))
    timings["resizing"] = t.time() - t0

    # Detection
    t0 = t.time()
    corners, ids, rejected = detector.detectMarkers(img_scaled)
    timings["detection"] = t.time() - t0

    # Filter unauthorized IDs and calculate real coordinates
    t0 = t.time()
    if ids is not None:
        for i, marker_id in enumerate(ids):
            mid = marker_id[0]

            # FILTERING: Ignore unauthorized IDs
            if mid not in ALLOWED_IDS:
                rejected_count += 1
                continue

            corner = corners[i][0].copy()

            # Rescale corners
            corner = corner / scale

            # Calculate center
            center_x = np.mean(corner[:, 0])
            center_y = np.mean(corner[:, 1])

            # Calculate real coordinates if homography provided
            real_coords = None
            expected_pos = None
            error_x = None
            error_y = None
            error_distance = None

            tvec = estimate_tag_pose_solvepnp(corner, K, D)
            if tvec is not None:
                real_coords = (tvec[0], tvec[1], tvec[2])

                # Calculer l'erreur par rapport à la position attendue
                closest = find_closest_expected_position(mid, real_coords)
                if closest is not None:
                    exp_x, exp_y, distance = closest
                    expected_pos = (exp_x, exp_y, 0)  # Z=0 pour les positions au sol
                    error_x = real_coords[0] - exp_x
                    error_y = real_coords[1] - exp_y
                    error_distance = distance

            all_detections.append(
                {
                    "id": mid,
                    "corners": corner,
                    "center": (center_x, center_y),
                    "real_coords": real_coords,
                    "expected_pos": expected_pos,
                    "error_x": error_x,
                    "error_y": error_y,
                    "error_distance": error_distance,
                }
            )
    timings["localization"] = t.time() - t0

    # Calculate total time before display
    timings["total_detection"] = t.time() - t_start

    if verbose:
        print(f"\nUnauthorized IDs rejected: \t{rejected_count}")
        print(f"Valid tags detected: \t\t{len(all_detections)}")
        print(f"Detection duration: \t\t{timings['total_detection']:.2f} seconds")

    # Display results
    if verbose and len(all_detections) > 0:
        if homography_inv is not None:
            print(
                f"\n{'#':<6}{'ID':<6}{'Image Position':<20}{'Detected (mm)':<25}{'Expected (mm)':<25}{'Error (mm)':<20}"
            )
            print("-" * 102)
        else:
            print(f"\n{'#':<6}{'ID':<10}{'Center Position':<25}")
            print("-" * 41)
        # Sort by ID then by position
        all_detections.sort(key=lambda x: (x["id"], x["center"][0], x["center"][1]))
        for idx, detection in enumerate(all_detections, 1):
            marker_id = detection["id"]
            center_x = int(detection["center"][0])
            center_y = int(detection["center"][1])
            if homography_inv is not None and detection["real_coords"] is not None:
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
    elif verbose:
        print("No valid tags detected")

    # Create output image
    t0 = t.time()
    output_image = image.copy()

    if len(all_detections) > 0:
        # Draw all markers at once (more efficient)
        all_corners = [
            detection["corners"].reshape(1, 4, 2).astype(np.float32)
            for detection in all_detections
        ]
        all_ids = np.array(
            [[detection["id"]] for detection in all_detections], dtype=np.int32
        )
        # cv2.aruco.drawDetectedMarkers(output_image, all_corners, all_ids)

        # Use annotate_img function to add annotations
        output_image = annotate_img(
            output_image,
            all_detections,
            id=True,
            real_coords=True,
            expected_pos=True,
            counter=True,
        )

    timings["annotation"] = t.time() - t0

    # Save with optimized compression
    t0 = t.time()
    # Fast PNG compression (level 1 instead of default 3)
    cv2.imwrite(output_path, output_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    timings["saving"] = t.time() - t0

    if verbose:
        print(f"\nAnnotated image saved: {output_path}")

    # Return statistics
    return {
        "detections": all_detections,
        "rejected_count": rejected_count,
        "valid_count": len(all_detections),
        "detection_time": timings["total_detection"],
        "timings": timings,
    }


def annotate_img(
    img, detected_tags, id=True, real_coords=True, expected_pos=True, counter=True
):
    """Add annotations to the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # If user want to display IDs, real coordinates, and expected positions
    for detected_tag in detected_tags:
        center_x = int(detected_tag["center"][0])
        center_y = int(detected_tag["center"][1])

        # Display tag ID
        if id:
            text = f"ID:{detected_tag['id']}"
            pos = (center_x, center_y)
            # Black outline then green text
            cv2.putText(img, text, pos, font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(img, text, pos, font, font_scale, (0, 255, 0), thickness)

        # Display real coordinates if available
        if real_coords and detected_tag["real_coords"] is not None:
            real_x = int(detected_tag["real_coords"][0])
            real_y = int(detected_tag["real_coords"][1])
            real_z = int(detected_tag["real_coords"][2])
            real_text = f"({real_x},{real_y},{real_z})mm"
            real_pos = (center_x + 50, center_y)

            # Black outline then cyan text
            cv2.putText(
                img,
                real_text,
                real_pos,
                font,
                font_scale * 0.8,
                (0, 0, 0),
                thickness + 2,
            )
            cv2.putText(
                img,
                real_text,
                real_pos,
                font,
                font_scale * 0.8,
                (255, 255, 0),
                thickness,
            )

        # Display expected position and error
        if expected_pos and detected_tag["expected_pos"] is not None:
            exp_x = int(detected_tag["expected_pos"][0])
            exp_y = int(detected_tag["expected_pos"][1])
            exp_z = int(detected_tag["expected_pos"][2])
            err_x = int(detected_tag["error_x"])
            err_y = int(detected_tag["error_y"])

            # Expected position text
            exp_text = f"Att: ({exp_x},{exp_y},{exp_z})mm"
            exp_pos = (center_x + 50, center_y + 20)
            cv2.putText(
                img,
                exp_text,
                exp_pos,
                font,
                font_scale * 0.8,
                (0, 0, 0),
                thickness + 2,
            )
            cv2.putText(
                img,
                exp_text,
                exp_pos,
                font,
                font_scale * 0.8,
                (255, 0, 255),  # Magenta for expected position
                thickness,
            )

    # Add counter at the top of the image
    if counter:
        counter_text = f"Tags detectes: {len(detected_tags)}"
        counter_pos = (50, 100)
        cv2.putText(
            img,
            counter_text,
            counter_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 0),
            8,
        )
        cv2.putText(
            img,
            counter_text,
            counter_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            5,
        )

    return img


def process_folder(input_folder, output_folder="output"):
    """
    Process every image in a folder and print statistics.
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

    # Global statistics
    total_rejected = 0
    total_valid = 0
    total_time = 0

    # Time statistics per step
    cumulative_timings = {}

    # Localization error statistics
    all_errors_x = []
    all_errors_y = []
    all_errors_distance = []

    # Get ArUco detector
    detector = get_aruco_detector()

    # Find mask and calculate inverse homography
    mask, homography_inv = find_mask(detector, image_files[0], scale_y=1.1)

    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"output_{image_name}")

        print(f"\n[{idx}/{total_images}] Processing: {image_name}")

        # Measure total time including all overheads
        t_image_start = t.time()

        # Detect tags (pass detector as parameter)
        stats = detect_aruco_tags(
            image_path,
            detector,
            output_path,
            verbose=True,
            mask=mask,
            homography_inv=homography_inv,
        )

        t_image_end = t.time()
        real_image_time = t_image_end - t_image_start

        # Time statistics for this image
        print_img_stats(stats)

        # Accumulate statistics (use REAL measured time)
        total_rejected += stats["rejected_count"]
        total_valid += stats["valid_count"]
        total_time += real_image_time  # CORRECTION: use real measured time

        # Accumulate time per step
        for key, value in stats["timings"].items():
            cumulative_timings[key] = cumulative_timings.get(key, 0) + value

        # Collect localization errors
        for detection in stats["detections"]:
            if detection["error_x"] is not None:
                all_errors_x.append(detection["error_x"])
                all_errors_y.append(detection["error_y"])
                all_errors_distance.append(detection["error_distance"])

    # Display averages
    print("\n" + "=" * 80)
    print("GLOBAL STATISTICS")
    print("=" * 80)
    print(f"Total number of images processed: \t{total_images}")
    print(f"\nAverage unauthorized IDs rejected: \t{total_rejected / total_images:.2f}")
    print(f"Average valid tags detected: \t\t{total_valid / total_images:.2f}")
    print(f"\nTotal processing time: \t\t{total_time:.2f} seconds")
    print(f"Total script execution time: \t\t{t.time() - t_init:.2f} seconds")
    print(f"Average time per image: \t\t{total_time / total_images:.3f} seconds")

    print("\n" + "-" * 80)
    print("TIME ANALYSIS PER STEP (averages)")
    print("-" * 80)
    for key in [
        "loading",
        "color_conversion",
        "mask_application",
        "sharpening",
        "resizing",
        "detection",
        "localization",
        "annotation",
        "saving",
    ]:
        avg_time = cumulative_timings.get(key, 0) / total_images
        percentage = (cumulative_timings.get(key, 0) / total_time) * 100
        print(f"{key.capitalize():<25} {avg_time:.3f}s\t({percentage:.1f}%)")

    # Display localization error statistics
    if len(all_errors_x) > 0:
        print("\n" + "=" * 80)
        print("LOCALIZATION ERROR STATISTICS")
        print("=" * 80)
        print(f"Number of tags localized with expected position: {len(all_errors_x)}")
        print(f"\nMean error in X: \t\t{np.mean(all_errors_x):.2f} mm")
        print(f"Mean error in Y: \t\t{np.mean(all_errors_y):.2f} mm")
        print(f"Mean absolute error in X: \t{np.mean(np.abs(all_errors_x)):.2f} mm")
        print(f"Mean absolute error in Y: \t{np.mean(np.abs(all_errors_y)):.2f} mm")
        print(f"Mean Euclidean distance: \t{np.mean(all_errors_distance):.2f} mm")
        print(f"\nMax error in X: \t\t{np.max(np.abs(all_errors_x)):.2f} mm")
        print(f"Max error in Y: \t\t{np.max(np.abs(all_errors_y)):.2f} mm")
        print(f"Max distance: \t\t\t{np.max(all_errors_distance):.2f} mm")
        print(f"\nMin error in X: \t\t{np.min(np.abs(all_errors_x)):.2f} mm")
        print(f"Min error in Y: \t\t{np.min(np.abs(all_errors_y)):.2f} mm")
        print(f"Min distance: \t\t\t{np.min(all_errors_distance):.2f} mm")
        print(f"\nStandard deviation in X: \t{np.std(all_errors_x):.2f} mm")
        print(f"Standard deviation in Y: \t{np.std(all_errors_y):.2f} mm")
    else:
        print("\n" + "=" * 80)
        print("No expected position found for detected tags")

    print("=" * 80)


def print_summary_stats(stats):
    """Displays a summary of detection statistics for all images."""
    print("\nDetection statistics summary:")
    print(f"  → Unauthorized IDs rejected: \t{stats['rejected_count']}")
    print(f"  → Valid tags detected: \t{stats['valid_count']}")
    print(f"  → Detection duration: \t{stats['detection_time']:.3f} seconds")


def print_img_stats(stats):
    """Displays detection statistics for an individual image."""
    print("\nDetection statistics:")
    print(f"  → Unauthorized IDs rejected: \t{stats['rejected_count']}")
    print(f"  → Valid tags detected: \t{stats['valid_count']}")
    print(f"  → Detection duration: \t{stats['detection_time']:.3f} seconds")
    print(f"  → Step-by-step time details:")
    for step, duration in stats["timings"].items():
        print(f"     • {step}: \t{duration:.3f} seconds")


def print_localisation_stats():
    """Displays localization error statistics."""
    pass  # Implementation left empty for now


def find_mask(
    detector,
    img_path,
    scale_y=1.0,
):
    """
    Generates a binary mask corresponding to the field.
    Allows extending or reducing mask height via scale_y.

    - scale_y > 1.0 : taller mask
    - scale_y < 1.0 : shorter mask
    """

    # Load reference image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Unable to load {img_path}")

    h, w = img.shape[:2]

    # Detection of fixed tags
    result = detect_aruco_tags(
        img_path,
        detector,
        verbose=False,
        mask=None,
    )

    if result is None:
        raise RuntimeError("No ArUco detection")

    # Known real-world positions of tags
    tag_irl = {
        20: (600, 600, 0),
        21: (600, 2400, 0),
        22: (1400, 600, 0),
        23: (1400, 2400, 0),
    }

    src_pts = []
    dst_pts = []

    for det in result["detections"]:
        tid = det["id"]
        if tid in tag_irl:
            src_pts.append(tag_irl[tid])
            dst_pts.append(det["center"])

    if len(src_pts) != 4:
        raise RuntimeError(f"Fixed tags detected: {len(src_pts)}/4")

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Undistort image points before calculating homography
    dst_pts_reshaped = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    dst_pts_undistorted = cv2.fisheye.undistortPoints(dst_pts_reshaped, K, D, None, K)
    dst_pts = dst_pts_undistorted.reshape(-1, 2)

    # Homography real-world → image
    H, _ = cv2.findHomography(src_pts, dst_pts)
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
    terrain_img = cv2.perspectiveTransform(terrain_irl, H)
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
    cv2.fillPoly(mask, [terrain_img], 255)

    return mask, H_inv


def get_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.minDistanceToBorder = 0
    params.minOtsuStdDev = 2.0
    params.perspectiveRemoveIgnoredMarginPerCell = 0.15
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return detector


def get_output_folder():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_folder = os.path.join("output", script_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def estimate_tag_pose_solvepnp(corners_img, K, D):
    """
    Estime la position 3D réelle du tag dans le repère caméra
    à partir des coins image (fisheye).
    Returns: tuple (X, Y, Z) en mm ou None si échec
    """

    # Coins 3D du tag (repère tag) - ordre OpenCV standard
    # Coin supérieur gauche, supérieur droit, inférieur droit, inférieur gauche
    s = TAG_SIZE  # en mm
    obj_points = np.array(
        [
            [-s / 2, s / 2, 0],  # Haut gauche
            [s / 2, s / 2, 0],  # Haut droit
            [s / 2, -s / 2, 0],  # Bas droit
            [-s / 2, -s / 2, 0],  # Bas gauche
        ],
        dtype=np.float32,
    )

    # Préparer les points image
    img_points = corners_img.astype(np.float32).reshape(-1, 1, 2)

    # Undistort les points fisheye avant solvePnP
    img_points_undistorted = cv2.fisheye.undistortPoints(img_points, K, D, None, K)
    img_points_undistorted = img_points_undistorted.reshape(4, 2)

    # solvePnP avec les points corrigés (distCoeffs=None car déjà undistorted)
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points_undistorted,
        K,
        None,  # Pas de distorsion car déjà corrigée
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not success:
        return None

    return tvec.flatten()  # (X, Y, Z) en mm


def main():
    input_folder = "2026-01-16-playground-ready"
    output_folder = get_output_folder()
    process_folder(input_folder, output_folder)


# Usage
if __name__ == "__main__":

    # Load expected ArUco positions
    EXPECTED_POSITIONS = aruco_initial_positions.get_expected_position()

    # Intrinsic matrix (calibration values)
    K = np.array(
        [
            [2.49362477e03, 0.00000000e00, 1.97718701e03],
            [0.00000000e00, 2.49311358e03, 2.03491176e03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    # Fisheye distortion coefficients (k1, k2, k3, k4)
    D = np.array([[-0.1203345, 0.06802544, -0.13779641, 0.08243704]])

    main()
