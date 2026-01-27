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


def detect_markers(detector, image, mask=None, scale=1.5, sharpen=True):
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
    
    # Apply mask if provided
    if mask is not None:
        t0 = t.time()
        image = cv2.bitwise_and(image, image, mask=mask)
        timings["mask_application"] = t.time() - t0
    
    # Preprocessing: Sharpness enhancement
    if sharpen:
        t0 = t.time()
        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel_sharpening)
        timings["sharpening"] = t.time() - t0
    
    # Resize if scale != 1.0
    if scale != 1.0:
        t0 = t.time()
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height))
        timings["resizing"] = t.time() - t0
    
    # Detection
    t0 = t.time()
    corners, ids, rejected = detector.detectMarkers(image)
    timings["detection"] = t.time() - t0
    
    # Rescale corners back to original coordinates
    if scale != 1.0 and corners:
        corners = [corner / scale for corner in corners]
    
    return corners, ids, rejected, timings


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


def convert_coord(points, homography_inv, K, D, z_values=None):
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
        point_undistorted = cv2.fisheye.undistortPoints(point_distorted, K, D, None, K)
        
        # Apply homography transformation
        point_real = cv2.perspectiveTransform(point_undistorted, homography_inv)
        
        # Extract coordinates and add Z
        x = point_real[0][0][0]
        y = point_real[0][0][1]
        z = z_values[i]
        
        real_coords.append((x, y, z))
    
    return real_coords


def print_detection_stats(ids, centers, real_coords=None, expected_positions=None):
    """
    Print statistics about the detected markers.
    
    Args:
        ids: Array of marker IDs (Nx1)
        centers: List of center coordinates (tuples)
        real_coords: Optional list of real-world coordinates
        expected_positions: Optional dictionary of expected positions by ID
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
            "center": centers[i],
            "real_coords": real_coords[i] if real_coords else None,
            "expected_pos": None,
            "error_x": None,
            "error_y": None,
        }
        
        # Calculate errors if expected positions provided
        if real_coords and expected_positions and mid in expected_positions:
            closest = find_closest_expected_position(mid, (real_coords[i][0], real_coords[i][1]))
            if closest:
                z_exp = real_coords[i][2]  # Use same Z as detected
                detection["expected_pos"] = (closest[0], closest[1], z_exp)
                detection["error_x"] = real_coords[i][0] - closest[0]
                detection["error_y"] = real_coords[i][1] - closest[1]
        
        detections.append(detection)
    
    # Print header
    if real_coords is not None:
        print(f"\n{'#':<6}{'ID':<6}{'Image Position':<20}{'Detected (mm)':<25}{'Expected (mm)':<25}{'Error (mm)':<20}")
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
                print(f"{idx:<6}{marker_id:<6}({center_x},{center_y}){'':>6}({real_x},{real_y},{real_z}){'':>6}({exp_x},{exp_y},{exp_z}){'':>6}(Δx:{err_x:+4}, Δy:{err_y:+4})")
            else:
                print(f"{idx:<6}{marker_id:<6}({center_x},{center_y}){'':>6}({real_x},{real_y},{real_z}){'':>6}(N/A){'':>12}(N/A)")
        else:
            print(f"{idx:<6}{marker_id:<10}({center_x}, {center_y})")


def detect_aruco_tags(
    image_path,
    detector,
    output_path="output.png",
    verbose=True,
    mask=None,
    homography_inv=None,
):
    """
    Orchestrates the complete ArUco detection and localization pipeline.
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

    # ========== STEP 1: DETECT MARKERS ==========
    corners, ids, rejected, detection_timings = detect_markers(
        detector, image, mask=mask, scale=1.5, sharpen=True
    )
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
                rejected_count += 1
                continue
            
            valid_ids.append(mid)
            valid_corners.append(corners[i][0])
    
    # ========== STEP 3: CALCULATE CENTERS ==========
    centers = []
    if valid_corners:
        centers = find_center_coord(valid_corners)
    
    # ========== STEP 4: CONVERT COORDINATES (if homography provided) ==========
    t0 = t.time()
    real_coords = None
    if homography_inv is not None and centers:
        # Determine Z values for each tag
        z_values = []
        for mid in valid_ids:
            z_expected = 30  # Default 30mm (common height of tags)
            if mid in EXPECTED_POSITIONS:
                # Retrieve Z from initial data
                for entry in aruco_initial_positions.initial_position:
                    if mid in entry[0]:
                        z_expected = entry[4]  # Index 4 = Z
                        break
            z_values.append(z_expected)
        
        # Convert image coordinates to real-world coordinates
        real_coords = convert_coord(centers, homography_inv, K, D, z_values)
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

    # ========== STEP 6: PRINT DETECTION STATISTICS ==========
    if verbose:
        print(f"\nUnauthorized IDs rejected: \t{rejected_count}")
        print(f"Valid tags detected: \t\t{len(all_detections)}")
        print(f"Detection duration: \t\t{timings['total_detection']:.2f} seconds")
        
        if len(all_detections) > 0:
            # Prepare data for print_detection_stats
            stats_ids = np.array([[d["id"]] for d in all_detections])
            stats_centers = [d["center"] for d in all_detections]
            stats_real_coords = [d["real_coords"] for d in all_detections] if real_coords else None
            
            print_detection_stats(stats_ids, stats_centers, stats_real_coords, EXPECTED_POSITIONS)
        else:
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
    global_stats = {
        "total_rejected": 0,
        "total_valid": 0,
        "total_time": 0
    }

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
        global_stats["total_rejected"] += stats["rejected_count"]
        global_stats["total_valid"] += stats["valid_count"]
        global_stats["total_time"] += real_image_time  # CORRECTION: use real measured time

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
    print(f"\nAverage unauthorized IDs rejected: \t{global_stats['total_rejected'] / total_images:.2f}")
    print(f"Average valid tags detected: \t\t{global_stats['total_valid'] / total_images:.2f}")
    print(f"\nTotal processing time: \t\t{global_stats['total_time']:.2f} seconds")
    print(f"Total script execution time: \t\t{t.time() - t_init:.2f} seconds")
    print(f"Average time per image: \t\t{global_stats['total_time'] / total_images:.3f} seconds")

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
        percentage = (cumulative_timings.get(key, 0) / global_stats['total_time']) * 100
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
    output_folder = os.path.join("pictures/debug", script_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def main():
    input_folder = "../pictures/2026-01-16-playground-ready"
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
