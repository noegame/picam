"""
3D localisation using ArUco tags with OpenCV solvePnP.
"""

import cv2
import numpy as np
import time as t


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
    timings["total_detection"] = t.time() - t_start

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

            all_detections.append(
                {
                    "id": mid,
                    "corners": corner,
                    "center": (center_x, center_y),
                }
            )

    return {
        "detections": all_detections,
        "rejected_count": rejected_count,
        "valid_count": len(all_detections),
        "detection_time": timings["total_detection"],
        "timings": timings,
    }


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


# =============================
# PARAMÈTRES
# =============================
TAG_SIZE = 0.05  # m

camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))

# =============================
# TAGS FIXES (REPÈRE TERRAIN)
# coins 3D de chaque tag connu
# =============================
TAG_SIZE = 0.12
HALF = TAG_SIZE / 2  # 0.06 m

fixed_tags = {
    20: np.array(
        [
            [0.600 - HALF, 0.600 + HALF, 0.0],
            [0.600 + HALF, 0.600 + HALF, 0.0],
            [0.600 + HALF, 0.600 - HALF, 0.0],
            [0.600 - HALF, 0.600 - HALF, 0.0],
        ],
        dtype=np.float32,
    ),
    21: np.array(
        [
            [0.600 - HALF, 2.400 + HALF, 0.0],
            [0.600 + HALF, 2.400 + HALF, 0.0],
            [0.600 + HALF, 2.400 - HALF, 0.0],
            [0.600 - HALF, 2.400 - HALF, 0.0],
        ],
        dtype=np.float32,
    ),
    22: np.array(
        [
            [1.400 - HALF, 0.600 + HALF, 0.0],
            [1.400 + HALF, 0.600 + HALF, 0.0],
            [1.400 + HALF, 0.600 - HALF, 0.0],
            [1.400 - HALF, 0.600 - HALF, 0.0],
        ],
        dtype=np.float32,
    ),
    23: np.array(
        [
            [1.400 - HALF, 2.400 + HALF, 0.0],
            [1.400 + HALF, 2.400 + HALF, 0.0],
            [1.400 + HALF, 2.400 - HALF, 0.0],
            [1.400 - HALF, 2.400 - HALF, 0.0],
        ],
        dtype=np.float32,
    ),
}


# Coins 3D d’un tag générique (repère tag)
tag_object_points = np.array(
    [
        [-TAG_SIZE / 2, TAG_SIZE / 2, 0],
        [TAG_SIZE / 2, TAG_SIZE / 2, 0],
        [TAG_SIZE / 2, -TAG_SIZE / 2, 0],
        [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],
    ],
    dtype=np.float32,
)

# =============================
# IMAGE
# =============================
image = cv2.imread(
    "pictures/2026-01-16-playground-ready/20260116_173858_506_4056x3040.png"
)
if image is None:
    raise RuntimeError("Impossible de charger l'image image1.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =============================
# DÉTECTION ARUCO
# =============================
detector = get_aruco_detector()

# Détection simple avec filtrage des IDs autorisés
ALLOWED_IDS = [20, 21, 22, 23, 41, 36, 47]
corners_all, ids_all, _ = detector.detectMarkers(gray)

# Filtrer les IDs autorisés
if ids_all is None:
    raise RuntimeError("Aucun tag détecté")

corners = []
ids = []
for i, marker_id in enumerate(ids_all):
    if marker_id[0] in ALLOWED_IDS:
        corners.append(corners_all[i])
        ids.append(marker_id[0])

if len(ids) == 0:
    raise RuntimeError("Aucun tag autorisé détecté")

ids = np.array(ids)

# =============================
# 1) POSE DES TAGS DANS LE REPÈRE CAMÉRA
# =============================
tag_poses_cam = {}

for i, tag_id in enumerate(ids):
    img_pts = corners[i].reshape(4, 2)

    success, rvec, tvec = cv2.solvePnP(
        tag_object_points, img_pts, camera_matrix, dist_coeffs
    )

    if success:
        tag_poses_cam[tag_id] = tvec

        print(f"Tag {tag_id} dans repère caméra : {tvec.ravel()}")

# =============================
# 2) ESTIMATION POSE CAMÉRA → TERRAIN
# =============================
object_points = []
image_points = []

for i, tag_id in enumerate(ids):
    if tag_id not in fixed_tags:
        continue

    for j in range(4):
        object_points.append(fixed_tags[tag_id][j])
        image_points.append(corners[i][0][j])

if len(object_points) < 4:
    detected_ids = ids.tolist()
    fixed_ids = list(fixed_tags.keys())
    raise RuntimeError(
        f"Pas assez de points de référence détectés ({len(object_points)} points). "
        f"Tags détectés: {detected_ids}. "
        f"Tags de référence requis: {fixed_ids}. "
        f"Au moins un tag de référence doit être visible."
    )

object_points = np.array(object_points, dtype=np.float32)
image_points = np.array(image_points, dtype=np.float32)

success, rvec_cw, tvec_cw = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs
)

if not success:
    raise RuntimeError("Impossible d’estimer la pose caméra")

R_cw, _ = cv2.Rodrigues(rvec_cw)

# Inversion : caméra → terrain
R_wc = R_cw.T
t_wc = -R_wc @ tvec_cw

# =============================
# 3) PASSAGE TAG → TERRAIN
# =============================
for tag_id, tvec_cam in tag_poses_cam.items():
    pos_terrain = R_wc @ tvec_cam + t_wc
    print(f"Tag {tag_id} dans repère terrain : {pos_terrain.ravel()}")
