# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
from vision_python.src.aruco.aruco import Aruco

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def init_aruco_detector(
    adaptive_thresh_constant=None,
    min_marker_perimeter_rate=None,
    max_marker_perimeter_rate=None,
    polygonal_approx_accuracy_rate=None,
    use_config_params=True,
) -> cv2.aruco.ArucoDetector:
    """
    Initialize and return an ArUco detector with configurable parameters.

    Args:
        adaptive_thresh_constant: Adaptive threshold constant (None = use config)
        min_marker_perimeter_rate: Min marker perimeter rate (None = use config)
        max_marker_perimeter_rate: Max marker perimeter rate (None = use config)
        polygonal_approx_accuracy_rate: Polygon approx accuracy rate (None = use config)
        use_config_params: If True, load missing params from config

    Returns:
        aruco_detector: Configured cv2.aruco.ArucoDetector object

    version of opencv: 4.6.0+dfsg-12
    """
    from vision_python.config import config

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # Use subpixel corner refinement for better accuracy
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # Load parameters from config if requested and not provided
    if use_config_params:
        img_params = config.get_image_processing_params()

        if adaptive_thresh_constant is None:
            adaptive_thresh_constant = img_params["adaptive_thresh_constant"]
        if min_marker_perimeter_rate is None:
            min_marker_perimeter_rate = img_params["min_marker_perimeter_rate"]
        if max_marker_perimeter_rate is None:
            max_marker_perimeter_rate = img_params["max_marker_perimeter_rate"]
        if polygonal_approx_accuracy_rate is None:
            polygonal_approx_accuracy_rate = img_params[
                "polygonal_approx_accuracy_rate"
            ]

    # Apply parameters if provided
    if adaptive_thresh_constant is not None:
        parameters.adaptiveThreshConstant = adaptive_thresh_constant
    if min_marker_perimeter_rate is not None:
        parameters.minMarkerPerimeterRate = min_marker_perimeter_rate
    if max_marker_perimeter_rate is not None:
        parameters.maxMarkerPerimeterRate = max_marker_perimeter_rate
    if polygonal_approx_accuracy_rate is not None:
        parameters.polygonalApproxAccuracyRate = polygonal_approx_accuracy_rate
    # aruco_params.minMarkerPerimeterRate = 0.005
    # aruco_params.maxMarkerPerimeterRate = 13
    # aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

    # # Seuillage adaptatif: fenêtres plus grandes pour compenser le flou
    # parameters.adaptiveThreshWinSizeMin = 5  # Au lieu de 3
    # parameters.adaptiveThreshWinSizeMax = 35  # Au lieu de 23
    # parameters.adaptiveThreshWinSizeStep = 5  # Au lieu de 10

    # # Réduire l'exigence de déviation standard pour Otsu
    # parameters.minOtsuStdDev = 2.0  # Au lieu de 5.0

    # # Augmenter la tolérance aux erreurs de lecture
    # parameters.errorCorrectionRate = 0.8  # Au lieu de 0.6

    # # Tolérer des erreurs dans la bordure (flou = pixels ambigus)
    # parameters.maxErroneousBitsInBorderRate = 0.5  # Au lieu de 0.35

    # # Augmenter les pixels par cellule lors du décodage
    # parameters.perspectiveRemovePixelPerCell = 6  # Au lieu de 4

    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return aruco_detector


def detect_aruco_in_img(
    img: np.ndarray, aruco_detector: cv2.aruco.ArucoDetector
) -> tuple[list[Aruco], list]:
    """
    Detect ArUco markers in an image.
    version of opencv :  4.6.0+dfsg-12

    Args:
        img: Image in which to detect ArUco markers.
        aruco_dic: Predefined ArUco dictionary from OpenCV.
        aruco_params: ArUco detector parameters.

    Returns:
        tuple: A tuple containing:
            - list of detected Aruco objects
            - list of rejected marker corners

    """

    # Detection
    corners_list, ids, rejected = aruco_detector.detectMarkers(img)

    detected_markers = []
    if ids is None or len(ids) == 0:
        return detected_markers, rejected if rejected is not None else []

    #
    ids = ids.flatten()
    for corners, id_val in zip(corners_list, ids):
        corners = corners.reshape((4, 2)).astype(np.float32)
        cX = float(np.mean(corners[:, 0]))
        cY = float(np.mean(corners[:, 1]))

        rect = cv2.minAreaRect(corners.astype(np.float32))
        angle = rect[2]
        w, h = rect[1]
        if w < h:
            corrected_angle = (angle + 90) % 360
        else:
            corrected_angle = angle % 360
        if corrected_angle > 180:
            corrected_angle -= 360

        a = Aruco(
            cX,
            cY,
            1,
            id_val,
            float(corrected_angle),
        )
        detected_markers.append(a)

    return detected_markers, rejected if rejected is not None else []


def create_annotated_image(
    img: np.ndarray,
    detected_markers: list[Aruco],
    rejected_markers: list = None,
) -> np.ndarray:
    """
    Create an annotated image with detected ArUco markers.
    version of opencv :  4.6.0+dfsg-12

    Args:
        img: Original image.
        detected_markers: List of detected Aruco objects.
        rejected_markers: List of rejected marker corners (optional).

    Returns:
        Annotated image with detected markers drawn.

    """
    annotated_img = img.copy()

    # Draw detected markers in green
    if len(detected_markers) > 0:
        corners_list = []
        ids = []
        for marker in detected_markers:
            corners = np.array(
                [
                    [marker.x - 10, marker.y - 10],
                    [marker.x + 10, marker.y - 10],
                    [marker.x + 10, marker.y + 10],
                    [marker.x - 10, marker.y + 10],
                ],
                dtype=np.float32,
            )
            # Reshape to (4, 1, 2) format required by drawDetectedMarkers
            corners = corners.reshape((4, 1, 2))
            corners_list.append(corners)
            ids.append(marker.aruco_id)

        ids = np.array(ids, dtype=np.int32)
        annotated_img = cv2.aruco.drawDetectedMarkers(annotated_img, corners_list, ids)

    # Draw rejected markers in red
    if rejected_markers is not None and len(rejected_markers) > 0:
        for corners in rejected_markers:
            corners = corners.reshape((4, 2)).astype(np.int32)
            cv2.polylines(annotated_img, [corners], True, (0, 0, 255), 2)
            # Add "X" mark for rejected
            center = corners.mean(axis=0).astype(np.int32)
            cv2.drawMarker(
                annotated_img,
                tuple(center),
                (0, 0, 255),
                cv2.MARKER_TILTED_CROSS,
                20,
                2,
            )

    return annotated_img
