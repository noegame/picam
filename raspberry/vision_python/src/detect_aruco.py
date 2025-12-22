# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
from vision_python.src.aruco import *

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def init_aruco_detector() -> cv2.aruco.ArucoDetector:
    """
    Initialize and return an ArUco detector with default parameters.
    version of opencv :  4.6.0+dfsg-12
    Returns:
        aruco_detector: An cv2.aruco.ArucoDetector object initialized with default parameters.

    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
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
) -> list[Aruco]:
    """
    Detect ArUco markers in an image.
    version of opencv :  4.6.0+dfsg-12

    Args:
        img: Image in which to detect ArUco markers.
        aruco_dic: Predefined ArUco dictionary from OpenCV.
        aruco_params: ArUco detector parameters.

    Returns:
        list: A list of detected Aruco objects.

    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

    cv2.imwrite("/home/roboteseo/dev/picam/output/aruco_gray.png", gray)
    cv2.imwrite("/home/roboteseo/dev/picam/output/aruco_sharp.png", sharp)
    cv2.imwrite("/home/roboteseo/dev/picam/output/aruco_img.png", img)

    # Detection
    corners_list, ids, rejected = aruco_detector.detectMarkers(sharp)

    detected_markers = []
    if ids is None or len(ids) == 0:
        return detected_markers

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

    return detected_markers


def create_annotated_image(
    img: np.ndarray,
    detected_markers: list[Aruco],
) -> np.ndarray:
    """
    Create an annotated image with detected ArUco markers.
    version of opencv :  4.6.0+dfsg-12

    Args:
        img: Original image.
        detected_markers: List of detected Aruco objects.

    Returns:
        Annotated image with detected markers drawn.

    """
    annotated_img = img.copy()
    if len(detected_markers) == 0:
        return annotated_img

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

    return annotated_img
