# version of opencv :  4.6.0+dfsg-12

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import math
from raspberry.src.aruco import *

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def detect_in_image(
    img: np.ndarray,
    aruco_dic,
    detector,
    aruco_params,
    draw=False,
    aruco_descriptions=None,
):
    """
    Détecte les tags ArUco dans une image et ajoute leurs descriptions si disponibles.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection
    if detector is not None:
        corners_list, ids, rejected = detector.detectMarkers(gray)
    else:
        # Utiliser la nouvelle API OpenCV 4.7+
        detector = cv2.aruco.ArucoDetector(aruco_dic, aruco_params)
        corners_list, ids, rejected = detector.detectMarkers(gray)

    results = []
    if ids is None or len(ids) == 0:
        return results, img

    ids = ids.flatten()
    for corners, id_val in zip(corners_list, ids):
        corners = corners.reshape((4, 2))
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

        results.append(
            {
                "id": int(id_val),
                "center_x": cX,
                "center_y": cY,
                "z": 1,
                "angle_deg": float(corrected_angle),
                "corners": corners,
            }
        )

        if draw:
            int_corners = corners.astype(int)
            cv2.polylines(
                img, [int_corners], isClosed=True, color=(0, 255, 0), thickness=2
            )
            cv2.circle(img, (int(round(cX)), int(round(cY))), 4, (255, 0, 0), -1)

            # Affiche l'ID et la description si disponible
            label = f"ID:{id_val}"
            if aruco_descriptions and str(id_val) in aruco_descriptions:
                label += f" - {aruco_descriptions[str(id_val)]}"

            cv2.putText(
                img,
                label,
                (int(cX) + 10, int(cY) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            L = max(40, int(min(img.shape[0], img.shape[1]) * 0.05))
            theta = math.radians(corrected_angle)
            dx = L * math.cos(theta)
            dy = -L * math.sin(theta)
            pt1 = (int(round(cX)), int(round(cY)))
            pt2 = (int(round(cX + dx)), int(round(cY + dy)))
            cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), 2, tipLength=0.2)

    return results, img


def convert_detected_tags(detected_tags: list) -> list[Aruco]:
    """
    Convertit la liste de tags détectés en une liste de points.

    """
    points = []
    for tag in detected_tags:
        p = Aruco(
            tag["center_x"], tag["center_y"], tag["z"], tag["id"], tag["angle_deg"]
        )
        points.append(p)
    return points
