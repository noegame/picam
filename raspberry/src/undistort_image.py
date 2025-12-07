"""
undistort_image.py
Correction de la distorsion d'une image en utilisant les paramètres de calibration de la caméra.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def import_camera_calibration(
    calibration_file: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Importe les paramètres de calibration de la caméra à partir d'un fichier.npz.

    Args:
        calibration_file (str): Chemin vers le fichier de calibration.

    Returns:
        tuple: Une tuple contenant la matrice de la caméra et les coefficients de distorsion.
    """
    with numpy.load(calibration_file) as data:
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs


def process_new_camera_matrix(
    camera_matrix: numpy.ndarray, dist_coeffs: numpy.ndarray, image_size: tuple
) -> numpy.ndarray:
    """
    Calcule une nouvelle matrice de caméra optimale pour la correction de la distorsion.

    Args:
        camera_matrix (numpy.ndarray): La matrice intrinsèque de la caméra.
        dist_coeffs (numpy.ndarray): Les coefficients de distorsion.
        image_size (tuple): La taille de l'image (largeur, hauteur).

    Returns:
        numpy.ndarray: La nouvelle matrice de caméra optimisée.
    """
    w, h = image_size
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    return newcameramtx


def undistort(
    img: numpy.ndarray,
    camera_matrix: numpy.ndarray,
    dist_coeffs: numpy.ndarray,
    newcameramtx: numpy.ndarray,
) -> numpy.ndarray:
    """
    Corrige la distorsion d'une image à l'aide de la matrice de la caméra et des coefficients de distorsion.

    Args:
        img (numpy.ndarray): L'image d'entrée (distordue).
        camera_matrix (numpy.ndarray): La matrice intrinsèque de la caméra.
        dist_coeffs (numpy.ndarray): Les coefficients de distorsion.

    Returns:
        numpy.ndarray: L'image corrigée (non distordue).
    """

    # Applique la correction de distorsion à l'image.
    # La nouvelle matrice de caméra (newcameramtx) est utilisée pour mapper les pixels de l'image corrigée.
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    return img_undistorted
