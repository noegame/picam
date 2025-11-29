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

def undistort(img : numpy.ndarray, camera_matrix : numpy.ndarray, dist_coeffs : numpy.ndarray):
    """
    Corrige la distorsion d'une image à l'aide de la matrice de la caméra et des coefficients de distorsion.

    Args:
        img (numpy.ndarray): L'image d'entrée (distordue).
        camera_matrix (numpy.ndarray): La matrice intrinsèque de la caméra.
        dist_coeffs (numpy.ndarray): Les coefficients de distorsion.

    Returns:
        numpy.ndarray: L'image corrigée (non distordue).
    """
    h, w = img.shape[:2]
    
    # Calcule la nouvelle matrice de caméra optimale pour conserver tous les pixels de l'image originale.
    # Le paramètre alpha=1 garantit que l'image entière est visible après correction, quitte à avoir des bords noirs.
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    
    # Applique la correction de distorsion à l'image.
    # La nouvelle matrice de caméra (newcameramtx) est utilisée pour mapper les pixels de l'image corrigée.
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    return img_undistorted
