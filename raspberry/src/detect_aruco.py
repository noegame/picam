# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from undistort_image import undistort
import cv2
import numpy

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------

def preprocess_image(img: numpy.ndarray, camera_matrix: numpy.ndarray, dist_coeffs: numpy.ndarray):
    
    # Correction de la distorsion
    img_undistorted = undistort(img, camera_matrix, dist_coeffs)
    
    return img_undistorted

