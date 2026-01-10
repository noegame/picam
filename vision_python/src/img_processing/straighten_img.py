"""
straighten_img.py
Straightens images using perspective transformation based on source and destination points.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def straighten_image(
    img: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    """
    Straighten the image using perspective transformation based on source and destination points.

    Args:
        img: Input image to be straightened.
        src_points: Source points in the original image.
        dst_points: Destination points in the straightened image.

    Returns:
        Straightened image.
    """

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation to straighten the image
    w, h = img.shape[:2]
    print(f"  w : {w} h : {h} ")
    straightened_img = cv2.warpPerspective(img, matrix, (w, h))
    return straightened_img
