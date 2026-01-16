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
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """
    Straighten the image using perspective transformation based on source and destination points.

    Args:
        img: Input image to be straightened.
        src_points: Source points in the original image (4 points).
        dst_points: Destination points in the straightened image (4 points).
        output_width: Width of the output straightened image.
        output_height: Height of the output straightened image.

    Returns:
        Straightened image with dimensions (output_height, output_width).
    """

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation to straighten the image
    straightened_img = cv2.warpPerspective(img, matrix, (output_width, output_height))
    return straightened_img
