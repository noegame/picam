"""
test_undistort_image.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import os
import cv2
import tkinter as tk
from undistort_image import undistort
from tkinter import filedialog

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def main():

    root = tk.Tk()
    root.withdraw()

    input_img_file = filedialog.askopenfilename(
        title="Select an image",
        initialdir=r"output",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )

    output_img_file = filedialog.askdirectory(
        title="Select an output folder",
        initialdir=r"output",
        mustexist=True
    )

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "camera_calibration.npz")

    # Chargement de la calibration
    data = np.load(CALIBRATION_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    # Importation de l'image 
    img = cv2.imread(input_img_file, cv2.IMREAD_COLOR)

    # Correction de la distorsion
    img_undistorted = undistort(img, camera_matrix, dist_coeffs)

    # Enregistrement de l'image
    output_img_path = os.path.join(output_img_file, os.path.basename(input_img_file))
    cv2.imwrite(output_img_path, img_undistorted)

    return
     
# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
