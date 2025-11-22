"""
undistort_image.py
Correction de la distorsion d'une image en utilisant les paramètres de calibration de la caméra.
"""

import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog


def undistord_2(calib_file, input_image_path, output_image_path):

    # Chargement de la calibration
    data = np.load(calib_file)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    print("Camera matrix :\n", camera_matrix)
    print("Distortion coefficients :\n", dist_coeffs.ravel())

    # Lecture de l'image d'entrée
    img = cv2.imread(input_image_path)
    if img is None:
        raise RuntimeError("Impossible de lire l'image d'entrée.")

    h, w = img.shape[:2]

    # Nouvelle matrice optimisée pour éviter de perdre du champ de vision
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Undistortion
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Optionnel : recadrage basé sur le ROI
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    # Sauvegarde
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    output_image_path = os.path.join(output_image_path, f"{base_name}_undistorted.jpg")
    cv2.imwrite(output_image_path, undistorted)
    print(f"Image corrigée enregistrée dans : {output_image_path}")


def undistort_1(image_path, camera_matrix, dist_coeffs, out_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    und = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    if out_path:
        # Si out_path est un dossier, créer un chemin complet vers un fichier
        if os.path.isdir(out_path):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(out_path, f"{base_name}_undistorted.jpg")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, und)
        print(f"Undistorted image saved to: {out_path}")
    return und

def main():

    root = tk.Tk()
    root.withdraw()

    INPUT_FILE_IMAGE = filedialog.askopenfilename(
        title="Select an image",
        initialdir=r"data",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    OUTPUT_FOLDER_IMAGE = r"output\\undistorted"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "camera_calibration.npz")

    # Chargement de la calibration
    data = np.load(CALIBRATION_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    undistort_1(INPUT_FILE_IMAGE, camera_matrix, dist_coeffs, OUTPUT_FOLDER_IMAGE)

if __name__ == "__main__":
    main()
