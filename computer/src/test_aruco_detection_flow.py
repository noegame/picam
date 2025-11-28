#!/usr/bin/env python3

"""
test_aruco_detection_flow.py
# 1. Prends une photo avec la caméra ou importe une image depuis un fichier.
# 2. Corrige la distorsion de l'image en utilisant les paramètres de distorsion de la caméra.
# 3. Détecte les tags ArUco dans l'image corrigée.
# 4. Grace au 4 tags aruco fixes dont on connait la position dans le monde réel, calcule la transformation 
#    entre le repère de la caméra et le repère du monde réel.
# 5. Utilise cette transformation pour estimer la position et l'orientation de tout autre tag ArUco détecté dans l'image.

repère de coordonnées du monde réel (en mm)
repère de coordonnées de l'image (en pixels)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from my_math import *
from detect_aruco import detect_aruco

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# Coordonnées des TAGS ARUCO fixes dans le repère du monde réel (en mm)
A1 = Point(600, 600, 20)
B1 = Point(1400, 600, 22)
C1 = Point(600, 2400, 21)
D1 = Point(1400, 2400, 23)

# A1 = Point(53, 53, 20)      #SO
# B1 = Point(123, 53, 22)     #SE
# C1 = Point(53, 213, 21)     #NO
# D1 = Point(123, 213, 23)    #NE

FIXED_IDS = {20, 21, 22, 23}

PRINT_TAGS_POSITIONS_PICTURE = True
PRINT_TAGS_POSITIONS_WORLD = True
PRINT_TRANSFORM_MATRIX = False

SHOW_DISTORTED_IMG = False
SHOW_TRANSFORMED_IMG = False
SHOW_ANNOTED_IMG = True

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------

def import_image():
    """Importe une image depuis un fichier en utilisant une boîte de dialogue.
    Retourne le chemin de l'image sélectionnée."""

    root = tk.Tk()
    root.withdraw()
    input_image_path = filedialog.askopenfilename(
        title="Select an image file",
        initialdir="data",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    if not input_image_path:
        return None

    return input_image_path

def show_image(image : np.ndarray):
    """Affiche une image."""
    image = cv2.resize(image, (int(image.shape[1] * 0.8), int(image.shape[0] * 0.8)))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    # ========= Importation de l'image ==========

    input_image_path = import_image()
    if not input_image_path:
        print("\nAucune image sélectionnée. Fin du programme.")
        return

    # Coordonnées des TAGS ARUCO fixes et mobiles détectés dans l'image (en pixels)
    tag_picture = detect_aruco(input_image_path)

    # Récupère les coordonnées des 4 points fixes détectés dans l'image
    A2 = find_point_by_id(tag_picture, 20)
    B2 = find_point_by_id(tag_picture, 22)
    C2 = find_point_by_id(tag_picture, 21)
    D2 = find_point_by_id(tag_picture, 23)

    # Vérification que tous les points fixes ont été détectés
    if not all([A2, B2, C2, D2]):
        print("Erreur: Tous les 4 tags fixes n'ont pas été détectés!")
        missing = []
        if not A2: missing.append("20")
        if not B2: missing.append("22")
        if not C2: missing.append("21")
        if not D2: missing.append("23")
        print(f"Tags manquants: {', '.join(missing)}")
        print(f"Tags trouvés: {', '.join([str(p.ID) for p in tag_picture])}")
        return

    # ========= Calcul de la transformation affine ==========

    # Calcul de la transformation affine entre les deux ensembles de points 
    src_points = np.array([[A2.x, A2.y], [B2.x, B2.y], [C2.x, C2.y], [D2.x, D2.y]], dtype=np.float32)
    dst_points = np.array([[A1.x, A1.y], [B1.x, B1.y], [C1.x, C1.y], [D1.x, D1.y]], dtype=np.float32)
    # Matrice de transformation affine
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Applique la transformation à l'image entière
    img = cv2.imread(input_image_path)
    transformed_img = cv2.warpPerspective(img, matrix, (600, 600))
 
    # ========= Estimation des positions des tags mobiles ==========

    # Transformation des points mobiles (tous les points sauf les 4 fixes)
    mobile_points = [p for p in tag_picture if p.ID not in FIXED_IDS]
    if mobile_points:
        print("\nPositions estimées des tags mobiles dans le repère du monde réel:")
        for i, mp in enumerate(mobile_points):
            src_pt = np.array([[mp.x, mp.y]], dtype=np.float32)
            dst_pt = cv2.perspectiveTransform(np.array([src_pt]), matrix)     
    else:
        print("\nAucun tag mobile détecté")
        return

    # sauvegarde les tags mobiles détectés avec leurs positions estimées et id dans une liste de type point
    tags_reel_world = [Point(dst_pt[0][0][0], dst_pt[0][0][1], mp.ID) for mp in mobile_points]

    print_points(tags_reel_world)

    # ========= Annotation de l'image originale avec les positions des tags mobiles ========== 
    
    annotated_img = img.copy()
    
    for mp in mobile_points:
        src_pt = np.array([[mp.x, mp.y]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(np.array([src_pt]), matrix)
        x_real, y_real = int(dst_pt[0][0][0]), int(dst_pt[0][0][1])
        cv2.circle(annotated_img, (int(mp.x), int(mp.y)), 5, (0, 0, 255), -1)
        cv2.putText(annotated_img, f"ID:{mp.ID} ({x_real:.2f}, {y_real:.2f})", (int(mp.x) + 10, int(mp.y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # ========== Sauvegarde de l'image annotée ==========

    output_dir = filedialog.askdirectory(
        title="Select output directory",
        initialdir="output/main"
    )
    if not output_dir:
        print("Aucun dossier sélectionné. Fin du programme.")
        return 
    try:
        cv2.imwrite(output_dir + "/" + "output.png", annotated_img)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image annotée: {e}")

    # ========== Gestion des affichages ==========

    if PRINT_TAGS_POSITIONS_PICTURE:
        print("\nPositions des tags détectés dans l'image (en pixels) :")
        for p in tag_picture:
            print(f"  TAG ID={p.ID} à ({p.x:.2f}, {p.y:.2f})")
    if PRINT_TAGS_POSITIONS_WORLD:
        print("\nPositions estimées des tags dans le repère du monde réel (en mm) :")
        for p in tags_reel_world:
            print(f"  TAG ID={p.ID} à ({p.x:.2f}, {p.y:.2f})")
    if PRINT_TRANSFORM_MATRIX:        
        print(matrix)
    if SHOW_TRANSFORMED_IMG:
        show_image(transformed_img)
    if SHOW_ANNOTED_IMG:
        show_image(annotated_img)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()