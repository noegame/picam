#!/usr/bin/env python3
"""
Tâche de détection ArUco
Capture des photos et les envoie à la queue pour le streaming
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import time
import os
import numpy as np
import cv2
from pathlib import Path
from multiprocessing import Queue
from logging import Logger
from camera_functions import initialize_camera, capture_and_save_image
from detect_aruco import preprocess_image
from detect_aruco2 import detect_aruco
from my_math import *

# ---------------------------------------------------------------------------
# Constantes globales
# --------------------------------------------------------------------------- 

A1 = Point(600, 600, 20)
B1 = Point(1400, 600, 22)
C1 = Point(600, 2400, 21)
D1 = Point(1400, 2400, 23)

# A1 = Point(53, 53, 20)      #SO
# B1 = Point(123, 53, 22)     #SE
# C1 = Point(53, 213, 21)     #NO
# D1 = Point(123, 213, 23)    #NE

FIXED_IDS = {20, 21, 22, 23}

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def task_aruco_detection(queue: Queue, logger: Logger):
    """
    Tâche de capture: prend une photo et l'envoie à la queue pour le streaming
    """
    try:
        repo_root = Path(__file__).resolve().parents[2]
        camera_pictures_dir = repo_root / "output" / "camera"
        processed_pictures_dir = repo_root / "output" / "processed_img"
        
        logger.info("Démarrage de la tâche de détection ArUco (capture de photos)")
        logger.info(f"Répertoire de sauvegarde: {camera_pictures_dir}")
        
        # Initialiser la caméra
        initialize_camera()
        logger.info("Caméra initialisée avec succès")
        
        # Importation des coefficients de distorsion (calibration)
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "camera_calibration.npz")
        data = np.load(CALIBRATION_FILE)
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]

        # Boucle de capture
        while True:
            try:
                # Capturer une image
                filepath = capture_and_save_image(camera_pictures_dir)
                logger.info(f"Image capturée: {filepath.name}")

            except Exception as e:
                logger.error(f"Erreur lors de la capture: {e}")
            
            # Pré-traitement de l'image
            img_distorted = preprocess_image(str(filepath), camera_matrix, dist_coeffs)

            # =============================================

            # Détection des tags ArUco fixes pour redressement de l'image

            # Coordonnées des TAGS ARUCO fixes et mobiles détectés dans l'image (en pixels)
            tag_picture = detect_aruco(img_distorted)

            # Récupère les coordonnées des 4 points fixes détectés dans l'image
            A2 = find_point_by_id(tag_picture, 20)
            B2 = find_point_by_id(tag_picture, 22)
            C2 = find_point_by_id(tag_picture, 21)
            D2 = find_point_by_id(tag_picture, 23)

            if not all([A2, B2, C2, D2]):
                    print("Erreur: Tous les 4 tags fixes n'ont pas été détectés!")
                    missing = []
                    if not A2: missing.append("20")
                    if not B2: missing.append("22")
                    if not C2: missing.append("21")
                    if not D2: missing.append("23")
                    print(f"Tags manquants: {', '.join(missing)}")
                    print(f"Tags trouvés: {', '.join([str(p.ID) for p in tag_picture])}")

            else :
                # Calcul de la transformation affine entre les deux ensembles de points 
                src_points = np.array([[A2.x, A2.y], [B2.x, B2.y], [C2.x, C2.y], [D2.x, D2.y]], dtype=np.float32)
                dst_points = np.array([[A1.x, A1.y], [B1.x, B1.y], [C1.x, C1.y], [D1.x, D1.y]], dtype=np.float32)
                # Matrice de transformation affine
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)

                # Applique la transformation à l'image entière
                transformed_img = cv2.warpPerspective(img_distorted, matrix, (1920,1080))

                # =============================================

                # Enregistrement de l'image détordue
                processed_filepath = processed_pictures_dir / f"processed_{filepath.name}"
                cv2.imwrite(str(processed_filepath), transformed_img)
                logger.info(f"Image prétraitée enregistrée: {processed_filepath.name}")
                
                # Envoyer le chemin du fichier de l'image détordue à la queue pour le streaming
                queue.put(str(processed_filepath))          
                
                # Petite pause entre les captures (environ 40ms pour ~25 FPS)
                # time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Erreur fatale dans la tâche ArUco: {e}")
        raise