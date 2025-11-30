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
import cv2
import logging

import numpy as np
import undistort_image as undistort

from pathlib import Path
from multiprocessing import Queue

from camera import initialize_camera, capture_image
from detect_aruco2 import detect_aruco
from my_math import *

# ---------------------------------------------------------------------------
# Constantes globales
# --------------------------------------------------------------------------- 

# A1 = Point(600, 600, 20)
# B1 = Point(1400, 600, 22)
# C1 = Point(600, 2400, 21)
# D1 = Point(1400, 2400, 23)

A1 = Point(53, 53, 20)      #SO
B1 = Point(123, 53, 22)     #SE
C1 = Point(53, 213, 21)     #NO
D1 = Point(123, 213, 23)    #NE

FIXED_IDS = {20, 21, 22, 23}

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def task_aruco_detection(queue: Queue):
    """
    Tâche de capture: 
    - prend une photo
    - corrige la distorsion de la photo
    - redresse la photo
    - enregistre la photo annotée
    - envoi le chemin de la photo à la queue pour le streaming
    """
    logger = logging.getLogger('task_aruco_detection')
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = Path(__file__).resolve().parents[2]
        camera_pictures_dir = repo_root / "output" / "camera"
        undistorted_pictures_dir = repo_root / "output" / "undistorted"
        warped_pictures_dir = repo_root / "output" / "warped"
        calibration_file = script_dir + "/camera_calibration.npz"
        
        logger.info("Démarrage de la tâche de détection ArUco (capture de photos)")
        logger.info(f"Répertoire de sauvegarde: {camera_pictures_dir}")
        
        # Initialiser la caméra
        initialize_camera(2000,2000)
        logger.info("Caméra initialisée avec succès")
        
        # Importation des coefficients de distorsion (calibration)
        camera_matrix, dist_coeffs = undistort.import_camera_calibration(calibration_file)
        logger.info("Paramètres de calibration de la caméra importés avec succès")

        # Calcule une nouvelle matrice de caméra optimale pour la correction de la distorsion.
        newcameramtx = undistort.process_new_camera_matrix(camera_matrix, dist_coeffs, (2000,2000))
        logger.info("Nouvelle matrice de caméra optimisée calculée avec succès")

        # Boucle de capture
        while True:
            try:
                # Capturer une image
                original_img, original_filepath = capture_image(2000,2000,camera_pictures_dir)
                logger.info(f"Image capturée: {original_filepath.name}")

            except Exception as e:
                logger.error(f"Erreur lors de la capture: {e}")
                time.sleep(1) # Eviter de surcharger en cas d'erreur de capture en boucle
                continue
            
            # Pré-traitement de l'image (correction de la distorsion)
            img_distorted = undistort.undistort(original_img, camera_matrix, dist_coeffs, newcameramtx)
            logger.debug("Distorsion de l'image corrigée avec succès")

            # Détection des tags ArUco fixes
            tags_from_img = detect_aruco(img_distorted)

            # Récupère les coordonnées des 4 points fixes
            A2 = find_point_by_id(tags_from_img, 20)
            B2 = find_point_by_id(tags_from_img, 22)
            C2 = find_point_by_id(tags_from_img, 21)
            D2 = find_point_by_id(tags_from_img, 23)

            # Compress img before sending to queue
            _, original_img_bytes = cv2.imencode('.jpg', original_img)
            _, undistorted_img_bytes = cv2.imencode('.jpg', img_distorted)

            aruco_tags_for_queue = [{"id": p.ID, "x": p.x, "y": p.y} for p in tags_from_img]

            if not all([A2, B2, C2, D2]):
                missing = [str(id) for id in [20, 22, 21, 23] if not find_point_by_id(tags_from_img, id)]
                logger.warning(f"Tags fixes {', '.join(missing)} non trouvé(s)")

                data_for_queue = {
                    "original_img": original_img_bytes.tobytes(),
                    "undistorted_img": undistorted_img_bytes.tobytes(),
                    "warped_img": undistorted_img_bytes.tobytes(),  # Fallback: utilise l'image non-distordue
                    "aruco_tags": aruco_tags_for_queue,
                    "filename": original_filepath.name,
                }
            else:
                # Redressement de l'image
                # Ordonner les points dans le sens horaire pour éviter les rotations aléatoires
                # Ordre: NO (20), NE (22), SE (23), SO (21)
                src_points = np.array([[A2.x, A2.y], [B2.x, B2.y], [D2.x, D2.y], [C2.x, C2.y]], dtype=np.float32)
                dst_points = np.array([[A1.x, A1.y], [B1.x, B1.y], [D1.x, D1.y], [C1.x, C1.y]], dtype=np.float32)
                
                # Matrice de transformation affine
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)

                h, w = img_distorted.shape[:2]
                transformed_img = cv2.warpPerspective(img_distorted, matrix, (h, w))
                
                # Compress img before sending to queue
                _, warped_img_bytes = cv2.imencode('.jpg', transformed_img)
                
                data_for_queue = {
                    "original_img": original_img_bytes.tobytes(),
                    "undistorted_img": undistorted_img_bytes.tobytes(),
                    "warped_img": warped_img_bytes.tobytes(),
                    "aruco_tags": aruco_tags_for_queue,
                    "filename": original_filepath.name,
                }

                # TODO : détecter les tags dynamiques et calculer leur position réelle à partir de l'image redressée
            
            queue.put(data_for_queue)          
            
            time.sleep(0.04) # Environ 25 FPS
            
    except Exception as e:
        logger.error(f"Erreur fatale dans la tâche ArUco: {e}")
        raise