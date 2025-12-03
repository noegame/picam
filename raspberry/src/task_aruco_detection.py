#!/usr/bin/env python3
"""
Tâche de détection ArUco
Capture des photos et les envoie à la queue pour le streaming
version de opencv :  4.10
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
from camera.camera_factory import get_camera
import detect_aruco as detect_aruco

from pathlib import Path
from multiprocessing import Queue

from my_math import *

# ---------------------------------------------------------------------------
# Constantes globales
# --------------------------------------------------------------------------- 

USE_FAKE_CAMERA = False

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

def task_aruco_detection(queue_to_stream: Queue, queue_to_com: Queue):
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

        # Create directories if they don't exist
        camera_pictures_dir.mkdir(parents=True, exist_ok=True)
        undistorted_pictures_dir.mkdir(parents=True, exist_ok=True)
        warped_pictures_dir.mkdir(parents=True, exist_ok=True)
        calibration_file = script_dir + "/camera_calibration.npz"
        
        logger.info("Démarrage de la tâche de détection ArUco (capture de photos)")

        # Initialiser la caméra via la factory
        image_width, image_height = 2000, 2000
        if USE_FAKE_CAMERA:
            # Le dossier d'images pour la fausse caméra
            fake_camera_folder = repo_root / "test_data" / "sample_images"
            logger.info(f"Utilisation de la fausse caméra avec les images de: {fake_camera_folder}")
            cam = get_camera(w=image_width, h=image_height, use_fake_camera=True, fake_camera_image_folder=fake_camera_folder)
        else:
            logger.info("Utilisation de la caméra réelle PiCamera2")
            cam = get_camera(w=image_width, h=image_height)
        
        # Importation des coefficients de distorsion (calibration)
        camera_matrix, dist_coeffs = undistort.import_camera_calibration(calibration_file)
        logger.info("Paramètres de calibration de la caméra importés avec succès")

        # Récupération de la taille de l'image
        image_size = (image_width, image_height)

        # Calcul une nouvelle matrice de caméra optimale pour la correction de la distorsion.
        newcameramtx = undistort.process_new_camera_matrix(camera_matrix, dist_coeffs, image_size)
        logger.info("Nouvelle matrice de caméra optimisée calculée avec succès")

        # Points de destination pour le redressement
        dst_points = np.array([[A1.x, A1.y], [B1.x, B1.y], [D1.x, D1.y], [C1.x, C1.y]], dtype=np.float32)

        # Initialise le détecteur openCV ArUco
        aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        logger.info("Détecteur ArUco initialisé avec succès")

        aruco_tags_for_queue = [Point]

        # Boucle de capture
        while True:
            try:
                # Capturer une image
                original_img, original_filepath = cam.capture_image(pictures_dir=camera_pictures_dir)
                
            except Exception as e:
                logger.error(f"Erreur lors de la capture: {e}")
                time.sleep(1) # Eviter de surcharger en cas d'erreur de capture en boucle
                continue
            
            # Pré-traitement de l'image (correction de la distorsion)
            img_distorted = undistort.undistort(original_img, camera_matrix, dist_coeffs, newcameramtx)
            logger.debug("Distorsion de l'image corrigée avec succès")

            # Détection des tags ArUco fixes
            tags_from_img, img_annotated = detect_aruco.detect_in_image(img_distorted, aruco_dict, None, aruco_params, draw=False)
            
            # artifice provisoire pour convertir la liste de tags détectés en liste de Point
            tags_from_img = detect_aruco.convert_detected_tags(tags_from_img)

            # Récupère les coordonnées des 4 points fixes
            A2 = find_point_by_id(tags_from_img, 20)
            B2 = find_point_by_id(tags_from_img, 22)
            C2 = find_point_by_id(tags_from_img, 21)
            D2 = find_point_by_id(tags_from_img, 23)

            # Compress img before sending to queue
            _, original_img_bytes = cv2.imencode('.jpg', original_img)
            _, undistorted_img_bytes = cv2.imencode('.jpg', img_distorted)

            # Prépare une liste des tags ArUco détectés pour la queue List[Points]
            aruco_tags_for_queue = tags_from_img
            logger.debug(f"Tags ArUco détectés et prêt pour queue: {[str(tag) for tag in tags_from_img]}")

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
                
                
                # Matrice de transformation affine
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)

                w,h = img_distorted.shape[:2]
                logger.debug(f" w : {w} h : {h} ")
                transformed_img = cv2.warpPerspective(img_distorted, matrix, (w, h))
                
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
            
            queue_to_stream.put(data_for_queue)          
            
    except Exception as e:
        logger.error(f"Erreur fatale dans la tâche ArUco: {e}")
        raise