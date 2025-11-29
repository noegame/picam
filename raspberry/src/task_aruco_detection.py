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
import logging
import logging.config
from pathlib import Path
from multiprocessing import Queue
from logging import Logger
from camera_functions import initialize_camera, capture_and_save_image
from detect_aruco import preprocess_image

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def task_aruco_detection(queue: Queue, logger: Logger):
    """
    Tâche de capture: prend une photo et l'envoie à la queue pour le streaming
    """
    try:
        repo_root = Path(__file__).resolve().parents[2]
        pictures_dir = repo_root / "output" / "camera"
        
        logger.info("Démarrage de la tâche de détection ArUco (capture de photos)")
        logger.info(f"Répertoire de sauvegarde: {pictures_dir}")
        
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
                filepath = capture_and_save_image(pictures_dir)
                logger.info(f"Image capturée: {filepath.name}")

                # Pré-traitement de l'image
                img_distorted = preprocess_image(str(filepath), camera_matrix, dist_coeffs)
                
                # Enregistrement de l'image détordue
                processed_filepath = pictures_dir / f"processed_{filepath.name}"
                cv2.imwrite(str(processed_filepath), img_distorted)
                logger.info(f"Image prétraitée enregistrée: {processed_filepath.name}")
                 
                # Envoyer le chemin du fichier de l'image détordue à la queue pour le streaming
                queue.put(str(processed_filepath))
                
            except Exception as e:
                logger.error(f"Erreur lors de la capture: {e}")
            
            # Petite pause entre les captures (environ 40ms pour ~25 FPS)
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Erreur fatale dans la tâche ArUco: {e}")
        raise