#!/usr/bin/env python3
"""
Tâche de détection ArUco
Capture des photos et les envoie à la queue pour le streaming
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Queue
from logging import Logger
from camera_functions import initialize_camera, capture_and_save_image
from detect_aruco import detect_aruco

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def task_aruco_detection(queue: Queue, logger: Logger):
    """
    Tâche de capture: prend une photo et l'envoie à la queue pour le streaming
    """
    try:
        # Déterminer le répertoire de sauvegarde
        # __file__ -> .../picam/raspberry/src/task_aruco_detection.py
        # parents[2] -> .../picam/raspberry
        # parents[3] -> .../picam
        repo_root = Path(__file__).resolve().parents[2]
        pictures_dir = repo_root / "output" / "camera"
        
        logger.info("Démarrage de la tâche de détection ArUco (capture de photos)")
        logger.info(f"Répertoire de sauvegarde: {pictures_dir}")
        
        # Initialiser la caméra
        initialize_camera()
        logger.info("Caméra initialisée avec succès")
        
        # Boucle de capture
        while True:
            try:
                # Capturer une image
                filepath = capture_and_save_image(pictures_dir)
                logger.info(f"Image capturée: {filepath.name}")

                # Pré-traitement de l'image
                # preprocess_image(filepath)
                
                # Detection des coordonnées des éléments de jeu
                # detect_aruco(filepath)
                
                # Envoyer le chemin du fichier à la queue pour le streaming
                queue.put(str(filepath))
                
            except Exception as e:
                logger.error(f"Erreur lors de la capture: {e}")
            
            # Petite pause entre les captures (environ 40ms pour ~25 FPS)
            time.sleep(0.04)
            
    except Exception as e:
        logger.error(f"Erreur fatale dans la tâche ArUco: {e}")
        raise