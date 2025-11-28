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
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# Variable globale pour la caméra (singleton)
camera = None

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------

def initialize_camera():
    """Initialise la caméra PiCamera2"""
    global camera
    if camera is None:
        try:
            camera = Picamera2()
            camera_config = camera.create_still_configuration(
                main={"size": (1920, 1080)}
            )
            camera.configure(camera_config)
            camera.start()
        except Exception as e:
            raise Exception(f"Erreur lors de l'initialisation de la caméra: {e}")

def capture_and_save_image(pictures_dir: Path) -> Path:
    """Capture une image et la sauvegarde"""
    global camera
    
    if camera is None:
        initialize_camera()
    
    try:
        # Créer le répertoire s'il n'existe pas
        pictures_dir.mkdir(parents=True, exist_ok=True)
        
        # Générer le nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # avec millisecondes
        filename = f"{timestamp}_capture.jpg"
        filepath = pictures_dir / filename
        
        # Capture avec PiCamera2
        camera.capture_file(str(filepath))
        
        return filepath
        
    except Exception as e:
        raise Exception(f"Erreur lors de la capture: {e}")

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
                
                # Envoyer le chemin du fichier à la queue pour le streaming
                queue.put(str(filepath))
                
            except Exception as e:
                logger.error(f"Erreur lors de la capture: {e}")
            
            # Petite pause entre les captures (environ 40ms pour ~25 FPS)
            time.sleep(0.04)
            
    except Exception as e:
        logger.error(f"Erreur fatale dans la tâche ArUco: {e}")
        raise