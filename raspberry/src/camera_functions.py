#!/usr/bin/env python3
"""
Fonctions liées à la caméra
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2
import numpy as np

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# Variable globale pour la caméra (singleton)
camera = None

# ---------------------------------------------------------------------------
# Fonctions
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

def capture_image(pictures_dir: Path) -> tuple[np.ndarray, Path]:
    """Capture une image, la sauvegarde et la retourne en tant que np.ndarray"""
    global camera
    
    if camera is None:
        initialize_camera()
    
    try:
        # Créer le répertoire s'il n'existe pas
        pictures_dir.mkdir(parents=True, exist_ok=True)
        
        # Générer le nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{timestamp}_capture.jpg"
        filepath = pictures_dir / filename
        
        # Capturer l'image en mémoire (array) et la sauvegarder
        image_array = camera.capture_array()
        camera.capture_file(str(filepath)) # On la sauvegarde pour le debug
        
        return image_array, filepath
        
    except Exception as e:
        raise Exception(f"Erreur lors de la capture: {e}")
