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
