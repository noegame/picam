#!/usr/bin/env python3
"""
Classe pour la gestion de la caméra PiCamera2.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import logging

# ---------------------------------------------------------------------------
# Classe
# ---------------------------------------------------------------------------

logger = logging.getLogger('camera')

class PiCamera:
    """Wrapper pour la caméra PiCamera2."""
    
    def __init__(self, w: int, h: int):
        """Initialise la caméra PiCamera2."""
        try:
            from picamera2 import Picamera2
            logger.info("Initialisation de la caméra...")
            self.camera = Picamera2()
            camera_config = self.camera.create_still_configuration(
                main={"size": (w, h)}
            )
            self.camera.configure(camera_config)
            self.camera.start()
            logger.info("Caméra initialisée avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la caméra: {e}")
            raise Exception(f"Erreur lors de l'initialisation de la caméra: {e}")

    def capture_image(self, pictures_dir: Path) -> tuple[np.ndarray, Path]:
        """Capture une image, la sauvegarde et la retourne en tant que np.ndarray"""
        try:
            pictures_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}_capture.jpg"
            filepath = pictures_dir / filename
            image_array = self.camera.capture_array()
            self.camera.capture_file(str(filepath))
            logger.info(f"Image capturée: {filepath.name}")
            return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), filepath
        except Exception as e:
            logger.error(f"Erreur lors de la capture: {e}")
            raise Exception(f"Erreur lors de la capture: {e}")
