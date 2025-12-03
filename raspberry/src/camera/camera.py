#!/usr/bin/env python3
"""
Classe d'abstraction pour la caméra PiCamera2.
"""

import logging
import time
from pathlib import Path
from datetime import datetime

import picamera2


class PiCamera:
    """Wrapper pour la librairie picamera2."""

    def __init__(self, w: int, h: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.width = w
        self.height = h
        self.camera = picamera2.Picamera2()
        # Configure la caméra pour la capture d'images fixes (haute résolution)
        # et pour le streaming (basse résolution, format compatible avec l'encodage rapide)
        config = self.camera.create_still_configuration(
            main={"size": (self.width, self.height)},
            lores={"size": (640, 480), "format": "YUV420"},
            controls={"FrameDurationLimits": (33333, 33333)} # Vise ~30fps pour le stream
        )
        self.camera.configure(config)
        self.logger.info(f"Caméra initialisée avec une résolution de {w}x{h}.")

    def start(self):
        """Démarre la caméra."""
        self.camera.start()
        self.logger.info("Caméra démarrée.")
        time.sleep(1) # Laisse le temps au capteur de s'initialiser

    def stop(self):
        """Arrête la caméra."""
        self.camera.stop()
        self.logger.info("Caméra arrêtée.")

    def capture_image(self, pictures_dir: Path) -> tuple[object, Path]:
        """Capture une image en haute résolution et la sauvegarde."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = pictures_dir / f"picam_{timestamp}.jpg"
        
        # La méthode capture_file utilise la configuration 'main' (haute résolution)
        self.camera.capture_file(str(filepath))
        self.logger.info(f"Image haute résolution sauvegardée dans {filepath}")
        
        # Pour retourner l'image, il faudrait la lire depuis le fichier,
        # ce qui est inefficace. Les clients actuels ne semblent pas utiliser l'image retournée.
        # On retourne None pour l'instant.
        return None, filepath

    def capture_array(self, stream_name: str = 'lores') -> object:
        """Capture une image basse résolution pour le streaming et la retourne comme un array numpy."""
        # La méthode capture_array utilise par défaut le stream 'main'.
        # On spécifie 'lores' pour obtenir l'image basse résolution, plus rapide.
        return self.camera.capture_array(stream_name)