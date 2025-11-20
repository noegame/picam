#!/usr/bin/env python3

"""

"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from picamera2 import Picamera2

class Camera:

    def __init__(self, pict_dir=None, resolution = (1920,1080)):
        """
        Initialise le système de capture
        
        Args:
            data_dir (str): Répertoire de sauvegarde des images
            interval (int): Intervalle entre les captures en secondes
        """
        # Si aucun répertoire passé, créer un dossier pictures
        if pict_dir:
            self.pictures_dir = Path("pictures")
        else:
            # __file__ est c:\...\picam\src\capture\capture.py
            # parents[2] -> c:\...\picam
            repo_root = Path(__file__).resolve().parents[2]
            self.pictures_dir = repo_root / "pictures"
        
        self.camera: Optional[Picamera2] = None
        self.running = False
        self.resolution = resolution

        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Créer le répertoire pictures s'il n'existe pas (avec parents)
        self.pictures_dir.mkdir(parents=True, exist_ok=True)

        self.setup_camera()

    def setup_camera(self):
        """Configure la caméra PiCamera2"""
        try:
            self.camera = Picamera2()
            camera_config = self.camera.create_still_configuration(
                main={"size": (self.resolution)}
            )
            self.camera.configure(camera_config)
            self.camera.start()
            self.logger.info("PiCamera2 initialisée avec succès")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de la caméra: {e}")
            raise

    def capture_image(self) -> Optional[Path]:
        """Capture une image et la sauvegarde"""
        try:
            if not self.camera:
                raise Exception("Caméra non initialisée")
                
            # Générer le nom de fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_capture_python.jpg"
            filepath = self.pictures_dir / filename
            
            # Capture avec PiCamera2
            self.camera.capture_file(str(filepath))
            
            self.logger.info(f"Image capturée: {filename}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la capture: {e}")
            return None