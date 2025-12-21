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

logger = logging.getLogger("camera")


class PiCamera:
    """Wrapper pour la caméra PiCamera2."""

    def __init__(self, w: int, h: int, config_mode: str = "preview"):
        """
        Initialise la caméra PiCamera2.

        :param w: Largeur de l'image
        :param h: Hauteur de l'image
        :param config_mode: Mode de configuration - "preview" (streaming continu) ou "still" (captures uniques)
        """
        try:
            from picamera2 import Picamera2
        except ImportError as ie:
            error_msg = (
                "Impossible d'importer picamera2. "
                "Assurez-vous que libcamera est installé et disponible sur le système. "
                "Si vous n'êtes pas sur un Raspberry Pi, utilisez fake_camera=True."
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from ie

        try:
            logger.info("Initialisation de la caméra...")
            self.camera = Picamera2()
            self.config_mode = config_mode

            if config_mode == "still":
                logger.info(f"Mode: STILL (captures uniques optimisées)")
                camera_config = self.camera.create_still_configuration(
                    main={"size": (w, h)}
                )
            else:  # "preview" by default
                logger.info(f"Mode: PREVIEW (streaming continu)")
                camera_config = self.camera.create_preview_configuration(
                    main={"format": "XRGB8888", "size": (w, h)}
                )

            self.camera.configure(camera_config)
            self.camera.start()
            logger.info("Caméra initialisée avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la caméra: {e}")
            raise Exception(f"Erreur lors de l'initialisation de la caméra: {e}")

    def capture_array(self) -> np.ndarray:
        """Capture une image et la retourne en tant que np.ndarray (format RGB)."""
        try:
            return self.camera.capture_array()
        except Exception as e:
            logger.error(f"Erreur lors de la capture du tableau: {e}")
            raise Exception(f"Erreur lors de la capture du tableau: {e}")

    def capture_image(self, pictures_dir: Path) -> tuple[np.ndarray, Path]:
        """Capture une image, la sauvegarde et la retourne en tant que np.ndarray"""
        try:
            pictures_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}_capture.jpg"
            filepath = pictures_dir / filename
            image_array = self.capture_array()
            self.camera.capture_file(str(filepath))
            logger.info(f"Image capturée: {filepath.name}")
            return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), filepath
        except Exception as e:
            logger.error(f"Erreur lors de la capture: {e}")
            raise Exception(f"Erreur lors de la capture: {e}")

    def close(self):
        """Ferme et nettoie la caméra proprement."""
        try:
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.stop()
                self.camera.close()
                logger.info("Caméra fermée correctement.")
            else:
                logger.warning("Caméra n'était pas initialisée, rien à fermer.")
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture de la caméra: {e}")
