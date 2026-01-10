#!/usr/bin/env python3
"""
emulated_camera.py
An emulated camera that reads images from a specified folder.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import cv2
import logging

# ---------------------------------------------------------------------------
# Classe
# ---------------------------------------------------------------------------

logger = logging.getLogger("emulated_camera")


class EmulatedCamera:
    """
    An emulated camera that reads images from a specified folder.
    """

    def __init__(self, w: int, h: int, image_folder: Path):
        """Initializes the emulated camera."""
        logger.info(f"Initializing the emulated camera from folder: {image_folder}")
        self.width = w
        self.height = h
        self.image_folder = image_folder
        self.image_files = sorted(
            [
                p
                for p in self.image_folder.glob("*")
                if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
            ]
        )

        if not self.image_files:
            raise FileNotFoundError(
                f"Aucune image trouvée dans le dossier: {self.image_folder}"
            )

        self.current_image_index = 0
        logger.info(f"Fausse caméra initialisée avec {len(self.image_files)} images.")

    def capture_array(self) -> np.ndarray:
        """'Capture' une image en lisant un fichier et la retourne en tant que np.ndarray (format RGB)."""
        try:
            source_path = self.image_files[self.current_image_index]
            image_array = cv2.imread(str(source_path))  # cv2 lit en BGR

            # Convertir BGR en RGB pour cohérence avec PiCamera
            image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            self.current_image_index = (self.current_image_index + 1) % len(
                self.image_files
            )
            logger.info(f"Image 'capturée' du tableau: {source_path.name}")

            return image_array_rgb
        except Exception as e:
            logger.error(f"Erreur lors de la capture simulée du tableau: {e}")
            raise Exception(f"Erreur lors de la capture simulée du tableau: {e}")

    def capture_image(self, pictures_dir: Path) -> tuple[np.ndarray, Path]:
        """'Capture' une image en lisant un fichier directement sans la sauvegarder."""
        try:
            source_path = self.image_files[self.current_image_index]

            # Lire l'image directement depuis le dossier source
            image_array = cv2.imread(str(source_path))  # cv2 lit en BGR

            # Convertir BGR en RGB pour cohérence avec PiCamera
            image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            self.current_image_index = (self.current_image_index + 1) % len(
                self.image_files
            )
            logger.info(
                f"Image 'capturée' depuis: {source_path.name} (pas de copie sauvegardée)"
            )

            return image_array_rgb, source_path
        except Exception as e:
            logger.error(f"Erreur lors de la capture simulée: {e}")
            raise Exception(f"Erreur lors de la capture simulée: {e}")

    def close(self):
        """Ferme et nettoie la fausse caméra (pas d'opération réelle nécessaire)."""
        logger.info("Fausse caméra fermée.")
