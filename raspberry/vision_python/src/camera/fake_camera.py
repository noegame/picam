#!/usr/bin/env python3
"""
Fonctions liées à la fausse caméra (émulation)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import logging
import shutil

# ---------------------------------------------------------------------------
# Classe
# ---------------------------------------------------------------------------

logger = logging.getLogger("fake_camera")


class FakeCamera:
    """
    Une fausse caméra qui lit des images depuis un dossier au lieu d'une vraie caméra.
    L'API est compatible avec la classe PiCamera.
    """

    def __init__(self, w: int, h: int, image_folder: Path):
        """Initialise la fausse caméra."""
        logger.info(
            f"Initialisation de la fausse caméra depuis le dossier: {image_folder}"
        )
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
        """'Capture' une image en lisant un fichier, la sauvegarde et la retourne."""
        try:
            # Créer le répertoire s'il n'existe pas
            pictures_dir.mkdir(parents=True, exist_ok=True)

            source_path = self.image_files[self.current_image_index]

            # Générer le nom de fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}_capture.jpg"
            filepath = pictures_dir / filename

            # Copier l'image et la charger
            shutil.copy(source_path, filepath)
            image_array = cv2.imread(str(filepath))  # cv2 lit en BGR

            # Convertir BGR en RGB pour cohérence avec PiCamera
            image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            self.current_image_index = (self.current_image_index + 1) % len(
                self.image_files
            )
            logger.info(
                f"Image 'capturée': {filepath.name} (source: {source_path.name})"
            )

            return image_array_rgb, filepath
        except Exception as e:
            logger.error(f"Erreur lors de la capture simulée: {e}")
            raise Exception(f"Erreur lors de la capture simulée: {e}")

    def close(self):
        """Ferme et nettoie la fausse caméra (pas d'opération réelle nécessaire)."""
        logger.info("Fausse caméra fermée.")
