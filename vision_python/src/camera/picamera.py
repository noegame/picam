"""
Module de gestion de la caméra Raspberry Pi via picamera2.
Implémente la classe PiCamera dérivée de Camera.
"""

# ---------------------------------------------------------------------------
# Importations
# ---------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import cv2
import logging
from typing import Dict, Any
from datetime import datetime
from .camera import Camera
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Classe
# ---------------------------------------------------------------------------

logger = logging.getLogger("picamera")


class PiCamera(Camera):
    """
    Implémentation de Camera pour le module caméra du Raspberry Pi (PiCamera2).

    Gère la caméra du Raspberry Pi via la bibliothèque picamera2.
    """

    def __init__(self):
        """
        Initialise la caméra PiCamera2.

        Note: Les paramètres (width, height, config_mode) doivent être configurés
        via set_parameters() avant d'appeler init().
        """
        self.width = None
        self.height = None
        self.config_mode = "preview"  # Valeur par défaut
        self.camera = None

    def init(self) -> None:
        """Initialise le matériel de la caméra Raspberry Pi."""
        if self.width is None or self.height is None:
            raise ValueError(
                "width et height doivent être configurés via set_parameters() avant l'initialisation"
            )

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

            if self.config_mode == "still":
                logger.info(f"Mode: STILL (captures uniques optimisées)")
                camera_config = self.camera.create_still_configuration(
                    main={"size": (self.width, self.height)}
                )
            else:  # "preview" by default
                logger.info(f"Mode: PREVIEW (streaming continu)")
                camera_config = self.camera.create_preview_configuration(
                    main={"format": "XRGB8888", "size": (self.width, self.height)}
                )

            self.camera.configure(camera_config)
            logger.info("Caméra initialisée avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la caméra: {e}")
            raise Exception(f"Erreur lors de l'initialisation de la caméra: {e}")

    def start(self) -> None:
        """Démarre le flux vidéo de la caméra."""
        try:
            if self.camera:
                self.camera.start()
                logger.info("Flux vidéo démarré.")
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de la caméra: {e}")
            raise

    def stop(self) -> None:
        """Arrête le flux vidéo et libère les ressources."""
        try:
            if hasattr(self, "camera") and self.camera is not None:
                self.camera.stop()
                self.camera.close()
                logger.info("Caméra fermée correctement.")
            else:
                logger.warning("Caméra n'était pas initialisée, rien à fermer.")
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture de la caméra: {e}")

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Configure les paramètres de la caméra.

        :param parameters: Dictionnaire de paramètres
                          Paramètres de configuration:
                          - width (int): Largeur de l'image
                          - height (int): Hauteur de l'image
                          - config_mode (str): Mode de configuration ("preview" ou "still")

                          Contrôles caméra (si la caméra est déjà initialisée):
                          - ExposureTime, AnalogueGain, etc.
        """
        # Paramètres de configuration (avant init)
        config_params = {"width", "height", "config_mode"}
        control_params = {}

        for key, value in parameters.items():
            if key == "width":
                self.width = value
                logger.info(f"Largeur configurée: {self.width}")
            elif key == "height":
                self.height = value
                logger.info(f"Hauteur configurée: {self.height}")
            elif key == "config_mode":
                if value not in ["preview", "still"]:
                    logger.warning(f"Mode invalide '{value}', utilisation de 'preview'")
                    value = "preview"
                self.config_mode = value
                logger.info(f"Mode configuré: {self.config_mode}")
            else:
                # Contrôles caméra (ExposureTime, AnalogueGain, etc.)
                control_params[key] = value

        # Appliquer les contrôles caméra si la caméra est initialisée
        if control_params:
            try:
                if self.camera:
                    self.camera.set_controls(control_params)
                    logger.info(f"Contrôles caméra configurés: {control_params}")
                else:
                    logger.warning(
                        f"Caméra non initialisée, contrôles ignorés: {control_params}"
                    )
            except Exception as e:
                logger.error(f"Erreur lors de la configuration des contrôles: {e}")
                raise

    def capture_photo(self) -> np.ndarray:
        """
        Capture une photo et la retourne.

        :return: Image capturée en tant que np.ndarray (format RGB)
        """
        return self.capture_array()

    def capture_array(self) -> np.ndarray:
        """Capture une image et la retourne en tant que np.ndarray (format RGB)."""
        try:
            assert self.camera is not None, "La caméra n'est pas initialisée"
            return self.camera.capture_array()
        except Exception as e:
            logger.error(f"Erreur lors de la capture du tableau: {e}")
            raise Exception(f"Erreur lors de la capture du tableau: {e}")

    def capture_image(self, pictures_dir: Path) -> tuple[np.ndarray, Path]:
        """Capture une image, la sauvegarde et la retourne en tant que np.ndarray"""
        try:
            assert self.camera is not None, "La caméra n'est pas initialisée"
            pictures_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}.jpg"
            filepath = pictures_dir / filename
            image_array = self.capture_array()
            self.camera.capture_file(str(filepath))
            logger.info(f"Image capturée: {filepath.name}")
            return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), filepath
        except Exception as e:
            logger.error(f"Erreur lors de la capture: {e}")
            raise Exception(f"Erreur lors de la capture: {e}")

    def capture_png(self, pictures_dir: Path) -> tuple[np.ndarray, Path]:
        """
        Capture une photo et l'enregistre en format PNG.

        :param pictures_dir: Répertoire où enregistrer la photo
        :return: Tuple (image en tant que np.ndarray, chemin du fichier)
        """
        try:
            pictures_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}.png"
            filepath = pictures_dir / filename

            # Capturer l'image
            image_array = self.capture_array()

            # Sauvegarder en PNG avec cv2 pour un meilleur contrôle
            cv2.imwrite(str(filepath), cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

            logger.info(f"Image PNG capturée: {filepath.name}")
            return image_array, filepath
        except Exception as e:
            logger.error(f"Erreur lors de la capture PNG: {e}")
            raise Exception(f"Erreur lors de la capture PNG: {e}")

    def close(self):
        """Ferme et nettoie la caméra proprement (alias pour stop())."""
        self.stop()
