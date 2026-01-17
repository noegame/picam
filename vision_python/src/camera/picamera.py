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

    def init(self) -> None:
        """Initialise le matériel de la caméra Raspberry Pi."""
        from picamera2 import Picamera2

        self.picamera2 = Picamera2()
        self.parameters = {}

    def start(self) -> None:
        """Démarre le flux vidéo de la caméra."""
        try:
            if self.picamera2:
                self.picamera2.start()
                logger.info("Flux vidéo démarré.")
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de la caméra: {e}")
            raise

    def stop(self) -> None:
        """Arrête le flux vidéo et libère les ressources."""
        try:
            if hasattr(self, "camera") and self.picamera2 is not None:
                self.picamera2.stop()
                self.picamera2.close()
                logger.info("Caméra fermée correctement.")
            else:
                logger.warning("Caméra n'était pas initialisée, rien à fermer.")
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture de la caméra: {e}")

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Configure les paramètres de la caméra.

        - output_folder (str): Dossier de sortie pour les images capturées
        - width (int): Largeur de l'image
        - height (int): Hauteur de l'image
        - config_mode (str): Mode de configuration ("preview" ou "still")
        - save_format (str): Format de sauvegarde des images ("jpeg", "png", etc.)

        Contrôles caméra (si la caméra est déjà initialisée):
        - ExposureTime, AnalogueGain, etc.
        """
        control_params = {}
        needs_reconfigure = False
        self.parameters = parameters

        for key, value in parameters.items():

            if key == "width":
                if self.parameters.get("width") != value:
                    self.parameters["width"] = value
                    needs_reconfigure = True
                logger.info(f"Largeur configurée: {self.parameters.get('width')}")

            elif key == "height":
                if self.parameters.get("height") != value:
                    self.parameters["height"] = value
                    needs_reconfigure = True
                logger.info(f"Hauteur configurée: {self.parameters.get('height')}")

            elif key == "output_folder":
                self.parameters["output_folder"] = value
                logger.info(
                    f"Dossier de sortie configuré: {self.parameters.get('output_folder')}"
                )

            elif key == "save_format":
                self.parameters["save_format"] = value
                logger.info(
                    f"Format de sauvegarde configuré: {self.parameters.get('save_format')}"
                )

            elif key == "config_mode":
                if value not in ["preview", "still"]:
                    logger.warning(f"Mode invalide '{value}', utilisation de 'preview'")
                    value = "preview"
                if self.parameters.get("config_mode") != value:
                    self.parameters["config_mode"] = value
                    needs_reconfigure = True
                logger.info(f"Mode configuré: {self.parameters.get('config_mode')}")
            else:
                # Contrôles caméra (ExposureTime, AnalogueGain, etc.)
                control_params[key] = value

        # Configurer la caméra si nécessaire et si elle est initialisée
        if needs_reconfigure and self.picamera2:
            try:
                if self.parameters.get("config_mode") == "still":
                    logger.info(f"Mode: STILL (captures uniques optimisées)")
                    camera_config = self.picamera2.create_still_configuration(
                        main={
                            "size": (
                                self.parameters.get("width"),
                                self.parameters.get("height"),
                            )
                        }
                    )
                else:  # "preview" by default
                    logger.info(f"Mode: PREVIEW (streaming continu)")
                    camera_config = self.picamera2.create_preview_configuration(
                        main={
                            "format": "XRGB8888",
                            "size": (
                                self.parameters.get("width"),
                                self.parameters.get("height"),
                            ),
                        }
                    )
                self.picamera2.configure(camera_config)
                logger.info("Configuration de la caméra mise à jour.")
            except Exception as e:
                logger.error(f"Erreur lors de la configuration de la caméra: {e}")
                raise

        # Appliquer les contrôles caméra si la caméra est initialisée
        if control_params:
            try:
                if self.picamera2:
                    self.picamera2.set_controls(control_params)
                    logger.info(f"Contrôles caméra configurés: {control_params}")
                else:
                    logger.warning(
                        f"Caméra non initialisée, contrôles ignorés: {control_params}"
                    )
            except Exception as e:
                logger.error(f"Erreur lors de la configuration des contrôles: {e}")
                raise

    def take_picture(self) -> np.ndarray:
        """
        Capture une photo et la retourne.

        :return: Image capturée en tant que np.ndarray (format RGB)
        """

        # Build filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{timestamp}.{self.parameters.get('save_format', 'jpg')}"
        output_folder = self.parameters.get("output_folder", ".")
        pictures_dir = Path(output_folder)
        pictures_dir.mkdir(parents=True, exist_ok=True)
        filepath = pictures_dir / filename

        # Capture image
        picture = self.picamera2.capture_array()

        # Sauvegarde de l'image
        try:
            if self.parameters.get("save_format", "jpg").lower() == "png":
                cv2.imwrite(
                    str(filepath),
                    cv2.cvtColor(picture, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 0],  # 0 for no compression
                )
            else:  # Default to JPEG
                cv2.imwrite(
                    str(filepath),
                    cv2.cvtColor(picture, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 100],  # 100 for best quality
                )
            logger.info(f"Image sauvegardée: {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'image: {e}")
            raise

        return picture

    def close(self):
        """Ferme et nettoie la caméra proprement (alias pour stop())."""
        self.stop()
