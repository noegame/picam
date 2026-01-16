#!/usr/bin/env python3
"""
Module pour la gestion des webcams USB standard.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import cv2
import logging
from typing import Dict, Any

from .camera import Camera

# ---------------------------------------------------------------------------
# Classe
# ---------------------------------------------------------------------------

logger = logging.getLogger("webcam")


class Webcam(Camera):
    """
    Implémentation de Camera pour les webcams USB standard.

    Utilise OpenCV pour capturer des images depuis une webcam USB.
    """

    def __init__(self):
        """
        Initialise la webcam.

        Note: Les paramètres (width, height, device_id) doivent être configurés
        via set_parameters() avant d'appeler init().
        """
        self.width = None
        self.height = None
        self.device_id = 0  # Valeur par défaut
        self.capture = None

    def init(self) -> None:
        """Initialise la webcam USB."""
        if self.width is None or self.height is None:
            raise ValueError(
                "width et height doivent être configurés via set_parameters() avant l'initialisation"
            )

        try:
            logger.info(f"Initialisation de la webcam (device {self.device_id})...")
            self.capture = cv2.VideoCapture(self.device_id)

            if not self.capture.isOpened():
                raise Exception(
                    f"Impossible d'ouvrir la webcam avec l'ID {self.device_id}"
                )

            # Configuration de la résolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Vérification de la résolution effective
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width != self.width or actual_height != self.height:
                logger.warning(
                    f"Résolution demandée ({self.width}x{self.height}) "
                    f"différente de la résolution obtenue ({actual_width}x{actual_height})"
                )

            logger.info(
                f"Webcam initialisée avec succès (résolution: {actual_width}x{actual_height})."
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la webcam: {e}")
            raise

    def start(self) -> None:
        """
        Démarre le flux vidéo de la webcam.

        Note: Pour OpenCV VideoCapture, le flux est déjà actif après init().
        """
        if self.capture is None or not self.capture.isOpened():
            logger.warning("La webcam n'est pas initialisée. Réinitialisation...")
            self.init()
        else:
            logger.info("Flux vidéo de la webcam déjà actif.")

    def stop(self) -> None:
        """Arrête le flux vidéo et libère les ressources."""
        try:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
                logger.info("Webcam fermée correctement.")
            else:
                logger.warning("Webcam n'était pas initialisée, rien à fermer.")
        except Exception as e:
            logger.warning(f"Erreur lors de la fermeture de la webcam: {e}")

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Configure les paramètres de la webcam.

        :param parameters: Dictionnaire de paramètres
                          Paramètres de configuration:
                          - width (int): Largeur de l'image
                          - height (int): Hauteur de l'image
                          - device_id (int): ID du périphérique de la webcam

                          Propriétés OpenCV (si la webcam est déjà initialisée):
                          - brightness, contrast, saturation, exposure, fps, gain, auto_exposure
        """
        # Paramètres de configuration (avant init)
        config_params = {"width", "height", "device_id"}
        opencv_params = {}

        for key, value in parameters.items():
            if key == "width":
                self.width = value
                logger.info(f"Largeur configurée: {self.width}")
            elif key == "height":
                self.height = value
                logger.info(f"Hauteur configurée: {self.height}")
            elif key == "device_id":
                self.device_id = value
                logger.info(f"Device ID configuré: {self.device_id}")
            else:
                # Propriétés OpenCV
                opencv_params[key] = value

        # Appliquer les propriétés OpenCV si la webcam est initialisée
        if opencv_params:
            try:
                if self.capture is None or not self.capture.isOpened():
                    logger.warning(
                        f"Webcam non initialisée, propriétés ignorées: {opencv_params}"
                    )
                    return

                # Mapping des noms de paramètres lisibles vers les propriétés OpenCV
                param_mapping = {
                    "brightness": cv2.CAP_PROP_BRIGHTNESS,
                    "contrast": cv2.CAP_PROP_CONTRAST,
                    "saturation": cv2.CAP_PROP_SATURATION,
                    "exposure": cv2.CAP_PROP_EXPOSURE,
                    "fps": cv2.CAP_PROP_FPS,
                    "gain": cv2.CAP_PROP_GAIN,
                    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
                }

                for param_name, value in opencv_params.items():
                    if param_name in param_mapping:
                        prop_id = param_mapping[param_name]
                        success = self.capture.set(prop_id, value)
                        if success:
                            logger.info(f"Paramètre '{param_name}' configuré à {value}")
                        else:
                            logger.warning(
                                f"Impossible de configurer le paramètre '{param_name}'"
                            )
                    else:
                        logger.warning(f"Paramètre inconnu: '{param_name}'")

            except Exception as e:
                logger.error(f"Erreur lors de la configuration des paramètres: {e}")
                raise

    def capture_photo(self) -> np.ndarray:
        """
        Capture une photo et la retourne.

        :return: Image capturée en tant que np.ndarray (format RGB)
        """
        return self.capture_array()

    def capture_array(self) -> np.ndarray:
        """
        Capture une image depuis la webcam.

        :return: Image en tant que np.ndarray (format RGB)
        """
        try:
            if self.capture is None:
                raise Exception("La webcam n'est pas initialisée.")

            ret, frame = self.capture.read()

            if not ret:
                raise Exception("Échec de la capture d'image depuis la webcam")

            # OpenCV lit les images en BGR, on convertit en RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return frame_rgb

        except Exception as e:
            logger.error(f"Erreur lors de la capture d'image: {e}")
            raise

    def close(self):
        """Ferme et nettoie la webcam proprement (alias pour stop())."""
        self.stop()
