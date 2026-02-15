"""
Module de gestion de la caméra Raspberry Pi via picamera2.
Implémente la classe PiCamera dérivée de Camera.
"""

# ---------------------------------------------------------------------------
# Importations
# ---------------------------------------------------------------------------

import numpy as np
import logging
from typing import Dict, Any
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
        """Initialise l'instance PiCamera."""
        self.picamera2 = None
        self.parameters = {}

    def init(self) -> None:
        """Initialise le matériel de la caméra Raspberry Pi."""
        from picamera2 import Picamera2

        try:
            self.picamera2 = Picamera2()
            logger.info("PiCamera2 initialisée avec succès")
        except Exception as e:
            logger.error(f"Impossible d'initialiser PiCamera2: {e}")
            logger.error("Vérifiez que la caméra est bien connectée et activée")
            raise RuntimeError(f"Échec de l'initialisation de PiCamera2: {e}") from e
        
        # Configure camera if parameters were set before init()
        if self.parameters.get("width") and self.parameters.get("height"):
            try:
                width = self.parameters.get("width")
                height = self.parameters.get("height")
                config_mode = self.parameters.get("config_mode", "preview")
                
                if config_mode == "still":
                    logger.info(f"Configuring camera in STILL mode (BGR): {width}x{height}")
                    camera_config = self.picamera2.create_still_configuration(
                        main={
                            "format": "BGR888",
                            "size": (width, height)
                        }
                    )
                else:  # "preview" by default
                    logger.info(f"Configuring camera in PREVIEW mode (BGR): {width}x{height}")
                    camera_config = self.picamera2.create_preview_configuration(
                        main={
                            "format": "BGR888",
                            "size": (width, height),
                        }
                    )
                self.picamera2.configure(camera_config)
                logger.info(f"Camera configured with resolution {width}x{height}")
            except Exception as e:
                logger.error(f"Erreur lors de la configuration de la caméra: {e}")
                raise

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

        - width (int): Largeur de l'image
        - height (int): Hauteur de l'image
        - config_mode (str): Mode de configuration ("preview" ou "still")

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
                    logger.info(f"Mode: STILL (captures uniques optimisées, BGR)")
                    camera_config = self.picamera2.create_still_configuration(
                        main={
                            "format": "BGR888",
                            "size": (
                                self.parameters.get("width"),
                                self.parameters.get("height"),
                            )
                        }
                    )
                else:  # "preview" by default
                    logger.info(f"Mode: PREVIEW (streaming continu, BGR)")
                    camera_config = self.picamera2.create_preview_configuration(
                        main={
                            "format": "BGR888",
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
        Capture une photo et la retourne directement (sans sauvegarde).

        :return: Image capturée en tant que np.ndarray (format RGB)
        """
        try:
            if not self.picamera2:
                raise Exception("La caméra n'est pas initialisée.")

            # Capture directe de l'image
            picture = self.picamera2.capture_array()
            logger.debug(f"Image capturée: {picture.shape}")
            return picture

        except Exception as e:
            logger.error(f"Erreur lors de la capture d'image: {e}")
            raise

    def close(self):
        """Ferme et nettoie la caméra proprement (alias pour stop())."""
        self.stop()
