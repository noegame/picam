#!/usr/bin/env python3
"""
camera_factory.py
Factory module to get camera instances
"""

import logging
from vision_python.config import config

logger = logging.getLogger("camera_factory")


def get_camera(w: int, h: int, camera: config.CameraMode, camera_param=None):
    """
    Retourne une instance de caméra.

    :param w: img width.
    :param h: img height.
    :param camera: type of camera to use (raspberry, computer, emulated).
    :param camera_param: paramètre spécifique à la caméra :
        - For Raspberry Pi camera: mode de configuration (ex: "still", "video")
        - For emulated camera: path to the image folder
    :return: A camera instance (PiCamera or FakeCamera).
    """
    if camera == config.CameraMode.EMULATED:
        from .emulated_camera import EmulatedCamera

        if camera_param is None:
            raise ValueError(
                "Le dossier d'images pour la fausse caméra doit être spécifié via camera_param."
            )
        return EmulatedCamera(w=w, h=h, image_folder=camera_param)
    else:
        from .camera import PiCamera

        # Pour Raspberry Pi, camera_param peut être "still" ou autre mode
        config_mode = camera_param if camera_param is not None else "still"

        try:
            return PiCamera(w=w, h=h, config_mode=config_mode)
        except (ImportError, Exception) as e:
            logger.error(f"Unable to initialize the real camera: {e}. ")
            raise
