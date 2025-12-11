#!/usr/bin/env python3
"""
Factory pour créer et retourner une instance de caméra (réelle ou simulée).
"""

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("camera_factory")


def get_camera(
    w: int,
    h: int,
    use_fake_camera: bool = False,
    config_mode: str = "preview",
    fake_camera_image_folder: Optional[Path] = None,
    allow_fallback: bool = True,
):
    """
    Retourne une instance de caméra.

    :param w: Largeur de l'image.
    :param h: Hauteur de l'image.
    :param use_fake_camera: Si True, retourne une instance de FakeCamera.
    :param config_mode: Mode de configuration - "preview" (streaming) ou "still" (captures uniques).
    :param fake_camera_image_folder: Dossier contenant les images pour la FakeCamera.
    :param allow_fallback: Si True et que la caméra réelle échoue, bascule automatiquement vers FakeCamera.
    :return: Une instance de caméra (PiCamera ou FakeCamera).
    """
    if use_fake_camera:
        from .fake_camera import FakeCamera

        if fake_camera_image_folder is None:
            raise ValueError(
                "Le dossier d'images pour la fausse caméra doit être spécifié."
            )
        return FakeCamera(w=w, h=h, image_folder=fake_camera_image_folder)
    else:
        from .camera import PiCamera

        try:
            return PiCamera(w=w, h=h, config_mode=config_mode)
        except (ImportError, Exception) as e:
            if allow_fallback:
                logger.warning(
                    f"Impossible d'initialiser la caméra réelle: {e}. "
                    f"Basculement vers la fausse caméra..."
                )
                if fake_camera_image_folder is None:
                    raise ValueError(
                        "La caméra réelle n'est pas disponible et aucun dossier d'images pour "
                        "la fausse caméra n'a été spécifié. Spécifiez 'fake_camera_image_folder' "
                        "ou installez libcamera sur le système."
                    )
                from .fake_camera import FakeCamera

                return FakeCamera(w=w, h=h, image_folder=fake_camera_image_folder)
            else:
                raise
