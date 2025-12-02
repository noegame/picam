#!/usr/bin/env python3
"""
Factory pour créer et retourner une instance de caméra (réelle ou simulée).
"""

from pathlib import Path
from typing import Optional

def get_camera(w: int, h: int, use_fake_camera: bool = False, fake_camera_image_folder: Optional[Path] = None):
    """
    Retourne une instance de caméra.

    :param w: Largeur de l'image.
    :param h: Hauteur de l'image.
    :param use_fake_camera: Si True, retourne une instance de FakeCamera.
    :param fake_camera_image_folder: Dossier contenant les images pour la FakeCamera.
    :return: Une instance de caméra (PiCamera ou FakeCamera).
    """
    if use_fake_camera:
        from .fake_camera import FakeCamera
        if fake_camera_image_folder is None:
            raise ValueError("Le dossier d'images pour la fausse caméra doit être spécifié.")
        return FakeCamera(w=w, h=h, image_folder=fake_camera_image_folder)
    else:
        from .camera import PiCamera
        return PiCamera(w=w, h=h)