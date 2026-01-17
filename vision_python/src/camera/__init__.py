"""
Camera Module - Implémentations de caméras basées sur POO.

Ce module fournit une hiérarchie de classes pour la gestion de différents types de caméras:
- Camera: Classe abstraite définissant l'interface commune
- PiCamera: Implémentation pour le module caméra du Raspberry Pi
- Webcam: Implémentation pour les webcams USB standard
- EmulatedCamera: Implémentation virtuelle pour le test/débogage

Utilisez camera_factory.get_camera() pour créer facilement des instances.
"""

from .camera import Camera
from .picamera import PiCamera
from .webcam import Webcam
from .emulated_camera import EmulatedCamera
from .camera_factory import get_camera

__all__ = [
    "Camera",
    "PiCamera",
    "Webcam",
    "EmulatedCamera",
    "get_camera",
]
