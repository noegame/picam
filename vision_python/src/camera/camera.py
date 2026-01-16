#!/usr/bin/env python3
"""
Module de gestion des caméras.
Définit la classe abstraite Camera et l'implémentation PiCamera.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np
import logging
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

logger = logging.getLogger("camera")


class Camera(ABC):
    """
    Classe abstraite définissant l'interface commune pour tous les types de caméras.

    Cette classe sert de base pour toutes les implémentations de caméras
    (PiCamera, Webcam, EmulatedCamera) et garantit une interface uniforme.
    """

    @abstractmethod
    def init(self) -> None:
        """
        Initialise le matériel de la caméra.

        Cette méthode doit configurer et préparer la caméra pour la capture.
        Lève une exception en cas d'échec d'initialisation.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """
        Démarre le flux vidéo de la caméra.

        Active la caméra et commence l'acquisition d'images.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Arrête le flux vidéo et libère les ressources.

        Ferme proprement la caméra et nettoie les ressources allouées.
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Configure les paramètres de la caméra.

        :param parameters: Dictionnaire contenant les paramètres à configurer
                          (ex: exposition, contraste, balance des blancs, etc.)
        """
        pass

    @abstractmethod
    def capture_photo(self) -> np.ndarray:
        """
        Capture une photo et la retourne.

        :return: Image capturée sous forme de tableau numpy (format RGB)
        """
        pass
