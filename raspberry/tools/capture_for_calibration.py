#!/usr/bin/env python3
"""
Script pour capturer des images pour la calibration de la caméra.

Ce script prend des photos en boucle en utilisant la camera_factory
et les enregistre dans le dossier 'output/calibration'.
La résolution (largeur et hauteur) est lue depuis un fichier de configuration.
"""

import configparser
import logging
import sys
import time
from pathlib import Path

# Ajoute le répertoire 'src' au chemin de recherche des modules pour trouver camera_factory
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / "raspberry" / "src"))

from camera.camera_factory import get_camera


def setup_logging():
    """Configure le logging pour l'application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )


def main():
    """Fonction principale du script de capture."""
    setup_logging()
    logger = logging.getLogger('capture_for_calibration')

    try:
        # Chemin vers le fichier de configuration
        config_path = Path(__file__).parent / 'config.ini'
        if not config_path.exists():
            logger.error(f"Le fichier de configuration '{config_path}' est introuvable.")
            return

        # Lire la configuration
        config = configparser.ConfigParser()
        config.read(config_path)
        image_width = config.getint('camera', 'width')
        image_height = config.getint('camera', 'height')

        # Créer le dossier de sortie
        calibration_pictures_dir = repo_root / "output" / "calibration"
        calibration_pictures_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Les images seront sauvegardées dans : {calibration_pictures_dir}")

        # Initialiser la caméra
        logger.info(f"Initialisation de la caméra avec une résolution de {image_width}x{image_height}")
        cam = get_camera(w=image_width, h=image_height, use_fake_camera=False)

        logger.info("Démarrage de la boucle de capture. Appuyez sur Ctrl+C pour arrêter.")

        # Boucle de capture
        while True:
            try:
                logger.info("Capture d'une image...")
                _, filepath = cam.capture_image(pictures_dir=calibration_pictures_dir)
                logger.info(f"Image sauvegardée : {filepath.name}")
                logger.info("Attente de 1 secondes avant la prochaine capture...")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Erreur lors de la capture de l'image : {e}")
                time.sleep(1) # Attendre un peu avant de réessayer

    except KeyboardInterrupt:
        logger.info("\nArrêt du script par l'utilisateur.")
    except Exception as e:
        logger.error(f"Une erreur fatale est survenue : {e}")

if __name__ == "__main__":
    main()