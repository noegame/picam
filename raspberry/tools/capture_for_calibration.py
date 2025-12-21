#!/usr/bin/env python3
"""
Script simple pour capturer des images pour la calibration de la caméra.

Ce script utilise la configuration pour obtenir la résolution,
initialise la caméra en mode still (haute qualité),
et capture une image chaque fois que l'utilisateur appuie sur [Entrée].

Les images sont sauvegardées dans 'output/calibration' avec la date et la résolution dans le nom.

Appuyez sur Ctrl+C pour arrêter le script.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import sys
from pathlib import Path
from datetime import datetime

from vision_python.src.camera.camera_factory import get_camera
from vision_python.config.env_loader import EnvConfig

# Load environment configuration
EnvConfig()


def setup_logging():
    """Configure le logging pour l'application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def main():
    """Fonction principale du script de capture."""
    setup_logging()
    logger = logging.getLogger("capture_for_calibration")

    repo_root = Path(__file__).resolve().parents[2]

    try:
        # Charger la configuration depuis .env
        logger.info("Chargement de la configuration...")
        image_width = EnvConfig.get_camera_width()
        image_height = EnvConfig.get_camera_height()
        logger.info(f"Résolution configurée : {image_width}x{image_height}")

        # Créer le dossier de sortie
        output_path = repo_root / "output" / "calibration"
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Les images seront sauvegardées dans : {output_path}")

        # Initialiser la caméra en mode still (haute qualité)
        logger.info(
            f"Initialisation de la caméra avec une résolution de {image_width}x{image_height}..."
        )
        cam = get_camera(
            w=image_width, h=image_height, config_mode="still", use_fake_camera=False
        )
        logger.info("✓ Caméra initialisée en mode STILL (haute qualité)")

        logger.info("=" * 60)
        logger.info("Appuyez sur [Entrée] pour capturer une image")
        logger.info("Appuyez sur Ctrl+C pour quitter")
        logger.info("=" * 60)

        # Boucle pour attendre l'input de l'utilisateur
        capture_count = 0
        while True:
            input()  # Attend que l'utilisateur appuie sur Entrée
            capture_count += 1
            logger.info(f"Capture #{capture_count} en cours...")

            try:
                # Créer le timestamp et le nom de fichier avec la résolution
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{timestamp}_{image_width}x{image_height}_capture.jpg"
                filepath = output_path / filename

                # Capturer et sauvegarder l'image
                _, returned_filepath = cam.capture_image(pictures_dir=output_path)

                # Renommer avec le format souhaité
                returned_filepath.rename(filepath)

                logger.info(f"✓ Image capturée : {filename}")

            except Exception as e:
                logger.error(f"✗ Erreur lors de la capture : {e}")

    except KeyboardInterrupt:
        logger.info("\nArrêt du script par l'utilisateur.")
    except Exception as e:
        logger.error(f"Une erreur fatale est survenue : {e}")
    finally:
        if cam is not None:
            try:
                cam.close()
            except Exception as e:
                logger.debug(f"Erreur lors de l'arrêt de la caméra : {e}")
        logger.info("Caméra arrêtée.")


if __name__ == "__main__":
    main()
