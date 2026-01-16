#!/usr/bin/env python3

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from datetime import datetime
import logging
import sys

from vision_python.config import config
from vision_python.src.camera.camera_factory import get_camera


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def main():
    """Main function of the capture script."""
    setup_logging()
    logger = logging.getLogger("capture_for_calibration")

    cam = None  # has to be initialized before the try block
    camera = (
        config.get_camera_mode()
    )  # type of camera from config(emulated, raspberry, computer)
    try:
        image_width, image_height = config.get_camera_resolution()
        output_path = config.get_output_directory() / "camera"

        # Initialiser la caméra en mode still (haute qualité)
        logger.info(
            f"Initialisation de la caméra avec une résolution de {image_width}x{image_height}..."
        )
        cam = get_camera(
            w=image_width,
            h=image_height,
            camera=camera,
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
                filename = (
                    f"{timestamp}_{image_width}x{image_height}_capture_debut_match.jpg"
                )
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
