#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aruco_detection_flow.py
# 1. Prends une photo avec la caméra ou importe une image depuis un fichier.
# 2. Corrige la distorsion de l'image en utilisant les paramètres de distorsion de la caméra.
# 3. Détecte les tags ArUco dans l'image corrigée.
# 4. Grace au 4 tags aruco fixes dont on connait la position dans le monde réel, calcule la transformation 
#    entre le repère de la caméra et le repère du monde réel.
# 5. Utilise cette transformation pour estimer la position et l'orientation de tout autre tag ArUco détecté dans l'image.

repère de coordonnées du monde réel (en mm)
repère de coordonnées de l'image (en pixels)
"""

import logging
import logging.config
from my_math import Point
from detect_aruco import detect_aruco
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Configuration du logging
# ---------------------------------------------------------------------------

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('aruco_detection_flow')

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# Coordonnées des TAGS ARUCO fixes dans le repère du monde réel (en mm)
A1 = Point(5.3, 5.3, 20)
B1 = Point(12.3, 5.3, 22)
C1 = Point(5.3, 21.2, 21)
D1 = Point(12.3, 21.2, 23)

# A1 = Point(600, 600, 20)
# B1 = Point(1400, 600, 22)
# C1 = Point(600, 2400, 21)
# D1 = Point(1400, 2400, 23)

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------
def test_logger():
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')

def capture_image(self) -> Optional[Path]:      # todo
    """Capture une image et la sauvegarde"""
    try:
        if not self.camera:
            raise Exception("Caméra non initialisée")
            
        # Générer le nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_capture_python.jpg"
        filepath = self.pictures_dir / filename
        
        # Capture avec PiCamera2
        self.camera.capture_file(str(filepath))
        
        self.logger.info(f"Image capturée: {filename}")
        return filepath
        
    except Exception as e:
        self.logger.error(f"Erreur lors de la capture: {e}")
        return None

def setup_camera():
    """Configure la caméra PiCamera2"""
    try:
        camera = Picamera2()
        camera_config = self.camera.create_still_configuration(
            main={"size": (1920, 1080)}
        )
        camera.configure(camera_config)
        camera.start()
        logger.info("PiCamera2 initialisée avec succès")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la caméra: {e}")
        raise


def main():
    
    # ========= Capture de l'image ==========
    
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (2200, 2200)}
    )
    picam2.configure(config)
    picam2.start()
    picam2.capture_file("photo_2200.jpg")
    picam2.stop()




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        pass
        # logger.exception("Erreur lors de l'exécution : %s", e)
        # sys.exit(1)
