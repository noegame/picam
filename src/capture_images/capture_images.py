#!/usr/bin/env python3
"""
Système PiCam - Capture automatique d'images
Prend une photo toutes les 15 secondes et les sauvegarde dans le dossier data/
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    try:
        import picamera
        PICAMERA_AVAILABLE = True
        LEGACY_PICAMERA = True
    except ImportError:
        PICAMERA_AVAILABLE = False
        LEGACY_PICAMERA = False
        import cv2  # Fallback pour les tests sans PiCamera

class PiCamCapture:
    def __init__(self, data_dir="data", interval=15):
        """
        Initialise le système de capture
        
        Args:
            data_dir (str): Répertoire de sauvegarde des images
            interval (int): Intervalle entre les captures en secondes
        """
        self.data_dir = Path(data_dir)
        self.interval = interval
        self.camera = None
        self.running = False
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Créer le répertoire data s'il n'existe pas
        self.data_dir.mkdir(exist_ok=True)
        
        self.setup_camera()
    
    def setup_camera(self):
        """Configure la caméra selon la disponibilité"""
        try:
            if PICAMERA_AVAILABLE:
                if 'LEGACY_PICAMERA' in globals() and LEGACY_PICAMERA:
                    # PiCamera legacy (Raspberry Pi OS Bullseye et antérieur)
                    import picamera
                    self.camera = picamera.PiCamera()
                    self.camera.resolution = (1920, 1080)
                    self.camera.framerate = 30
                    self.camera_type = "picamera_legacy"
                    self.logger.info("PiCamera legacy initialisée")
                else:
                    # PiCamera2 (Raspberry Pi OS Bookworm et ultérieur)
                    self.camera = Picamera2()
                    camera_config = self.camera.create_still_configuration(
                        main={"size": (1920, 1080)}
                    )
                    self.camera.configure(camera_config)
                    self.camera.start()
                    self.camera_type = "picamera2"
                    self.logger.info("PiCamera2 initialisée")
            else:
                # Fallback avec OpenCV (pour tests)
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.camera_type = "opencv"
                self.logger.warning("PiCamera non disponible, utilisation d'OpenCV")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de la caméra: {e}")
            raise
    
    def capture_image(self):
        """Capture une image et la sauvegarde"""
        try:
            # Générer le nom de fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"picam_{timestamp}.jpg"
            filepath = self.data_dir / filename
            
            if self.camera_type == "picamera2":
                # PiCamera2
                self.camera.capture_file(str(filepath))
                
            elif self.camera_type == "picamera_legacy":
                # PiCamera legacy
                self.camera.capture(str(filepath))
                
            elif self.camera_type == "opencv":
                # OpenCV fallback
                ret, frame = self.camera.read()
                if ret:
                    cv2.imwrite(str(filepath), frame)
                else:
                    raise Exception("Impossible de capturer l'image avec OpenCV")
            
            self.logger.info(f"Image capturée: {filename}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la capture: {e}")
            return None
    
    def start_capture_loop(self):
        """Démarre la boucle de capture automatique"""
        self.running = True
        self.logger.info(f"Démarrage de la capture automatique (intervalle: {self.interval}s)")
        
        try:
            while self.running:
                # Capturer une image
                filepath = self.capture_image()
                
                if filepath:
                    self.logger.info(f"Prochaine capture dans {self.interval} secondes...")
                else:
                    self.logger.warning("Échec de la capture, nouvelle tentative...")
                
                # Attendre l'intervalle spécifié
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            self.logger.info("Interruption détectée (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Erreur dans la boucle de capture: {e}")
        finally:
            self.stop_capture()
    
    def stop_capture(self):
        """Arrête la capture et libère les ressources"""
        self.running = False
        
        try:
            if self.camera:
                if self.camera_type == "picamera2":
                    self.camera.stop()
                elif self.camera_type == "picamera_legacy":
                    self.camera.close()
                elif self.camera_type == "opencv":
                    self.camera.release()
                    
            self.logger.info("Système de capture arrêté")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'arrêt: {e}")
    
    def get_stats(self):
        """Retourne les statistiques de capture"""
        if not self.data_dir.exists():
            return {"total_images": 0, "disk_usage": "0 MB"}
        
        images = list(self.data_dir.glob("picam_*.jpg"))
        total_size = sum(img.stat().st_size for img in images)
        
        return {
            "total_images": len(images),
            "disk_usage": f"{total_size / (1024*1024):.1f} MB",
            "data_directory": str(self.data_dir.absolute())
        }


def main():
    """Fonction principale"""
    print("=" * 50)
    print("Système PiCam - Capture automatique")
    print("=" * 50)
    
    try:
        # Créer l'instance de capture
        picam = PiCamCapture(data_dir="../data", interval=5)
        
        # Afficher les statistiques initiales
        stats = picam.get_stats()
        print(f"Répertoire de sauvegarde: {stats['data_directory']}")
        print(f"Images existantes: {stats['total_images']}")
        print(f"Espace utilisé: {stats['disk_usage']}")
        print()
        print("Appuyez sur Ctrl+C pour arrêter la capture")
        print("-" * 50)
        
        # Démarrer la capture
        picam.start_capture_loop()
        
    except Exception as e:
        print(f"Erreur: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())