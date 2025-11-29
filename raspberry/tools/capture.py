#!/usr/bin/env python3
"""
Capture automatique d'images
Prend une photo toutes les 5 secondes et les sauvegarde dans le dossier picam/pictures
Utilise la bibliothèque PiCamera2 (Raspberry Pi OS Bookworm et ultérieur)
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from picamera2 import Picamera2

class PiCamCapture:
    def __init__(self, pict_dir=None, interval=5):
        """
        Initialise le système de capture
        
        Args:
            data_dir (str): Répertoire de sauvegarde des images
            interval (int): Intervalle entre les captures en secondes
        """
        # Si aucun répertoire passé, utiliser le dossier repo_root/pictures
        if pict_dir:
            self.pictures_dir = Path(pict_dir)
        else:
            # __file__ est c:\...\picam\src\capture\capture.py
            # parents[2] -> c:\...\picam
            repo_root = Path(__file__).resolve().parents[2]
            self.pictures_dir = repo_root / "pictures"
        self.interval = interval
        self.camera: Optional[Picamera2] = None
        self.running = False
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Créer le répertoire pictures s'il n'existe pas (avec parents)
        self.pictures_dir.mkdir(parents=True, exist_ok=True)

        self.setup_camera()
    
    def setup_camera(self):
        """Configure la caméra PiCamera2"""
        try:
            self.camera = Picamera2()
            camera_config = self.camera.create_still_configuration(
                main={"size": (1920, 1080)}
            )
            self.camera.configure(camera_config)
            self.camera.start()
            self.logger.info("PiCamera2 initialisée avec succès")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de la caméra: {e}")
            raise
    
    def capture_image(self) -> Optional[Path]:
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
                self.camera.stop()
                    
            self.logger.info("Système de capture arrêté")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'arrêt: {e}")
    
    def get_stats(self):
        """Retourne les statistiques de capture"""
        if not self.pictures_dir.exists():
            return {"total_images": 0, "disk_usage": "0 MB"}
        
        images = list(self.pictures_dir.glob("picam_*.jpg"))
        total_size = sum(img.stat().st_size for img in images)
        
        return {
            "total_images": len(images),
            "disk_usage": f"{total_size / (1024*1024):.1f} MB",
            "data_directory": str(self.pictures_dir.absolute())
        }


def main():
    print("=" * 50)
    print("Système PiCam - Capture automatique")
    print("=" * 50)
    try:
        # Créer l'instance de capture (utilise repo/pictures par défaut)
        picam = PiCamCapture(interval=5)

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