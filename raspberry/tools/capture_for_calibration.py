#!/usr/bin/env python3
"""
Script pour capturer des images pour la calibration de la caméra avec un aperçu en direct.

Ce script démarre un serveur web qui diffuse le flux vidéo de la caméra.
Vous pouvez visualiser ce flux depuis un navigateur à l'adresse http://<ip_raspberry>:5000.

Pour capturer une image, appuyez sur la touche 'Entrée' dans la console où le script est lancé.
Les images sont sauvegardées dans 'output/calibration'.

Appuyez sur Ctrl+C pour arrêter le script.
"""

import configparser
import io
import logging
import sys
import time
from pathlib import Path

# Ajoute le répertoire 'src' au chemin de recherche des modules pour trouver camera_factory
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / "raspberry" / "src"))

from camera.camera_factory import get_camera
from flask import Flask, Response
import threading


def setup_logging():
    """Configure le logging pour l'application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )


app = Flask(__name__)
cam = None
output_path = None

def generate_frames():
    """Générateur pour le flux vidéo."""
    global cam
    while True:
        # Utilise capture_array pour obtenir l'image pour le streaming
        frame = cam.capture_array()
        if frame is not None:
            # cv2.imencode s'attend à du BGR, mais fonctionne souvent avec RGB.
            # Si les couleurs sont inversées dans le stream, décommentez la ligne ci-dessous
            # import cv2
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            import cv2
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05) # Limite le framerate pour ne pas surcharger le CPU

@app.route('/video_feed')
def video_feed():
    """Route pour le streaming vidéo."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Page d'accueil qui affiche le flux vidéo."""
    return """
    <html>
        <head>
            <title>Aperçu pour Calibration</title>
            <style>
                body { font-family: sans-serif; text-align: center; margin-top: 2rem; background-color: #f0f0f0; }
                h1 { color: #333; }
                img { border: 2px solid #ccc; max-width: 90%; height: auto; background-color: #fff; }
            </style>
        </head>
        <body>
            <h1>Aperçu pour Calibration</h1>
            <p>Positionnez le damier et appuyez sur 'Entrée' dans la console pour capturer une image.</p>
            <img src="/video_feed">
        </body>
    </html>
    """

def main():
    """Fonction principale du script de capture."""
    setup_logging()
    logger = logging.getLogger('capture_for_calibration')

    global cam, output_path

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
        output_path = repo_root / "output" / "calibration" # Défini pour la capture d'image
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Les images seront sauvegardées dans : {output_path}")

        # Initialiser la caméra
        logger.info(f"Initialisation de la caméra avec une résolution de {image_width}x{image_height}")
        cam = get_camera(w=image_width, h=image_height, use_fake_camera=False)
        # Pour le streaming, la caméra doit être démarrée si ce n'est pas déjà fait
        if hasattr(cam, 'start'):
             cam.start()
        
        # Démarrer le serveur Flask dans un thread séparé
        flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False), daemon=True)
        flask_thread.start()
        logger.info("="*50)
        logger.info("Serveur de streaming démarré.")
        logger.info("Ouvrez votre navigateur et allez sur http://<ip_du_raspberry>:5000")
        logger.info("="*50)

        logger.info("\nAppuyez sur [Entrée] pour prendre une photo, ou Ctrl+C pour quitter.\n")

        # Boucle pour attendre l'input de l'utilisateur
        while True:
            input() # Attend que l'utilisateur appuie sur Entrée
            logger.info("Capture d'une image...")
            _, filepath = cam.capture_image(pictures_dir=output_path)
            logger.info(f"Image sauvegardée : {filepath.name}")

    except KeyboardInterrupt:
        logger.info("\nArrêt du script par l'utilisateur.")
    except Exception as e:
        logger.error(f"Une erreur fatale est survenue : {e}")
    finally:
        global cam
        if 'cam' in globals() and cam and hasattr(cam, 'stop'):
            cam.stop()
        logger.info("Caméra arrêtée.")

if __name__ == "__main__":
    main()